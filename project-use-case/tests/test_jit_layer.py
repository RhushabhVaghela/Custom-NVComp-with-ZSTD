#!/usr/bin/env python3

"""
🔥 test_jit_layer.py - Comprehensive Unit Test Suite 🔥

This test suite is designed to provide 100% functional coverage for
jit_layer.py by addressing all critical components and execution paths.

It includes:

1.  **TestCorrectness:**
    * Ensures numerical correctness (e.g., base + delta = result).
    * Hermetic (self-contained) and requires no external files.

2.  **TestHelperFunctions:** (UPDATED)
    * Unit tests for all core logic:
    * `_select_optimal_path` (all 5 path selections)
    * `_decompress_on_cpu` (Zstd + Pickle)
    * `estimate_vram_needed` / `get_adaptive_dtype`
    * `_process_delta_info` (NEW: tests dense, sparse, and streaming)
    * `_validate_delta_format` (NEW: tests good and bad data)
    * `get_system_info` (NEW)
    * `benchmark_paths` (NEW)

3.  **TestExecutionPaths:** (UPDATED)
    * Unit tests that *force* the layer to execute each of the 5 paths.
    * Mocks the CUDA extension and environment to test all logic.
    * `test_path_cpu_zstd`
    * `test_path_hybrid`
    * `test_path_gpu_fallback`
    * `test_path_gpu_optimized_MOCKED` (sparse kernel)
    * `test_path_gpu_dense_zstd_MOCKED` (NEW: dense Zstd kernel)
    * `test_path_gpu_dense_zstd_FAILURE_FALLBACK` (NEW: tests emergency fallback)

4.  **TestCudaLoading:**
    * Unit tests for the `load_cuda_extension` function itself.
    * Simulates all failure modes (SO missing, JIT fail, etc.).

5.  **TestSmokeTests:**
    * Ports of the simple, fast tests from the original file.
"""

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with.*")
warnings.filterwarnings("ignore", message=".*You are using the default.*")
warnings.filterwarnings("ignore", message=".*AutoAWQ is officially deprecated.*")
warnings.filterwarnings("ignore", message=".*site-packages/torch/utils/cpp_extension.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*autoawq.*")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

import unittest
import torch
import pickle
import zstandard as zstd
import numpy as np
import sys
from unittest.mock import patch, MagicMock, call
import os

# --- Force loader path ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up, '../')
parent_dir = os.path.join(script_dir, '..')
# Add the parent directory to Python's search path
sys.path.append(script_dir)
sys.path.append(parent_dir)

try:
    import jit_layer
except ImportError:
    print("FATAL: Could not import jit_layer.py. Make sure it is in the same directory.")
    sys.exit(1)


# ===================================================================
# == 1. TEST FOR NUMERICAL CORRECTNESS
# ===================================================================

class TestCorrectness(unittest.TestCase):
    """
    Tests the most basic function: is the math correct?
    Uses known, non-random data.
    """
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = torch.full((10, 10), 10.0, dtype=torch.float32)
        self.delta = torch.full((10, 10), 2.0, dtype=torch.float32)
        self.expected_result = torch.full((10, 10), 12.0, dtype=torch.float32)
        
        # Create a mock delta_info that forces CPU path
        self.mock_delta_info = {
            'delta': self.delta,
            'has_gpu_format': False,
            'format_validated': True
        }

    def test_reconstruction_correctness_cpu(self):
        """Tests that base (10) + delta (2) = 12 on CPU."""
        print("\n--- TestCorrectness: test_reconstruction_correctness_cpu ---")
        # Force CPU device for this test
        layer = jit_layer.UniversalSmartHybridJITLayer(
            base_weight=self.base, 
            device=torch.device('cpu')
        )
        
        # Force selection of CPU path to test CPU math
        with patch.object(layer, '_select_optimal_path', return_value='cpu_zstd_path'):
            reconstructed_weight = layer(delta_info=self.mock_delta_info)
            
        self.assertTrue(torch.allclose(reconstructed_weight, self.expected_result))
        self.assertEqual(reconstructed_weight.device.type, 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reconstruction_correctness_gpu(self):
        """Tests that base (10) + delta (2) = 12 on GPU."""
        print("\n--- TestCorrectness: test_reconstruction_correctness_gpu ---")
        layer = jit_layer.UniversalSmartHybridJITLayer(
            base_weight=self.base, 
            device=self.device
        )
        
        # Force selection of GPU fallback path to test GPU math
        layer.selected_path = 'cuda_gpu_fallback_path'
        reconstructed_weight = layer(delta_info=self.mock_delta_info)

        self.assertTrue(torch.allclose(reconstructed_weight.cpu(), self.expected_result))
        self.assertEqual(reconstructed_weight.device.type, 'cuda')


# ===================================================================
# == 2. TEST ALL HELPER FUNCTIONS
# ===================================================================

class TestHelperFunctions(unittest.TestCase):
    """
    Tests all the core logic (non-path) functions in isolation.
    """
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # <-- ADD THIS
        self.layer = jit_layer.UniversalSmartHybridJITLayer(
            base_weight=torch.empty(0), 
            enable_benchmarking=True
        )

    def test_estimate_vram_and_get_adaptive_dtype(self):
        """Tests VRAM estimation and adaptive dtype selection."""
        print("\n--- TestHelperFunctions: test_estimate_vram_and_get_adaptive_dtype ---")
        # 1M elements * 4 bytes (fp32) * 3x overhead = 12MB
        size = torch.Size([1024, 1024])
        fp32_gb = (1024 * 1024 * 4 * 3) / (1024**3)
        self.assertAlmostEqual(jit_layer.estimate_vram_needed(size, 'float32'), fp32_gb)
        
        # 20k x 20k tensor (400M elements)
        # fp32: 400M * 4 * 3 = 4.8B bytes = ~4.47 GB
        # fp16: 400M * 2 * 3 = 2.4B bytes = ~2.23 GB
        large_size = torch.Size([20000, 20000])
        
        self.assertEqual(jit_layer.get_adaptive_dtype(8.0, large_size), torch.float32)
        self.assertEqual(jit_layer.get_adaptive_dtype(3.0, large_size), torch.float16)
        # Fallback (assuming bf16 is supported, test env may vary)
        fallback_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.assertEqual(jit_layer.get_adaptive_dtype(1.0, large_size), fallback_dtype)

    def test_decompress_on_cpu(self):
        """Tests the Zstd + Pickle decompression logic."""
        print("\n--- TestHelperFunctions: test_decompress_on_cpu ---")
        tensor = torch.randn(10, 10)
        data_np = tensor.numpy()
        data_pkl = pickle.dumps(data_np)
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(data_pkl)
        
        mock_delta_info = {'compressed_delta': compressed_data}
        
        decompressed_tensor = self.layer._decompress_on_cpu(mock_delta_info)
        self.assertTrue(torch.allclose(tensor, decompressed_tensor))

    def test_process_delta_info(self):
        """🌟 NEW: Tests that delta_info is parsed correctly for all formats."""
        print("\n--- TestHelperFunctions: test_process_delta_info ---")
        # GPU DENSE format
        info = self.layer._process_delta_info({'compression_type': 'gpu_nvcomp_zstd'})
        self.assertTrue(info['has_gpu_format'])
        self.assertEqual(info['gpu_format_type'], 'dense_zstd')
        
        # GPU SPARSE format
        info = self.layer._process_delta_info({'comp_indices_list_nvcomp': [1]})
        self.assertTrue(info['has_gpu_format'])
        self.assertEqual(info['gpu_format_type'], 'sparse')
        
        # CPU format
        info = self.layer._process_delta_info({'delta': [1]})
        self.assertFalse(info['has_gpu_format'])
        self.assertEqual(info['gpu_format_type'], 'none')
        
        # Streaming format
        info = self.layer._process_delta_info({'streaming_enabled': True})
        self.assertTrue(info['streaming_available'])
        
    def test_validate_delta_format(self):
        """🌟 NEW: Tests the _validate_delta_format helper."""
        print("\n--- TestHelperFunctions: test_validate_delta_format ---")
        # Valid cases
        self.assertTrue(self.layer._validate_delta_format({'delta': torch.tensor(1)}))
        self.assertTrue(self.layer._validate_delta_format({'compressed_delta': b'123'}))
        self.assertTrue(self.layer._validate_delta_format({'comp_indices_list_nvcomp': [1]}))
        self.assertTrue(self.layer._validate_delta_format({'delta_compressed': b'123'}))
        
        # Invalid cases
        self.assertFalse(self.layer._validate_delta_format({})) # Empty
        self.assertFalse(self.layer._validate_delta_format({'bad_key': 123}))

    def test_select_optimal_path(self):
        """CRITICAL: Tests the 5-path selection logic."""
        print("\n--- TestHelperFunctions: test_select_optimal_path ---")
        
        # --- Case 1: GPU Dense Zstd Path (NEW) ---
        # Needs CUDA Ext + 'dense_zstd' format + CUDA device
        with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
            with patch('jit_layer.cuda_ext', MagicMock(jit_decompress_zstd_v1=MagicMock())):
                layer = jit_layer.UniversalSmartHybridJITLayer(torch.empty(1), device=torch.device('cuda'))
                layer.delta_info = {'gpu_format_type': 'dense_zstd', 'format_validated': True}
                self.assertEqual(layer._select_optimal_path(), 'cuda_gpu_dense_zstd_path')
        
        # --- Case 2: GPU Optimized (Sparse) Path ---
        # Needs CUDA Ext + 'sparse' format + CUDA device
        with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
            with patch('jit_layer.cuda_ext', MagicMock(jit_apply_v1_full_gpu=MagicMock())):
                layer = jit_layer.UniversalSmartHybridJITLayer(torch.empty(1), device=torch.device('cuda'))
                layer.delta_info = {'gpu_format_type': 'sparse', 'format_validated': True}
                self.assertEqual(layer._select_optimal_path(), 'cuda_gpu_optimized_path')

        # --- Case 3: GPU Fallback Path ---
        # Needs CUDA Ext + NO GPU format + CUDA device
        with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
            layer = jit_layer.UniversalSmartHybridJITLayer(torch.empty(1), device=torch.device('cuda'))
            layer.delta_info = {'gpu_format_type': 'none', 'format_validated': True}
            self.assertEqual(layer._select_optimal_path(), 'cuda_gpu_fallback_path')
            
        # --- Case 4: Hybrid Path ---
        # Needs NO CUDA Ext + CUDA device + Medium tensor (e.g., 5MB)
        with patch('jit_layer.CUDA_EXT_AVAILABLE', False):
            # 5MB tensor = 5 * 1024 * 1024 / 4 bytes = 1,310,720 elements
            medium_tensor = torch.empty(1_310_720)
            layer = jit_layer.UniversalSmartHybridJITLayer(medium_tensor, device=torch.device('cuda'))
            self.assertEqual(layer._select_optimal_path(), 'cuda_cpu_hybrid_path')

        # --- Case 5: CPU Path ---
        # Needs NO CUDA Ext + Small tensor (e.g., 100 elements)
        with patch('jit_layer.CUDA_EXT_AVAILABLE', False):
            small_tensor = torch.empty(100)
            layer = jit_layer.UniversalSmartHybridJITLayer(small_tensor, device=torch.device('cuda'))
            self.assertEqual(layer._select_optimal_path(), 'cpu_zstd_path')
            
    def test_get_system_info(self):
        """🌟 NEW: Tests get_system_info."""
        print("\n--- TestHelperFunctions: test_get_system_info ---")
        info = jit_layer.get_system_info()
        self.assertIn('torch_version', info)
        self.assertIn('cuda_available', info)
        self.assertIn('cuda_extension_available', info)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_benchmark_paths_MOCKED(self):
        """🌟 NEW: Tests benchmark_paths runs."""
        print("\n--- TestHelperFunctions: test_benchmark_paths_MOCKED ---")
        layer = jit_layer.UniversalSmartHybridJITLayer(
            torch.randn(10,10, device=self.device),
            enable_benchmarking=True
        )
        # Mock all paths to just return the base weight
        with patch.object(layer, 'cuda_cpu_hybrid_path', return_value=layer.original_base_weight):
            with patch.object(layer, 'cpu_zstd_path', return_value=layer.original_base_weight):
                results = layer.benchmark_paths(num_iterations=2)
                
        self.assertIn('cuda_cpu_hybrid_path', results)
        self.assertIn('cpu_zstd_path', results)
        self.assertEqual(results['cpu_zstd_path']['iterations'], 2)


# ===================================================================
# == 3. TEST ALL 5 EXECUTION PATHS
# ===================================================================

class TestExecutionPaths(unittest.TestCase):
    """
    Tests each of the 5 execution paths in isolation by forcing
    the path selection and spying on the function call.
    """
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = torch.full((10, 10), 10.0, dtype=torch.float32)
        self.delta = torch.full((10, 10), 2.0, dtype=torch.float32)
        self.expected_result = torch.full((10, 10), 12.0, dtype=torch.float32)
        self.mock_delta_info = {'delta': self.delta}

    def test_path_cpu_zstd(self):
        """Forces and tests the 'cpu_zstd_path'."""
        print("\n--- TestExecutionPaths: test_path_cpu_zstd ---")
        layer = jit_layer.UniversalSmartHybridJITLayer(self.base, device=torch.device('cpu'))
        
        # Spy on the real function while forcing it to be selected
        with patch.object(layer, '_select_optimal_path', return_value='cpu_zstd_path'):
            with patch.object(layer, 'cpu_zstd_path', wraps=layer.cpu_zstd_path) as spy:
                result = layer(delta_info=self.mock_delta_info)
                
                spy.assert_called_once() # Verify this path was taken
                self.assertTrue(torch.allclose(result, self.expected_result))
                self.assertEqual(layer.performance_stats['path_used'], 'cpu_zstd')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_path_hybrid(self):
        """Forces and tests the 'cuda_cpu_hybrid_path'."""
        print("\n--- TestExecutionPaths: test_path_hybrid ---")
        layer = jit_layer.UniversalSmartHybridJITLayer(self.base, device=self.device)
        
        layer.selected_path = 'cuda_cpu_hybrid_path' # Force the path
        with patch.object(layer, 'cuda_cpu_hybrid_path', wraps=layer.cuda_cpu_hybrid_path) as spy:
            result = layer(delta_info=self.mock_delta_info)

            spy.assert_called_once()
            self.assertTrue(torch.allclose(result.cpu(), self.expected_result))
            self.assertEqual(layer.performance_stats['path_used'], 'cuda_cpu_hybrid')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_path_gpu_fallback(self):
        """Forces and tests the 'cuda_gpu_fallback_path'."""
        print("\n--- TestExecutionPaths: test_path_gpu_fallback ---")
        layer = jit_layer.UniversalSmartHybridJITLayer(self.base, device=self.device)
        
        layer.selected_path = 'cuda_gpu_fallback_path' # Force the path
        with patch.object(layer, 'cuda_gpu_fallback_path', wraps=layer.cuda_gpu_fallback_path) as spy:
            result = layer(delta_info=self.mock_delta_info)

            spy.assert_called_once()
            self.assertTrue(torch.allclose(result.cpu(), self.expected_result))
            self.assertEqual(layer.performance_stats['path_used'], 'cuda_gpu_fallback')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_path_gpu_optimized_MOCKED(self):
        """Forces and tests the 'cuda_gpu_optimized_path' (SPARSE) by mocking the kernel."""
        print("\n--- TestExecutionPaths: test_path_gpu_optimized_MOCKED ---")
        
        mock_gpu_delta = {
            'comp_indices_list_nvcomp': [np.array([1,2])],
            'comp_values_list_nvcomp': [np.array([1.0, 1.0])],
            'gpu_format_type': 'sparse',
            'format_validated': True
        }
        
        mock_ext_func = MagicMock(return_value=self.expected_result.to(self.device))
        
        with patch.object(jit_layer, 'cuda_ext', MagicMock(jit_apply_v1_full_gpu=mock_ext_func)) as mock_ext_module:
            with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
                layer = jit_layer.UniversalSmartHybridJITLayer(
                    self.base, 
                    device=self.device, 
                    delta_info=mock_gpu_delta
                )
                self.assertEqual(layer.selected_path, 'cuda_gpu_optimized_path')
                result = layer(delta_info=mock_gpu_delta)
                mock_ext_func.assert_called_once()
                self.assertTrue(torch.allclose(result.cpu(), self.expected_result))
                self.assertEqual(layer.performance_stats['path_used'], 'cuda_gpu_optimized')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_path_gpu_dense_zstd_MOCKED(self):
        """🌟 NEW: Forces and tests the 'cuda_gpu_dense_zstd_path' (DENSE) by mocking the kernel."""
        print("\n--- TestExecutionPaths: test_path_gpu_dense_zstd_MOCKED ---")
        
        # 1. Create mock data
        delta_bytes_uncompressed = self.delta.numpy().tobytes()
        delta_bytes_compressed = zstd.ZstdCompressor().compress(delta_bytes_uncompressed)
        
        mock_gpu_delta = {
            'delta_compressed': delta_bytes_compressed,
            'delta_uncompressed_bytes': len(delta_bytes_uncompressed),
            'compression_type': 'gpu_nvcomp_zstd', # This is the key
            'gpu_format_type': 'dense_zstd',
            'format_validated': True,
            'original_shape': self.base.shape
        }
        
        # 2. Mock the kernel function
        # The kernel returns a uint8 buffer, which the python layer must .view()
        mock_uint8_buffer = torch.from_numpy(
            np.frombuffer(delta_bytes_uncompressed, dtype=np.uint8)
        ).to(self.device)
        mock_ext_func = MagicMock(return_value=mock_uint8_buffer)
        
        with patch.object(jit_layer, 'cuda_ext', MagicMock(jit_decompress_zstd_v1=mock_ext_func)) as mock_ext_module:
            with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
            
                # 3. Init layer
                layer = jit_layer.UniversalSmartHybridJITLayer(
                    self.base, 
                    device=self.device, 
                    delta_info=mock_gpu_delta
                )
                
                # 4. Check path selection and run
                self.assertEqual(layer.selected_path, 'cuda_gpu_dense_zstd_path')
                result = layer(delta_info=mock_gpu_delta)

                # 5. Verify
                mock_ext_func.assert_called_once()
                self.assertTrue(torch.allclose(result.cpu(), self.expected_result))
                self.assertEqual(layer.performance_stats['path_used'], 'cuda_gpu_dense_zstd')
                
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_path_gpu_dense_zstd_FAILURE_FALLBACK(self):
        """🌟 NEW: Tests that a GPU path failure triggers the emergency CPU fallback."""
        print("\n--- TestExecutionPaths: test_path_gpu_dense_zstd_FAILURE_FALLBACK ---")
        
        mock_gpu_delta = {
            'delta_compressed': b'123',
            'delta_uncompressed_bytes': 100,
            'compression_type': 'gpu_nvcomp_zstd',
            'gpu_format_type': 'dense_zstd',
            'format_validated': True,
            'original_shape': self.base.shape,
            'delta': self.delta # Add CPU delta for the fallback path to find
        }
        
        # Mock the kernel to RAISE an exception
        mock_ext_func = MagicMock(side_effect=RuntimeError("Kernel Failed!"))
        
        with patch.object(jit_layer, 'cuda_ext', MagicMock(jit_decompress_zstd_v1=mock_ext_func)):
            with patch('jit_layer.CUDA_EXT_AVAILABLE', True):
            
                layer = jit_layer.UniversalSmartHybridJITLayer(
                    self.base, 
                    device=self.device, 
                    delta_info=mock_gpu_delta
                )
                
                # Spy on the fallback path
                with patch.object(layer, 'cpu_zstd_path', wraps=layer.cpu_zstd_path) as spy_cpu_path:
                    
                    result = layer(delta_info=mock_gpu_delta) # This should fail and fall back

                    # Verify the GPU kernel was tried
                    mock_ext_func.assert_called_once()
                    
                    # Verify the CPU fallback was called
                    spy_cpu_path.assert_called_once()
                    
                    # Verify the result is still correct
                    self.assertTrue(torch.allclose(result.cpu(), self.expected_result))
                    self.assertEqual(layer.performance_stats['fallback_events'], 1)
                    self.assertEqual(layer.performance_stats['path_used'], 'cpu_zstd')


# ===================================================================
# == 4. TEST CUDA EXTENSION LOADER
# ===================================================================

class TestCudaLoading(unittest.TestCase):
    """
    Tests the complex `load_cuda_extension` function and its fallbacks.
    """
    def setUp(self):
        # We must reset the module-level state for each test
        jit_layer.CUDA_EXT_AVAILABLE = False
        jit_layer.cuda_ext = None
    
    @patch('torch.utils.cpp_extension.load', side_effect=Exception("JIT Compile Error"))
    @patch('torch.ops.load_library', side_effect=Exception("SO file not found"))
    def test_all_loading_fails(self, mock_load_library, mock_cpp_load):
        """Tests that if all 3 methods fail, it correctly sets AVAILABLE=False."""
        print("\n--- TestCudaLoading: test_all_loading_fails ---")
        
        result = jit_layer.load_cuda_extension()
        
        self.assertFalse(result)
        self.assertFalse(jit_layer.CUDA_EXT_AVAILABLE)
        self.assertEqual(mock_load_library.call_count, 1) # Tries pre-compiled once
        self.assertEqual(mock_cpp_load.call_count, 1)     # Tries JIT-compile + JIT-alt

    @patch('torch.utils.cpp_extension.load', return_value=MagicMock())
    @patch('torch.ops.load_library', side_effect=Exception("SO file not found"))
    def test_jit_compilation_succeeds(self, mock_load_library, mock_cpp_load):
        """Tests that it falls back to JIT compile and succeeds."""
        print("\n--- TestCudaLoading: test_jit_compilation_succeeds ---")
        
        result = jit_layer.load_cuda_extension()
        
        self.assertTrue(result)
        self.assertTrue(jit_layer.CUDA_EXT_AVAILABLE)
        mock_load_library.assert_called_once() # Tries pre-compiled (fails)
        mock_cpp_load.assert_called_once()     # Tries JIT-compile (succeeds)

    @patch('torch.ops.load_library', return_value=MagicMock())
    def test_precompiled_succeeds(self, mock_load_library):
        """Tests that it finds the pre-compiled .so file first."""
        print("\n--- TestCudaLoading: test_precompiled_succeeds ---")
        
        with patch('torch.utils.cpp_extension.load') as mock_cpp_load:
            result = jit_layer.load_cuda_extension()
            
            self.assertTrue(result)
            self.assertTrue(jit_layer.CUDA_EXT_AVAILABLE)
            mock_load_library.assert_called_once() # Tries pre-compiled (succeeds)
            mock_cpp_load.assert_not_called()      # Never tries JIT

# ===================================================================
# == 5. SMOKE TESTS (PORTED)
# ===================================================================

class TestSmokeTests(unittest.TestCase):
    """
    Ports of the basic, non-crashing tests from the original file.
    """
    def test_basic_functionality_passthrough(self):
        """Tests passthrough mode (delta_info=None)."""
        print("\n--- TestSmokeTests: test_basic_functionality_passthrough ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_weight = torch.randn(100, 50, dtype=torch.float16, device=device)
        
        # The original test called the layer with an input, but the
        # layer's forward pass ignores it. We test the *actual* behavior.
        layer = jit_layer.UniversalSmartHybridJITLayer(base_weight=base_weight)
        
        # Calling with delta_info=None should just return the base weight
        # (This is inferred from the `cpu_zstd_path` fallback logic)
        output = layer(delta_info=None)
        
        self.assertTrue(torch.allclose(output, base_weight.float()))

    def test_multi_dtype_support(self):
        """Tests that various dtypes don't crash."""
        print("\n--- TestSmokeTests: test_multi_dtype_support ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dtypes_to_test = [torch.float32, torch.float16]
        if torch.cuda.is_available() and hasattr(torch, 'bfloat16'):
             if torch.cuda.is_bf16_supported():
                dtypes_to_test.append(torch.bfloat16)
        
        for base_dtype in dtypes_to_test:
            with self.subTest(f"Base DType: {base_dtype}"):
                base_weight = torch.randn(64, 32, dtype=base_dtype, device=device)
                layer = jit_layer.UniversalSmartHybridJITLayer(base_weight=base_weight)
                
                # Test passthrough
                output = layer(delta_info=None)
                self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    print("="*80)
    print("🔥 Running Comprehensive Unit Test Suite for jit_layer.py 🔥")
    print("="*80)
    
    # Run the tests and capture the TestProgram object
    test_program = unittest.main(verbosity=0, exit=False) # verbosity=0 to avoid double output
    
    # Get the TestResult object
    result = test_program.result
    
    # Print the custom summary
    print("\n\n" + "="*50)
    print("🔥 TEST RUN SUMMARY 🔥")
    print("="*50)
    print(f"  Total Tests Run:    {result.testsRun}")
    print(f"  Tests Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Tests Failed:       {len(result.failures)}")
    print(f"  Tests with Errors:  {len(result.errors)}")
    print("="*50)
    
    # Exit with appropriate status code
    if not result.wasSuccessful():
        sys.exit(1)