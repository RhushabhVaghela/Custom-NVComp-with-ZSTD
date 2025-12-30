#!/usr/bin/env python3
"""
test_framework.py - Comprehensive Unit Test Suite for framework.py

This test suite is 100% self-contained and provides full functional coverage
for all logic in framework.py.

It includes:
1.  **TestAssetLoading:** Tests all 5 asset loading paths:
    * Pickle (standard) success
    * Zstandard (compressed) success
    * Missing base_model.pth failure
    * Missing delta_dequantization.pkl failure
    * Corrupted data failure
2.  **TestDeltaModelWrapper:** Tests the core model wrapper:
    * `__init__`: Correctly creates JIT layers and skips 1D tensors.
    * `__init__`: Correctly handles layers with missing delta payloads.
    * `forward`: Tests default key, valid key, and invalid key paths.
    * `get_stats`: Tests all statistics-gathering methods.
3.  **TestAdvancedJITModelFramework:** Tests the high-level orchestration class:
    * `test_all_layers`: Ensures it correctly handles different layer shapes.
    * `benchmark_performance`: Tests both the "run on CUDA" and "skip on CPU" paths.
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
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# --- Force loader path ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up, '../')
parent_dir = os.path.join(script_dir, '..')
# Add the parent directory to Python's search path
sys.path.append(script_dir)
sys.path.append(parent_dir)

try:
    import framework
    from jit_layer import UniversalSmartHybridJITLayer
except ImportError as e:
    print(f"FATAL: Could not import framework or jit_layer. Ensure they are in the same directory.")
    print(f"Error: {e}")
    sys.exit(1)


# ===================================================================
# == 1. TEST ASSET LOADING
# ===================================================================

class TestAssetLoading(unittest.TestCase):
    """
    Tests the `find_and_load_assets` function with all scenarios.
    """
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Define mock data
        self.mock_base_weights = {
            'layer1.weight': torch.randn(10, 10),
            'layer1.bias': torch.randn(10)
        }
        self.mock_delta_payload = {
            'layer1.weight': {'delta': torch.ones(10, 10), 'streaming_available': True}
        }
        
        # Define file paths
        self.base_path = os.path.join(self.temp_dir, "base_model.pth")
        self.delta_path = os.path.join(self.temp_dir, "delta_dequantization.pkl")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_load_success_pickle(self):
        """Tests the happy path: loading a standard pickle payload."""
        print("\n--- TestAssetLoading: test_load_success_pickle ---")
        
        # Save mock files
        torch.save(self.mock_base_weights, self.base_path)
        with open(self.delta_path, 'wb') as f:
            pickle.dump(self.mock_delta_payload, f)
            
        base, delta = framework.find_and_load_assets(self.temp_dir)
        
        self.assertIsInstance(base, dict)
        self.assertIsInstance(delta, dict)
        self.assertEqual(delta['layer1.weight']['delta'][0, 0], 1.0)
        self.assertTrue(torch.allclose(base['layer1.weight'], self.mock_base_weights['layer1.weight']))

    def test_load_success_zstd(self):
        """Tests the zstd fallback path: loading a zstd-compressed pickle."""
        print("\n--- TestAssetLoading: test_load_success_zstd ---")
        
        # Save mock files
        torch.save(self.mock_base_weights, self.base_path)
        
        # Save delta as Zstd-compressed pickle
        cctx = zstd.ZstdCompressor()
        pickled_data = pickle.dumps(self.mock_delta_payload)
        compressed_data = cctx.compress(pickled_data)
        with open(self.delta_path, 'wb') as f:
            f.write(compressed_data)
            
        base, delta = framework.find_and_load_assets(self.temp_dir)
        
        self.assertIsInstance(base, dict)
        self.assertIsInstance(delta, dict)
        self.assertEqual(delta['layer1.weight']['delta'][0, 0], 1.0) # Verifies correct decompression

    def test_load_fail_base_missing(self):
        """Tests failure when base_model.pth is missing."""
        print("\n--- TestAssetLoading: test_load_fail_base_missing ---")
        
        # Save *only* delta file
        with open(self.delta_path, 'wb') as f:
            pickle.dump(self.mock_delta_payload, f)
            
        with self.assertRaises(FileNotFoundError):
            framework.find_and_load_assets(self.temp_dir)

    def test_load_fail_delta_missing(self):
        """Tests failure when delta_dequantization.pkl is missing."""
        print("\n--- TestAssetLoading: test_load_fail_delta_missing ---")
        
        # Save *only* base file
        torch.save(self.mock_base_weights, self.base_path)
            
        with self.assertRaises(FileNotFoundError):
            framework.find_and_load_assets(self.temp_dir)

    def test_load_fail_corrupted_data(self):
        """Tests failure when delta file is corrupted (not pickle or zstd)."""
        print("\n--- TestAssetLoading: test_load_fail_corrupted_data ---")
        
        # Save base file
        torch.save(self.mock_base_weights, self.base_path)
        
        # Save garbage to delta file
        with open(self.delta_path, 'wb') as f:
            f.write(b"this is not pickle data")
            
        # It will raise an exception (e.g., pickle.UnpicklingError)
        with self.assertRaises(Exception):
            framework.find_and_load_assets(self.temp_dir)


# ===================================================================
# == 2. TEST DELTA MODEL WRAPPER
# ===================================================================

class TestDeltaModelWrapper(unittest.TestCase):
    """
    Tests the `DeltaModelWrapper` class logic.
    """
    def setUp(self):
        self.device = 'cpu'
        # Mock base weights: one 2D layer, one 1D bias
        self.base_weights = {
            'layer.2d': torch.randn(10, 10, device=self.device),
            'layer.1d_bias': torch.randn(10, device=self.device),
            'layer.no_delta': torch.randn(5, 5, device=self.device)
        }
        # Mock delta payload, missing 'layer.no_delta'
        self.delta_payload = {
            'layer.2d': {'delta': torch.ones(10, 10), 'streaming_available': True}
        }
        
        # Mock print to suppress warnings during tests
        patcher = patch('builtins.print')
        self.mock_print = patcher.start()
        
        # Initialize the model
        self.model = framework.DeltaModelWrapper(
            self.base_weights, 
            self.delta_payload, 
            self.device
        )
        patcher.stop()

    def test_init_skips_1d_tensors(self):
        """Tests that __init__ correctly skips 1D tensors (biases/norms)."""
        print("\n--- TestDeltaModelWrapper: test_init_skips_1d_tensors ---")
        
        # Check that the 2D layer was converted to a JIT layer
        self.assertIn('layer_2d', self.model.layers)
        self.assertIsInstance(self.model.layers['layer_2d'], UniversalSmartHybridJITLayer)
        
        # Check that the 1D bias was skipped
        self.assertNotIn('layer_1d_bias', self.model.layers)
        
        # Check that the 2D layer *without* a delta was still created
        self.assertIn('layer_no_delta', self.model.layers)

    def test_init_handles_missing_delta_warning(self):
        """Tests that __init__ prints a warning for missing deltas."""
        print("\n--- TestDeltaModelWrapper: test_init_handles_missing_delta_warning ---")
        
        # We re-initialize here with print mocked
        with patch('builtins.print') as mock_print:
            model = framework.DeltaModelWrapper(
                self.base_weights, 
                self.delta_payload, 
                self.device
            )
        
        # Check that a warning was printed for 'layer.no_delta'
        mock_print.assert_any_call(" > WARNING: No delta payload found for layer layer.no_delta.")

    def test_forward_valid_key(self):
        """Tests forward pass with a specific, valid layer key."""
        print("\n--- TestDeltaModelWrapper: test_forward_valid_key ---")
        mock_input = torch.randn(4, 10, device=self.device)
        
        # Patch the JIT layer itself to just return its input
        with patch.object(self.model.layers['layer_2d'], 'forward', return_value=mock_input) as mock_jit:
            output = self.model(mock_input, layer_key='layer_2d')
            mock_jit.assert_called_once()
            self.assertTrue(torch.allclose(output, mock_input))

    def test_forward_default_key(self):
        """Tests forward pass with no key (uses first available layer)."""
        print("\n--- TestDeltaModelWrapper: test_forward_default_key ---")
        mock_input = torch.randn(4, 10, device=self.device)
        
        # The first layer is 'layer_2d'
        with patch.object(self.model.layers['layer_2d'], 'forward', return_value=mock_input) as mock_jit:
            output = self.model(mock_input)
            mock_jit.assert_called_once()
            self.assertTrue(torch.allclose(output, mock_input))

    def test_forward_invalid_key(self):
        """Tests that an invalid key raises a ValueError."""
        print("\n--- TestDeltaModelWrapper: test_forward_invalid_key ---")
        mock_input = torch.randn(4, 10, device=self.device)
        
        with self.assertRaises(ValueError) as cm:
            self.model(mock_input, layer_key='bad_key')
        self.assertIn("Layer bad_key not found", str(cm.exception))

    def test_get_stats_methods(self):
        """Tests that all statistics methods run and return dicts."""
        print("\n--- TestDeltaModelWrapper: test_get_stats_methods ---")
        stats = self.model.get_layer_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('layer_2d', stats)
        
        mem = self.model.get_memory_usage()
        self.assertIsInstance(mem, dict)
        
        stream_stats = self.model.get_streaming_stats()
        self.assertIsInstance(stream_stats, dict)
        self.assertEqual(stream_stats['streaming_events'], 1) # From layer.2d
        
        info = self.model.get_layer_info()
        self.assertIsInstance(info, dict)
        self.assertTrue(info['layer_2d']['has_delta'])
        self.assertFalse(info['layer_no_delta']['has_delta'])


# ===================================================================
# == 3. TEST ADVANCED JIT FRAMEWORK
# ===================================================================

class TestAdvancedJITModelFramework(unittest.TestCase):
    """
    Tests the `AdvancedJITModelFramework` orchestration class.
    """
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock data with a 2D layer and a 1D layer
        self.mock_base_weights = {
            'layer.2d': torch.randn(10, 10),
            'layer.1d_bias': torch.randn(10)
        }
        self.mock_delta_payload = {
            'layer.2d': {'delta': torch.ones(10, 10), 'streaming_available': True}
        }
        
        # Save mock files
        self.base_path = os.path.join(self.temp_dir, "base_model.pth")
        self.delta_path = os.path.join(self.temp_dir, "delta_dequantization.pkl")
        torch.save(self.mock_base_weights, self.base_path)
        with open(self.delta_path, 'wb') as f:
            pickle.dump(self.mock_delta_payload, f)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_test_all_layers_logic(self):
        """Tests `test_all_layers` to ensure it runs on 2D and handles shapes."""
        print("\n--- TestAdvancedJITModelFramework: test_test_all_layers_logic ---")
        # We test on CPU
        adv_framework = framework.AdvancedJITModelFramework(
            self.base_path, self.delta_path, device='cpu'
        )
        
        # The framework's model should have 1 JIT layer (layer.2d)
        # and should have skipped the 1D bias
        self.assertEqual(adv_framework.get_layer_count(), 1)
        
        # Mock the forward pass of the model
        with patch.object(adv_framework.model, 'forward', return_value=torch.randn(2, 10)) as mock_forward:
            results = adv_framework.test_all_layers(batch_size=2)
            
            # Check that it was called
            mock_forward.assert_called_once()
            
            # Check args: The input tensor should have shape (batch_size, in_features)
            # which is (2, 10) for 'layer.2d'
            call_args = mock_forward.call_args[0]
            input_tensor = call_args[0]
            self.assertEqual(input_tensor.shape, (2, 10))
            
            # Check results
            self.assertEqual(results['successful'], 1)
            self.assertEqual(results['failed'], 0)
            self.assertTrue(results['layer_results']['layer_2d']['streaming'])

    def test_benchmark_skips_on_cpu(self):
        """Tests that `benchmark_performance` correctly skips on CPU."""
        print("\n--- TestAdvancedJITModelFramework: test_benchmark_skips_on_cpu ---")
        adv_framework = framework.AdvancedJITModelFramework(
            self.base_path, self.delta_path, device='cpu'
        )
        
        results = adv_framework.benchmark_performance()
        
        # Should return an empty dict
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_benchmark_runs_on_cuda(self):
        """Tests that `benchmark_performance` runs on CUDA."""
        print("\n--- TestAdvancedJITModelFramework: test_benchmark_runs_on_cuda ---")
        adv_framework = framework.AdvancedJITModelFramework(
            self.base_path, self.delta_path, device='cuda'
        )
        
        # Patch the model's forward pass to just return a valid tensor
        mock_output = torch.randn(1, 10, device='cuda')
        with patch.object(adv_framework.model, 'forward', return_value=mock_output):
            results = adv_framework.benchmark_performance(batch_sizes=[1])
            
            # Should return a populated dict
            self.assertIn('batch_1', results)
            self.assertGreater(results['batch_1']['time_ms'], 0)


if __name__ == "__main__":
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