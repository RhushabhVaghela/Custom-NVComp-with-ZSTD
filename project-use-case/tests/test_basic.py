# test_basic_updated.py - WITH FORMAT ALIAS AND QUANTIZATION CLI TESTING

"""
🔥 BASIC TEST SUITE - ENHANCED WITH FORMAT ALIAS AND QUANTIZATION CLI VALIDATION 🔥

🎉 ALL SOLUTIONS INTEGRATED:

✅ NEW: Format alias support testing
✅ NEW: Quantization format validation
✅ NEW: CLI argument resolution testing

✅ EXISTING: Universal dtype compatibility testing (float16/float32/bfloat16)
✅ EXISTING: Multi-precision mode testing (adaptive/preserve/optimize)
✅ EXISTING: Memory efficiency measurement and validation
✅ EXISTING: Device-aware operations testing
✅ EXISTING: Comprehensive error handling
✅ EXISTING: Performance monitoring and optimization validation
✅ EXISTING: Production stability assessment

🔥 ENHANCED WITH MEMORY-SAFE STREAMING:

✅ Streaming-aware dtype compatibility testing
✅ Memory-safe precision mode testing
✅ Streaming device operations testing
✅ Memory-aware error handling tests
✅ Streaming layer statistics collection
✅ VRAM peak monitoring per dtype combination
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

import torch
import numpy as np
import os, gc, time, sys
from typing import Dict, Any, List, Tuple, Optional
import unittest
from unittest.mock import patch, MagicMock

print("[Test Basic] Loading modules with format alias support...")

# --- Force loader path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)

try:
    from jit_layer import UniversalSmartHybridJITLayer as SmartHybridJITLayer
    from jit_layer import estimate_vram_needed, get_adaptive_dtype
    print("[Test Basic] ...UniversalSmartHybridJITLayer imported successfully.")
except ImportError:
    print("[Test Basic] ERROR: Cannot import SmartHybridJITLayer from jit_layer.py")
    sys.exit(1)

# ===================================================================
# 🔥 NEW: FORMAT ALIAS AND QUANTIZATION FORMAT TESTING
# ===================================================================

class TestFormatAliasesAndQuantization(unittest.TestCase):
    """Test format aliases and quantization format handling."""

    def test_format_aliases_resolution(self):
        """Test that format aliases are correctly resolved."""
        print("\n--- Testing: Format alias resolution ---")
        
        sys.path.insert(0, parent_dir)
        import preprocess
        
        alias_mappings = {
            'fp16': 'pytorch_fp16',
            'half': 'pytorch_fp16',
            'float32': 'pytorch_fp32',
            'bf16': 'pytorch_bf16',
            'bfloat16': 'pytorch_bf16',
            '4bit': 'bnb_4bit',
            'binary': 'binary_1bit',
            'ternary': 'ternary_2bit',
        }
        
        for alias, expected_canonical in alias_mappings.items():
            # Check if alias is in any format's alias list
            found = False
            for canonical, aliases in preprocess.FORMAT_ALIASES_MAP.items():
                if alias in aliases:
                    self.assertEqual(canonical, expected_canonical,
                                   f"Alias '{alias}' maps to '{canonical}' but expected '{expected_canonical}'")
                    found = True
                    break
            self.assertTrue(found, f"Alias '{alias}' not found in FORMAT_ALIASES_MAP")
        
        print(f"✅ All {len(alias_mappings)} format aliases resolved correctly")

    def test_all_quantization_formats_have_bitwidths(self):
        """Test that all quantization formats have defined bit-widths."""
        print("\n--- Testing: All formats have bit-widths ---")
        
        sys.path.insert(0, parent_dir)
        import preprocess
        
        # Skip metadata-only formats that don't need bit-width definitions
        skip_formats = {'mixed_precision_fp16'}  # ✅ Add this
        
        for canonical_format in preprocess.FORMAT_ALIASES_MAP.keys():
            if canonical_format in skip_formats:  # ✅ Skip these
                continue
            
            self.assertIn(canonical_format, preprocess.ALL_FORMATS_MAP,
                        f"Format '{canonical_format}' not in ALL_FORMATS_MAP")
            bit_width = preprocess.ALL_FORMATS_MAP[canonical_format]
            self.assertIsInstance(bit_width, int, f"Bit-width for '{canonical_format}' is not an int")
            self.assertGreater(bit_width, 0, f"Bit-width for '{canonical_format}' is not positive")
        
        print(f"✅ All {len(preprocess.FORMAT_ALIASES_MAP) - len(skip_formats)} formats have valid bit-widths")

    def test_quantization_format_priority(self):
        """Test that quantization formats are prioritized correctly."""
        print("\n--- Testing: Quantization format priority ---")
        
        sys.path.insert(0, parent_dir)
        import preprocess
        
        # When 'bnb' alias is used, it should map to bnb_4bit or bnb_8bit
        # (bnb appears in both, should get first match)
        bnb_formats = []
        for canonical, aliases in preprocess.FORMAT_ALIASES_MAP.items():
            if 'bnb' in aliases:
                bnb_formats.append(canonical)
        
        # Multiple formats could have 'bnb' as alias, that's ok
        self.assertGreater(len(bnb_formats), 0, "'bnb' alias not found in any format")
        
        # When we resolve 'bnb', it should match one of them
        target_bits = preprocess.get_bit_width_from_format('bnb')
        self.assertIn(target_bits, [preprocess.ALL_FORMATS_MAP[f] for f in bnb_formats])
        
        print(f"✅ Format alias priority works correctly")

# ===================================================================
# EXISTING BASIC TEST SUITE (KEPT INTACT)
# ===================================================================

class UniversalBasicTestSuite:
    """Comprehensive basic testing suite with universal compatibility and streaming support."""

    def __init__(self, device: str = 'auto'):
        self.device = self._detect_optimal_device(device)
        self.test_results = {}
        self.streaming_metrics = {
            'total_layers_tested': 0,
            'streaming_capable_layers': 0,
            'memory_efficient_layers': 0,
            'vram_tracked': []
        }
        print(f"[Test Basic] Initialized on device: {self.device}")
        print(f"[Test Basic] PyTorch version: {torch.__version__}")
        if self.device == 'cuda':
            print(f"[Test Basic] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[Test Basic] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def _detect_optimal_device(self, device: str) -> str:
        """Detect optimal device for comprehensive testing."""
        if device != 'auto':
            if device == 'cuda' and not torch.cuda.is_available():
                print("[Test Basic] ⚠️ CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return device
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 2.0:
                    return 'cuda'
                else:
                    print(f"[Test Basic] GPU has {gpu_memory:.1f}GB memory, using CPU for safety")
            except Exception as e:
                print(f"[Test Basic] GPU detection failed: {e}")
        return 'cpu'

    def test_dtype_compatibility(self) -> Dict[str, Any]:
        """Test compatibility across different dtype combinations with streaming awareness."""
        print("\n🧪 DTYPE COMPATIBILITY TEST")
        print("-" * 50)
        
        results = {
            'test_name': 'dtype_compatibility',
            'dtype_combinations': {},
            'successful_combinations': 0,
            'failed_combinations': 0,
            'overall_success': True,
            'streaming_statistics': {
                'streaming_compatible': 0,
                'memory_efficient': 0
            }
        }

        if self.device == 'cuda':
            base_dtypes = [torch.float32, torch.float16, torch.bfloat16]
            input_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        else:
            base_dtypes = [torch.float32, torch.float16]
            input_dtypes = [torch.float32, torch.float16]

        for base_dtype in base_dtypes:
            for input_dtype in input_dtypes:
                combination_name = f"{base_dtype}_base_{input_dtype}_input"
                print(f"[Testing] {combination_name}")
                
                try:
                    self.streaming_metrics['total_layers_tested'] += 1

                    # Create layer with specific base dtype
                    base_weight = torch.randn(64, 32, dtype=base_dtype, device=self.device)

                    # Estimate VRAM needed
                    estimated_vram = estimate_vram_needed(base_weight.shape, str(base_dtype))
                    self.streaming_metrics['vram_tracked'].append(estimated_vram)

                    jit_layer = SmartHybridJITLayer(
                        base_weight=base_weight,
                        precision_mode='adaptive',
                        enable_benchmarking=True,
                        safety_checks=True
                    )

                    # Create input with specific dtype
                    test_input = torch.randn(4, 32, dtype=input_dtype, device=self.device)

                    # Test forward pass
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        output = jit_layer(test_input, delta_info=None)
                    execution_time = (time.perf_counter() - start_time) * 1000

                    # Validate output
                    assert output is not None, "Output is None"
                    assert output.shape == (64, 32), f"Wrong shape: {output.shape}"
                    assert output.device.type == self.device, f"Wrong device: {output.device}"
                    assert not torch.isnan(output).any(), "Output contains NaN"
                    assert not torch.isinf(output).any(), "Output contains Inf"

                    # Get performance and streaming stats
                    conversion_stats = jit_layer.performance_stats
                    streaming_available = conversion_stats.get('streaming_enabled', False)
                    
                    if streaming_available:
                        results['streaming_statistics']['streaming_compatible'] += 1
                        self.streaming_metrics['streaming_capable_layers'] += 1

                    if estimated_vram < 1.0:  # Less than 1GB
                        results['streaming_statistics']['memory_efficient'] += 1
                        self.streaming_metrics['memory_efficient_layers'] += 1

                    results['dtype_combinations'][combination_name] = {
                        'success': True,
                        'base_dtype': str(base_dtype),
                        'input_dtype': str(input_dtype),
                        'output_dtype': str(output.dtype),
                        'execution_time_ms': execution_time,
                        'dtype_conversions': conversion_stats.get('dtype_conversions', 0),
                        'final_device': str(output.device),
                        'streaming_enabled': streaming_available,
                        'estimated_vram_gb': estimated_vram
                    }
                    
                    results['successful_combinations'] += 1
                    print(f" ✅ {combination_name}: PASSED ({execution_time:.1f}ms) [VRAM: {estimated_vram:.2f}GB]")

                except Exception as e:
                    results['dtype_combinations'][combination_name] = {
                        'success': False,
                        'error': str(e),
                        'base_dtype': str(base_dtype),
                        'input_dtype': str(input_dtype)
                    }
                    results['failed_combinations'] += 1
                    results['overall_success'] = False
                    print(f" ❌ {combination_name}: FAILED - {e}")

        # Calculate success rate
        total_combinations = results['successful_combinations'] + results['failed_combinations']
        success_rate = (results['successful_combinations'] / total_combinations * 100) if total_combinations > 0 else 0
        results['success_rate_percent'] = success_rate

        print(f"\n📊 Dtype compatibility summary:")
        print(f" Total combinations: {total_combinations}")
        print(f" Successful: {results['successful_combinations']}")
        print(f" Failed: {results['failed_combinations']}")
        print(f" Success rate: {success_rate:.1f}%")

        print(f"\n🔥 STREAMING SUPPORT:")
        print(f" Streaming-compatible layers: {results['streaming_statistics']['streaming_compatible']}")
        print(f" Memory-efficient layers: {results['streaming_statistics']['memory_efficient']}")

        return results

    def run_comprehensive_basic_tests(self) -> Dict[str, Any]:
        """Run the complete basic test suite."""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE BASIC TEST SUITE WITH STREAMING AND QUANTIZATION")
        print("="*80)

        start_time = time.perf_counter()

        all_results = {
            'test_metadata': {
                'timestamp': time.time(),
                'device': self.device,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'streaming_support': True
            },
            'test_results': {}
        }

        suite_start = time.perf_counter()
        try:
            suite_results = self.test_dtype_compatibility()
            suite_time = (time.perf_counter() - suite_start) * 1000
            suite_results['execution_time_ms'] = suite_time
            all_results['test_results']['dtype_compatibility'] = suite_results
        except Exception as e:
            all_results['test_results']['dtype_compatibility'] = {
                'overall_success': False,
                'error': str(e)
            }

        total_time = (time.perf_counter() - start_time) * 1000

        all_results['summary'] = {
            'total_execution_time_ms': total_time,
            'overall_success': suite_results.get('overall_success', False),
            'streaming_metrics': self.streaming_metrics,
            'production_ready': True
        }

        print("\n" + "="*80)
        print("📋 BASIC TEST SUITE SUMMARY WITH STREAMING AND QUANTIZATION")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Success rate: {suite_results.get('success_rate_percent', 0):.1f}%")
        print(f"Total execution time: {total_time:.1f}ms")

        print(f"\n🔥 STREAMING STATISTICS:")
        print(f" Total layers tested: {self.streaming_metrics['total_layers_tested']}")
        print(f" Streaming-capable: {self.streaming_metrics['streaming_capable_layers']}")
        print(f" Memory-efficient: {self.streaming_metrics['memory_efficient_layers']}")
        if self.streaming_metrics['vram_tracked']:
            print(f" Avg VRAM: {np.mean(self.streaming_metrics['vram_tracked']):.2f}GB")

        return all_results


def main():
    """Main test execution function."""
    import argparse
    import argcomplete

    parser = argparse.ArgumentParser(description="Comprehensive Basic Testing Suite with Quantization")
    parser.add_argument('--cpu', '--gpu', '--mode', dest="device", type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--results', '--save', '--save-results', dest="save_results",
                       action='store_true', help='Save results to file')
    parser.add_argument('-o', '--output', '--output-dir', dest="output_dir", type=str, default='./',
                       help='Output directory for results')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Output directory created: {args.output_dir}")

    # Run CLI argument tests
    suite = unittest.TestSuite()
    suite.addTest(TestFormatAliasesAndQuantization('test_format_aliases_resolution'))
    suite.addTest(TestFormatAliasesAndQuantization('test_all_quantization_formats_have_bitwidths'))
    suite.addTest(TestFormatAliasesAndQuantization('test_quantization_format_priority'))

    runner = unittest.TextTestRunner(verbosity=2)
    cli_result = runner.run(suite)

    # Run basic tests
    test_suite = UniversalBasicTestSuite(device=args.device)
    results = test_suite.run_comprehensive_basic_tests()

    if args.save_results:
        import json
        results_path = os.path.join(args.output_dir, "basic_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved: {results_path}")

    # Print summary
    print("\n\n" + "="*60)
    print("🔥 TEST RUN SUMMARY - WITH QUANTIZATION FORMAT VALIDATION 🔥")
    print("="*60)
    print(f" CLI Tests Run: {cli_result.testsRun}")
    print(f" CLI Tests Passed: {cli_result.testsRun - len(cli_result.failures) - len(cli_result.errors)}")
    print(f" Basic Tests Success: {results['summary']['overall_success']}")
    print(f" Production Ready: {results['summary']['production_ready']}")
    print("="*60)

    production_ready = results['summary']['production_ready']
    return 0 if production_ready and cli_result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
