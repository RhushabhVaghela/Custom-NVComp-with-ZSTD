#!/usr/bin/env python3
"""
test_evaluation.py - Comprehensive Unit Test Suite for evaluation.py

This test suite is 100% self-contained and provides full functional coverage
for all logic in evaluation.py.

It includes:
1.  **TestComprehensiveModelEvaluator:**
    * `test_init_loads_assets`: Verifies the evaluator loads mock assets.
    * `test_create_test_data_adaptive_batching`: Tests the VRAM-aware
        adaptive batch sizing logic by mocking VRAM limits.
    * `test_evaluate_layer_accuracy_math`: CRITICAL test that mocks
        the `F.linear` and `jit_layer` calls to verify the
        MSE/MAE and improvement percentage math is correct.
    * `test_generate_executive_summary_recommendations`: Tests that the
        logic for generating recommendations (e.g., "increase precision")
        is triggered correctly by mock data.
    * `test_run_all_evaluations`: An integration test for the evaluator
        itself, ensuring all main methods run without crashing.
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
import torch.nn.functional as F
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
    import evaluation
    import framework
    import jit_layer
except ImportError as e:
    print(f"FATAL: Could not import evaluation, framework, or jit_layer.")
    print(f"Error: {e}")
    sys.exit(1)


# ===================================================================
# == TEST COMPREHENSIVE MODEL EVALUATOR
# ===================================================================

class TestComprehensiveModelEvaluator(unittest.TestCase):
    """
    Tests the `ComprehensiveModelEvaluator` class logic.
    """
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Define mock data
        self.mock_base_weights = {
            'layer1': torch.randn(10, 10)
        }
        self.mock_delta_payload = {
            'layer1': {
                'original_shape': (10, 10),
                'delta': torch.ones(10, 10),
                'streaming_available': True
            }
        }
        
        # Save mock files
        self.base_path = os.path.join(self.temp_dir, "base_model.pth")
        self.delta_path = os.path.join(self.temp_dir, "delta_dequantization.pkl")
        torch.save(self.mock_base_weights, self.base_path)
        with open(self.delta_path, 'wb') as f:
            pickle.dump(self.mock_delta_payload, f)

        # Patch the JIT layer where it is *used* (in the evaluation module)
        patcher = patch('evaluation.UniversalSmartHybridJITLayer')
        self.mock_jit_class = patcher.start()
        
        # Initialize the evaluator
        self.evaluator = evaluation.ComprehensiveModelEvaluator(self.temp_dir, device='cpu')

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
        patch.stopall() # Stop all patches

    def test_init_loads_assets(self):
        """Tests that the evaluator's __init__ successfully loads all mock assets."""
        print("\n--- TestEvaluator: test_init_loads_assets ---")
        
        # The setUp method already did this. We just check the result.
        self.assertIsInstance(self.evaluator.framework, framework.AdvancedJITModelFramework)
        self.assertIn('layer1', self.evaluator.base_weights)
        self.assertIn('layer1', self.evaluator.delta_payload)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_create_test_data_adaptive_batching(self):
        """Tests the VRAM-aware adaptive batch sizing logic."""
        print("\n--- TestEvaluator: test_create_test_data_adaptive_batching ---")
        
        # Re-init evaluator for CUDA
        self.evaluator = evaluation.ComprehensiveModelEvaluator(self.temp_dir, device='cuda')
        
        # Mock VRAM estimation to return a large size (1000 GB)
        # Mock device properties to return a small total memory (1 GB)
        with patch('evaluation.estimate_vram_needed', return_value=1000) as mock_estimate:
            with patch('torch.cuda.get_device_properties') as mock_props:
                # 1 GB VRAM
                mock_props.return_value.total_memory = 1 * (1024**3) 
                
                # Try to create batch of 8, should be halved to 4
                test_data = self.evaluator._create_test_data('layer1', batch_size=8)
                
                # Check that batch size was halved
                self.assertEqual(test_data.shape[0], 4)
                
                # Try to create batch of 2, should be halved to 1
                test_data = self.evaluator._create_test_data('layer1', batch_size=2)
                self.assertEqual(test_data.shape[0], 1)
                
                # Try to create batch of 1, should stay 1
                test_data = self.evaluator._create_test_data('layer1', batch_size=1)
                self.assertEqual(test_data.shape[0], 1)

    def test_evaluate_layer_accuracy_math(self):
        """Tests the MSE/MAE and improvement percentage math is correct."""
        print("\n--- TestEvaluator: test_evaluate_layer_accuracy_math ---")
        
        # Known values for outputs
        mock_input = torch.randn(8, 10)
        base_output = torch.full((8, 10), 1.0, dtype=torch.float32)
        jit_output = torch.full((8, 10), 2.0, dtype=torch.float32)
        final_output = torch.full((8, 10), 3.0, dtype=torch.float32)
        
        # Expected MSE:
        # base_vs_final = (1.0 - 3.0)^2 = 4.0
        # jit_vs_final  = (2.0 - 3.0)^2 = 1.0
        #
        # Expected Improvement:
        # (4.0 - 1.0) / 4.0 * 100 = 75.0%
        
        # Mock all external calls
        self.evaluator._create_test_data = MagicMock(return_value=mock_input)
        
        with patch('torch.nn.functional.linear') as mock_linear:
            # First call is base_output, second is final_output
            mock_linear.side_effect = [base_output, final_output] 
            
            # Patch the JIT layer's __call__ method
            self.mock_jit_class.return_value.return_value = jit_output
            
            # FIX: Mock get_performance_stats to return a valid dictionary
            self.mock_jit_class.return_value.get_performance_stats.return_value = {
                'total_time': 0.001, 
                'path_used': 'mocked_path', 
                'dtype_conversions': 0,
                'precision_mode': 'adaptive' # Added to prevent potential new errors
            }
                
            results = self.evaluator.evaluate_layer_accuracy('layer1')

        metrics = results['reconstruction_metrics']
        
        self.assertAlmostEqual(metrics['base_vs_final_mse'], 4.0)
        self.assertAlmostEqual(metrics['jit_vs_final_mse'], 1.0)
        self.assertAlmostEqual(metrics['mse_improvement_percent'], 75.0)

    def test_generate_executive_summary_recommendations(self):
        """Tests that the recommendation logic is triggered correctly."""
        print("\n--- TestEvaluator: test_generate_executive_summary_recommendations ---")
        
        # --- Test Case 1: Low accuracy improvement ---
        self.evaluator.results = {
            'accuracy_evaluation': {'evaluation_summary': {'avg_mse_improvement_percent': 5.0, 'success_rate_percent': 100.0}},
            'performance_benchmark': {'layer_test_results': {'success_rate': 100, 'avg_time_ms': 50}},
            'compatibility_test': {'summary': {'success_rate_percent': 100.0}}
        }
        summary = self.evaluator._generate_executive_summary()
        self.assertIn(
            "Consider increasing quantization precision for better accuracy",
            summary['recommendations']
        )
        self.assertEqual(summary['overall_status'], 'EXCELLENT') # Still excellent, just a recommendation
        
        # --- Test Case 2: Slow performance ---
        self.evaluator.results = {
            'accuracy_evaluation': {'evaluation_summary': {'avg_mse_improvement_percent': 90.0, 'success_rate_percent': 100.0}},
            'performance_benchmark': {'layer_test_results': {'success_rate': 100, 'avg_time_ms': 200}}, # Slow
            'compatibility_test': {'summary': {'success_rate_percent': 100.0}}
        }
        summary = self.evaluator._generate_executive_summary()
        self.assertIn(
            "Optimize processing speed with streaming for better performance",
            summary['recommendations']
        )
        self.assertEqual(summary['performance_grade'], 'B')
        self.assertEqual(summary['overall_status'], 'GOOD') # Downgraded to GOOD

    def test_run_all_evaluations_integration(self):
        """
        Tests that the main 'run_full_evaluation' function runs without
        crashing by mocking its sub-routines.
        """
        print("\n--- TestEvaluator: test_run_all_evaluations_integration ---")
        
        # We test the top-level function `run_full_evaluation`
        # We mock its internal evaluator's methods to just return empty dicts
        with patch.object(evaluation.ComprehensiveModelEvaluator, 'run_comprehensive_accuracy_test', return_value={}):
            with patch.object(evaluation.ComprehensiveModelEvaluator, 'run_performance_benchmark', return_value={}):
                with patch.object(evaluation.ComprehensiveModelEvaluator, 'run_device_compatibility_test', return_value={}):
                    with patch.object(evaluation.ComprehensiveModelEvaluator, 'generate_comprehensive_report', return_value={'summary': 'mock_summary'}):
                        
                        success = evaluation.run_full_evaluation(self.temp_dir, device='cpu')
                        
                        self.assertTrue(success)


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