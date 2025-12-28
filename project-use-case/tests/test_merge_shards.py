#!/usr/bin/env python3
"""
test_merge_shards.py - Unit Test Suite for merge_shards.py

This test suite verifies the Stage 2 shard merging logic.
It uses mocks to simulate a file system with shards and an index,
then checks if the script correctly loads, merges, and saves
the final .pth file.
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
import json
import sys
from unittest.mock import patch, MagicMock, mock_open
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
    import merge_shards
except ImportError as e:
    print(f"FATAL: Could not import merge_shards.py.")
    print(f"Error: {e}")
    sys.exit(1)

class TestMergeShards(unittest.TestCase):
    """
    Tests the `merge_shards` script.
    """
    def setUp(self):
        # --- Mock Data ---
        self.mock_tensor_1 = torch.randn(10, 10).cpu()
        self.mock_tensor_2 = torch.randn(20, 20).cpu()
        
        # This is the final, merged dictionary we expect
        self.expected_full_dict = {
            'layer.1': self.mock_tensor_1,
            'layer.2': self.mock_tensor_2
        }
        
        # This is the mock index.json file content
        self.mock_index_content = {
            "metadata": {},
            "weight_map": {
                "layer.1": "model-00001.safetensors",
                "layer.2": "model-00002.safetensors"
            }
        }
        
        # This is what load_file will return for each shard
        self.mock_shard_data = {
            "model-00001.safetensors": {"layer.1": self.mock_tensor_1},
            "model-00002.safetensors": {"layer.2": self.mock_tensor_2}
        }

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    @patch('merge_shards.load_file')
    @patch('torch.save')
    @patch('os.remove')
    def test_merge_model_success(self, mock_os_remove, mock_torch_save, mock_load_file, mock_os_exists, mock_open_file): # <-- ADD mock_os_remove HERE
        print("\n--- TestMergeShards: test_merge_model_success ---")
        
        # Configure builtins.open to return our mock index content
        mock_open_file.return_value.read.return_value = json.dumps(self.mock_index_content)
        
        # Configure load_file to return the correct shard based on its input
        def load_side_effect(path, device="cpu"):
            filename = os.path.basename(path)
            if filename in self.mock_shard_data:
                return self.mock_shard_data[filename]
            raise FileNotFoundError(f"Mock file not found: {path}")
            
        mock_load_file.side_effect = load_side_effect

        # --- Run the function to test ---
        merge_shards.merge_model(
            model_name="base_model",
            index_path="/fake/dir/base_model.safetensors.index.json",
            output_dir="/fake/dir",
            prefix=""
        )

        # --- Verify the results ---
        # 1. Check that load_file was called for both shards
        self.assertEqual(mock_load_file.call_count, 2)
        mock_load_file.assert_any_call("/fake/dir/model-00001.safetensors", device="cpu")
        mock_load_file.assert_any_call("/fake/dir/model-00002.safetensors", device="cpu")
        
        # 2. Check that torch.save was called ONCE
        mock_torch_save.assert_called_once()
        
        # 3. Check that os.remove was called 3 times (2 shards + 1 index file)
        self.assertEqual(mock_os_remove.call_count, 3)
        mock_os_remove.assert_any_call("/fake/dir/model-00001.safetensors")
        mock_os_remove.assert_any_call("/fake/dir/model-00002.safetensors")
        mock_os_remove.assert_any_call("/fake/dir/base_model.safetensors.index.json")

        # 4. Check that torch.save was called with the CORRECT merged data
        saved_dict = mock_torch_save.call_args[0][0]
        saved_path = mock_torch_save.call_args[0][1]
        
        self.assertEqual(saved_path, "/fake/dir/base_model.pth")
        self.assertTrue(torch.equal(saved_dict['layer.1'], self.expected_full_dict['layer.1']))
        self.assertTrue(torch.equal(saved_dict['layer.2'], self.expected_full_dict['layer.2']))

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=False) # Simulate index file is missing
    @patch('torch.save')
    def test_merge_model_index_missing(self, mock_torch_save, mock_os_exists, mock_open_file):
        """Tests that merging is skipped if the index is missing."""
        print("\n--- TestMergeShards: test_merge_model_index_missing ---")
        
        merge_shards.merge_model(
            model_name="base_model",
            index_path="/fake/dir/base_model.safetensors.index.json",
            output_dir="/fake/dir",
            prefix=""
        )
        
        # torch.save should NEVER have been called
        mock_torch_save.assert_not_called()


if __name__ == "__main__":
    # Run the tests and capture the TestProgram object
    test_program = unittest.main(verbosity=1, exit=False)
    
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
    # If there are failures or errors, exit with 1, else exit with 0
    if not result.wasSuccessful():
        sys.exit(1)