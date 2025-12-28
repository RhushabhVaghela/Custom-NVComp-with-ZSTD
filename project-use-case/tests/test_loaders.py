# test_loaders.py

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
import sys
from unittest.mock import patch, MagicMock
import os

# --- 1. Mock External Libraries ---
# We must do this *before* importing the loaders, otherwise, the
# 'from gguf import GGUFReader' line will fail if gguf-py isn't installed.
mock_awq = MagicMock()
mock_awq_models = MagicMock()
mock_awq_models_auto = MagicMock()

# --- Force loader path ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up, '../')
parent_dir = os.path.join(script_dir, '..')
# Get the loaders directory
loader_dir = os.path.join(parent_dir, 'loaders')

# Add the parent directory to Python's search path
sys.path.append(parent_dir)
sys.path.append(loader_dir)

sys.modules['awq'] = mock_awq
sys.modules['awq.models'] = mock_awq_models
sys.modules['awq.models.auto'] = mock_awq_models_auto

# --- 2. Now we can safely import our loaders ---
from safetensors_loader import SafetensorsLoader
from gguf_loader import GGUFLoader
from awq_loader import AWQLoader


class TestLoaderSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up mock data and configure the mock libraries once."""

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Mock Data ---
        cls.mock_tensor_1 = torch.randn(10, 10).cpu()
        cls.mock_tensor_2 = torch.randn(20, 20).cpu()
        
        # --- Configure safetensors.safe_open ---
        # This mock simulates: with safe_open(...) as f:
        cls.mock_sf_context = MagicMock()
        cls.mock_sf_context.keys.return_value = ['tensor.a', 'tensor.b']
        cls.mock_sf_context.get_tensor.side_effect = [
            cls.mock_tensor_1, 
            cls.mock_tensor_2
        ]

        # --- Configure gguf.GGUFReader ---
        # Mock the tensor field objects that GGUFReader.tensors contains
        mock_gguf_tensor_field_1 = MagicMock()
        mock_gguf_tensor_field_1.name = 'gguf.tensor.0'
        mock_gguf_tensor_field_1.data = cls.mock_tensor_1.numpy() # GGUF returns numpy

        mock_gguf_tensor_field_2 = MagicMock()
        mock_gguf_tensor_field_2.name = 'gguf.tensor.1'
        mock_gguf_tensor_field_2.data = cls.mock_tensor_2.numpy()

        # Configure the GGUFReader *instance*
        cls.mock_gguf_reader_instance = MagicMock()
        cls.mock_gguf_reader_instance.tensors = [
            mock_gguf_tensor_field_1, 
            mock_gguf_tensor_field_2
        ]

        # --- Configure autoawq.AutoAWQForCausalLM ---
        mock_awq_model = MagicMock()
        mock_awq_model.state_dict.return_value = {
            'awq.tensor.0': cls.mock_tensor_1,
            'awq.tensor.1': cls.mock_tensor_2,
        }
        # When .from_quantized is called, return the mock model
        mock_awq_models_auto.AutoAWQForCausalLM.from_quantized.return_value = mock_awq_model

    @patch('safetensors_loader.safe_open')
    def test_01_safetensors_loader_safetensors_file(self, mock_safe_open_patched):
        """Tests the SafetensorsLoader with a .safetensors file."""
        print("\n--- Testing: SafetensorsLoader (.safetensors) ---")
        
        # Configure the *patched* mock to use the behavior from setUpClass
        mock_safe_open_patched.return_value.__enter__.return_value = self.mock_sf_context
        
        loader = SafetensorsLoader(['fake.safetensors'])
        self.assertEqual(len(loader), 2)

        # Iterate and check
        data = list(loader)
        self.assertEqual(len(data), 2)
        
        # Check tensor 1
        self.assertEqual(data[0][0], 'checkpoint_fake_safetensors_tensor.a')
        self.assertTrue(torch.equal(data[0][1], self.mock_tensor_1.to(data[0][1].device)))
        self.assertEqual(data[0][1].device.type, self.device)
        
        # Check tensor 2
        self.assertEqual(data[1][0], 'checkpoint_fake_safetensors_tensor.b')
        self.assertTrue(torch.equal(data[1][1], self.mock_tensor_2.to(data[1][1].device)))
        self.assertEqual(data[1][1].device.type, self.device)

    @patch('safetensors_loader.torch.load')
    @patch('safetensors_loader.safe_open')
    def test_02_safetensors_loader_pth_file(self, mock_safe_open_patched, mock_torch_load):
        """Tests the SafetensorsLoader with a .pth file fallback."""
        print("\n--- Testing: SafetensorsLoader (.pth) ---")
        
        # Configure the mock for the __init__ scan.
        # We'll pretend the .safetensors scan found no keys.
        mock_safe_open_patched.return_value.__enter__.return_value.keys.return_value = []
        
        # Configure the mock for torch.load
        mock_torch_load.return_value = {
            'pth.tensor.0': self.mock_tensor_1
        }
        
        loader = SafetensorsLoader(['fake.pth'])
        # Note: __len__ relies on the quick pre-load
        self.assertEqual(len(loader), 1)
        
        data = list(loader)
        self.assertEqual(len(data), 1)
        
        self.assertEqual(data[0][0], 'checkpoint_fake_pth_pth.tensor.0')
        self.assertTrue(torch.equal(data[0][1], self.mock_tensor_1.to(data[0][1].device)))

    @patch('gguf_loader.GGUFReader')
    def test_03_gguf_loader(self, mock_gguf_reader_patched):
        """Tests the GGUFLoader."""
        print("\n--- Testing: GGUFLoader ---")
        
        # Configure the *patched* mock to use the behavior from setUpClass
        mock_gguf_reader_patched.return_value = self.mock_gguf_reader_instance
        
        loader = GGUFLoader(['fake.gguf'])
        self.assertEqual(len(loader), 2)
        
        data = list(loader)
        self.assertEqual(len(data), 2)
        
        # Check tensor 1
        self.assertEqual(data[0][0], 'gguf.tensor.0')
        self.assertTrue(torch.equal(data[0][1], self.mock_tensor_1.to(data[0][1].device)))
        self.assertEqual(data[0][1].device.type, self.device)
        
        # Check tensor 2
        self.assertEqual(data[1][0], 'gguf.tensor.1')
        self.assertTrue(torch.equal(data[1][1], self.mock_tensor_2.to(data[1][1].device)))
        self.assertEqual(data[1][1].device.type, self.device)

    @patch('os.path.isdir', return_value=True) # Mock that the path is a directory
    @patch('os.path.dirname', return_value='fake_awq_dir')
    def test_04_awq_loader(self, mock_dirname, mock_isdir):
        """Tests the AWQLoader."""
        print("\n--- Testing: AWQLoader ---")
        
        # AWQLoader is initialized with a directory path
        loader = AWQLoader(['fake_awq_dir'])
        self.assertEqual(len(loader), 2)
        
        data = list(loader)
        self.assertEqual(len(data), 2)
        
        # Check tensor 1
        self.assertEqual(data[0][0], 'awq.tensor.0')
        self.assertTrue(torch.equal(data[0][1], self.mock_tensor_1.to(data[0][1].device)))
        self.assertEqual(data[0][1].device.type, self.device)
        
        # Check tensor 2
        self.assertEqual(data[1][0], 'awq.tensor.1')
        self.assertTrue(torch.equal(data[1][1], self.mock_tensor_2.to(data[1][1].device)))
        self.assertEqual(data[1][1].device.type, self.device)

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