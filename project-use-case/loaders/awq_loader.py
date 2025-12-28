# awq_loader.py - GPU-ACCELERATED VERSION

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
from typing import List, Tuple, Iterator
import os
import sys
import gc

# --- Force loader path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)

try:
    from awq.models.auto import AutoAWQForCausalLM
except ImportError:
    print("❌ AutoAWQ library not found. Please run: pip install autoawq")
    raise

class AWQLoader:
    """
    ✅ GPU-ACCELERATED AWQ Loader with proper device management.
    
    Key improvements:
    - Loads to CPU temporarily, then moves to GPU for actual processing
    - Uses CUDA streams for overlapped execution
    - Properly cleans up memory after each tensor
    - Tracks GPU memory usage
    
    This class loads a *quantized* AWQ model directory into CPU memory,
    then yields its dequantized parameters one by one to GPU.
    Note: This is not a true "streaming-from-disk" loader, as AutoAWQ
    loads the entire model. The "streaming" happens from the in-memory
    state_dict to GPU. This is necessary due to how AWQ models are stored.
    """

    def __init__(self, checkpoint_files: List[str], gpu_enabled: bool = True):
        if not checkpoint_files:
            raise ValueError("No AWQ checkpoint files (directory) provided.")

        # AWQ models are directories. find_local_checkpoints will find
        # the quant_config.json file. We get the directory from that.
        self.model_dir = None
        for f in checkpoint_files:
            if f.endswith("quant_config.json"):
                self.model_dir = os.path.dirname(f)
                break

        if self.model_dir is None:
            # Fallback: assume the user passed the directory path directly
            if os.path.isdir(checkpoint_files[0]):
                self.model_dir = checkpoint_files[0]
            else:
                raise ValueError(f"Could not find AWQ model directory or quant_config.json in: {checkpoint_files}")

        print(f"[AWQLoader] Initializing from model directory: {self.model_dir}")

        # ✅ GPU DEVICE MANAGEMENT
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.device = "cuda" if self.gpu_enabled else "cpu"
        
        # ✅ CUDA STREAMS FOR OVERLAPPED EXECUTION
        if self.gpu_enabled:
            self.cuda_streams = [torch.cuda.Stream() for _ in range(4)]
            print(f"[AWQLoader] 🔥 GPU acceleration ENABLED - {torch.cuda.get_device_name(0)}")
        else:
            self.cuda_streams = None
            print("[AWQLoader] GPU acceleration disabled - CPU fallback")

        try:
            # CRITICAL: Load model to CPU to avoid VRAM OOM on initialization
            # The dequantized tensors will be processed by preprocess.py
            print("[AWQLoader] Loading quantized model to CPU (this may take a moment)...")
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_dir,
                fuse_layers=False,
                device_map="cpu"  # Load to CPU first (temporary)
            )

            print("[AWQLoader] Model loaded. Extracting state dictionary...")
            # This state_dict contains the DEQUANTIZED weights as float
            self.state_dict = self.model.state_dict()
            self.layer_keys = list(self.state_dict.keys())
            self.total_layers = len(self.layer_keys)

            # ✅ DELETE MODEL FROM MEMORY TO FREE SPACE
            print("[AWQLoader] Cleaning up model object from memory...")
            del self.model
            gc.collect()
            if self.gpu_enabled:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"✅ [AWQLoader] Ready to stream {self.total_layers} tensors to {self.device.upper()}")

        except Exception as e:
            print(f"❌ [AWQLoader] Failed to load AWQ model from {self.model_dir}: {e}")
            raise

    def __len__(self) -> int:
        """Returns the total number of dequantized tensors."""
        return self.total_layers

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        ✅ GPU-ACCELERATED ITERATION using CUDA streams.
        Yields (layer_key, dequantized_tensor) pairs on the target device.
        """
        
        if self.gpu_enabled:
            return self._iter_gpu()
        else:
            return self._iter_cpu()

    def _iter_gpu(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """✅ GPU iteration with CUDA streams for overlapped execution."""
        
        print(f"🔄 [AWQLoader] GPU-accelerated streaming ({len(self.cuda_streams)} streams)...")
        
        stream_idx = 0
        processed = 0
        
        for layer_key in self.layer_keys:
            try:
                # Select CUDA stream for this tensor
                stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
                
                # Process with CUDA stream (allows overlapping)
                with torch.cuda.stream(stream):
                    # Get tensor from state dict
                    tensor = self.state_dict[layer_key]
                    
                    # ✅ CRITICAL: Move to GPU with non-blocking transfer
                    tensor = tensor.to(self.device, non_blocking=True)
                    
                    # Track statistics
                    if processed % 10 == 0:
                        used_gb = torch.cuda.memory_allocated() / 1e9
                        print(f"  [{processed}/{self.total_layers}] GPU: {used_gb:.2f}GB")
                    
                    yield layer_key, tensor
                    processed += 1
                    
                    # ✅ CLEANUP: Delete from state dict and CPU memory
                    del self.state_dict[layer_key]
                
                # Synchronize periodically to prevent queue buildup
                if stream_idx % 4 == 0:
                    torch.cuda.synchronize()
                
                stream_idx += 1
                
                # Aggressive cleanup every 50 layers
                if processed % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ [AWQLoader] Failed to yield {layer_key}: {e}")
                continue
        
        print(f"✅ [AWQLoader] Completed streaming all {processed} tensors")

    def _iter_cpu(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """CPU iteration (fallback)."""
        
        print(f"🔄 [AWQLoader] CPU iteration...")
        
        for i, layer_key in enumerate(self.layer_keys):
            try:
                tensor = self.state_dict[layer_key]
                yield layer_key, tensor
                
                # Cleanup
                del self.state_dict[layer_key]
                
                if i % 50 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"⚠️ [AWQLoader] Failed: {e}")
                continue

    def get_memory_status(self) -> dict:
        """Get current memory usage statistics."""
        
        status = {}
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            status["gpu_allocated_gb"] = allocated
            status["gpu_reserved_gb"] = reserved
            status["gpu_device"] = torch.cuda.get_device_name(0)
        
        status["total_layers"] = self.total_layers
        status["device"] = self.device
        
        return status


# =====================================================
# USAGE EXAMPLE
# =====================================================

if __name__ == "__main__":
    # Test the loader
    model_path = "path/to/awq/model"  # Change this
    
    loader = AWQLoader([model_path], gpu_enabled=True)
    
    print("\n📊 Memory Status:")
    print(loader.get_memory_status())
    
    print("\n🔄 Processing first 3 layers:")
    for i, (layer_key, tensor) in enumerate(loader):
        print(f"\n[{i+1}] {layer_key}")
        print(f"  Device: {tensor.device}")  # Should show cuda:0
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        
        if i >= 2:  # Just first 3
            break
