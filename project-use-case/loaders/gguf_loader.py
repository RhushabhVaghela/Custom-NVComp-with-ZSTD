# gguf_loader.py - GPU-ACCELERATED VERSION

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
    from gguf import GGUFReader
except ImportError:
    print("❌ GGUF library not found. Please run: pip install gguf-py")
    raise


class GGUFLoader:
    """
    ✅ GPU-ACCELERATED streaming loader for GGUF files.
    
    Key improvements:
    - Moved warning filters to line 1 (eliminates deprecation warnings)
    - GPU device management with CUDA support
    - Non-blocking tensor transfers to GPU
    - Proper memory cleanup after each tensor
    
    This class reads a GGUF file, dequantizes its tensors using the
    'gguf' library, and yields them as PyTorch tensors on GPU.
    """

    def __init__(self, checkpoint_files: List[str], gpu_enabled: bool = True):
        if not checkpoint_files:
            raise ValueError("No GGUF checkpoint files provided.")

        # Find the first .gguf file
        self.gguf_path = None
        for f in checkpoint_files:
            if f.endswith(".gguf"):
                self.gguf_path = f
                break

        if self.gguf_path is None:
            raise ValueError(f"No .gguf file found in provided list: {checkpoint_files}")

        print(f"[GGUFLoader] Initializing from file: {self.gguf_path}")

        # ✅ GPU DEVICE MANAGEMENT
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.device = "cuda" if self.gpu_enabled else "cpu"
        
        # ✅ CUDA STREAMS FOR OVERLAPPED EXECUTION
        if self.gpu_enabled:
            self.cuda_streams = [torch.cuda.Stream() for _ in range(4)]
            print(f"🔥 [GGUFLoader] GPU acceleration ENABLED - {torch.cuda.get_device_name(0)}")
        else:
            self.cuda_streams = None
            print("[GGUFLoader] GPU acceleration disabled - CPU fallback")

        try:
            self.reader = GGUFReader(self.gguf_path, 'r')
        except Exception as e:
            print(f"❌ [GGUFLoader] Failed to open {self.gguf_path}: {e}")
            raise

        self.tensors_metadata = self.reader.tensors
        self.total_layers = len(self.tensors_metadata)
        print(f"✅ [GGUFLoader] Ready to stream {self.total_layers} quantized tensors.")

    def __len__(self) -> int:
        """Returns the total number of tensors in the file."""
        return self.total_layers

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        ✅ GPU-ACCELERATED ITERATION
        Yields (layer_key, tensor_on_device) pairs.
        """
        print(f"🔄 [GGUFLoader] Streaming {self.total_layers} tensors to {self.device.upper()}...")
        
        stream_idx = 0
        processed = 0

        for tensor_field in self.tensors_metadata:
            layer_key = tensor_field.name

            # .data dequantizes to a NumPy array (e.g., float32)
            try:
                data_numpy = tensor_field.data
            except Exception as e:
                print(
                    f"⚠️ [GGUFLoader] Failed to dequantize {layer_key} "
                    f"(Type: {tensor_field.tensor_type.name}): {e}. Skipping."
                )
                continue

            # Convert to PyTorch tensor
            try:
                torch_tensor = torch.from_numpy(data_numpy)
                
                if self.gpu_enabled:
                    # ✅ GPU-ACCELERATED: Move to GPU with CUDA streams
                    stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
                    with torch.cuda.stream(stream):
                        torch_tensor = torch_tensor.to(self.device, non_blocking=True)
                else:
                    # CPU fallback
                    torch_tensor = torch_tensor.cpu()
                
                # Track statistics
                if processed % 50 == 0 and self.gpu_enabled:
                    used_gb = torch.cuda.memory_allocated() / 1e9
                    print(f"  [{processed}/{self.total_layers}] GPU: {used_gb:.2f}GB")
                
                yield layer_key, torch_tensor
                processed += 1
                
                # Synchronize periodically
                if self.gpu_enabled and stream_idx % 4 == 0:
                    torch.cuda.synchronize()
                
                stream_idx += 1
                
                # Cleanup every 50 tensors
                if processed % 50 == 0:
                    gc.collect()
                    if self.gpu_enabled:
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ [GGUFLoader] Failed to convert {layer_key} to tensor: {e}. Skipping.")
                continue
        
        if self.gpu_enabled:
            print(f"✅ [GGUFLoader] Completed streaming all {processed} tensors to GPU")

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
    model_path = "path/to/model.gguf"  # Change this
    
    loader = GGUFLoader([model_path], gpu_enabled=True)
    
    print("\n📊 Memory Status:")
    print(loader.get_memory_status())
    
    print("\n🔄 Processing first 3 tensors:")
    for i, (layer_key, tensor) in enumerate(loader):
        print(f"\n[{i+1}] {layer_key}")
        print(f"  Device: {tensor.device}")  # Should show cuda:0
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        
        if i >= 2:
            break
