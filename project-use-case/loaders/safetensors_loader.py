# safetensors_loader.py - GPU-ACCELERATED VERSION

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
from safetensors import safe_open
from typing import List, Tuple, Iterator
import os
import sys
import gc

# --- Force loader path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)


class SafetensorsLoader:
    """
    ✅ GPU-ACCELERATED Streaming loader for Safetensors, .pth, and .bin files.
    
    Key improvements:
    - Moved warning filters to line 1 (eliminates deprecation warnings)
    - GPU device management with CUDA support
    - Non-blocking tensor transfers to GPU
    - Proper memory cleanup after each tensor
    
    Features:
    - For .safetensors: Streams one tensor at a time (memory-efficient)
    - For .pth/.bin: Loads one *file* at a time, yields its tensors, then unloads
    - GPU acceleration when available (RTX 5080 supported)
    """

    def __init__(self, checkpoint_files: List[str], gpu_enabled: bool = True):
        self.checkpoint_files = checkpoint_files
        self.total_layers = 0
        
        # ✅ GPU DEVICE MANAGEMENT
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.device = "cuda" if self.gpu_enabled else "cpu"
        
        # ✅ CUDA STREAMS FOR OVERLAPPED EXECUTION
        if self.gpu_enabled:
            self.cuda_streams = [torch.cuda.Stream() for _ in range(4)]
            print(f"🔥 [SafetensorsLoader] GPU acceleration ENABLED - {torch.cuda.get_device_name(0)}")
        else:
            self.cuda_streams = None
            print("[SafetensorsLoader] GPU acceleration disabled - CPU fallback")

        print(f"🔍 [SafetensorsLoader] Scanning {len(self.checkpoint_files)} file(s)...")
        
        for checkpoint_file in self.checkpoint_files:
            try:
                if checkpoint_file.endswith(".safetensors"):
                    # ✅ GPU-AWARE: Open safetensors with device support
                    with safe_open(checkpoint_file, framework="pt", device=self.device) as f:
                        self.total_layers += len(list(f.keys()))
                else:
                    # .pth or .bin file
                    print(f" [Info] Found .pth/.bin file: {checkpoint_file}. Layer count will be discovered during processing.")
                    
                    # Quick load to get count
                    state_dict = self._load_local_checkpoint(checkpoint_file)
                    self.total_layers += len(state_dict)
                    del state_dict  # Don't keep in memory
                    
            except Exception as e:
                print(f"⚠️ [SafetensorsLoader] Could not read keys from {checkpoint_file}: {e}")

        print(f"✅ [SafetensorsLoader] Found approximately {self.total_layers} total tensors.")

    def _load_local_checkpoint(self, checkpoint_path: str) -> dict:
        """Loads a .pth or .bin file, handling nested dicts."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle common nested formats
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
            
            if not isinstance(state_dict, dict):
                raise TypeError(f"Checkpoint did not contain a state_dict: {checkpoint_path}")
            
            return state_dict
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return {}

    def __len__(self) -> int:
        """Returns the total number of tensors across all files."""
        return self.total_layers

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        ✅ GPU-ACCELERATED ITERATION
        Yields (prefixed_layer_key, tensor_on_device) pairs.
        """
        print(f"🔄 [SafetensorsLoader] Streaming {self.total_layers} tensors to {self.device.upper()}...")
        
        stream_idx = 0
        processed = 0
        
        for checkpoint_file in self.checkpoint_files:
            file_prefix = f"checkpoint_{os.path.basename(checkpoint_file).replace('.', '_')}"
            
            try:
                if checkpoint_file.endswith(".safetensors"):
                    # ✅ GPU-ACCELERATED: Safetensors streaming
                    with safe_open(checkpoint_file, framework="pt", device=self.device) as f:
                        for layer_key in f.keys():
                            if self.gpu_enabled:
                                # Use CUDA stream for overlapped execution
                                stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
                                with torch.cuda.stream(stream):
                                    tensor = f.get_tensor(layer_key)
                                    # ✅ Already on GPU if device="cuda" was used
                                    if tensor.device.type != self.device:
                                        tensor = tensor.to(self.device, non_blocking=True)
                            else:
                                tensor = f.get_tensor(layer_key)
                            
                            prefixed_layer_key = f"{file_prefix}_{layer_key}"
                            
                            # Track statistics
                            if processed % 50 == 0 and self.gpu_enabled:
                                used_gb = torch.cuda.memory_allocated() / 1e9
                                print(f"  [{processed}/{self.total_layers}] GPU: {used_gb:.2f}GB")
                            
                            yield prefixed_layer_key, tensor
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
                                    
                else:
                    # .pth/.bin file - load entire file
                    state_dict = self._load_local_checkpoint(checkpoint_file)
                    
                    for layer_key, tensor in state_dict.items():
                        if self.gpu_enabled:
                            # ✅ GPU-ACCELERATED: Move to GPU
                            stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
                            with torch.cuda.stream(stream):
                                tensor = tensor.to(self.device, non_blocking=True)
                        else:
                            # CPU fallback
                            if tensor.device.type != "cpu":
                                tensor = tensor.cpu()
                        
                        prefixed_layer_key = f"{file_prefix}_{layer_key}"
                        
                        # Track statistics
                        if processed % 50 == 0 and self.gpu_enabled:
                            used_gb = torch.cuda.memory_allocated() / 1e9
                            print(f"  [{processed}/{self.total_layers}] GPU: {used_gb:.2f}GB")
                        
                        yield prefixed_layer_key, tensor
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
                    
                    # Unload entire file from memory
                    del state_dict
                    
            except Exception as e:
                print(f"⚠️ [SafetensorsLoader] Failed to load from {checkpoint_file}: {e}")
                continue
        
        if self.gpu_enabled:
            print(f"✅ [SafetensorsLoader] Completed streaming all {processed} tensors to GPU")

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
    model_path = "path/to/model"  # Change this
    
    loader = SafetensorsLoader([model_path], gpu_enabled=True)
    
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
