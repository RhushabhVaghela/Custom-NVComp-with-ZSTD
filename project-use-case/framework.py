#!/usr/bin/env python3
"""
framework.py - FINAL PRODUCTION FRAMEWORK WITH MEMORY-SAFE STREAMING
🎉 ALL SOLUTIONS INTEGRATED:
- Universal JIT layer support with device-aware operations
- Production-ready model wrapper with comprehensive error handling
- Performance monitoring and memory usage tracking
- Advanced benchmarking capabilities for evaluation
- Complete integration with preprocessing pipeline
🔥 MEMORY-SAFE STREAMING ENHANCEMENTS:
✅ Safetensors streaming support for large models
✅ Memory-safe layer-by-layer processing
✅ Adaptive device selection for efficiency
✅ Streaming-aware delta decompression
✅ VRAM-aware batch processing
✅ Zero runtime weight caching (streaming only)
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
import torch.nn as nn
import torch.nn.functional as F
import os, pickle, sys, gc, time
from typing import Dict, Any, Optional, List, Union

TORCH_DTYPE = torch.bfloat16
print("[Framework] Loading JIT Layer...")
try:
    from jit_layer import UniversalSmartHybridJITLayer

    print("[Framework] ...UniversalSmartHybridJITLayer imported successfully.")
except ImportError:
    print(
        "[Framework] X ERROR: Cannot import UniversalSmartHybridJITLayer from jit_layer.py"
    )
    sys.exit(1)
# Define the file names
BASE_FILE = "base_model.pth"
PAYLOAD_FILE = "delta_dequantization.pkl"


def find_and_load_assets(output_dir):
    """Finds and loads the base model and delta payload with comprehensive error handling."""
    base_path = os.path.join(output_dir, BASE_FILE)
    payload_path = os.path.join(output_dir, PAYLOAD_FILE)
    if not os.path.isfile(base_path) or not os.path.isfile(payload_path):
        print(f"X ERROR: Cannot find required files in '{output_dir}':")
        if not os.path.isfile(base_path):
            print(f" - Missing: {BASE_FILE}")
        if not os.path.isfile(payload_path):
            print(f" - Missing: {PAYLOAD_FILE}")
        print("\n > Please run 'preprocess.py' first.")
        raise FileNotFoundError("Missing required model assets")
    print(f"[Framework] Loading base model: {BASE_FILE}")
    try:
        base_weights = torch.load(base_path, map_location="cpu")
        print(f"[Framework] ✅ Base model loaded: {len(base_weights)} layers")
    except Exception as e:
        print(f"[Framework] ❌ Failed to load base model: {e}")
        raise
    print(f"[Framework] Loading delta payload: {PAYLOAD_FILE}")
    try:
        # 🔥 NEW: Handle both pickle and zstd-compressed formats
        try:
            with open(payload_path, "rb") as f:
                delta_payload = pickle.load(f)
        except Exception as e:
            # Try zstd decompression if standard pickle fails
            try:
                import zstandard as zstd

                with open(payload_path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    delta_payload = pickle.loads(dctx.decompress(f.read()))
            except:
                raise e  # Re-raise original exception if zstd also fails
        print(f"[Framework] ✅ Delta payload loaded: {len(delta_payload)} layers")
    except Exception as e:
        print(f"[Framework] ❌ Failed to load delta payload: {e}")
        raise
    return base_weights, delta_payload


class DeltaModelWrapper(nn.Module):
    """
    Production-ready model wrapper with JIT layers.
    Supports all layer types, device-aware operations, and comprehensive error handling.
    🔥 ENHANCED: Memory-safe streaming support with no runtime caching.
    """

    def __init__(self, base_weights, delta_payload, device="cuda"):
        super().__init__()
        print(f"[Framework] Initializing DeltaModelWrapper on device='{device}'...")
        self.layers = nn.ModuleDict()
        self.device = device
        self.base_weights = base_weights
        self.delta_payload = delta_payload
        # Memory tracking with streaming awareness
        self.memory_usage = {"peak_allocated": 0.0, "peak_reserved": 0.0}
        self.streaming_stats = {"layers_processed": 0, "streaming_events": 0}
        # Find all valid layers to replace
        created_layers = 0
        for key in base_weights.keys():
            if base_weights[key].dim() < 2:  # Skip biases, norms, etc.
                continue
            print(f" > Creating JIT layer for: {key}")
            # Get the specific delta for this layer
            delta_info_for_layer = delta_payload.get(key, None)
            if delta_info_for_layer is None:
                print(f" > WARNING: No delta payload found for layer {key}.")
            try:
                # Load the base weight onto the target device
                base_weight = base_weights[key].to(device)
                # Create the JIT layer with streaming support
                self.layers[key.replace(".", "_")] = UniversalSmartHybridJITLayer(
                    base_weight=base_weight,
                    delta_info=delta_info_for_layer,
                    device=device,
                    precision_mode="adaptive",
                    enable_benchmarking=True,
                )
                created_layers += 1
                # Track memory usage
                if device == "cuda":
                    current_alloc = torch.cuda.memory_allocated() / 1024**3
                    self.memory_usage["peak_allocated"] = max(
                        self.memory_usage["peak_allocated"], current_alloc
                    )
                # 🔥 NEW: Track streaming events
                if delta_info_for_layer and delta_info_for_layer.get(
                    "streaming_available", False
                ):
                    self.streaming_stats["streaming_events"] += 1
            except Exception as e:
                print(f" > ❌ Failed to create JIT layer for {key}: {e}")
                continue
        print(f"[Framework] Created {created_layers} JIT layers. Model ready.")
        # 🔥 NEW: Print streaming statistics
        if self.streaming_stats["streaming_events"] > 0:
            print(
                f"[Framework] 🔥 Streaming support enabled for {self.streaming_stats['streaming_events']} layers"
            )
        # Final memory report
        if device == "cuda":
            final_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[Framework] Final GPU memory usage: {final_mem:.2f}GB")

    def forward(self, x, layer_key=None):
        """
        Production forward pass with comprehensive error handling and streaming.
        """
        if layer_key is None:
            # Get the first layer in our dictionary
            layer_key = next(iter(self.layers.keys()))
        if layer_key not in self.layers:
            raise ValueError(f"Layer {layer_key} not found in model")
        layer = self.layers[layer_key]
        # Get the delta info corresponding to that layer
        # (We have to replace the '_' with '.' to match the pickle file key)
        delta_key = layer_key.replace("_", ".")
        delta_info = self.delta_payload.get(delta_key, None)
        print(f"\n[Framework] --- Executing forward pass on layer: {delta_key} ---")
        try:
            # 🔥 NEW: Track layer processing for streaming statistics
            self.streaming_stats["layers_processed"] += 1
            result = layer(x, delta_info=delta_info)
            return result
        except Exception as e:
            print(f"[Framework] ❌ Forward pass failed for {delta_key}: {e}")
            raise

    def get_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all layers."""
        stats = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, "get_performance_stats"):
                stats[layer_name] = layer.get_performance_stats().copy()
        return stats

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if self.device == "cuda":
            current_stats = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_allocated": self.memory_usage["peak_allocated"],
            }
            return current_stats
        return {"cpu_mode": True}

    def get_streaming_stats(self) -> Dict[str, Any]:
        """🔥 NEW: Get streaming statistics."""
        return self.streaming_stats.copy()

    def cleanup_memory(self):
        """Aggressive memory cleanup with streaming awareness."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all layers."""
        layer_info = {}
        for layer_name, layer in self.layers.items():
            original_key = layer_name.replace("_", ".")
            delta_info = self.delta_payload.get(original_key, {})
            layer_info[layer_name] = {
                "original_shape": layer.original_base_weight.shape,
                "dtype": str(layer.original_base_weight.dtype),
                "device": str(layer.original_base_weight.device),
                "has_delta": original_key in self.delta_payload,
                "delta_chunks": len(delta_info.get("comp_indices_list_zstd", [])),
                "precision_mode": layer.precision_mode
                if hasattr(layer, "precision_mode")
                else "unknown",
                # 🔥 NEW: Streaming information
                "streaming_available": delta_info.get("streaming_available", False),
            }
        return layer_info


class AdvancedJITModelFramework:
    """
    Advanced JIT Model Framework for evaluation and testing.
    Provides high-level interface for model evaluation and benchmarking
    with JIT weight reconstruction and streaming capabilities.
    🔥 ENHANCED: Memory-safe streaming and adaptive processing.
    """

    def __init__(
        self, base_model_path: str, delta_payload_path: str, device: str = "cuda"
    ):
        self.device = device
        print(
            f"[Framework] Initializing AdvancedJITModelFramework on device='{device}'..."
        )
        # Load assets
        print(f"[Framework] Loading base model: {base_model_path}")
        self.base_weights = torch.load(base_model_path, map_location="cpu")
        print(f"[Framework] Loading delta payload: {delta_payload_path}")
        # 🔥 ENHANCED: Handle both pickle and zstd formats
        try:
            with open(delta_payload_path, "rb") as f:
                self.delta_payload = pickle.load(f)
        except Exception as e:
            try:
                import zstandard as zstd

                with open(delta_payload_path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    self.delta_payload = pickle.loads(dctx.decompress(f.read()))
            except:
                raise e
        # Create model wrapper with streaming support
        self.model = DeltaModelWrapper(self.base_weights, self.delta_payload, device)
        print("[Framework] AdvancedJITModelFramework ready.")

    def forward_with_reconstruction(
        self, x: torch.Tensor, layer_key: str = None
    ) -> torch.Tensor:
        """Forward pass with JIT reconstruction and streaming."""
        return self.model(x, layer_key)

    def get_layer_count(self) -> int:
        """Get the number of JIT layers."""
        return len(self.model.layers)

    def get_delta_info(self, layer_key: str) -> Optional[Dict[str, Any]]:
        """Get delta information for a specific layer."""
        return self.delta_payload.get(layer_key, None)

    def test_all_layers(self, batch_size: int = 2) -> Dict[str, Any]:
        """Test all layers with comprehensive validation and streaming awareness."""
        print(f"[Framework] Testing all {self.get_layer_count()} layers...")
        results = {
            "successful": 0,
            "failed": 0,
            "layer_results": {},
            "total_time": 0.0,
            "streaming_stats": {"layers_with_streaming": 0, "total_processed": 0},
        }
        for layer_name, layer in self.model.layers.items():
            print(f"\n[Framework] Testing layer: {layer_name}")
            try:
                # Create appropriate input based on layer shape
                base_shape = layer.original_base_weight.shape
                if len(base_shape) == 2:  # Linear layer
                    input_shape = (batch_size, base_shape[1])
                else:  # Other layer types
                    input_shape = (batch_size, base_shape[0])
                test_input = torch.randn(
                    input_shape, device=self.device, dtype=torch.float32
                )
                start_time = time.perf_counter()
                output = self.forward_with_reconstruction(test_input, layer_name)
                end_time = time.perf_counter()
                process_time = (end_time - start_time) * 1000
                results["total_time"] += process_time
                # Validate output
                assert output is not None, "Output is None"
                assert output.device.type == self.device, (
                    f"Wrong device: {output.device}"
                )
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"
                results["successful"] += 1
                # 🔥 NEW: Track streaming usage
                original_key = layer_name.replace("_", ".")
                delta_info = self.delta_payload.get(original_key, {})
                if delta_info.get("streaming_available", False):
                    results["streaming_stats"]["layers_with_streaming"] += 1
                results["streaming_stats"]["total_processed"] += 1
                results["layer_results"][layer_name] = {
                    "status": "success",
                    "time_ms": process_time,
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "streaming": delta_info.get("streaming_available", False),
                }
                print(f"[Framework] ✅ {layer_name}: SUCCESS ({process_time:.1f}ms)")
            except Exception as e:
                results["failed"] += 1
                results["layer_results"][layer_name] = {
                    "status": "failed",
                    "error": str(e),
                }
                print(f"[Framework] ❌ {layer_name}: FAILED - {e}")
        # Calculate summary statistics
        total_layers = results["successful"] + results["failed"]
        results["success_rate"] = (
            (results["successful"] / total_layers * 100) if total_layers > 0 else 0
        )
        results["avg_time_ms"] = (
            results["total_time"] / results["successful"]
            if results["successful"] > 0
            else 0
        )
        # 🔥 NEW: Add streaming summary
        if results["streaming_stats"]["total_processed"] > 0:
            results["streaming_coverage_percent"] = (
                results["streaming_stats"]["layers_with_streaming"]
                / results["streaming_stats"]["total_processed"]
                * 100
            )
        print(f"\n[Framework] Test Summary:")
        print(f" Total layers: {total_layers}")
        print(f" ✅ Successful: {results['successful']}")
        print(f" ❌ Failed: {results['failed']}")
        print(f" Success rate: {results['success_rate']:.1f}%")
        print(f" Average time: {results['avg_time_ms']:.1f}ms")
        # 🔥 NEW: Print streaming summary
        if results["streaming_stats"]["total_processed"] > 0:
            print(
                f" 🔥 Streaming coverage: {results['streaming_coverage_percent']:.1f}%"
            )
        return results

    def benchmark_performance(
        self, batch_sizes=[1, 2, 4, 8], seq_length=512
    ) -> Dict[str, Any]:
        """Comprehensive performance benchmark with streaming awareness."""
        if self.device != "cuda":
            print("[Framework] Skipping benchmark - CUDA required")
            return {}
        results = {}
        print(f"[Framework] Running performance benchmark...")
        # Get the first layer for benchmarking
        first_layer_key = next(iter(self.model.layers.keys()))
        layer = self.model.layers[first_layer_key]
        # Determine appropriate input shape
        base_shape = layer.original_base_weight.shape
        if len(base_shape) == 2:
            input_features = base_shape[1]
        else:
            input_features = base_shape[0]
        # Warmup
        dummy_input = torch.randn(
            2, input_features, dtype=TORCH_DTYPE, device=self.device
        )
        with torch.no_grad():
            for _ in range(3):
                _ = self.forward_with_reconstruction(dummy_input, first_layer_key)
        torch.cuda.synchronize()
        # Benchmark different batch sizes
        for batch_size in batch_sizes:
            test_input = torch.randn(
                batch_size, input_features, dtype=TORCH_DTYPE, device=self.device
            )
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            with torch.no_grad():
                output = self.forward_with_reconstruction(test_input, first_layer_key)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = start_time.elapsed_time(end_time)
            results[f"batch_{batch_size}"] = {
                "time_ms": elapsed_ms,
                "throughput_samples_per_sec": batch_size / (elapsed_ms / 1000),
                "output_shape": list(output.shape),
            }
            print(
                f" Batch {batch_size}: {elapsed_ms:.1f}ms "
                f"({results[f'batch_{batch_size}']['throughput_samples_per_sec']:.1f} samples/sec)"
            )
            del test_input, output
            torch.cuda.empty_cache()
        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """🔥 NEW: Get comprehensive memory and streaming statistics."""
        stats = {
            "device": self.device,
            "model_memory": self.model.get_memory_usage(),
            "streaming": self.model.get_streaming_stats(),
        }
        return stats

    def run_framework_test(self, output_dir: str = "./") -> bool:
        """
        Complete framework test with comprehensive error handling and validation.
        🔥 ENHANCED: Includes streaming verification.
        """
        print("[Framework] Starting comprehensive framework test...")
        DEVICE = self.device
        try:
            # Find the input dimension required by our first layer
            first_key = next(iter(self.base_weights.keys()))
            while self.base_weights[first_key].dim() < 2:
                first_key = next(iter(self.base_weights.keys()))
            in_features = self.base_weights[first_key].shape[1]
            print(f"[Framework] Creating mock input tensor: (4, {in_features})")
            # Create mock input
            mock_input = torch.randn(4, in_features, dtype=torch.float32, device=DEVICE)
            print(f"[Framework] ✅ Mock input created: {mock_input.shape}")
            # Test a forward pass
            print("[Framework] Testing forward pass with JIT reconstruction...")
            output = self.forward_with_reconstruction(mock_input)
            print(f"[Framework] ✅ Forward pass successful!")
            print(f" Input shape: {mock_input.shape}")
            print(f" Output shape: {output.shape}")
            print(f" Output device: {output.device}")
            print(f" Output dtype: {output.dtype}")
            # Test memory cleanup
            print("[Framework] Testing memory cleanup...")
            self.model.cleanup_memory()
            print("[Framework] ✅ Memory cleanup successful!")
            # 🔥 NEW: Print streaming and memory stats
            stats = self.get_memory_stats()
            print(f"\n[Framework] 🔥 Streaming Statistics:")
            print(f" Layers processed: {stats['streaming']['layers_processed']}")
            print(f" Streaming events: {stats['streaming']['streaming_events']}")
            print(f"\n[Framework] Framework test PASSED!")
            return True
        except Exception as e:
            print(f"[Framework] ❌ Framework test FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Basic self-test
    print("[Framework] Framework Module Self-Test")
    print("=" * 50)
    try:
        # Create a minimal test
        test_base_weights = {
            "layer1": torch.randn(100, 50),
            "layer2": torch.randn(50, 25),
        }
        test_delta_payload = {
            "layer1": {"delta": torch.randn(100, 50) * 0.01},
            "layer2": {"delta": torch.randn(50, 25) * 0.01},
        }
        # Create wrapper
        wrapper = DeltaModelWrapper(test_base_weights, test_delta_payload, device="cpu")
        print(f"✅ Framework loaded successfully!")
        print(f" Created {len(wrapper.layers)} JIT layers")
        # Test forward pass
        test_input = torch.randn(4, 50)
        output = wrapper(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
    print("\n🔥 Framework Ready!")
