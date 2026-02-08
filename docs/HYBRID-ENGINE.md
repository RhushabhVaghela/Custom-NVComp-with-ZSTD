# Hybrid Engine Guide

## Overview

The `HybridEngine` is a smart routing layer that automatically selects the best execution backend (CPU or GPU) for each compression and decompression operation. It considers data location, buffer size, and historical performance to maximize throughput.

**Why does this exist?** GPU compression excels at batch and device-resident workloads, but for single-buffer host-to-host operations under ~1MB, CPU `libzstd` is faster due to PCIe transfer and kernel launch overhead. The hybrid engine makes this decision transparent.

---

## Routing Decision Matrix

The engine evaluates these factors in order:

```
1. Explicit mode override (FORCE_CPU / FORCE_GPU)
2. Data location (HOST / DEVICE / MANAGED)
3. Buffer size vs. thresholds
4. ADAPTIVE profiling history (if enabled)
5. Default AUTO heuristics
```

| Mode | Behavior |
|:-----|:---------|
| `AUTO` | GPU if data is on device, CPU if data is on host and above `cpu_size_threshold` |
| `PREFER_CPU` | CPU unless data is already on device and small enough to avoid transfer cost |
| `PREFER_GPU` | GPU unless data is on host and large (transfer cost exceeds compute savings) |
| `FORCE_CPU` | Always CPU, transfers data from device to host if needed |
| `FORCE_GPU` | Always GPU, uploads data from host to device if needed |
| `ADAPTIVE` | Profiles both paths over time, picks the faster one based on rolling history |

---

## HybridConfig Reference

```cpp
struct HybridConfig {
    HybridMode mode = HybridMode::AUTO;
    size_t cpu_size_threshold = 1024 * 1024;      // 1MB
    size_t gpu_device_threshold = 64 * 1024;       // 64KB
    bool enable_profiling = false;
    int compression_level = 3;
    u32 cpu_thread_count = 0;                      // 0 = auto-detect
    bool use_pinned_memory = true;
    bool overlap_transfers = true;
};
```

| Field | Default | Description |
|:------|:--------|:------------|
| `mode` | `AUTO` | Routing strategy (see table above) |
| `cpu_size_threshold` | 1 MB | Host data smaller than this routes to CPU in AUTO mode |
| `gpu_device_threshold` | 64 KB | Device data smaller than this may still route to CPU |
| `enable_profiling` | `false` | Collect timing data for ADAPTIVE mode |
| `compression_level` | 3 | ZSTD level (1-22) |
| `cpu_thread_count` | 0 | CPU threads for parallel CPU path (0 = auto) |
| `use_pinned_memory` | `true` | Use pinned host memory for transfers |
| `overlap_transfers` | `true` | Overlap PCIe transfers with compute |

---

## HybridResult Reference

Every compress/decompress call can optionally return a `HybridResult`:

```cpp
struct HybridResult {
    ExecutionBackend backend_used;       // CPU_LIBZSTD or GPU_KERNELS
    DataLocation input_location;         // Where input was detected
    DataLocation output_location;        // Where output was written
    double total_time_ms;                // Wall-clock time
    double transfer_time_ms;             // PCIe transfer time only
    double compute_time_ms;              // Compression/decompression time only
    double throughput_mbps;              // Effective throughput
    size_t input_bytes;                  // Input buffer size
    size_t output_bytes;                 // Output buffer size
    float compression_ratio;             // input_bytes / output_bytes
    const char* routing_reason;          // Human-readable routing explanation
};
```

The `routing_reason` string explains why a particular backend was chosen, e.g., `"AUTO: host data, using CPU"` or `"FORCE_GPU: user override"`.

---

## Data Location Auto-Detection

The engine uses `cudaPointerGetAttributes()` to automatically classify pointers:

| `cudaMemoryType` | Detected As |
|:-----------------|:------------|
| `cudaMemoryTypeHost` | `DataLocation::HOST` |
| `cudaMemoryTypeDevice` | `DataLocation::DEVICE` |
| `cudaMemoryTypeManaged` | `DataLocation::MANAGED` |
| Error / unregistered | `DataLocation::HOST` (safe default) |

You can also specify locations explicitly to skip detection:

```cpp
engine->compress(input, size, output, &out_size,
                 DataLocation::DEVICE,   // I know input is on GPU
                 DataLocation::HOST,     // I want output on host
                 &result);
```

---

## GPU Fallback Behavior

When GPU decompression fails (typically because the compressed data contains Huffman-encoded literals from CPU `libzstd`), the engine automatically retries via CPU -- unless the mode is `FORCE_GPU`.

This ensures cross-compatibility: data compressed by CPU `libzstd` can always be decompressed through the hybrid engine, regardless of routing mode.

---

## C++ API

### Creating an Engine

```cpp
#include "cuda_zstd_hybrid.h"

// With default config (AUTO mode, level 3)
auto engine = cuda_zstd::create_hybrid_engine();

// With custom config
cuda_zstd::HybridConfig config;
config.mode = cuda_zstd::HybridMode::ADAPTIVE;
config.compression_level = 5;
config.enable_profiling = true;
auto engine = cuda_zstd::create_hybrid_engine(config);

// Shorthand: just a compression level
auto engine = cuda_zstd::create_hybrid_engine(9);
```

### Single-Item Compress/Decompress

```cpp
cuda_zstd::HybridResult result;
size_t compressed_size = engine->get_max_compressed_size(input_size);

cuda_zstd::Status status = engine->compress(
    input, input_size,
    output, &compressed_size,
    cuda_zstd::DataLocation::HOST,    // input location
    cuda_zstd::DataLocation::HOST,    // output location
    &result                           // optional timing info
);

if (status == cuda_zstd::Status::SUCCESS) {
    printf("Backend: %s, Time: %.2f ms, Throughput: %.1f MB/s\n",
           result.backend_used == cuda_zstd::ExecutionBackend::CPU_LIBZSTD
               ? "CPU" : "GPU",
           result.total_time_ms,
           result.throughput_mbps);
}
```

### Batch Operations

```cpp
// Prepare batch items
std::vector<const void*> inputs = {buf1, buf2, buf3, buf4};
std::vector<size_t> input_sizes = {size1, size2, size3, size4};
std::vector<void*> outputs = {out1, out2, out3, out4};
std::vector<size_t> output_sizes = {max1, max2, max3, max4};
std::vector<cuda_zstd::DataLocation> in_locs(4, cuda_zstd::DataLocation::HOST);
std::vector<cuda_zstd::DataLocation> out_locs(4, cuda_zstd::DataLocation::HOST);

cuda_zstd::Status status = engine->compress_batch(
    inputs.data(), input_sizes.data(),
    outputs.data(), output_sizes.data(),
    4,
    in_locs.data(), out_locs.data()
);
```

Each batch item is routed independently -- some may go to CPU, others to GPU.

### Convenience Functions

For one-off operations without creating an engine:

```cpp
size_t compressed_size = output_capacity;
cuda_zstd::Status s = cuda_zstd::hybrid_compress(
    input, input_size,
    output, &compressed_size,
    cuda_zstd::DataLocation::HOST,
    cuda_zstd::DataLocation::HOST
);

size_t decompressed_size = output_capacity;
s = cuda_zstd::hybrid_decompress(
    compressed, compressed_size,
    output, &decompressed_size,
    cuda_zstd::DataLocation::HOST,
    cuda_zstd::DataLocation::HOST
);
```

These create a temporary engine per call. For repeated use, create an engine and reuse it.

### Advisory Routing Query

```cpp
cuda_zstd::ExecutionBackend backend = engine->query_routing(
    input_size,
    cuda_zstd::DataLocation::HOST,
    cuda_zstd::DataLocation::DEVICE,
    true  // is_compress
);
// Returns CPU_LIBZSTD or GPU_KERNELS without doing any work
```

### ADAPTIVE Mode Profiling

```cpp
cuda_zstd::HybridConfig config;
config.mode = cuda_zstd::HybridMode::ADAPTIVE;
config.enable_profiling = true;
auto engine = cuda_zstd::create_hybrid_engine(config);

// First N calls profile both backends
for (int i = 0; i < 100; i++) {
    engine->compress(input, size, output, &out_size,
                     cuda_zstd::DataLocation::HOST,
                     cuda_zstd::DataLocation::HOST);
}

// Check what it learned
double cpu_tp = engine->get_observed_throughput(
    cuda_zstd::ExecutionBackend::CPU_LIBZSTD, true);
double gpu_tp = engine->get_observed_throughput(
    cuda_zstd::ExecutionBackend::GPU_KERNELS, true);

printf("Observed CPU: %.1f MB/s, GPU: %.1f MB/s\n", cpu_tp, gpu_tp);

// Reset profiling data
engine->reset_profiling();
```

The profiling history uses a 64-sample rolling window per backend per operation type (compress/decompress).

---

## C API

The C API provides 7 functions for use from C, Python (via ctypes/cffi), Rust, Go, etc.

### Lifecycle

```c
#include "cuda_zstd_hybrid.h"

// Create with config
cuda_zstd_hybrid_config_t config = {0};
config.mode = 0;              // AUTO
config.compression_level = 3;
config.cpu_size_threshold = 1024 * 1024;
config.gpu_device_threshold = 64 * 1024;
config.use_pinned_memory = 1;
config.overlap_transfers = 1;

cuda_zstd_hybrid_engine_t* engine = cuda_zstd_hybrid_create(&config);

// Or create with defaults
cuda_zstd_hybrid_engine_t* engine = cuda_zstd_hybrid_create_default();

// Use engine...

// Cleanup
cuda_zstd_hybrid_destroy(engine);
```

### Compress/Decompress

```c
// Compress
cuda_zstd_hybrid_result_t result = {0};
size_t compressed_size = input_size * 2;

int err = cuda_zstd_hybrid_compress(
    engine,
    src, src_size,
    dst, &compressed_size,
    0,  // input location: HOST
    0,  // output location: HOST
    &result,
    0   // stream (0 = default)
);

if (err == 0) {
    printf("Compressed %zu -> %zu bytes (%.1f MB/s)\n",
           src_size, compressed_size, result.throughput_mbps);
}

// Decompress
size_t decompressed_size = original_size;
err = cuda_zstd_hybrid_decompress(
    engine,
    dst, compressed_size,
    output, &decompressed_size,
    0, 0,   // HOST -> HOST
    &result,
    0
);
```

### Utility Functions

```c
// Query max compressed size
size_t max_size = cuda_zstd_hybrid_max_compressed_size(engine, input_size);

// Advisory routing query
unsigned int backend = cuda_zstd_hybrid_query_routing(
    engine,
    input_size,
    0,    // input: HOST
    1,    // output: DEVICE
    1     // is_compress: true
);
// backend == 0 means CPU_LIBZSTD, 1 means GPU_KERNELS
```

### C API Function Summary

| Function | Description |
|:---------|:------------|
| `cuda_zstd_hybrid_create(config)` | Create engine with config |
| `cuda_zstd_hybrid_create_default()` | Create engine with defaults |
| `cuda_zstd_hybrid_destroy(engine)` | Destroy engine |
| `cuda_zstd_hybrid_compress(...)` | Compress with auto-routing |
| `cuda_zstd_hybrid_decompress(...)` | Decompress with auto-routing |
| `cuda_zstd_hybrid_max_compressed_size(engine, size)` | Query output bound |
| `cuda_zstd_hybrid_query_routing(...)` | Advisory routing query |

---

## Measured Performance (RTX 5080 Laptop GPU)

These numbers are from `benchmark_hybrid` with `--quick` mode:

| Path | Pattern | Size | Compress | Decompress |
|:-----|:--------|:-----|:---------|:-----------|
| HOST-to-HOST | Repetitive | 64 KB | 1780 MB/s | 7499 MB/s |
| HOST-to-HOST | Repetitive | 1 MB | 1690 MB/s | 6200 MB/s |
| HOST-to-DEV | Repetitive | 1 MB | 2163 MB/s | -- |
| DEV-to-HOST | Repetitive | 1 MB | 863 MB/s | -- |
| DEV-to-DEV | Repetitive | 64 KB | 54 MB/s | -- |
| HOST-to-HOST | Random | 1 MB | 520 MB/s | 1100 MB/s |

HOST-to-HOST and HOST-to-DEV paths route to CPU for compression, which is why they achieve high throughput. DEV-to-DEV routes to GPU kernels.

---

## Thread Safety

- `HybridEngine` instances are **NOT** thread-safe. Create one per thread.
- The internal GPU manager is lazily initialized on first GPU-path use.
- Dictionary objects can be shared across engines (read-only after creation).

---

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_hybrid.h` | Header with C++ and C API declarations |
| `src/cuda_zstd_hybrid.cu` | Full implementation (~1200 lines) |
| `tests/test_hybrid.cu` | 26 test cases |
| `benchmarks/benchmark_hybrid.cu` | 4-path x 4-pattern x 6-size benchmark |

---

## Related Documentation

- [Architecture Overview](ARCHITECTURE-OVERVIEW.md) -- Where HybridEngine fits in the system
- [Performance Tuning](PERFORMANCE-TUNING.md) -- General optimization tips
- [C API Reference](C-API-REFERENCE.md) -- Full C API documentation
- [Quick Reference](QUICK-REFERENCE.md) -- One-page cheat sheet
