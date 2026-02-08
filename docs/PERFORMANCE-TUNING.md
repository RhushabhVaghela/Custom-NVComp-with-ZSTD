# Performance Tuning Guide

> *"Measure first, optimize second. The hybrid engine does the rest."*

## Why Performance Matters

Proper tuning can improve throughput by orders of magnitude. This guide covers the key techniques to go from "it works" to "it flies."

---

## Measured Performance Baselines

These numbers were measured on an RTX 5080 Laptop GPU (Blackwell sm_120):

| Metric | Measured Value | Configuration |
|:-------|:---------------|:--------------|
| **Peak Compress Throughput (GPU)** | 800.3 MB/s | Single stream, 256 KB blocks |
| **Peak Decompress Throughput (GPU)** | 1.77 GB/s | Single stream, 256 KB blocks |
| **Peak Batch Throughput (GPU)** | 9.81 GB/s | 250 x 256 KB blocks |
| **HOST→HOST Compress (CPU)** | 1,171-6,320 MB/s | Hybrid engine, CPU libzstd |
| **HOST→HOST Decompress (CPU)** | 1,983-20,738 MB/s | Hybrid engine, CPU libzstd |
| **GPU/CPU Crossover (FSE)** | ~64 KB | Below this, CPU is faster for FSE |
| **Optimal Block Size** | 64-256 KB | Best throughput per item |
| **Stream Pool Speedup** | 4.7x | vs. cudaStreamCreate per operation |

---

## The Speed vs. Size Tradeoff

Compression levels work like gears:

| Level | Speed | Compression | Best For |
|:-----:|:-----:|:-----------:|:---------|
| **1-3** | Fastest | Good | Real-time streaming, logs |
| **4-6** | Fast | Better | General purpose |
| **7-12** | Moderate | Great | Archival, storage |
| **13-22** | Slow | Maximum | Long-term cold storage |

> **Rule of Thumb**: Level 3 gives you 80% of the compression at 5x the speed. Start there.

---

## Quick Wins (Do These First)

### 1. Use Pinned Memory

```cpp
// Slow: Regular memory
void* buffer = malloc(size);

// Fast: Pinned memory (2-3x faster transfers)
void* buffer;
cudaMallocHost(&buffer, size);
```

Pinned memory can be transferred directly to/from the GPU without intermediate copies.

### 2. Choose the Right Chunk Size

| Chunk Size | GPU Utilization | When to Use |
|:----------:|:---------------:|:------------|
| < 1 MB | **N/A** | Handled by Smart Router (CPU) for latency |
| 16 KB | 60-70% | Many tiny files (forced GPU) |
| **64 KB** | **80-85%** | **Most use cases (sweet spot)** |
| 128 KB | 85-90% | Large files |
| 256 KB | 90-95% | Maximum throughput |

### 3. Reuse Your Workspace

```cpp
// Slow: Allocate every time
for (auto& file : files) {
    void* workspace;
    cudaMalloc(&workspace, size);
    compress(file, workspace);
    cudaFree(workspace);
}

// Fast: Allocate once, reuse
void* workspace;
cudaMalloc(&workspace, size);
for (auto& file : files) {
    compress(file, workspace);
}
```

---

## Stream Pool

The `StreamPool` class in `include/cuda_zstd_stream_pool.h` provides pre-allocated CUDA streams with RAII-based checkout/return. This avoids the overhead of `cudaStreamCreate`/`cudaStreamDestroy` per operation.

```cpp
#include "cuda_zstd_stream_pool.h"

// Create a pool of 8 streams
cuda_zstd::StreamPool pool(8);

{
    // RAII guard: acquires a stream, releases on scope exit
    auto guard = pool.acquire();
    cudaStream_t stream = guard.get_stream();

    // Use the stream for compression
    mgr->compress(d_input, input_size, d_output, &out_size,
                  d_temp, temp_size, nullptr, 0, stream);
}
// Stream automatically returned to pool here
```

Measured speedup: **4.7x** compared to creating/destroying streams per operation.

---

## Advanced Techniques

### Multi-Stream Overlap (The Pipeline Trick)

Overlap upload, compute, and download across multiple streams:

```
Time ->   | Stream 1 | Stream 2 | Stream 3 |
----------+----------+----------+----------+
Step 1    | Upload   |          |          |
Step 2    | Compress | Upload   |          |
Step 3    | Download | Compress | Upload   |
Step 4    |          | Download | Compress |
Step 5    |          |          | Download |
```

**Result**: 3x throughput with no extra hardware.

```cpp
// Create multiple streams
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

// Use them in rotation
for (int i = 0; i < num_chunks; i++) {
    int s = i % 4;
    upload(streams[s]);
    compress(streams[s]);
    download(streams[s]);
}
```

### The OpenMP Secret Weapon (Maximum Aggregate Throughput)

This is the highest-performing pattern:

```cpp
#pragma omp parallel num_threads(8)
{
    // Each thread: own manager + own stream
    auto manager = cuda_zstd::create_manager(3);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    #pragma omp for
    for (int i = 0; i < num_files; i++) {
        manager->compress(files[i], stream);
    }
}
```

**Why it works**: Multiple CPU threads saturate the GPU with work, hiding all latency.

---

## Benchmark Your Setup

Run the built-in benchmark:

```bash
./benchmark_batch_throughput
```

**Reference output (RTX 5080 Laptop GPU):**

```
=== ZSTD GPU Batch Performance ===
Parallel Batch 64KB  x 1000:   9.81 GB/s
```

Your numbers will vary depending on GPU model, clock speeds, and thermal conditions. If you are not hitting similar numbers, check:
1. Is the GPU being thermally throttled?
2. Are you using pinned memory?
3. Is another process using the GPU?
4. Did you build with `-DCMAKE_BUILD_TYPE=Release`?

---

## Environment Variables

| Variable | What It Does | Recommendation |
|:---------|:-------------|:---------------|
| `CUDA_ZSTD_DEBUG_LEVEL=0` | Disable debug output | Set to 0 in production |
| `CUDA_ZSTD_POOL_MAX_SIZE=4G` | Allow larger memory pool | For big workloads |

---

## Performance Checklist

Before going to production, verify:

- [ ] Using compression level 3 (unless you need higher ratios)
- [ ] Chunk size is 64KB or larger
- [ ] Using pinned memory for host buffers
- [ ] Workspace is pre-allocated and reused
- [ ] Using multiple streams, StreamPool, or OpenMP pattern
- [ ] Using HybridEngine for automatic CPU/GPU routing
- [ ] Debug logging is disabled (`CUDA_ZSTD_DEBUG_LEVEL=0`)
- [ ] Built in Release mode (`-DCMAKE_BUILD_TYPE=Release`)

---

## Hybrid Engine Tuning

The `HybridEngine` automatically routes work to CPU or GPU based on data location and size. For host-memory data, CPU `libzstd` is **100-790x faster** than GPU due to CUDA API overhead.

### When CPU Wins (HOST→HOST Path)

| Size | CPU Compress | GPU Compress | CPU Advantage |
|------|-------------|-------------|---------------|
| 1 MB | 895-1,853 MB/s | 6-9 MB/s | **100-300x** |
| 100 MB | 1,171-1,502 MB/s | 7.5 MB/s | **150-200x** |

### When GPU Wins

- **Batch processing**: Many small buffers already on GPU (DEV→DEV)
- **Device-resident data**: Data that is already in GPU memory from prior CUDA operations
- **Decompression**: GPU decompression is competitive for large buffers (1+ GB/s)

### HybridConfig Tuning

```cpp
cuda_zstd::HybridConfig config;
config.mode = HybridMode::AUTO;            // Auto-select (default)
config.cpu_size_threshold = 1024 * 1024;   // CPU for host data above this (1 MB)
config.gpu_device_threshold = 64 * 1024;   // GPU for device data above this (64 KB)
config.compression_level = 3;               // Level 3 (default)
config.use_pinned_memory = true;            // Pinned staging buffers
config.overlap_transfers = true;            // Overlap D2H/H2D with compute

HybridEngine engine(config);
```

### Routing Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `AUTO` | Detect location, pick fastest | **Default -- use this** |
| `PREFER_CPU` | CPU unless data is on GPU | Latency-sensitive host workloads |
| `PREFER_GPU` | GPU unless data is on host | GPU-heavy pipelines |
| `FORCE_CPU` | Always CPU `libzstd` | Baseline comparison, debugging |
| `FORCE_GPU` | Always GPU kernels | Benchmark GPU path only |
| `ADAPTIVE` | Profile-guided (64-sample window) | Long-running mixed workloads |

### DataLocation Selection

| Data Is In | Set `DataLocation` To | Engine Routes To |
|------------|----------------------|-----------------|
| `malloc()`/`new` | `HOST` | CPU libzstd |
| `cudaMalloc()` | `DEVICE` | GPU kernels (or D2H→CPU→H2D if large) |
| `cudaMallocManaged()` | `MANAGED` | CPU (direct access) |
| Unknown | `UNKNOWN` | Auto-detected via `cudaPointerGetAttributes` |

---

## Learn More

- [Hybrid Engine](HYBRID-ENGINE.md) -- CPU/GPU routing engine deep dive
- [Batch Processing](BATCH-PROCESSING.md) -- Process thousands of files at once
- [Stream Optimization](STREAM-OPTIMIZATION.md) -- Deep dive on stream-based parallelism
- [Memory Pool](MEMORY-POOL-IMPLEMENTATION.md) -- Optimize GPU memory usage
- [Benchmarking Guide](BENCHMARKING-GUIDE.md) -- All 30 benchmark executables
- [Debugging Guide](DEBUGGING-GUIDE.md) -- When things do not go as planned

---

*Remember: Measure first, optimize second.*
