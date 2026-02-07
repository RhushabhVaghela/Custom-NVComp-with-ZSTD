# Performance Tuning Guide

> *"The difference between good and great? About 50 GB/s."*

## Why Performance Matters

Proper tuning can improve throughput by orders of magnitude. This guide covers the key techniques to go from "it works" to "it flies."

---

## Measured Performance Baselines

These numbers were measured on an RTX 5080 Laptop GPU (Blackwell sm_120):

| Metric | Measured Value | Configuration |
|:-------|:---------------|:--------------|
| **Peak Compress Throughput** | 800.3 MB/s | Single stream, 256 KB blocks |
| **Peak Decompress Throughput** | 1.77 GB/s | Single stream, 256 KB blocks |
| **Peak Batch Throughput** | 9.81 GB/s | 250 x 256 KB blocks |
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
- [ ] Debug logging is disabled (`CUDA_ZSTD_DEBUG_LEVEL=0`)
- [ ] Built in Release mode (`-DCMAKE_BUILD_TYPE=Release`)

---

## Learn More

- [Batch Processing](BATCH-PROCESSING.md) -- Process thousands of files at once
- [Stream Optimization](STREAM-OPTIMIZATION.md) -- Deep dive on stream-based parallelism
- [Memory Pool](MEMORY-POOL-IMPLEMENTATION.md) -- Optimize GPU memory usage
- [Benchmarking Guide](BENCHMARKING-GUIDE.md) -- All 30 benchmark executables
- [Debugging Guide](DEBUGGING-GUIDE.md) -- When things do not go as planned

---

*Remember: Measure first, optimize second.*
