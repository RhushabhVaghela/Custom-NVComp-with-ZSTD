# CUDA-ZSTD Performance Tuning Guide

## Overview

This guide covers optimization strategies for achieving maximum throughput with CUDA-ZSTD, including configuration tuning, memory optimization, and parallel execution patterns.

## Performance Targets

| Scenario | Target | Key Optimizations |
|:---------|:------:|:------------------|
| Single-shot (>1MB) | 10+ GB/s | Level 1-3, large blocks |
| Streaming | 5-8 GB/s | Chunk overlap, prefetch |
| Batch (>100 items) | 30-60+ GB/s | OpenMP parallel managers |
| Low Latency | <100µs | Small chunks, pinned memory |

## Compression Level Selection

| Level | Speed | Ratio | Use Case |
|:-----:|:-----:|:-----:|:---------|
| 1 | ★★★★★ | ★★ | Real-time streaming, logs |
| 3 | ★★★★ | ★★★ | General purpose (default) |
| 5 | ★★★ | ★★★★ | Balanced workloads |
| 9 | ★★ | ★★★★★ | Archival, cold storage |
| 15+ | ★ | ★★★★★★ | Maximum compression |

## Memory Configuration

### Optimal Block Sizes

```
┌─────────────────────────────────────────────────────────────┐
│ Block Size Selection Guide                                   │
├───────────────┬───────────────┬─────────────────────────────┤
│ Block Size    │ GPU Util      │ Recommendation              │
├───────────────┼───────────────┼─────────────────────────────┤
│ 16 KB         │ 60-70%        │ Many small files            │
│ 64 KB         │ 80-85%        │ Balanced (recommended)      │
│ 128 KB        │ 85-90%        │ Large files, high throughput│
│ 256 KB        │ 90-95%        │ Maximum throughput          │
│ 512 KB+       │ 85-90%        │ Diminishing returns         │
└───────────────┴───────────────┴─────────────────────────────┘
```

### Memory Layout Optimization

```cpp
// Optimal memory alignment (256 bytes for coalescing)
size_t aligned_size = (size + 255) & ~255;

// Use pinned memory for H2D/D2H transfers
void* h_pinned;
cudaMallocHost(&h_pinned, size);  // 2-3x faster transfers

// Pre-allocate workspace
size_t workspace_size = manager->get_compress_temp_size(max_size);
void* d_workspace;
cudaMalloc(&d_workspace, workspace_size);
```

## Parallelization Strategies

### Strategy 1: Multi-Stream Overlap
```cpp
const int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];

for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamCreate(&streams[i]);
}

// Overlap H2D, compute, D2H across streams
for (int i = 0; i < num_chunks; ++i) {
    int s = i % NUM_STREAMS;
    
    cudaMemcpyAsync(d_input[s], h_input[i], size, 
                    cudaMemcpyHostToDevice, streams[s]);
    
    manager->compress(d_input[s], size,
                      d_output[s], &out_size,
                      d_temp[s], temp_size,
                      nullptr, 0, streams[s]);
    
    cudaMemcpyAsync(h_output[i], d_output[s], out_size,
                    cudaMemcpyDeviceToHost, streams[s]);
}

for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamSynchronize(streams[i]);
}
```

### Strategy 2: OpenMP Multi-Manager (Highest Throughput)
```cpp
#include <omp.h>

const int NUM_THREADS = 8;

#pragma omp parallel num_threads(NUM_THREADS)
{
    // Each thread: own manager + stream
    auto manager = cuda_zstd::create_manager(3);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < num_chunks; ++i) {
        manager->compress(d_inputs[i], sizes[i],
                          d_outputs[i], &out_sizes[i],
                          d_temps[omp_get_thread_num()], temp_sz,
                          nullptr, 0, stream);
    }
    
    cudaStreamSynchronize(stream);
}
```

### Strategy 3: CUDA Graphs (Reduced Launch Overhead)
```cpp
// Note: Limited applicability due to hybrid CPU-GPU architecture
// Use for decompression or repeated identical operations

cudaGraph_t graph;
cudaGraphExec_t instance;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... compression operations ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// Execute captured graph
for (int i = 0; i < iterations; ++i) {
    cudaGraphLaunch(instance, stream);
}
```

## Profiling and Monitoring

### Enable Built-in Profiling
```cpp
auto manager = cuda_zstd::create_manager(3);
manager->enable_profiling(true);

// After operations
auto stats = manager->get_stats();
printf("Compressed: %zu MB\n", stats.bytes_compressed / (1024*1024));
printf("Ratio: %.2fx\n", stats.compression_ratio);
printf("Throughput: %.2f GB/s\n", stats.throughput_gbps);
```

### NVIDIA Nsight Integration
```bash
# Profile with Nsight Systems
nsys profile --stats=true ./my_app

# Detailed kernel analysis
ncu --set full ./my_app
```

## Environment Variables

| Variable | Default | Description |
|:---------|:-------:|:------------|
| `CUDA_ZSTD_DEBUG_LEVEL` | 0 | Debug verbosity (0-3) |
| `CUDA_ZSTD_POOL_MAX_SIZE` | 2GB | Max memory pool size |
| `CUDA_ZSTD_ENABLE_PROFILING` | 0 | Enable timing stats |
| `CUDA_ZSTD_BLOCK_SIZE` | 128KB | Default block size |

## Common Bottlenecks

| Symptom | Cause | Solution |
|:--------|:------|:---------|
| Low GPU utilization | Small chunks | Increase chunk size |
| High latency | Sync overhead | Use streams, async |
| OOM errors | Large workspace | Use memory pool |
| Slow H2D/D2H | Pageable memory | Use pinned memory |
| Poor scaling | Manager contention | Multi-manager pattern |

## Benchmarking

### Quick Performance Test
```bash
cd build
./benchmark_batch_throughput
```

### Expected Output
```
=== ZSTD GPU Batch Performance ===
Target: >10GB/s (Batched)
      Size |    Batch | Compress (MB/s) | Decompress (MB/s)
-------------------------------------------------------------
Parallel Batch 4096 | 2000 | 3.31 ms | 2.47 GB/s
Parallel Batch 16384 | 1500 | 2.71 ms | 9.08 GB/s
Parallel Batch 65536 | 1000 | 2.23 ms | 29.42 GB/s
Parallel Batch 262144 | 500 | 2.12 ms | 61.91 GB/s
```

## Source Files

| File | Description |
|:-----|:------------|
| `benchmarks/benchmark_batch_throughput.cu` | Parallel benchmark |
| `benchmarks/run_performance_suite.cu` | Full suite |
| `include/performance_profiler.h` | Profiling API |

## Related Documentation
- [BATCH-PROCESSING.md](BATCH-PROCESSING.md)
- [STREAMING-API.md](STREAMING-API.md)
- [MEMORY-POOL-IMPLEMENTATION.md](MEMORY-POOL-IMPLEMENTATION.md)
