# CUDA-ZSTD Batch Processing Guide

## Overview

Batch processing enables parallel compression/decompression of multiple independent data chunks, achieving **>60 GB/s throughput** through GPU parallelism.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZstdBatchManager                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Chunk 0 │  │ Chunk 1 │  │ Chunk 2 │  ...  │ Chunk N │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                  │            │
│       ▼            ▼            ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Parallel GPU Kernel Execution                   ││
│  │  (LZ77 → Parse → Sequence → FSE/Huffman) × N                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Performance Results

| Chunk Size | Batch Count | Throughput | Notes |
|:-----------|:-----------:|:----------:|:------|
| 4 KB | 2000 | 2.5 GB/s | Limited by launch overhead |
| 16 KB | 1500 | 9.1 GB/s | Near linear scaling |
| 64 KB | 1000 | **29.4 GB/s** | Optimal for many use cases |
| 256 KB | 500 | **61.9 GB/s** | Maximum throughput |

## API Reference

### ZstdBatchManager Class

```cpp
class ZstdBatchManager {
public:
    // Create batch manager
    static std::unique_ptr<ZstdBatchManager> create(int level = 3);
    
    // Batch compression
    Status compress_batch(
        const void* const* d_inputs,    // Array of input pointers
        const size_t* input_sizes,      // Array of input sizes
        void* const* d_outputs,         // Array of output pointers
        size_t* output_sizes,           // Array of output sizes
        size_t batch_count,             // Number of items
        void* d_workspace,              // Shared workspace
        size_t workspace_size,          // Workspace size
        cudaStream_t stream = 0
    );
    
    // Batch decompression
    Status decompress_batch(
        const void* const* d_inputs,
        const size_t* input_sizes,
        void* const* d_outputs,
        size_t* output_sizes,
        size_t batch_count,
        void* d_workspace,
        size_t workspace_size,
        cudaStream_t stream = 0
    );
    
    // Query workspace size for batch
    size_t get_batch_workspace_size(size_t max_chunk_size, size_t batch_count);
};
```

## Usage Examples

### Basic Batch Compression

```cpp
#include "cuda_zstd_manager.h"
#include <vector>

void compress_batch_example() {
    using namespace cuda_zstd;
    
    // Create batch manager (level 3 for speed)
    auto manager = ZstdBatchManager::create(3);
    
    // Prepare batch data
    const size_t batch_count = 100;
    const size_t chunk_size = 64 * 1024;  // 64KB chunks
    
    // Allocate device arrays
    std::vector<void*> d_inputs(batch_count);
    std::vector<void*> d_outputs(batch_count);
    std::vector<size_t> input_sizes(batch_count, chunk_size);
    std::vector<size_t> output_sizes(batch_count);
    
    for (size_t i = 0; i < batch_count; ++i) {
        cudaMalloc(&d_inputs[i], chunk_size);
        cudaMalloc(&d_outputs[i], chunk_size * 2);
        output_sizes[i] = chunk_size * 2;
    }
    
    // Allocate workspace
    size_t ws_size = manager->get_batch_workspace_size(chunk_size, batch_count);
    void* d_workspace;
    cudaMalloc(&d_workspace, ws_size);
    
    // Compress batch
    Status status = manager->compress_batch(
        d_inputs.data(), input_sizes.data(),
        d_outputs.data(), output_sizes.data(),
        batch_count,
        d_workspace, ws_size,
        0  // default stream
    );
    
    // output_sizes now contains actual compressed sizes
    size_t total_input = batch_count * chunk_size;
    size_t total_output = 0;
    for (size_t s : output_sizes) total_output += s;
    
    printf("Batch: %zu items, %zu -> %zu bytes (%.2fx)\n",
           batch_count, total_input, total_output,
           (float)total_input / total_output);
}
```

### OpenMP Parallel Multi-Manager Pattern

For maximum throughput, use multiple managers with OpenMP:

```cpp
#include <omp.h>
#include "cuda_zstd_manager.h"

void parallel_batch_compression() {
    const int num_threads = 8;
    const size_t items_per_thread = 100;
    const size_t total_items = num_threads * items_per_thread;
    
    // Each thread gets its own manager and stream
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        // Create per-thread manager
        auto manager = cuda_zstd::create_manager(3);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Process this thread's portion
        for (size_t i = 0; i < items_per_thread; ++i) {
            size_t idx = tid * items_per_thread + i;
            
            size_t compressed_size;
            manager->compress(
                d_inputs[idx], sizes[idx],
                d_outputs[idx], &compressed_size,
                d_temps[tid], temp_size,
                nullptr, 0,
                stream
            );
        }
        
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}
```

## Workspace Management

### Sizing Guidelines

| Batch Count | Chunk Size | Recommended Workspace |
|:-----------:|:----------:|:---------------------:|
| 10 | 64 KB | ~50 MB |
| 100 | 64 KB | ~100 MB |
| 1000 | 64 KB | ~500 MB |

### Workspace Structure
```
┌────────────────────────────────────────┐
│ Per-Chunk Buffers (N × chunk_size)     │
├────────────────────────────────────────┤
│ Hash Tables (shared, 2 MB)             │
├────────────────────────────────────────┤
│ FSE/Huffman Tables (N × 64 KB)         │
├────────────────────────────────────────┤
│ Sequence Buffers (N × 512 KB)          │
└────────────────────────────────────────┘
```

## Best Practices

1. **Use Appropriate Chunk Sizes**: 64KB-256KB for best GPU utilization
2. **Pin Host Memory**: Use `cudaMallocHost` for input/output staging
3. **Stream Overlap**: Use multiple streams for H2D/D2H overlap
4. **Pre-allocate Workspace**: Reuse workspace across batches
5. **Batch Similar Sizes**: Group chunks by size for efficient packing

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_manager.cu` | ZstdBatchManager implementation |
| `benchmarks/benchmark_batch_throughput.cu` | Batch performance tests |
| `tests/test_nvcomp_batch.cu` | Batch correctness tests |

## Related Documentation
- [PERFORMANCE-TUNING.md](PERFORMANCE-TUNING.md)
- [STREAMING-API.md](STREAMING-API.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
