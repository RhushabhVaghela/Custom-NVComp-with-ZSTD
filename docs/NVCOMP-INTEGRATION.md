# CUDA-ZSTD NVComp Integration Guide

## Overview

CUDA-ZSTD provides a compatibility layer with NVIDIA's nvCOMP library, enabling drop-in replacement for existing nvCOMP-based applications.

## Compatibility Matrix

| nvCOMP API | CUDA-ZSTD Equivalent | Status |
|:-----------|:---------------------|:------:|
| `nvcompBatchedZstdCompressAsync` | `ZstdBatchManager::compress_batch` | ✅ |
| `nvcompBatchedZstdDecompressAsync` | `ZstdBatchManager::decompress_batch` | ✅ |
| `nvcompBatchedZstdGetDecompressSizeAsync` | `get_decompressed_size` | ✅ |
| `nvcompBatchedZstdCompressGetTempSize` | `get_batch_workspace_size` | ✅ |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   nvCOMP Compatibility Layer                 │
│                 (src/cuda_zstd_nvcomp.cpp)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ nvcomp-style API │ →  │ CUDA-ZSTD Native │              │
│  │   (Batched)      │    │   (Batch Mgr)    │              │
│  └──────────────────┘    └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Compression

```cpp
#include "cuda_zstd_nvcomp.h"

nvcompStatus_t nvcompBatchedZstdCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const nvcompBatchedZstdOpts_t format_opts,
    cudaStream_t stream
);
```

### Decompression

```cpp
nvcompStatus_t nvcompBatchedZstdDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    nvcompStatus_t* device_statuses,
    cudaStream_t stream
);
```

## Migration Guide

### Before (nvCOMP)
```cpp
#include <nvcomp/zstd.h>

nvcompBatchedZstdOpts_t opts = {0};
nvcompStatus_t status = nvcompBatchedZstdCompressAsync(
    inputs, input_sizes, max_size, batch_size,
    temp, temp_size, outputs, output_sizes,
    opts, stream
);
```

### After (CUDA-ZSTD)
```cpp
#include "cuda_zstd_nvcomp.h"

nvcompBatchedZstdOpts_t opts = {0};
nvcompStatus_t status = nvcompBatchedZstdCompressAsync(
    inputs, input_sizes, max_size, batch_size,
    temp, temp_size, outputs, output_sizes,
    opts, stream
);
// Same API! Just change the include
```

## Performance Comparison

| Operation | nvCOMP | CUDA-ZSTD | Notes |
|:----------|:------:|:---------:|:------|
| Batch 64KB × 100 | 6.2 GB/s | 8.5 GB/s | +37% |
| Batch 256KB × 50 | 8.1 GB/s | 12.3 GB/s | +52% |
| Decompress | 15.2 GB/s | 18.7 GB/s | +23% |

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_nvcomp.cpp` | nvCOMP compatibility layer |
| `include/cuda_zstd_nvcomp.h` | Public API header |
| `tests/test_nvcomp_interface.cu` | Compatibility tests |
| `tests/test_nvcomp_batch.cu` | Batch API tests |
| `benchmarks/benchmark_nvcomp_interface.cu` | Performance comparison |

## Limitations

1. **Dictionary Compression**: Not exposed via nvCOMP API (use native API)
2. **Streaming**: nvCOMP batched API is single-shot only
3. **Compression Levels**: Limited to nvCOMP-compatible levels (1-9)

## Related Documentation
- [BATCH-PROCESSING.md](BATCH-PROCESSING.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
