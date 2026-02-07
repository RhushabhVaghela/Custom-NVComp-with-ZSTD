# CUDA-ZSTD NVComp Integration Guide

## Overview

CUDA-ZSTD provides a complete NVComp v5 compatibility layer, enabling drop-in replacement for existing NVComp-based applications. The integration uses the `cuda_zstd::nvcomp_v5` namespace internally and exposes a C API with the `nvcomp_zstd_*_v5()` naming pattern.

## API Surface

The NVComp v5 integration provides two interfaces:

1. **C++ API** -- `NvcompV5BatchManager` class and related types in `cuda_zstd::nvcomp_v5` namespace.
2. **C API** -- 7 extern "C" functions in `cuda_zstd_nvcomp.h` for language-agnostic access.

### C API Functions

| Function | Purpose |
|:---------|:--------|
| `nvcomp_zstd_create_manager_v5()` | Create an NVComp-compatible ZSTD manager |
| `nvcomp_zstd_destroy_manager_v5()` | Destroy manager and free resources |
| `nvcomp_zstd_compress_async_v5()` | Asynchronous GPU compression |
| `nvcomp_zstd_decompress_async_v5()` | Asynchronous GPU decompression |
| `nvcomp_zstd_get_compress_temp_size_v5()` | Query compression workspace size |
| `nvcomp_zstd_get_decompress_temp_size_v5()` | Query decompression workspace size |
| `nvcomp_zstd_get_metadata_v5()` | Extract metadata from compressed data (C++ only) |

### C++ Types

| Type | Header | Purpose |
|:-----|:-------|:--------|
| `NvcompV5BatchManager` | `cuda_zstd_nvcomp.h` | Batch compression/decompression manager |
| `NvcompV5Options` | `cuda_zstd_nvcomp.h` | Configuration for NVComp operations |
| `NvcompV5Metadata` | `cuda_zstd_nvcomp.h` | Metadata extracted from compressed frames |

---

## Architecture

```
+-------------------------------------------------------------+
|                NVComp v5 Compatibility Layer                 |
|              (include/cuda_zstd_nvcomp.h)                    |
+-------------------------------------------------------------+
|                                                              |
|  C API (7 functions)    C++ API (NvcompV5BatchManager)       |
|  nvcomp_zstd_*_v5()     cuda_zstd::nvcomp_v5 namespace      |
|         |                         |                          |
|         +----------+--------------+                          |
|                    |                                         |
|                    v                                         |
|          CUDA-ZSTD Core Engine                               |
|          (Manager, LZ77, FSE, Huffman)                       |
+-------------------------------------------------------------+
```

---

## Usage: C API

```c
#include <cuda_runtime.h>
#include "cuda_zstd_nvcomp.h"

int main() {
    // 1. Create manager (compression level 1-22)
    nvcompZstdManagerHandle handle = nvcomp_zstd_create_manager_v5(3);

    // 2. Query workspace size
    size_t input_size = 1024 * 1024;  // 1MB
    size_t temp_size = nvcomp_zstd_get_compress_temp_size_v5(handle, input_size);

    // 3. Allocate buffers
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);
    cudaMalloc(&d_temp, temp_size);
    cudaMemset(d_input, 'A', input_size);

    // 4. Compress
    size_t compressed_size = input_size * 2;
    int status = nvcomp_zstd_compress_async_v5(
        handle, d_input, input_size,
        d_output, &compressed_size,
        d_temp, temp_size, 0);
    cudaDeviceSynchronize();

    // 5. Decompress
    size_t decomp_temp = nvcomp_zstd_get_decompress_temp_size_v5(handle, compressed_size);
    void *d_decomp_temp, *d_decomp_out;
    cudaMalloc(&d_decomp_temp, decomp_temp);
    cudaMalloc(&d_decomp_out, input_size);

    size_t decompressed_size = input_size;
    status = nvcomp_zstd_decompress_async_v5(
        handle, d_output, compressed_size,
        d_decomp_out, &decompressed_size,
        d_decomp_temp, decomp_temp, 0);
    cudaDeviceSynchronize();

    // 6. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_decomp_temp);
    cudaFree(d_decomp_out);
    nvcomp_zstd_destroy_manager_v5(handle);

    return status;
}
```

---

## Usage: C++ API

```cpp
#include "cuda_zstd_nvcomp.h"

using namespace cuda_zstd::nvcomp_v5;

// Create batch manager
NvcompV5Options opts;
opts.compression_level = 3;
auto batch_mgr = NvcompV5BatchManager::create(opts);

// Compress batch
batch_mgr->compress_batch(
    d_input_ptrs, d_input_sizes,
    d_output_ptrs, d_output_sizes,
    batch_count,
    d_workspace, workspace_size,
    stream);

// Decompress batch
batch_mgr->decompress_batch(
    d_compressed_ptrs, d_compressed_sizes,
    d_output_ptrs, d_output_sizes,
    batch_count,
    d_workspace, workspace_size,
    stream);
```

---

## Migration from NVIDIA nvCOMP

Migrating from NVIDIA's nvCOMP library to CUDA-ZSTD requires changing the include path and using the v5 naming convention. The core workflow remains the same.

### Before (NVIDIA nvCOMP)

```cpp
#include <nvcomp/zstd.h>

// NVIDIA nvCOMP batched API
nvcompBatchedZstdOpts_t opts = {0};
nvcompStatus_t status = nvcompBatchedZstdCompressAsync(
    inputs, input_sizes, max_size, batch_size,
    temp, temp_size, outputs, output_sizes,
    opts, stream);
```

### After (CUDA-ZSTD NVComp v5)

```cpp
#include "cuda_zstd_nvcomp.h"

// CUDA-ZSTD NVComp v5 API
nvcompZstdManagerHandle handle = nvcomp_zstd_create_manager_v5(3);
int status = nvcomp_zstd_compress_async_v5(
    handle, d_input, input_size,
    d_output, &compressed_size,
    d_temp, temp_size, stream);
```

The key differences:

| Aspect | NVIDIA nvCOMP | CUDA-ZSTD NVComp v5 |
|:-------|:-------------|:---------------------|
| Header | `<nvcomp/zstd.h>` | `"cuda_zstd_nvcomp.h"` |
| Naming | `nvcompBatchedZstd*` | `nvcomp_zstd_*_v5()` |
| Manager | Implicit (opts struct) | Explicit handle creation |
| Levels | Vendor-defined | Full ZSTD range: 1-22 |
| Dependency | Closed-source `libnvcomp` | Open-source, statically linked |

---

## Compression Levels

CUDA-ZSTD supports the full ZSTD compression level range (1-22) through the NVComp v5 API. This is not limited to a subset.

---

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_nvcomp.h` | NVComp v5 C and C++ API declarations |
| `src/cuda_zstd_nvcomp.cu` | NVComp v5 implementation |
| `tests/test_nvcomp_interface.cu` | NVComp v5 interface tests |
| `tests/test_nvcomp_batch.cu` | Batch API tests |
| `benchmarks/benchmark_nvcomp_interface.cu` | NVComp v5 benchmarks |

---

## Related Documentation

- [C API Reference](C-API-REFERENCE.md) -- Full C API documentation including NVComp v5 functions
- [Batch Processing](BATCH-PROCESSING.md) -- Batch compression patterns
- [Manager Implementation](MANAGER-IMPLEMENTATION.md) -- Internal manager architecture
