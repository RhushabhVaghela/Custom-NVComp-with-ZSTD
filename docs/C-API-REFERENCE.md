# CUDA-ZSTD C API Reference

## Overview

The C API provides a stable, ABI-compatible interface for integrating CUDA-ZSTD into C applications, dynamic language bindings (Python, Rust, etc.), and legacy systems. There are two C APIs:

1. **CUDA-ZSTD C API** -- 11 functions in `cuda_zstd_manager.h` for direct compression/decompression with dictionary support.
2. **NVComp v5 C API** -- 7 functions in `cuda_zstd_nvcomp.h` providing an NVComp-compatible interface.

## Header Files

```c
#include "cuda_zstd_manager.h"  // CUDA-ZSTD C API (11 functions)
#include "cuda_zstd_nvcomp.h"   // NVComp v5 C API (7 functions)
```

---

## Opaque Types

### CUDA-ZSTD Types

```c
typedef struct cuda_zstd_manager_t cuda_zstd_manager_t;
typedef struct cuda_zstd_dict_t cuda_zstd_dict_t;
```

- `cuda_zstd_manager_t` -- Opaque handle to a compression/decompression manager.
- `cuda_zstd_dict_t` -- Opaque handle to a trained dictionary.

### NVComp v5 Type

```c
typedef void* nvcompZstdManagerHandle;
```

- `nvcompZstdManagerHandle` -- Opaque void pointer to an NVComp-compatible manager.

---

## CUDA-ZSTD C API (11 Functions)

All functions are declared in `include/cuda_zstd_manager.h` (lines 433-479).

### Manager Lifecycle

#### cuda_zstd_create_manager

```c
cuda_zstd_manager_t* cuda_zstd_create_manager(int compression_level);
```

Creates a new compression manager.

**Parameters:**
- `compression_level`: ZSTD compression level, 1-22. Lower is faster, higher compresses more.

**Returns:** Pointer to a new manager, or `NULL` on failure.

---

#### cuda_zstd_destroy_manager

```c
void cuda_zstd_destroy_manager(cuda_zstd_manager_t* manager);
```

Destroys a manager and frees all associated resources.

**Parameters:**
- `manager`: Manager to destroy. Safe to pass `NULL`.

---

### Compression

#### cuda_zstd_get_compress_workspace_size

```c
size_t cuda_zstd_get_compress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t src_size
);
```

Queries the required workspace buffer size for compression.

**Parameters:**
- `manager`: Active manager handle.
- `src_size`: Size of the uncompressed input in bytes.

**Returns:** Required workspace size in bytes.

---

#### cuda_zstd_compress

```c
int cuda_zstd_compress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
);
```

Compresses data on the GPU.

**Parameters:**
- `manager`: Active manager handle.
- `src`: Device pointer to uncompressed input data.
- `src_size`: Size of input data in bytes.
- `dst`: Device pointer to output buffer.
- `dst_size`: On input, maximum output buffer size. On output, actual compressed size.
- `workspace`: Device pointer to temporary workspace buffer.
- `workspace_size`: Size of workspace buffer (from `cuda_zstd_get_compress_workspace_size`).
- `stream`: CUDA stream for async execution. Use `0` for default stream.

**Returns:** `0` on success, non-zero error code on failure.

---

### Decompression

#### cuda_zstd_get_decompress_workspace_size

```c
size_t cuda_zstd_get_decompress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t compressed_size
);
```

Queries the required workspace buffer size for decompression.

**Parameters:**
- `manager`: Active manager handle.
- `compressed_size`: Size of the compressed data in bytes.

**Returns:** Required workspace size in bytes.

---

#### cuda_zstd_decompress

```c
int cuda_zstd_decompress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
);
```

Decompresses data on the GPU.

**Parameters:**
- `manager`: Active manager handle.
- `src`: Device pointer to compressed input data.
- `src_size`: Size of compressed data in bytes.
- `dst`: Device pointer to output buffer.
- `dst_size`: On input, maximum output buffer size. On output, actual decompressed size.
- `workspace`: Device pointer to temporary workspace buffer.
- `workspace_size`: Size of workspace buffer (from `cuda_zstd_get_decompress_workspace_size`).
- `stream`: CUDA stream for async execution. Use `0` for default stream.

**Returns:** `0` on success, non-zero error code on failure.

---

### Dictionary API

Dictionaries improve compression ratios for small inputs that share common patterns (e.g., JSON records, log lines). The workflow is: train, set on manager, compress/decompress, destroy.

#### cuda_zstd_train_dictionary

```c
cuda_zstd_dict_t* cuda_zstd_train_dictionary(
    const void** samples,
    const size_t* sample_sizes,
    size_t num_samples,
    size_t dict_size
);
```

Trains a compression dictionary from representative samples.

**Parameters:**
- `samples`: Array of pointers to sample data buffers (host memory).
- `sample_sizes`: Array of sizes for each sample.
- `num_samples`: Number of samples in the arrays.
- `dict_size`: Desired dictionary size in bytes (typically 16KB-112KB).

**Returns:** Pointer to a trained dictionary, or `NULL` on failure.

---

#### cuda_zstd_destroy_dictionary

```c
void cuda_zstd_destroy_dictionary(cuda_zstd_dict_t* dict);
```

Destroys a dictionary and frees associated memory.

**Parameters:**
- `dict`: Dictionary to destroy. Safe to pass `NULL`.

---

#### cuda_zstd_set_dictionary

```c
int cuda_zstd_set_dictionary(
    cuda_zstd_manager_t* manager,
    cuda_zstd_dict_t* dict
);
```

Assigns a trained dictionary to a manager. All subsequent compress/decompress calls will use this dictionary.

**Parameters:**
- `manager`: Active manager handle.
- `dict`: Trained dictionary handle.

**Returns:** `0` on success, non-zero error code on failure.

---

### Error Handling

#### cuda_zstd_get_error_string

```c
const char* cuda_zstd_get_error_string(int error_code);
```

Returns a human-readable string for an error code.

**Parameters:**
- `error_code`: Error code returned by a C API function.

**Returns:** Static string describing the error.

---

#### cuda_zstd_is_error

```c
int cuda_zstd_is_error(int code);
```

Tests whether a return code represents an error.

**Parameters:**
- `code`: Return code from a C API function.

**Returns:** Non-zero if `code` is an error, `0` if success.

---

## NVComp v5 C API (7 Functions)

All functions are declared in `include/cuda_zstd_nvcomp.h` (lines 276-336). This API provides compatibility with NVIDIA's NVComp library interface.

### Manager Lifecycle

#### nvcomp_zstd_create_manager_v5

```c
nvcompZstdManagerHandle nvcomp_zstd_create_manager_v5(
    int compression_level
);
```

Creates an NVComp-compatible ZSTD manager.

**Parameters:**
- `compression_level`: ZSTD compression level, 1-22.

**Returns:** Opaque handle (`void*`), or `NULL` on failure.

---

#### nvcomp_zstd_destroy_manager_v5

```c
void nvcomp_zstd_destroy_manager_v5(
    nvcompZstdManagerHandle handle
);
```

Destroys an NVComp manager and frees resources.

---

### Compression

#### nvcomp_zstd_get_compress_temp_size_v5

```c
size_t nvcomp_zstd_get_compress_temp_size_v5(
    nvcompZstdManagerHandle handle,
    size_t uncompressed_size
);
```

Queries temporary buffer size needed for compression.

**Returns:** Required temporary buffer size in bytes.

---

#### nvcomp_zstd_compress_async_v5

```c
int nvcomp_zstd_compress_async_v5(
    nvcompZstdManagerHandle handle,
    const void* d_uncompressed,
    size_t uncompressed_size,
    void* d_compressed,
    size_t* compressed_size,
    void* d_temp,
    size_t temp_size,
    cudaStream_t stream
);
```

Asynchronously compresses data on the GPU.

**Parameters:**
- `handle`: NVComp manager handle.
- `d_uncompressed`: Device pointer to input data.
- `uncompressed_size`: Input data size in bytes.
- `d_compressed`: Device pointer to output buffer.
- `compressed_size`: On input, max output size. On output, actual compressed size.
- `d_temp`: Device pointer to temporary workspace.
- `temp_size`: Workspace size (from `nvcomp_zstd_get_compress_temp_size_v5`).
- `stream`: CUDA stream.

**Returns:** `0` on success, non-zero on failure.

---

### Decompression

#### nvcomp_zstd_get_decompress_temp_size_v5

```c
size_t nvcomp_zstd_get_decompress_temp_size_v5(
    nvcompZstdManagerHandle handle,
    size_t compressed_size
);
```

Queries temporary buffer size needed for decompression.

**Returns:** Required temporary buffer size in bytes.

---

#### nvcomp_zstd_decompress_async_v5

```c
int nvcomp_zstd_decompress_async_v5(
    nvcompZstdManagerHandle handle,
    const void* d_compressed,
    size_t compressed_size,
    void* d_uncompressed,
    size_t* uncompressed_size,
    void* d_temp,
    size_t temp_size,
    cudaStream_t stream
);
```

Asynchronously decompresses data on the GPU.

**Parameters:**
- `handle`: NVComp manager handle.
- `d_compressed`: Device pointer to compressed input.
- `compressed_size`: Compressed data size in bytes.
- `d_uncompressed`: Device pointer to output buffer.
- `uncompressed_size`: On input, max output size. On output, actual decompressed size.
- `d_temp`: Device pointer to temporary workspace.
- `temp_size`: Workspace size (from `nvcomp_zstd_get_decompress_temp_size_v5`).
- `stream`: CUDA stream.

**Returns:** `0` on success, non-zero on failure.

---

### Metadata (C++ Only)

#### nvcomp_zstd_get_metadata_v5

```c
// Only available in C++ translation units (behind #ifdef __cplusplus)
int nvcomp_zstd_get_metadata_v5(
    const void* d_compressed_data,
    size_t compressed_size,
    cuda_zstd::nvcomp_v5::NvcompV5Metadata* h_metadata,
    cudaStream_t stream
);
```

Extracts metadata from compressed data. This function is only available from C++ code because it uses the `NvcompV5Metadata` struct.

**Parameters:**
- `d_compressed_data`: Device pointer to compressed data.
- `compressed_size`: Size of compressed data.
- `h_metadata`: Host pointer to metadata output struct.
- `stream`: CUDA stream.

**Returns:** `0` on success, non-zero on failure.

---

## Complete Example (CUDA-ZSTD C API)

```c
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_zstd_manager.h"

int main() {
    cuda_zstd_manager_t* manager = NULL;
    int status;

    // 1. Create manager (level 5)
    manager = cuda_zstd_create_manager(5);
    if (manager == NULL) {
        fprintf(stderr, "Failed to create manager\n");
        return 1;
    }

    // 2. Prepare data
    size_t input_size = 1024 * 1024;  // 1MB
    void *d_src, *d_dst, *d_workspace;
    size_t workspace_size = cuda_zstd_get_compress_workspace_size(manager, input_size);

    cudaMalloc(&d_src, input_size);
    cudaMalloc(&d_dst, input_size * 2);
    cudaMalloc(&d_workspace, workspace_size);

    // Fill input with test data
    cudaMemset(d_src, 'A', input_size);

    // 3. Compress
    size_t dst_size = input_size * 2;
    status = cuda_zstd_compress(manager,
                                d_src, input_size,
                                d_dst, &dst_size,
                                d_workspace, workspace_size,
                                0);

    if (cuda_zstd_is_error(status)) {
        fprintf(stderr, "Compression failed: %s\n",
                cuda_zstd_get_error_string(status));
    } else {
        printf("Compressed %zu -> %zu bytes (%.2fx)\n",
               input_size, dst_size,
               (float)input_size / dst_size);
    }

    // 4. Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_workspace);
    cuda_zstd_destroy_manager(manager);

    return status;
}
```

## Dictionary Workflow Example

```c
#include "cuda_zstd_manager.h"

int main() {
    // 1. Prepare training samples (host memory)
    const char* samples[] = {
        "{\"user\":\"alice\",\"action\":\"login\"}",
        "{\"user\":\"bob\",\"action\":\"logout\"}",
        "{\"user\":\"charlie\",\"action\":\"upload\"}"
    };
    size_t sample_sizes[] = { 37, 36, 38 };

    // 2. Train dictionary
    cuda_zstd_dict_t* dict = cuda_zstd_train_dictionary(
        (const void**)samples, sample_sizes, 3, 16384);
    if (!dict) {
        fprintf(stderr, "Dictionary training failed\n");
        return 1;
    }

    // 3. Create manager and attach dictionary
    cuda_zstd_manager_t* manager = cuda_zstd_create_manager(3);
    int status = cuda_zstd_set_dictionary(manager, dict);
    if (cuda_zstd_is_error(status)) {
        fprintf(stderr, "Set dictionary failed: %s\n",
                cuda_zstd_get_error_string(status));
    }

    // 4. Compress/decompress as normal (dictionary used automatically)
    // ...

    // 5. Cleanup
    cuda_zstd_destroy_manager(manager);
    cuda_zstd_destroy_dictionary(dict);
    return 0;
}
```

## NVComp v5 Example

```c
#include <cuda_runtime.h>
#include "cuda_zstd_nvcomp.h"

int main() {
    // 1. Create NVComp manager
    nvcompZstdManagerHandle handle = nvcomp_zstd_create_manager_v5(3);

    // 2. Query workspace size
    size_t input_size = 1024 * 1024;
    size_t temp_size = nvcomp_zstd_get_compress_temp_size_v5(handle, input_size);

    // 3. Allocate buffers
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);
    cudaMalloc(&d_temp, temp_size);
    cudaMemset(d_input, 'A', input_size);

    // 4. Compress asynchronously
    size_t compressed_size = input_size * 2;
    int status = nvcomp_zstd_compress_async_v5(
        handle, d_input, input_size,
        d_output, &compressed_size,
        d_temp, temp_size, 0);

    cudaDeviceSynchronize();

    // 5. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    nvcomp_zstd_destroy_manager_v5(handle);

    return status;
}
```

## Thread Safety

- Manager instances are **NOT** thread-safe.
- Create separate managers for each thread.
- CUDA streams provide async safety within a single manager.
- Dictionary objects are read-only after training and can be shared across managers.

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_manager.h` | C API declarations (lines 433-479) |
| `include/cuda_zstd_nvcomp.h` | NVComp v5 C API declarations (lines 276-336) |
| `src/cuda_zstd_c_api.cpp` | C API implementation (212 lines) |
| `tests/test_c_api.cpp` | C API test suite |
| `tests/test_c_api_edge_cases.cu` | Edge case tests |

## Related Documentation

- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
- [QUICK-REFERENCE.md](QUICK-REFERENCE.md)
