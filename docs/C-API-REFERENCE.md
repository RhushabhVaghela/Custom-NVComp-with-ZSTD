# CUDA-ZSTD C API Reference

## Overview

The C API provides a stable, ABI-compatible interface for integrating CUDA-ZSTD into C applications, dynamic language bindings (Python, Rust, etc.), and legacy systems. There are three C APIs (25 functions total):

1. **CUDA-ZSTD C API** -- 11 functions in `cuda_zstd_manager.h` for direct compression/decompression with dictionary support.
2. **NVComp v5 C API** -- 7 functions in `cuda_zstd_nvcomp.h` providing an NVComp-compatible interface.
3. **Hybrid C API** -- 7 functions in `cuda_zstd_hybrid.h` for automatic CPU/GPU routing with zero-copy transfers.

## Header Files

```c
#include "cuda_zstd_manager.h"  // CUDA-ZSTD C API (11 functions)
#include "cuda_zstd_nvcomp.h"   // NVComp v5 C API (7 functions)
#include "cuda_zstd_hybrid.h"   // Hybrid C API (7 functions)
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

### Hybrid Type

```c
typedef struct cuda_zstd_hybrid_engine_t cuda_zstd_hybrid_engine_t;
```

- `cuda_zstd_hybrid_engine_t` -- Opaque handle to a hybrid CPU/GPU compression engine.

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

---

## Hybrid C API (7 Functions)

All functions are declared in `include/cuda_zstd_hybrid.h` (lines 296-359). This API provides automatic CPU/GPU routing: it selects the fastest execution backend (CPU libzstd or GPU kernels) based on data size, location, and operation type.

### Configuration Structures

```c
typedef struct {
    unsigned int mode;              // HybridMode: 0=AUTO, 1=PREFER_CPU, 2=PREFER_GPU,
                                    //             3=FORCE_CPU, 4=FORCE_GPU, 5=ADAPTIVE
    size_t cpu_size_threshold;      // Below this size, prefer CPU (default: 1MB)
    size_t gpu_device_threshold;    // Device data below this size still uses GPU (default: 64KB)
    int compression_level;          // ZSTD level 1-22 (default: 3)
    int enable_profiling;           // Non-zero to enable timing/profiling (default: 0)
    unsigned int cpu_thread_count;  // CPU threads for parallel compression (0 = auto)
} cuda_zstd_hybrid_config_t;

typedef struct {
    unsigned int backend_used;      // ExecutionBackend: 0=CPU_LIBZSTD, 1=GPU_KERNELS,
                                    //                  2=CPU_PARALLEL, 3=GPU_BATCH
    unsigned int input_location;    // DataLocation of actual input
    unsigned int output_location;   // DataLocation of actual output
    double total_time_ms;           // Wall-clock time for the operation
    double transfer_time_ms;        // Time spent on H2D/D2H transfers
    double compute_time_ms;         // Time spent on compression/decompression
    double throughput_mbps;          // Effective throughput in MB/s
    size_t input_bytes;             // Input size in bytes
    size_t output_bytes;            // Output size in bytes
    float compression_ratio;        // input_bytes / output_bytes
} cuda_zstd_hybrid_result_t;
```

### Engine Lifecycle

#### cuda_zstd_hybrid_create

```c
cuda_zstd_hybrid_engine_t* cuda_zstd_hybrid_create(
    const cuda_zstd_hybrid_config_t* config
);
```

Creates a hybrid engine with the specified configuration.

**Parameters:**
- `config`: Pointer to a configuration struct. Must not be `NULL`.

**Returns:** Pointer to a new engine, or `NULL` on failure.

---

#### cuda_zstd_hybrid_create_default

```c
cuda_zstd_hybrid_engine_t* cuda_zstd_hybrid_create_default(void);
```

Creates a hybrid engine with default settings (AUTO mode, level 3, 1MB CPU threshold).

**Returns:** Pointer to a new engine, or `NULL` on failure.

---

#### cuda_zstd_hybrid_destroy

```c
void cuda_zstd_hybrid_destroy(cuda_zstd_hybrid_engine_t* engine);
```

Destroys a hybrid engine and frees all associated resources.

**Parameters:**
- `engine`: Engine to destroy. Safe to pass `NULL`.

---

### Compression & Decompression

#### cuda_zstd_hybrid_compress

```c
int cuda_zstd_hybrid_compress(
    cuda_zstd_hybrid_engine_t* engine,
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    unsigned int input_location,
    unsigned int output_location,
    cuda_zstd_hybrid_result_t* result,
    cudaStream_t stream
);
```

Compresses data using the optimal backend (CPU or GPU) based on the engine's configuration and data characteristics.

**Parameters:**
- `engine`: Active hybrid engine handle.
- `input`: Pointer to uncompressed input data (host or device, as indicated by `input_location`).
- `input_size`: Size of input data in bytes.
- `output`: Pointer to output buffer (host or device, as indicated by `output_location`).
- `output_size`: On input, maximum output buffer size. On output, actual compressed size.
- `input_location`: DataLocation of input: `0` = HOST, `1` = DEVICE, `2` = MANAGED.
- `output_location`: DataLocation of output: `0` = HOST, `1` = DEVICE, `2` = MANAGED.
- `result`: Optional pointer to a result struct to receive profiling data. May be `NULL`.
- `stream`: CUDA stream for async GPU execution. Use `0` for default stream.

**Returns:** `0` on success, non-zero error code on failure.

---

#### cuda_zstd_hybrid_decompress

```c
int cuda_zstd_hybrid_decompress(
    cuda_zstd_hybrid_engine_t* engine,
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    unsigned int input_location,
    unsigned int output_location,
    cuda_zstd_hybrid_result_t* result,
    cudaStream_t stream
);
```

Decompresses data using the optimal backend.

**Parameters:**
- Same as `cuda_zstd_hybrid_compress`, except `input` is compressed data and `output` receives the decompressed result.

**Returns:** `0` on success, non-zero error code on failure.

---

### Utilities

#### cuda_zstd_hybrid_max_compressed_size

```c
size_t cuda_zstd_hybrid_max_compressed_size(
    cuda_zstd_hybrid_engine_t* engine,
    size_t input_size
);
```

Returns the maximum possible compressed output size for a given input size. Use this to allocate the output buffer before calling `cuda_zstd_hybrid_compress`.

**Parameters:**
- `engine`: Active hybrid engine handle.
- `input_size`: Size of uncompressed input in bytes.

**Returns:** Maximum compressed output size in bytes.

---

#### cuda_zstd_hybrid_query_routing

```c
unsigned int cuda_zstd_hybrid_query_routing(
    cuda_zstd_hybrid_engine_t* engine,
    size_t data_size,
    unsigned int input_location,
    unsigned int output_location,
    int is_compression
);
```

Queries which execution backend would be selected for the given parameters, without actually performing the operation.

**Parameters:**
- `engine`: Active hybrid engine handle.
- `data_size`: Size of the data in bytes.
- `input_location`: DataLocation of input (`0`=HOST, `1`=DEVICE, `2`=MANAGED).
- `output_location`: DataLocation of output.
- `is_compression`: Non-zero for compression, `0` for decompression.

**Returns:** ExecutionBackend enum value: `0` = CPU_LIBZSTD, `1` = GPU_KERNELS, `2` = CPU_PARALLEL, `3` = GPU_BATCH.

---

### Hybrid C API Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cuda_zstd_hybrid.h"

int main() {
    // 1. Create engine with default settings (AUTO mode, level 3)
    cuda_zstd_hybrid_engine_t* engine = cuda_zstd_hybrid_create_default();
    if (!engine) {
        fprintf(stderr, "Failed to create hybrid engine\n");
        return 1;
    }

    // 2. Prepare host data
    size_t input_size = 1024 * 1024;  // 1MB
    char* input = (char*)malloc(input_size);
    memset(input, 'A', input_size);

    // 3. Allocate output buffer
    size_t max_out = cuda_zstd_hybrid_max_compressed_size(engine, input_size);
    char* output = (char*)malloc(max_out);

    // 4. Compress (HOST → HOST, engine routes to CPU automatically)
    size_t output_size = max_out;
    cuda_zstd_hybrid_result_t result;
    int status = cuda_zstd_hybrid_compress(
        engine, input, input_size,
        output, &output_size,
        0, 0,       // HOST input, HOST output
        &result, 0  // result struct, default stream
    );

    if (status != 0) {
        fprintf(stderr, "Compression failed\n");
    } else {
        printf("Compressed %zu -> %zu bytes (%.2fx) via %s in %.2f ms\n",
               input_size, output_size,
               (float)input_size / output_size,
               result.backend_used == 0 ? "CPU" : "GPU",
               result.total_time_ms);
    }

    // 5. Decompress
    char* decompressed = (char*)malloc(input_size);
    size_t dec_size = input_size;
    status = cuda_zstd_hybrid_decompress(
        engine, output, output_size,
        decompressed, &dec_size,
        0, 0, &result, 0
    );

    if (status == 0 && dec_size == input_size) {
        printf("Decompressed successfully: %zu bytes\n", dec_size);
    }

    // 6. Query routing (without compressing)
    unsigned int backend = cuda_zstd_hybrid_query_routing(
        engine, 512, 0, 0, 1  // 512 bytes, HOST→HOST, compression
    );
    printf("512 bytes HOST→HOST would use: %s\n",
           backend == 0 ? "CPU_LIBZSTD" : "GPU_KERNELS");

    // 7. Cleanup
    free(input);
    free(output);
    free(decompressed);
    cuda_zstd_hybrid_destroy(engine);

    return 0;
}
```

## Thread Safety

- Manager and HybridEngine instances are **NOT** thread-safe.
- Create separate managers/engines for each thread.
- CUDA streams provide async safety within a single manager.
- Dictionary objects are read-only after training and can be shared across managers.
- HybridEngine routing decisions are stateless and deterministic for the same configuration.

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_manager.h` | C API declarations (lines 433-479) |
| `include/cuda_zstd_nvcomp.h` | NVComp v5 C API declarations (lines 276-336) |
| `include/cuda_zstd_hybrid.h` | Hybrid C API declarations (lines 296-359) |
| `src/cuda_zstd_c_api.cpp` | C API implementation (212 lines) |
| `src/cuda_zstd_hybrid.cu` | Hybrid engine implementation (~1200 lines) |
| `tests/test_c_api.cpp` | C API test suite |
| `tests/test_c_api_edge_cases.cu` | Edge case tests |
| `tests/test_hybrid.cu` | Hybrid engine test suite (26 tests) |

## Related Documentation

- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
- [QUICK-REFERENCE.md](QUICK-REFERENCE.md)
