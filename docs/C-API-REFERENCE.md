# CUDA-ZSTD C API Reference

## Overview

The C API provides a stable, ABI-compatible interface for integrating CUDA-ZSTD into C applications, dynamic language bindings (Python, Rust, etc.), and legacy systems.

## Header Files

```c
#include "cuda_zstd_types.h"    // Core types and enums
#include "cuda_zstd_manager.h"  // C++ API (optional)
```

## Core Types

### Status Codes
```c
typedef enum {
    CUDA_ZSTD_SUCCESS = 0,
    CUDA_ZSTD_ERROR_GENERIC = 1,
    CUDA_ZSTD_ERROR_INVALID_PARAMETER = 2,
    CUDA_ZSTD_ERROR_BUFFER_TOO_SMALL = 3,
    CUDA_ZSTD_ERROR_OUT_OF_MEMORY = 4,
    CUDA_ZSTD_ERROR_CUDA_ERROR = 5,
    CUDA_ZSTD_ERROR_UNSUPPORTED_FORMAT = 6,
    CUDA_ZSTD_ERROR_CHECKSUM_MISMATCH = 7,
    CUDA_ZSTD_ERROR_INVALID_HEADER = 8,
    CUDA_ZSTD_ERROR_CORRUPTED_DATA = 9,
    // ... 18 total error codes
} cuda_zstd_status_t;
```

### Compression Configuration
```c
typedef struct {
    int compression_level;        // 1-22
    size_t block_size;            // Block size (default: 128KB)
    int enable_checksum;          // 0=disabled, 1=enabled
    int enable_dictionaries;      // 0=disabled, 1=enabled
    void* reserved[4];            // Future expansion
} cuda_zstd_config_t;
```

## API Functions

### Manager Lifecycle

#### Create Manager
```c
cuda_zstd_manager_t* cuda_zstd_create_manager(
    int compression_level
);
```
**Parameters:**
- `compression_level`: 1-22 (1=fastest, 22=best compression)

**Returns:** Pointer to created manager, or `NULL` on failure.


#### Destroy Manager
```c
cuda_zstd_status_t cuda_zstd_destroy_manager(
    cuda_zstd_manager_t* manager
);
```

### Compression Functions

#### Get Temporary Buffer Size
```c
size_t cuda_zstd_get_compress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t input_size
);
```

#### Compress

```c
cuda_zstd_status_t cuda_zstd_compress(
    cuda_zstd_manager_t* manager,
    const void* d_input,          // Device input buffer
    size_t input_size,            // Input size in bytes
    void* d_output,               // Device output buffer
    size_t* output_size,          // In: max size, Out: actual size
    void* d_temp,                 // Device temp buffer
    size_t temp_size,             // Temp buffer size
    cudaStream_t stream           // CUDA stream (0 for default)
);
```

### Decompression Functions

#### Get Temporary Buffer Size
```c
size_t cuda_zstd_get_decompress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t compressed_size
);
```

#### Decompress

```c
cuda_zstd_status_t cuda_zstd_decompress(
    cuda_zstd_manager_t* manager,
    const void* d_input,          // Compressed data
    size_t input_size,
    void* d_output,               // Output buffer
    size_t* output_size,          // In: max, Out: actual
    void* d_temp,
    size_t temp_size,
    cudaStream_t stream
);
```

## Complete Example

```c
#include <stdio.h>
#include <cuda_runtime.h>

// Assuming C API declarations available
extern cuda_zstd_status_t cuda_zstd_create_manager(void**, int);
extern cuda_zstd_status_t cuda_zstd_compress(void*, const void*, size_t,
                                             void*, size_t*, void*, size_t,
                                             cudaStream_t);
extern cuda_zstd_status_t cuda_zstd_destroy_manager(void*);

int main() {
    void* manager = NULL;
    cuda_zstd_status_t status;
    
    // 1. Create manager (level 5)
    manager = cuda_zstd_create_manager(5);
    if (manager == NULL) {
        fprintf(stderr, "Failed to create manager\n");
        return 1;
    }
    
    // 2. Prepare data
    size_t input_size = 1024 * 1024;  // 1MB
    void *d_input, *d_output, *d_temp;
    size_t temp_size = cuda_zstd_get_compress_workspace_size(manager, input_size);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);
    cudaMalloc(&d_temp, temp_size);
    
    // Fill input with test data
    cudaMemset(d_input, 'A', input_size);
    
    // 3. Compress
    size_t output_size = input_size * 2;
    status = cuda_zstd_compress(manager,
                                d_input, input_size,
                                d_output, &output_size,
                                d_temp, temp_size,
                                0);

    
    if (status == 0) {
        printf("Compressed %zu -> %zu bytes (%.2fx)\n",
               input_size, output_size,
               (float)input_size / output_size);
    }
    
    // 4. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cuda_zstd_destroy_manager(manager);
    
    return status;
}
```

## Error Handling

```c
const char* cuda_zstd_get_error_string(cuda_zstd_status_t status);

// Example
cuda_zstd_status_t status = cuda_zstd_compress(...);
if (status != CUDA_ZSTD_SUCCESS) {
    fprintf(stderr, "Compression failed: %s\n",
            cuda_zstd_get_error_string(status));
}
```

## Thread Safety

- Manager instances are **NOT** thread-safe
- Create separate managers for each thread
- CUDA streams provide async safety within a manager

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_c_api.cpp` | C API implementation |
| `tests/test_c_api.c` | C API tests |
| `tests/test_c_api_edge_cases.cu` | Edge case tests |

## Related Documentation
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
