# CUDA-ZSTD C API Reference

## Overview

The C API provides a stable, ABI-compatible interface for integrating CUDA-ZSTD into C applications, dynamic language bindings (Python, Rust, etc.), and legacy systems.

## Header Files

```c
#include "cuda_zstd_types.h"    // Core types and enums
#include "cuda_zstd_manager.h"  // C API declarations
```

## Core Types

### Status Codes
Status values are returned as `int` codes compatible with nvCOMP error values.

### Compression Configuration
This API uses `CompressionConfig` from `cuda_zstd_types.h` on the C++ side.

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
void cuda_zstd_destroy_manager(
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
int cuda_zstd_compress(
    cuda_zstd_manager_t* manager,
    const void* src,              // Device input buffer
    size_t src_size,              // Input size in bytes
    void* dst,                    // Device output buffer
    size_t* dst_size,             // In: max size, Out: actual size
    void* workspace,              // Device workspace buffer
    size_t workspace_size,        // Workspace buffer size
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
int cuda_zstd_decompress(
    cuda_zstd_manager_t* manager,
    const void* src,              // Compressed data
    size_t src_size,
    void* dst,                    // Output buffer
    size_t* dst_size,             // In: max, Out: actual
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
);
```

## Complete Example

```c
#include <stdio.h>
#include <cuda_runtime.h>

// Assuming C API declarations available
extern cuda_zstd_manager_t* cuda_zstd_create_manager(int);
extern int cuda_zstd_compress(cuda_zstd_manager_t*, const void*, size_t,
                              void*, size_t*, void*, size_t, cudaStream_t);
extern void cuda_zstd_destroy_manager(cuda_zstd_manager_t*);
extern size_t cuda_zstd_get_compress_workspace_size(cuda_zstd_manager_t*, size_t);

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

    
    if (status == 0) {
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

## Error Handling

```c
const char* cuda_zstd_get_error_string(int status);

// Example
int status = cuda_zstd_compress(...);
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
