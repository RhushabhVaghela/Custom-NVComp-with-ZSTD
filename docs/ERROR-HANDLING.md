# Error Handling

## Overview

Every function in CUDA-ZSTD returns a `Status` code. The full enum is defined in `include/cuda_zstd_types.h` with 29 entries (26 unique values + 3 deprecated aliases).

```cpp
Status result = manager->compress(...);

if (result != Status::SUCCESS) {
    printf("Error: %s\n", status_to_string(result));
}
```

---

## Complete Status Code Reference

The following table lists all status codes exactly as defined in `cuda_zstd_types.h`.

### Success

| Value | Name | Description |
|:-----:|:-----|:------------|
| 0 | `SUCCESS` | Operation completed successfully. |

### Parameter and Input Errors

| Value | Name | Description |
|:-----:|:-----|:------------|
| 2 | `ERROR_INVALID_PARAMETER` | Invalid argument passed to function. |
| 5 | `ERROR_INVALID_MAGIC` | Invalid ZSTD magic number in input. |
| 6 | `ERROR_CORRUPT_DATA` | Input data is corrupted or malformed. |
| 7 | `ERROR_BUFFER_TOO_SMALL` | Provided output buffer is too small. |
| 8 | `ERROR_UNSUPPORTED_VERSION` | Unsupported ZSTD format version. |
| 9 | `ERROR_DICTIONARY_MISMATCH` | Dictionary does not match the compressed data. |
| 10 | `ERROR_CHECKSUM_FAILED` | Frame checksum verification failed. |
| 28 | `ERROR_UNSUPPORTED_FORMAT` | Unsupported frame or block format. |

### Memory and Resource Errors

| Value | Name | Description |
|:-----:|:-----|:------------|
| 3 | `ERROR_OUT_OF_MEMORY` | Memory allocation failed. |
| 16 | `ERROR_ALLOCATION_FAILED` | Resource allocation failed. |
| 14 | `ERROR_WORKSPACE_INVALID` | Temporary workspace is invalid or too small. |
| 17 | `ERROR_HASH_TABLE_FULL` | Internal hash table limit reached. |

### CUDA and Runtime Errors

| Value | Name | Description |
|:-----:|:-----|:------------|
| 4 | `ERROR_CUDA_ERROR` | CUDA runtime error occurred. |
| 15 | `ERROR_STREAM_ERROR` | CUDA stream error. |
| 22 | `ERROR_TIMEOUT` | Operation timed out. |
| 23 | `ERROR_CANCELLED` | Operation was cancelled. |

### Compression Pipeline Errors

| Value | Name | Description |
|:-----:|:-----|:------------|
| 1 | `ERROR_GENERIC` | General error. |
| 11 | `ERROR_IO` | I/O operation failed. |
| 12 | `ERROR_COMPRESSION` | Error occurred during compression. |
| 13 | `ERROR_DECOMPRESSION` | Error occurred during decompression. |
| 18 | `ERROR_SEQUENCE_ERROR` | Invalid match sequences encountered. |
| 27 | `ERROR_DICTIONARY_FAILED` | Dictionary training or parsing failed. |

### State Errors

| Value | Name | Description |
|:-----:|:-----|:------------|
| 19 | `ERROR_NOT_INITIALIZED` | Manager or context not initialized. |
| 20 | `ERROR_ALREADY_INITIALIZED` | Resource already initialized. |
| 21 | `ERROR_INVALID_STATE` | Operation invalid in current state. |

### Other

| Value | Name | Description |
|:-----:|:-----|:------------|
| 24 | `ERROR_NOT_IMPLEMENTED` | Requested feature not yet implemented. |
| 25 | `ERROR_INTERNAL` | Internal consistency check failed. |
| 26 | `ERROR_UNKNOWN` | Unknown error occurred. |

### Deprecated Aliases

The following names are deprecated and map to existing values:

| Deprecated Name | Use Instead | Value |
|:----------------|:------------|:-----:|
| `ERROR_CORRUPTED_DATA` | `ERROR_CORRUPT_DATA` | 6 |
| `ERROR_COMPRESSION_FAILED` | `ERROR_COMPRESSION` | 12 |
| `ERROR_DECOMPRESSION_FAILED` | `ERROR_DECOMPRESSION` | 13 |

---

## Error Handling Patterns

### Basic Pattern

```cpp
Status status = manager->compress(...);

if (status != Status::SUCCESS) {
    printf("Error: %s\n", status_to_string(status));
    // Handle the error
}
```

### Detailed Error Context

```cpp
#include "cuda_zstd_types.h"

Status status = manager->compress(...);

if (status != Status::SUCCESS) {
    ErrorContext ctx = cuda_zstd::error_handling::get_last_error();

    printf("Error: %s\n", status_to_string(ctx.status));
    printf("  Location: %s:%d\n", ctx.file, ctx.line);
    printf("  Function: %s\n", ctx.function);

    if (ctx.message) {
        printf("  Details: %s\n", ctx.message);
    }

    if (ctx.cuda_error != cudaSuccess) {
        printf("  CUDA: %s\n", cudaGetErrorString(ctx.cuda_error));
    }
}
```

The `ErrorContext` struct (defined in `cuda_zstd_types.h`) captures the status code, source file/line, function name, an optional message, and the underlying `cudaError_t` if applicable.

---

## Common Scenarios

### Buffer Too Small (ERROR_BUFFER_TOO_SMALL = 7)

```cpp
// Wrong: guessing the output size
void* output = malloc(input_size);  // May be too small

// Right: query the maximum compressed size first
size_t max_size = manager->get_max_compressed_size(input_size);
void* output = malloc(max_size);
```

### Out of Memory (ERROR_OUT_OF_MEMORY = 3)

```cpp
// Wrong: allocating too much at once
cudaMalloc(&huge_buffer, 16ULL * 1024 * 1024 * 1024);

// Right: process in chunks
for (auto chunk : split_into_chunks(data, 128 * MB)) {
    manager->compress(chunk, ...);
}
```

### Checksum Failed (ERROR_CHECKSUM_FAILED = 10)

```cpp
Status status = manager->decompress(...);

if (status == Status::ERROR_CHECKSUM_FAILED) {
    // Data was corrupted in transit -- do not trust the output
    fprintf(stderr, "Checksum verification failed. Re-transfer data.\n");
}
```

### CUDA Error (ERROR_CUDA_ERROR = 4)

```cpp
Status status = manager->compress(...);

if (status == Status::ERROR_CUDA_ERROR) {
    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

## C API Error Handling

The C API returns `int` error codes that map directly to the `Status` enum values.

```c
int status = cuda_zstd_compress(mgr, src, src_size,
                                dst, &dst_size,
                                workspace, ws_size, stream);

if (cuda_zstd_is_error(status)) {
    fprintf(stderr, "Error: %s\n", cuda_zstd_get_error_string(status));
}
```

---

## Debugging Tips

### Enable Debug Logging

```bash
CUDA_ZSTD_DEBUG_LEVEL=3 ./my_app
```

### Force Synchronous Execution

```bash
CUDA_LAUNCH_BLOCKING=1 ./my_app
```

### Check for CUDA Errors

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

## Related Documentation

- [Debugging Guide](DEBUGGING-GUIDE.md) -- Troubleshooting techniques
- [Testing Guide](TESTING-GUIDE.md) -- Running and writing tests
- [Quick Reference](QUICK-REFERENCE.md) -- Common usage patterns
