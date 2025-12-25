# CUDA-ZSTD Error Handling Guide

## Overview

The error handling system provides comprehensive error detection, reporting, and recovery mechanisms with 18 distinct error codes and detailed context information.

## Error Code Reference

| Code | Name | Description | Recovery |
|:----:|:-----|:------------|:---------|
| 0 | `SUCCESS` | Operation completed successfully | N/A |
| 1 | `ERROR_GENERIC` | Unspecified error | Check logs |
| 2 | `ERROR_INVALID_PARAMETER` | Invalid function argument | Fix parameter |
| 3 | `ERROR_BUFFER_TOO_SMALL` | Output buffer insufficient | Resize buffer |
| 4 | `ERROR_OUT_OF_MEMORY` | GPU/Host memory exhausted | Free memory |
| 5 | `ERROR_CUDA_ERROR` | CUDA runtime error | Check GPU |
| 6 | `ERROR_UNSUPPORTED_FORMAT` | Unknown data format | Verify input |
| 7 | `ERROR_NOT_INITIALIZED` | Manager not initialized | Call init() |
| 8 | `ERROR_CHECKSUM_MISMATCH` | Data integrity failure | Re-transfer |
| 9 | `ERROR_CORRUPTED_DATA` | Invalid compressed data | Verify source |
| 10 | `ERROR_INVALID_HEADER` | Malformed frame header | Check format |
| 11 | `ERROR_WORKSPACE_TOO_SMALL` | Workspace buffer too small | Resize |
| 12 | `ERROR_STREAM_ERROR` | CUDA stream error | Reset stream |
| 13 | `ERROR_DICTIONARY_INVALID` | Invalid dictionary format | Retrain |
| 14 | `ERROR_DICTIONARY_MISMATCH` | Dictionary ID mismatch | Use correct dict |
| 15 | `ERROR_TIMEOUT` | Operation timed out | Retry or abort |
| 16 | `ERROR_CANCELLED` | Operation cancelled | User action |
| 17 | `ERROR_DICTIONARY_FAILED` | Dictionary training failed | Check samples |
| 18 | `ERROR_UNSUPPORTED_FORMAT` | Unsupported ZSTD features | Check version |

## ErrorContext Structure

```cpp
struct ErrorContext {
    Status status;              // Error code
    const char* file;           // Source file
    int line;                   // Line number
    const char* function;       // Function name
    const char* message;        // Detailed message
    cudaError_t cuda_error;     // CUDA error (if applicable)
};
```

## Usage Patterns

### Basic Error Checking
```cpp
#include "cuda_zstd_manager.h"

Status status = manager->compress(...);
if (status != Status::SUCCESS) {
    printf("Compression failed: %s\n", status_to_string(status));
    return status;
}
```

### Detailed Error Context
```cpp
#include "error_context.h"

Status status = manager->compress(...);
if (status != Status::SUCCESS) {
    ErrorContext ctx = cuda_zstd::error_handling::get_last_error();
    
    fprintf(stderr, "Error: %s\n", status_to_string(ctx.status));
    fprintf(stderr, "  File: %s:%d\n", ctx.file, ctx.line);
    fprintf(stderr, "  Function: %s\n", ctx.function);
    if (ctx.message) {
        fprintf(stderr, "  Message: %s\n", ctx.message);
    }
    if (ctx.cuda_error != cudaSuccess) {
        fprintf(stderr, "  CUDA: %s\n", cudaGetErrorString(ctx.cuda_error));
    }
}
```

### Error Callbacks
```cpp
void my_error_handler(const ErrorContext& ctx) {
    // Log to file, send alert, etc.
    log_to_file("[CUDA-ZSTD] %s in %s at %s:%d",
                status_to_string(ctx.status),
                ctx.function, ctx.file, ctx.line);
}

// Register handler
cuda_zstd::set_error_callback(my_error_handler);
```

## Debug Macros

### CHECK_STATUS
```cpp
#define CHECK_STATUS(status) do { \
    if ((status) != Status::SUCCESS) { \
        ErrorContext ctx((status), __FILE__, __LINE__, __FUNCTION__); \
        error_handling::log_error(ctx); \
        return (status); \
    } \
} while(0)

// Usage
CHECK_STATUS(compress(...));  // Logs and returns on error
```

### VALIDATE_NOT_NULL
```cpp
#define VALIDATE_NOT_NULL(ptr, name) do { \
    if (!(ptr)) { \
        ErrorContext ctx(Status::ERROR_INVALID_PARAMETER, \
                        __FILE__, __LINE__, __FUNCTION__, \
                        name " is null"); \
        error_handling::log_error(ctx); \
        return Status::ERROR_INVALID_PARAMETER; \
    } \
} while(0)

// Usage
VALIDATE_NOT_NULL(d_input, "d_input");
```

## CUDA Error Handling

### Automatic Detection
```cpp
// Internal CUDA_CHECK macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        ErrorContext ctx(Status::ERROR_CUDA_ERROR, \
                        __FILE__, __LINE__, __FUNCTION__); \
        ctx.cuda_error = err; \
        error_handling::log_error(ctx); \
        return Status::ERROR_CUDA_ERROR; \
    } \
} while(0)
```

### Stream Error Recovery
```cpp
// Check for async errors
cudaStreamSynchronize(stream);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Reset stream state
    cudaStreamDestroy(stream);
    cudaStreamCreate(&stream);
    // Retry operation or fail gracefully
}
```

## Best Practices

1. **Always Check Return Values**: Never ignore Status returns
2. **Use Detailed Logging**: Enable verbose mode for debugging
3. **Implement Recovery Logic**: Handle recoverable errors gracefully
4. **Monitor Memory**: Track allocations to prevent OOM
5. **Validate Input**: Check parameters before GPU operations

## Source Files

| File | Description |
|:-----|:------------|
| `include/error_context.h` | Error context definitions |
| `include/cuda_zstd_types.h` | Status enum (lines 120-170) |
| `src/error_context.cpp` | Error context implementation |
| `src/cuda_zstd_types.cpp` | status_to_string() |
| `tests/test_error_handling.cu` | Error handling tests |

## Related Documentation
- [C-API-REFERENCE.md](C-API-REFERENCE.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
