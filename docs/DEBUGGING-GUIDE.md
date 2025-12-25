# CUDA-ZSTD Debugging Guide

## Overview

This guide covers debugging techniques for identifying and resolving issues in CUDA-ZSTD applications.

## Debug Environment Variables

| Variable | Values | Description |
|:---------|:------:|:------------|
| `CUDA_ZSTD_DEBUG_LEVEL` | 0-3 | Verbosity level |
| `CUDA_LAUNCH_BLOCKING` | 0/1 | Synchronous kernel launches |
| `CUDA_ZSTD_DEBUG_KERNEL_VERIFY` | 0/1 | Verify after each kernel |
| `CUDA_ZSTD_PRINT_STATS` | 0/1 | Print timing statistics |

## Debug Levels

```bash
# Level 0: Errors only (default)
CUDA_ZSTD_DEBUG_LEVEL=0 ./my_app

# Level 1: Warnings + Errors
CUDA_ZSTD_DEBUG_LEVEL=1 ./my_app

# Level 2: Info + Warnings + Errors
CUDA_ZSTD_DEBUG_LEVEL=2 ./my_app

# Level 3: All debug output
CUDA_ZSTD_DEBUG_LEVEL=3 ./my_app
```

## Common Issues

### 1. Illegal Memory Access

**Symptoms:**
```
CUDA Error: an illegal memory access was encountered
```

**Debugging:**
```bash
# Enable memory checking
compute-sanitizer --tool memcheck ./my_app

# With stacktrace
compute-sanitizer --tool memcheck --show-backtrace yes ./my_app
```

**Common Causes:**
- Buffer too small (use `get_max_compressed_size()`)
- Workspace insufficient (use `get_compress_temp_size()`)
- Null pointer passed

### 2. Checksum Mismatch

**Symptoms:**
```
Status: ERROR_CHECKSUM_MISMATCH
```

**Debugging:**
```cpp
// Disable checksum temporarily
config.checksum = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
// If compression works, data was corrupted in transfer
```

**Common Causes:**
- GPU memory corruption
- Async copy not synchronized
- Buffer overwritten before read

### 3. Buffer Too Small

**Symptoms:**
```
Status: ERROR_BUFFER_TOO_SMALL
```

**Fix:**
```cpp
// Always use proper sizing
size_t max_size = manager->get_max_compressed_size(input_size);
size_t temp_size = manager->get_compress_temp_size(input_size);
```

### 4. CUDA Errors

**Symptoms:**
```
Status: ERROR_CUDA_ERROR
```

**Debugging:**
```cpp
// Get detailed error
ErrorContext ctx = cuda_zstd::error_handling::get_last_error();
printf("CUDA error: %s\n", cudaGetErrorString(ctx.cuda_error));
```

## Memory Debugging

### Detect Leaks
```bash
compute-sanitizer --tool memcheck --leak-check full ./my_app
```

### Track Allocations
```cpp
// Enable pool diagnostics
auto& pool = MemoryPoolManager::get_instance();
pool.enable_tracking(true);

// ... run operations ...

pool.print_stats();
// Output:
// Allocated: 128 MB
// Peak: 256 MB
// Current: 64 MB
```

## Kernel Debugging

### Synchronous Execution
```bash
CUDA_LAUNCH_BLOCKING=1 ./my_app
```

### Kernel-by-Kernel Verification
```cpp
// Enable in code
#define CUDA_ZSTD_DEBUG_KERNELS 1

// Or via environment
CUDA_ZSTD_DEBUG_KERNEL_VERIFY=1 ./my_app
```

## Profiling

### NVIDIA Nsight Systems
```bash
nsys profile --stats=true -o profile ./my_app
nsys stats profile.nsys-rep
```

### NVIDIA Nsight Compute
```bash
ncu --set full -o kernel_profile ./my_app
ncu -i kernel_profile.ncu-rep
```

## Assertion Macros

```cpp
// In debug builds
#ifdef DEBUG
  #define CUDA_ZSTD_ASSERT(cond, msg) \
    if (!(cond)) { \
      fprintf(stderr, "ASSERT FAILED: %s at %s:%d\n", msg, __FILE__, __LINE__); \
      abort(); \
    }
#else
  #define CUDA_ZSTD_ASSERT(cond, msg) ((void)0)
#endif
```

## Source Files

| File | Description |
|:-----|:------------|
| `include/error_context.h` | Error context utilities |
| `src/cuda_zstd_stacktrace.cpp` | Stacktrace capture |
| `tests/test_error_handling.cu` | Error handling tests |

## Related Documentation
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
- [TESTING-GUIDE.md](TESTING-GUIDE.md)
