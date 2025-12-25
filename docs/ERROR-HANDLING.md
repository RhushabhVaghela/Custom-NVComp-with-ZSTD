# 🚨 Error Handling: When Things Go Wrong

> *"Good error handling is the difference between 'it crashed' and 'here's exactly what went wrong.'"*

## Understanding Errors

Every function in CUDA-ZSTD returns a **Status** code. Think of it like a report card:

```cpp
Status result = manager->compress(...);

if (result == Status::SUCCESS) {
    🎉 Everything worked!
} else {
    🚨 Something went wrong - check what!
}
```

---

## 📋 Error Code Reference

### The Good Ones
| Code | Name | Meaning |
|:----:|:-----|:--------|
| ✅ 0 | `SUCCESS` | All good! Continue on. |

### The Recoverable Ones
| Code | Name | What Happened | How to Fix |
|:----:|:-----|:--------------|:-----------|
| 🔧 3 | `BUFFER_TOO_SMALL` | Output buffer too small | Make buffer bigger |
| 🔧 4 | `OUT_OF_MEMORY` | GPU ran out of memory | Free some memory, try again |
| 🔧 15 | `TIMEOUT` | Operation took too long | Retry or use smaller chunks |

### The "Check Your Input" Ones
| Code | Name | What Happened | How to Fix |
|:----:|:-----|:--------------|:-----------|
| ⚠️ 2 | `INVALID_PARAMETER` | Bad argument passed | Check your inputs |
| ⚠️ 9 | `CORRUPTED_DATA` | Data is broken | Verify source data |
| ⚠️ 10 | `INVALID_HEADER` | Not a valid ZSTD file | Check file format |

### The Serious Ones
| Code | Name | What Happened | How to Fix |
|:----:|:-----|:--------------|:-----------|
| 🔴 5 | `CUDA_ERROR` | GPU had a problem | Restart, check GPU |
| 🔴 8 | `CHECKSUM_MISMATCH` | Data corrupted in transit | Re-transfer data |

---

## 🛠️ How to Handle Errors

### The Basic Pattern
```cpp
Status status = manager->compress(...);

if (status != Status::SUCCESS) {
    printf("Error: %s\n", status_to_string(status));
    // Handle the error...
}
```

### The Complete Pattern (Recommended)
```cpp
#include "error_context.h"

Status status = manager->compress(...);

if (status != Status::SUCCESS) {
    // Get detailed information
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

---

## 🎯 Common Scenarios

### "Buffer Too Small"
```cpp
// ❌ Wrong: Guessing the size
void* output = malloc(input_size);  // Too small!

// ✅ Right: Ask for the correct size
size_t max_size = manager->get_max_compressed_size(input_size);
void* output = malloc(max_size);
```

### "Out of Memory"
```cpp
// ❌ Wrong: Allocating too much
cudaMalloc(&huge_buffer, 16 * GB);  // 💥

// ✅ Right: Process in chunks
for (auto chunk : split_into_chunks(data, 128 * MB)) {
    manager->compress(chunk, ...);
}
```

### "Checksum Mismatch"
```cpp
Status status = manager->decompress(...);

if (status == Status::ERROR_CHECKSUM_MISMATCH) {
    printf("⚠️ Data was corrupted! Re-download and try again.\n");
    // Don't trust the output!
}
```

---

## 🔍 Debugging Tips

### 1. Enable Debug Logging
```bash
CUDA_ZSTD_DEBUG_LEVEL=3 ./my_app
```

### 2. Force Synchronous Execution
```bash
CUDA_LAUNCH_BLOCKING=1 ./my_app
```

### 3. Check for CUDA Errors
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

---

## 📚 Error Code Quick Reference

```
 0 = SUCCESS              ✅ All good
 1 = ERROR_GENERIC        🤷 Something went wrong
 2 = INVALID_PARAMETER    ⚠️ Bad input
 3 = BUFFER_TOO_SMALL     📏 Need bigger buffer
 4 = OUT_OF_MEMORY        💾 No more memory
 5 = CUDA_ERROR           🔴 GPU problem
 6 = UNSUPPORTED_FORMAT   📄 Unknown format
 7 = NOT_INITIALIZED      🚫 Call init() first
 8 = CHECKSUM_MISMATCH    🔐 Data corrupted
 9 = CORRUPTED_DATA       💔 Invalid data
10 = INVALID_HEADER       📝 Bad header
```

---

## 📚 Related Guides

- [Debugging Guide](DEBUGGING-GUIDE.md) — Deep dive into troubleshooting
- [Testing Guide](TESTING-GUIDE.md) — Test your error handling
- [Quick Reference](QUICK-REFERENCE.md) — Common patterns

---

*"Expect the best, handle the worst." 🛡️*
