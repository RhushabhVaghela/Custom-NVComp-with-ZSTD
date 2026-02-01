# 🚀 Release Notes - v2.2.0 (Stable)

**Date**: January 31, 2026
**Status**: Production Ready (100% Test Pass Rate, RFC 8878 Compliant)

## ✨ New Features

### 🏛️ RFC 8878 Compliance
- **Full Interoperability**: GPU-compressed frames are now fully compatible with the standard Zstandard (RFC 8878) format.
- **Verification**: Successfully validated using official `zstd` CLI and internal compliance suite.
- **Details**: Correct frame headers, block structures, and magic numbers.

### 🧠 Smart Router (Hybrid Execution)

- **Automatic CPU Fallback**: Files smaller than 1MB are automatically routed to the CPU (`libzstd`) to avoid GPU launch latency.
- **Transparent**: No code changes required by the user; the Manager handles the copy/execution transparently.

### ⚖️ Scalability Testing
- **New Test**: `test_scale_repro.cu` validates thread safety and concurrency limits.
- **WSL2 Support**: Documented limitations for high concurrency on WSL2 (recommend <4 threads).

---

## 🛠️ Critical Bug Fixes

### 1. Batch API Initialization
- **Fixed**: `ERROR_BUFFER_TOO_SMALL` in `NvcompV5BatchManager` due to uninitialized output capacity.
- **Impact**: Batch compression now works correctly for variable-sized inputs.

### 2. Concurrency Stability
- **Fixed**: Race condition in `ZstdBatchManager` destructor causing SEGFAULTS.
- **Fix**: Replaced global `cudaDeviceSynchronize()` with per-stream synchronization.

### 3. Large File Support (>64MB)
- **Fixed**: `Buffer Overflow` and `Correction Error` (Status 4) on large files.
- **Root Cause**: Fixed O(1) buffer allocation logic in `cuda_zstd_manager.cu` that failed for large block counts.

### 4. Memory Leaks
- **Fixed**: >10MB leak in `test_memory_efficiency` resolved by improving error-path cleanup in `compress()`.

---

## 🧪 Testing Status

| Component | Status | Check |
|:----------|:------:|:-----:|
| Unit Tests (47/47) | **PASS** | ✅ |
| Integration Tests | **PASS** | ✅ |
| Large File (>128MB) | **PASS** | ✅ |
| Batch Processing | **PASS** | ✅ |
| Dictionary Mode | **PASS** | ✅ |

---

## 📚 Updated Documentation
- **[README.md](../README.md)**: Updated with Smart Router and current status.
- **[ARCHITECTURE-OVERVIEW.md](ARCHITECTURE-OVERVIEW.md)**: Added Hybrid Execution section.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: New guide for common errors.
- **[stream_based_parallelism.md](stream_based_parallelism.md)**: Design doc for future stateless architecture.
