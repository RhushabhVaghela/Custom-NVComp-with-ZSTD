# 🔧 Troubleshooting Guide

> *"It works on my machine" is not a valid error code.*

## 🚨 Common Error Codes

### `ERROR_BUFFER_TOO_SMALL` (Code 7)
- **Meaning**: The output buffer provided is smaller than the compressed size.
- **Cause**: 
  - `ZSTD_compressBound(input_size)` was not used to calculate buffer size.
  - In `NvcompV5BatchManager`, the `d_output_sizes` input array was not initialized with capacities.
- **Fix**: Ensure `d_output_sizes[i]` contains the size of the buffer *before* calling compress.

### `ERROR_GENERIC` (Code 1) on Large Files
- **Meaning**: Internal GPU error (often illegal memory access).
- **Cause**: Usually related to LZ77 or FSE kernel buffer overflows on >64MB files.
- **Fix**: Update to the latest version (Issues #3 and #3a resolved this).

### `Status 4` (Correction/Corruption)
- **Meaning**: Decompression mismatch.
- **Cause**: Overlapping matches in LZ77 decoding or incorrect offset encoding.
- **Fix**: Fixed in `execute_sequences` kernel.

---

## 🖥️ WSL2 Limitations

### Concurrency Crashes (>4 Threads)
- **Symptom**: `test_scale_repro` fails or crashes with `ESRCH` or driver errors when running 8+ threads.
- **Cause**: WSL2's WDDM (Windows Display Driver Model) has overhead when context-switching rapidly between many GPU contexts.
- **Workaround**: 
  - Use **Native Linux** for high-concurrency workloads.
  - Limit concurrent `ZstdBatchManager` instances to 4 on WSL2.
  - Use the **Batch API** (single thread, multiple files) instead of multi-threading.

---

## 🐛 Debugging Tips

### 1. Enable Debug Logs
Rebuild with debug logging enabled to see kernel-level traces:
```bash
cmake -DCUDA_ZSTD_ENABLE_DEBUG_LOGS=ON ..
```

### 2. Use Compute Sanitizer
Run your application under the sanitizer to catch memory violations:
```bash
compute-sanitizer --tool memcheck ./your_app
```
