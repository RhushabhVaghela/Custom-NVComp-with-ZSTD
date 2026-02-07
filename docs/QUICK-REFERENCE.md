# Quick Reference Card

> *Everything you need on one page.*

---

## 30-Second Start (C++)

```cpp
#include "cuda_zstd_manager.h"

// Create manager (level 1-22, lower = faster)
auto mgr = cuda_zstd::create_manager(3);

// Compress
size_t out_size;
mgr->compress(d_input, input_size, d_output, &out_size,
              d_temp, temp_size, nullptr, 0, stream);

// Decompress
mgr->decompress(d_compressed, comp_size, d_output, &out_size,
                d_temp, temp_size, nullptr, 0, stream);
```

---

## Python API

```python
import cuda_zstd

# Simple compress/decompress
compressed = cuda_zstd.compress(b"hello world" * 1000)
original = cuda_zstd.decompress(compressed)

# With a Manager (context manager for resource cleanup)
with cuda_zstd.Manager(level=3) as mgr:
    compressed = mgr.compress(data)
    decompressed = mgr.decompress(compressed)

# Batch compression
results = cuda_zstd.compress_batch([buf1, buf2, buf3])

# Install: cd python/ && pip install -e .
```

---

## C API

```c
#include "cuda_zstd_manager.h"

// Create/destroy
cuda_zstd_manager_t* mgr = cuda_zstd_create_manager(3);
cuda_zstd_destroy_manager(mgr);

// Compress
size_t ws = cuda_zstd_get_compress_workspace_size(mgr, input_size);
int status = cuda_zstd_compress(mgr, d_src, src_size,
                                d_dst, &dst_size,
                                d_workspace, ws, stream);

// Error handling
if (cuda_zstd_is_error(status)) {
    fprintf(stderr, "%s\n", cuda_zstd_get_error_string(status));
}
```

See [C-API-REFERENCE.md](C-API-REFERENCE.md) for all 11 C API functions and 7 NVComp v5 functions.

---

## Size Queries

```cpp
// How big should my output buffer be?
size_t max = mgr->get_max_compressed_size(input_size);

// How big should my temp buffer be?
size_t temp = mgr->get_compress_temp_size(input_size);
```

---

## Compression Levels

| Level | Speed | Use For |
|:-----:|:-----:|:--------|
| 1-3 | Fastest | Real-time, logs |
| 4-6 | Fast | General purpose |
| 7-12 | Moderate | Storage, archival |
| 13-22 | Slow | Maximum compression |

**TIP**: Level 3 is the sweet spot for most uses.

---

## Memory Allocation

```cpp
// Allocate GPU buffers
cudaMalloc(&d_input, input_size);
cudaMalloc(&d_output, mgr->get_max_compressed_size(input_size));
cudaMalloc(&d_temp, mgr->get_compress_temp_size(input_size));

// For faster transfers, use pinned memory
cudaMallocHost(&h_buffer, size);
```

---

## Streaming

```cpp
auto sm = cuda_zstd::create_streaming_manager(3);
sm->init_compression(stream, max_chunk_size);

while (has_data) {
    sm->compress_chunk(d_in, size, d_out, &out_size, is_last, stream);
}
```

---

## Batch (Fastest)

```cpp
auto bm = cuda_zstd::ZstdBatchManager::create(3);

bm->compress_batch(
    input_ptrs, input_sizes,  // Arrays of inputs
    output_ptrs, output_sizes, // Arrays of outputs
    batch_count,               // How many items
    workspace, ws_size,        // Shared workspace
    stream                     // CUDA stream
);
```

---

## Error Checking

```cpp
Status status = mgr->compress(...);

if (status != Status::SUCCESS) {
    printf("Error: %s\n", status_to_string(status));
}
```

| Code | Meaning |
|:----:|:--------|
| 0 | Success |
| 3 | Buffer too small |
| 4 | Out of memory |
| 5 | CUDA error |
| 8 | Checksum mismatch |

---

## Environment Variables

```bash
CUDA_ZSTD_DEBUG_LEVEL=3    # Debug output (0-3)
CUDA_LAUNCH_BLOCKING=1     # Sync execution
```

---

## Build Commands

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure
```

---

## Need More?

| Topic | Guide |
|:------|:------|
| Many files fast | [Batch Processing](BATCH-PROCESSING.md) |
| Large files | [Streaming API](STREAMING-API.md) |
| Go faster | [Performance Tuning](PERFORMANCE-TUNING.md) |
| C API details | [C API Reference](C-API-REFERENCE.md) |
| Something broke | [Error Handling](ERROR-HANDLING.md) |

---

*Keep this page bookmarked.*
