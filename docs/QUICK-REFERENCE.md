# CUDA-ZSTD Quick Reference Card

## One-Liner Examples

### Compress
```cpp
manager->compress(d_in, size, d_out, &out_size, d_temp, temp_size, nullptr, 0, stream);
```

### Decompress
```cpp
manager->decompress(d_in, size, d_out, &out_size, d_temp, temp_size, nullptr, 0, stream);
```

### Create Manager
```cpp
auto manager = cuda_zstd::create_manager(5);  // Level 5
```

## Size Queries

| Query | Code |
|:------|:-----|
| Max compressed size | `manager->get_max_compressed_size(input_size)` |
| Temp buffer size | `manager->get_compress_temp_size(input_size)` |
| Decompressed size | `manager->get_decompressed_size(d_compressed, size)` |

## Compression Levels

| Level | Speed | Ratio | Use |
|:-----:|:-----:|:-----:|:----|
| 1 | ★★★★★ | ★★ | Real-time |
| 3 | ★★★★ | ★★★ | Default |
| 9 | ★★ | ★★★★ | Archive |

## Status Codes

| Code | Name |
|:----:|:-----|
| 0 | SUCCESS |
| 3 | BUFFER_TOO_SMALL |
| 4 | OUT_OF_MEMORY |
| 5 | CUDA_ERROR |
| 8 | CHECKSUM_MISMATCH |

## Memory Allocation

```cpp
// Input/output buffers
cudaMalloc(&d_input, input_size);
cudaMalloc(&d_output, manager->get_max_compressed_size(input_size));
cudaMalloc(&d_temp, manager->get_compress_temp_size(input_size));
```

## Streaming

```cpp
auto sm = ZstdStreamingManager::create(3);
sm->init_compression();
sm->compress_chunk(d_in, size, d_out, &out, is_last, stream);
```

## Batch

```cpp
auto bm = ZstdBatchManager::create(3);
bm->compress_batch(inputs, sizes, outputs, out_sizes, count, workspace, ws_size, stream);
```

## Environment Variables

| Variable | Default | Description |
|:---------|:-------:|:------------|
| `CUDA_ZSTD_DEBUG_LEVEL` | 0 | Debug verbosity |
| `CUDA_ZSTD_POOL_MAX_SIZE` | 2GB | Max pool size |

## Build Commands

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure
```

## Doc Links

- [ARCHITECTURE-OVERVIEW.md](ARCHITECTURE-OVERVIEW.md)
- [BATCH-PROCESSING.md](BATCH-PROCESSING.md)
- [STREAMING-API.md](STREAMING-API.md)
- [PERFORMANCE-TUNING.md](PERFORMANCE-TUNING.md)
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
