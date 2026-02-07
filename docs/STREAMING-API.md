# Streaming API

## What is Streaming?

Streaming compression processes data incrementally in chunks rather than requiring the entire input in memory at once. This is useful for large files, live data feeds, or memory-constrained environments.

**Important**: The current implementation produces a separate ZSTD frame per chunk. It does not maintain a single continuous frame across chunks. The `compress_chunk_with_history` path retains a sliding window for better compression ratios, but still emits independent frames.

```
Traditional Compression:          Streaming Compression:

Wait... Wait... Wait...           Start immediately!
        |                                |
[================] 100%           [=...............] -> [C1]
        |                         [====............] -> [C1][C2]
       [C]                        [========........] -> [C1][C2][C3]
                                  [============....] -> [C1][C2][C3][C4]
Total time: 10 seconds            [================] -> [C1][C2][C3][C4][C5]
                                  Total time: 10 seconds
                                  BUT: First output at 2 seconds!
```

## When to Use Streaming

### Good Fit:

| Scenario | Why Streaming Wins |
|:---------|:-------------------|
| Live data feeds | Data has no defined end; must process continuously |
| Huge files | No need to fit entire file in memory |
| Network transfers | Start sending compressed data immediately |
| Limited memory | Process 100 GB file with only 128 KB buffer |

### Skip Streaming When:
- You have many small files (use [Batch Processing](BATCH-PROCESSING.md) instead)
- The entire file fits easily in GPU memory

---

## Complete API Reference

### ZstdStreamingManager Class

From `include/cuda_zstd_manager.h` lines 300-352:

#### Construction

| Method | Description |
|:-------|:------------|
| `ZstdStreamingManager()` | Default constructor |
| `ZstdStreamingManager(const CompressionConfig& config)` | Construct with configuration |
| Factory: `create_streaming_manager(int compression_level = 3)` | Returns `std::unique_ptr<ZstdStreamingManager>` |

#### Initialization

| Method | Description |
|:-------|:------------|
| `Status init_compression(cudaStream_t stream = 0, size_t max_chunk_size = 0)` | Initialize for compression (independent chunks) |
| `Status init_compression_with_history(cudaStream_t stream = 0, size_t max_chunk_size = 0)` | Initialize for compression with sliding window history |
| `Status init_decompression(cudaStream_t stream = 0)` | Initialize for decompression |

#### Compression

| Method | Description |
|:-------|:------------|
| `Status compress_chunk(const void* input, size_t input_size, void* output, size_t* output_size, bool is_last_chunk, cudaStream_t stream = 0)` | Compress one chunk (independent frame) |
| `Status compress_chunk_with_history(const void* input, size_t input_size, void* output, size_t* output_size, bool is_last_chunk, cudaStream_t stream = 0)` | Compress one chunk with sliding window history for better ratios |

#### Decompression

| Method | Description |
|:-------|:------------|
| `Status decompress_chunk(const void* input, size_t input_size, void* output, size_t* output_size, bool* is_last_chunk, cudaStream_t stream = 0)` | Decompress one chunk; sets `is_last_chunk` on final frame |

#### Control

| Method | Description |
|:-------|:------------|
| `Status reset()` | Reset all internal state |
| `Status reset_streaming()` | Reset streaming state only |
| `Status flush(cudaStream_t stream = 0)` | Flush pending output |
| `Status flush_streaming(cudaStream_t stream = 0)` | Flush streaming-specific buffers |

#### Configuration

| Method | Description |
|:-------|:------------|
| `Status set_config(const CompressionConfig& config)` | Set compression configuration |
| `Status set_dictionary(const dictionary::Dictionary& dict)` | Set dictionary for compression |
| `CompressionConfig get_config() const` | Get current configuration |

#### Queries

| Method | Description |
|:-------|:------------|
| `size_t get_temp_size() const` | Get required temporary workspace size |
| `bool is_compression_initialized() const` | Check if compression is ready |
| `bool is_decompression_initialized() const` | Check if decompression is ready |

---

## Usage Examples

### Basic Streaming Compression (Independent Chunks)

```cpp
#include "cuda_zstd_manager.h"

void compress_file_streaming(const std::string& filename) {
    // 1. Create a streaming manager
    auto stream_mgr = cuda_zstd::create_streaming_manager(5);
    stream_mgr->init_compression(0, 128 * 1024);  // stream=default, max_chunk=128KB

    // 2. Process the file in chunks
    std::ifstream input(filename, std::ios::binary);
    std::ofstream output(filename + ".zst", std::ios::binary);
    std::vector<uint8_t> chunk(128 * 1024);
    std::vector<uint8_t> compressed(256 * 1024);

    while (input) {
        input.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
        size_t bytes_read = input.gcount();
        bool is_last = input.eof();

        size_t compressed_size;
        stream_mgr->compress_chunk(
            chunk.data(), bytes_read,
            compressed.data(), &compressed_size,
            is_last
        );

        output.write(reinterpret_cast<char*>(compressed.data()), compressed_size);
    }
}
```

### Streaming with History (Better Compression)

The `compress_chunk_with_history` path maintains a sliding window across chunks, improving compression ratios by 5-10% compared to independent chunks:

```cpp
#include "cuda_zstd_manager.h"

void compress_with_history(const std::string& filename) {
    auto stream_mgr = cuda_zstd::create_streaming_manager(5);

    // Use init_compression_with_history instead
    stream_mgr->init_compression_with_history(0, 128 * 1024);

    std::ifstream input(filename, std::ios::binary);
    std::ofstream output(filename + ".zst", std::ios::binary);
    std::vector<uint8_t> chunk(128 * 1024);
    std::vector<uint8_t> compressed(256 * 1024);

    while (input) {
        input.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
        size_t bytes_read = input.gcount();
        bool is_last = input.eof();

        size_t compressed_size;
        // Use the history-enabled variant
        stream_mgr->compress_chunk_with_history(
            chunk.data(), bytes_read,
            compressed.data(), &compressed_size,
            is_last
        );

        output.write(reinterpret_cast<char*>(compressed.data()), compressed_size);
    }
}
```

### Streaming Decompression

```cpp
void decompress_streaming(const std::string& compressed_file) {
    auto stream_mgr = cuda_zstd::create_streaming_manager(3);
    stream_mgr->init_decompression();

    std::ifstream input(compressed_file, std::ios::binary);
    // Read frame by frame
    while (input) {
        // Read compressed chunk (application-specific framing)
        auto [data, size] = read_next_frame(input);

        size_t decompressed_size;
        bool is_last = false;
        stream_mgr->decompress_chunk(
            data, size,
            output_buffer, &decompressed_size,
            &is_last
        );

        process_output(output_buffer, decompressed_size);
        if (is_last) break;
    }
}
```

### Network Stream

```cpp
// Compress data as it arrives from the network
auto stream_mgr = cuda_zstd::create_streaming_manager(3);
stream_mgr->init_compression(0, 64 * 1024);

while (socket.has_data()) {
    auto data = socket.receive();

    size_t compressed_size;
    stream_mgr->compress_chunk(
        data.ptr, data.size,
        output, &compressed_size,
        socket.is_closing()  // Is this the last chunk?
    );

    socket.send(output, compressed_size);
}
```

---

## Architecture

```
+-------------------------------------------------------------+
|                  Streaming Manager                           |
+-------------------------------------------------------------+
|                                                              |
|   Your Data     ===>  [Internal Buffer]  ===>  Compressed    |
|   (arrives in         (accumulates if         (output when   |
|    chunks)             needed)                 ready)        |
|                                                              |
|   State Machine:                                             |
|   +------+    +------+    +------+    +------+              |
|   | INIT | -> | RUN  | -> |FLUSH | -> | END  |              |
|   +------+    +------+    +------+    +------+              |
|                                                              |
+-------------------------------------------------------------+
```

The streaming manager keeps per-chunk state. The history-enabled path (`compress_chunk_with_history`) improves compression ratios by maintaining a sliding window, but each chunk still produces a standalone ZSTD frame.

---

## Configuration Options

### Chunk Size Guidelines

| Scenario | Recommended Chunk Size | Latency | Throughput |
|:---------|:----------------------:|:-------:|:----------:|
| Real-time (video, audio) | 8-16 KB | Ultra-low | Medium |
| General files | 64-128 KB | Low | Fast |
| Maximum throughput | 256 KB+ | Higher | Maximum |

### Independent Frame Limitation

Each chunk produces a complete ZSTD frame with full headers. This means:

- Decompression can start from any chunk (random access)
- Slightly lower compression ratio than a single continuous frame
- The `compress_chunk_with_history` path partially mitigates this with sliding window context
- For maximum compression on large contiguous data, consider using the batch manager with larger block sizes

---

## Common Issues

| Problem | Likely Cause | Solution |
|:--------|:-------------|:---------|
| Output is empty | Forgot to call with `is_last=true` | Always set `is_last` on final chunk |
| Decompression fails | Chunks out of order | Process chunks sequentially |
| Memory growing | Not writing output | Write compressed data after each chunk |
| Low compression ratio | Using `compress_chunk` (independent) | Switch to `compress_chunk_with_history` |
| Init fails | Wrong init method | Use `init_compression_with_history` before `compress_chunk_with_history` |

---

## Related Guides

- [Batch Processing](BATCH-PROCESSING.md) -- For many small files
- [Performance Tuning](PERFORMANCE-TUNING.md) -- Optimize throughput and latency
- [Error Handling](ERROR-HANDLING.md) -- Handle edge cases gracefully
- [Quick Reference](QUICK-REFERENCE.md) -- Copy-paste code snippets
