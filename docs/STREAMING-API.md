# CUDA-ZSTD Streaming API Guide

## Overview

The Streaming API enables compression/decompression of data that arrives incrementally, maintaining state across chunks while producing valid ZSTD frames.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ZstdStreamingManager                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Internal State                         │   │
│  │  - Frame context (window, checksum)                       │   │
│  │  - Block buffer (partial data)                            │   │
│  │  - Dictionary context                                     │   │
│  │  - Flush mode                                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Input Chunks → [Compress Chunk] → Output Chunks                │
│                                                                  │
│  Modes:                                                          │
│    • CONTINUE - More data coming                                 │
│    • FLUSH    - Produce output, reset block                      │
│    • END      - Finalize frame                                   │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Class Definition

```cpp
class ZstdStreamingManager {
public:
    // Create streaming manager
    static std::unique_ptr<ZstdStreamingManager> create(int level = 3);
    
    // Initialize for compression
    Status init_compression();
    
    // Initialize for decompression
    Status init_decompression();
    
    // Compress a chunk
    Status compress_chunk(
        const void* d_input,
        size_t input_size,
        void* d_output,
        size_t* output_size,
        bool is_last_chunk,
        cudaStream_t stream = 0
    );
    
    // Decompress a chunk
    Status decompress_chunk(
        const void* d_input,
        size_t input_size,
        void* d_output,
        size_t* output_size,
        bool* frame_complete,
        cudaStream_t stream = 0
    );
    
    // Reset for new stream
    Status reset();
    
    // Get internal statistics
    StreamingStats get_stats();
};
```

## Usage Examples

### File Compression

```cpp
#include "cuda_zstd_manager.h"
#include <fstream>

void compress_file_streaming(const std::string& input_path,
                             const std::string& output_path) {
    using namespace cuda_zstd;
    
    // Create manager
    auto manager = ZstdStreamingManager::create(5);
    manager->init_compression();
    
    // Setup buffers
    const size_t CHUNK_SIZE = 128 * 1024;  // 128KB chunks
    std::vector<uint8_t> h_input(CHUNK_SIZE);
    std::vector<uint8_t> h_output(CHUNK_SIZE * 2);
    
    void *d_input, *d_output;
    cudaMalloc(&d_input, CHUNK_SIZE);
    cudaMalloc(&d_output, CHUNK_SIZE * 2);
    
    std::ifstream fin(input_path, std::ios::binary);
    std::ofstream fout(output_path, std::ios::binary);
    
    while (fin) {
        // Read chunk
        fin.read((char*)h_input.data(), CHUNK_SIZE);
        size_t bytes_read = fin.gcount();
        if (bytes_read == 0) break;
        
        bool is_last = fin.eof();
        
        // Transfer to GPU
        cudaMemcpy(d_input, h_input.data(), bytes_read, 
                   cudaMemcpyHostToDevice);
        
        // Compress chunk
        size_t out_size = CHUNK_SIZE * 2;
        Status status = manager->compress_chunk(
            d_input, bytes_read,
            d_output, &out_size,
            is_last, 0
        );
        
        if (status != Status::SUCCESS) {
            fprintf(stderr, "Compression error: %s\n", 
                    status_to_string(status));
            break;
        }
        
        // Write compressed data
        cudaMemcpy(h_output.data(), d_output, out_size,
                   cudaMemcpyDeviceToHost);
        fout.write((char*)h_output.data(), out_size);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}
```

### Network Stream Compression

```cpp
void compress_network_stream(Socket& socket) {
    auto manager = cuda_zstd::ZstdStreamingManager::create(3);
    manager->init_compression();
    
    void *d_buf, *d_out;
    cudaMalloc(&d_buf, 64 * 1024);
    cudaMalloc(&d_out, 128 * 1024);
    
    while (socket.connected()) {
        // Receive data (non-blocking)
        std::vector<uint8_t> data = socket.receive(64 * 1024);
        if (data.empty()) continue;
        
        bool is_last = socket.is_closing();
        
        cudaMemcpy(d_buf, data.data(), data.size(), 
                   cudaMemcpyHostToDevice);
        
        size_t out_size;
        manager->compress_chunk(d_buf, data.size(),
                                d_out, &out_size, is_last);
        
        std::vector<uint8_t> compressed(out_size);
        cudaMemcpy(compressed.data(), d_out, out_size,
                   cudaMemcpyDeviceToHost);
        
        // Send compressed data
        socket.send(compressed);
        
        if (is_last) break;
    }
}
```

### Decompression with Unknown Size

```cpp
void decompress_streaming(const void* compressed_data,
                          size_t compressed_size,
                          std::vector<uint8_t>& output) {
    auto manager = cuda_zstd::ZstdStreamingManager::create();
    manager->init_decompression();
    
    const size_t chunk_size = 64 * 1024;
    void *d_input, *d_output;
    cudaMalloc(&d_input, chunk_size);
    cudaMalloc(&d_output, chunk_size * 10);  // Decompression expands
    
    size_t offset = 0;
    bool frame_complete = false;
    
    while (offset < compressed_size && !frame_complete) {
        size_t this_chunk = std::min(chunk_size, compressed_size - offset);
        
        cudaMemcpy(d_input, (uint8_t*)compressed_data + offset,
                   this_chunk, cudaMemcpyHostToDevice);
        
        size_t out_size = chunk_size * 10;
        manager->decompress_chunk(
            d_input, this_chunk,
            d_output, &out_size,
            &frame_complete, 0
        );
        
        // Append to output
        std::vector<uint8_t> chunk(out_size);
        cudaMemcpy(chunk.data(), d_output, out_size,
                   cudaMemcpyDeviceToHost);
        output.insert(output.end(), chunk.begin(), chunk.end());
        
        offset += this_chunk;
    }
}
```

## Configuration Options

### Chunk Size Guidelines

| Use Case | Recommended Size | Latency | Throughput |
|:---------|:----------------:|:-------:|:----------:|
| Real-time | 8-16 KB | Low | Medium |
| Balanced | 64-128 KB | Medium | High |
| Maximum | 256+ KB | High | Maximum |

### Flush Modes

| Mode | Behavior |
|:-----|:---------|
| `FLUSH_NONE` | Buffer data, output when full |
| `FLUSH_BLOCK` | Complete current block, output |
| `FLUSH_FRAME` | End frame, output, reset |

## Thread Safety

- Single manager: **NOT** thread-safe
- Multiple managers: Thread-safe (separate instances)
- CUDA stream: Provides async safety within one manager

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_manager.cu` | ZstdStreamingManager |
| `tests/test_streaming.cu` | Streaming tests |
| `benchmarks/benchmark_streaming.cu` | Streaming benchmark |

## Related Documentation
- [BATCH-PROCESSING.md](BATCH-PROCESSING.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
- [PERFORMANCE-TUNING.md](PERFORMANCE-TUNING.md)
