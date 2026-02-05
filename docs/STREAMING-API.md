# 🌊 Streaming API: Process Data as It Flows

> *"Why wait for the whole river when you can drink as it flows?"*

## What is Streaming?

Imagine you're watching a live video. You don't wait for the entire movie to download—you watch it as it arrives. **Streaming compression** works the same way!

Note: The current implementation produces a separate Zstd frame per chunk. It does not maintain a single continuous frame across chunks. The `compress_chunk_with_history` path retains a sliding window for better ratios, but still emits independent frames.

```
Traditional Compression:          Streaming Compression:
                                   
Wait... Wait... Wait...           Start immediately!
        ↓                                ↓
[████████████████] 100%           [█░░░░░░░░░░░░░░░] → 📦
        ↓                         [████░░░░░░░░░░░] → 📦📦
       📦                         [████████░░░░░░░] → 📦📦📦
                                  [████████████░░░] → 📦📦📦📦
Total time: 10 seconds            [████████████████] → 📦📦📦📦📦
                                  Total time: 10 seconds
                                  BUT: Output starts at 2 seconds!
```

## 🎯 When to Use Streaming

### ✅ Perfect For:
| Scenario | Why Streaming Wins |
|:---------|:-------------------|
| 📡 **Live data feeds** | Can't wait for "the end"—there isn't one! |
| 📁 **Huge files** | Don't need to fit entire file in memory |
| 🌐 **Network transfers** | Start sending compressed data immediately |
| 💾 **Limited memory** | Process 100GB file with only 128KB buffer |

### ❌ Skip Streaming When:
- You have many small files (use [Batch Processing](BATCH-PROCESSING.md) instead)
- The entire file fits easily in memory

---

## 🛠️ How to Use It

### Basic Example: Compress a File Piece by Piece

```cpp
#include "cuda_zstd_manager.h"

void compress_huge_file(const std::string& filename) {
    // 1. Create a streaming manager
    auto stream_mgr = cuda_zstd::create_streaming_manager(5);
    // Use init_compression_with_history to enable better ratios across chunks
    stream_mgr->init_compression_with_history();
    
    // 2. Process the file in 128KB chunks
...
        // Compress it (GPU does the heavy lifting!)
        size_t compressed_size;
        stream_mgr->compress_chunk_with_history(
            chunk.data(), bytes_read,
            output_buffer, &compressed_size,
            is_last
        );
        
        // Write compressed data immediately
        output.write((char*)output_buffer, compressed_size);
    }
    
    // That's it! File compressed in chunks 🎉
}
```

### Real-World Example: Network Stream

```cpp
// Compress data as it arrives from the network
while (socket.has_data()) {
    auto data = socket.receive();
    
    size_t compressed_size;
    stream_mgr->compress_chunk(
        data.ptr, data.size,
        output, &compressed_size,
        socket.is_closing()  // Is this the last chunk?
    );
    
    // Send compressed data immediately
    socket.send(output, compressed_size);
}
```

---

## 🎨 Visual: How Streaming Works Inside

```
┌─────────────────────────────────────────────────────────────┐
│                  Streaming Manager                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Your Data     ═══▶  [Internal Buffer]  ═══▶  Compressed   │
│   (arrives in         (accumulates if         (output when  │
│    chunks)             needed)                 ready)        │
│                                                              │
│   State Machine:                                             │
│   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐            │
│   │ INIT │ →  │ RUN  │ →  │ FLUSH│ →  │ END  │            │
│   └──────┘    └──────┘    └──────┘    └──────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The streaming manager keeps per-chunk state, and the history-enabled path improves ratios, but each chunk is still a standalone frame.

---

## ⚙️ Configuration Options

### Chunk Size Guidelines

| Your Situation | Recommended Chunk Size | Latency | Speed |
|:---------------|:----------------------:|:-------:|:-----:|
| Real-time (video, audio) | 8-16 KB | ⚡ Ultra-low | Medium |
| General files | 64-128 KB | Low | Fast |
| Maximum throughput | 256 KB+ | Higher | 🚀 Maximum |

### Flush Modes

| Mode | What Happens | When to Use |
|:-----|:-------------|:------------|
| **Continue** | Buffer data, output when optimal | Normal operation |
| **Flush** | Output everything now | Need immediate output |
| **End** | Finalize the frame | Last chunk of data |

---

## 🧪 Testing Your Streaming Code

```cpp
// Test with a file you can verify
void test_streaming_roundtrip() {
    std::vector<uint8_t> original = load_file("test.bin");
    
    // Compress in streaming mode
    auto compressed = streaming_compress(original);
    
    // Decompress
    auto decompressed = streaming_decompress(compressed);
    
    // Verify
    assert(original == decompressed);
    printf("✅ Roundtrip successful!\n");
}
```

---

## 🔍 Common Issues

| Problem | Likely Cause | Solution |
|:--------|:-------------|:---------|
| Output is empty | Forgot to call with `is_last=true` | Always set `is_last` on final chunk |
| Decompression fails | Chunks out of order | Process chunks sequentially |
| Memory growing | Not writing output | Write compressed data after each chunk |

---

## 📚 Related Guides

- [Batch Processing](BATCH-PROCESSING.md) — For many small files
- [Performance Tuning](PERFORMANCE-TUNING.md) — Optimize your streaming
- [Error Handling](ERROR-HANDLING.md) — Handle edge cases gracefully

---

*Streaming: Because the best time to start compressing is right now! 🌊*
