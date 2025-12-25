# 🚀 Batch Processing: Compress at Warp Speed

> *"Imagine compressing 1,000 files in the time it takes to blink. That's batch processing."*

## What is Batch Processing?

Think of batch processing like a **factory assembly line** for data compression. Instead of compressing one file at a time (like making sandwiches one by one), we compress hundreds or thousands simultaneously—like having 1,000 chefs all making sandwiches at once!

```
Traditional (One at a time):        Batch (All at once):
📄 → 📦                              📄📄📄📄📄📄📄📄 → 📦📦📦📦📦📦📦📦
📄 → 📦                              (Happens in parallel!)
📄 → 📦
📄 → 📦
⏱️ 4 seconds                         ⏱️ 0.1 seconds
```

## 🎯 Why Should You Care?

| What You Get | The Benefit |
|:-------------|:------------|
| **60+ GB/s Throughput** | Compress a 4K movie in under 1 second |
| **Linear Scaling** | 2x more files = same time (GPU handles it) |
| **Lower Latency** | Process thousands of small files instantly |

---

## 🏎️ Performance: See the Numbers

Here's what we achieved on real hardware:

| Chunk Size | Files Processed | Speed | That Means... |
|:-----------|:---------------:|:-----:|:--------------|
| 4 KB | 2,000 | 2.5 GB/s | 500,000 small files per second! |
| 64 KB | 1,000 | **29.4 GB/s** | A Blu-ray disc in 1.5 seconds |
| 256 KB | 500 | **61.9 GB/s** | 4 USB drives per second |

> 💡 **Pro Tip**: Use 64KB-256KB chunks for the sweet spot between speed and compression ratio.

---

## 🛠️ How to Use It

### The Simple Way (5 Lines of Code)

```cpp
// 1. Create a batch manager
auto manager = cuda_zstd::ZstdBatchManager::create(3);  // Level 3 = fast

// 2. Tell it what to compress
manager->compress_batch(
    my_file_pointers,     // Your 1000 files
    my_file_sizes,        // How big each one is
    output_pointers,      // Where to put compressed data
    output_sizes,         // Will tell you compressed sizes
    1000,                 // Number of files
    workspace, ws_size,   // GPU scratch space
    stream                // GPU stream
);

// Done! 1000 files compressed in milliseconds! 🎉
```

### The Power User Way (Maximum Speed)

For absolute maximum performance, use the **OpenMP Multi-Manager** pattern:

```cpp
#pragma omp parallel num_threads(8)
{
    // Each CPU thread gets its own GPU compression manager
    auto my_manager = cuda_zstd::create_manager(3);
    
    #pragma omp for
    for (int i = 0; i < num_files; ++i) {
        my_manager->compress(files[i], ...);
    }
}
// Result: >60 GB/s throughput! 🚀
```

---

## 📊 When to Use Batch Processing

### ✅ Perfect For:
- 📁 **Backup systems** — Compress thousands of files overnight
- 📊 **Log aggregation** — Compress server logs in real-time
- 🎮 **Game assets** — Package game files lightning-fast
- 🔬 **Scientific data** — Compress simulation outputs

### ❌ Not Ideal For:
- Single large files (use streaming instead)
- Files that change frequently (overhead not worth it)

---

## 🧠 How It Works (The Fun Version)

```
Your Files                      GPU (The Compression Factory)
                               ┌─────────────────────────────────┐
📄 File 1 ──────────────────▶ │  🔨 Worker 1: Compressing...    │
📄 File 2 ──────────────────▶ │  🔨 Worker 2: Compressing...    │
📄 File 3 ──────────────────▶ │  🔨 Worker 3: Compressing...    │
   ...                         │        ... (thousands more)     │
📄 File N ──────────────────▶ │  🔨 Worker N: Compressing...    │
                               └─────────────────────────────────┘
                                            │
                                            ▼
                               📦📦📦📦📦📦📦📦 All done!
```

The GPU has **thousands of workers** (CUDA cores) that all work simultaneously. While your CPU might have 8-16 cores, a GPU has **10,000+** parallel workers!

---

## 📚 Learn More

- [Performance Tuning Guide](PERFORMANCE-TUNING.md) — Squeeze out every last bit of speed
- [Streaming API](STREAMING-API.md) — For large files that come in chunks
- [Architecture Overview](ARCHITECTURE-OVERVIEW.md) — How everything fits together

---

*Ready to compress at warp speed? Check out the [Quick Reference](QUICK-REFERENCE.md) for copy-paste code snippets!*
