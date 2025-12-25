# ⚡ Performance Tuning: Make Your Compression Fly

> *"The difference between good and great? About 50 GB/s."*

## Why Performance Matters

Imagine you're copying files. Now imagine doing it **100x faster**. That's what proper tuning can achieve. This guide will take you from "it works" to "it flies."

---

## 🎚️ The Speed vs. Size Tradeoff

Think of compression levels like gears in a car:

| Level | Speed | Compression | Best For |
|:-----:|:-----:|:-----------:|:---------|
| 🏎️ **1-3** | *Blazing fast* | Good | Real-time streaming, logs |
| 🚗 **4-6** | Fast | Better | General purpose |
| 🚌 **7-12** | Moderate | Great | Archival, storage |
| 🐢 **13-22** | Slow | Maximum | Long-term cold storage |

> **Rule of Thumb**: Level 3 gives you 80% of the compression at 5x the speed. Start there!

---

## 🎯 Quick Wins (Do These First!)

### 1. Use Pinned Memory
```cpp
// ❌ Slow: Regular memory
void* buffer = malloc(size);

// ✅ Fast: Pinned memory (2-3x faster transfers!)
void* buffer;
cudaMallocHost(&buffer, size);
```
**What it does**: Pinned memory can be transferred directly to/from the GPU without copying through the CPU first.

### 2. Choose the Right Chunk Size

| Chunk Size | GPU Utilization | When to Use |
|:----------:|:---------------:|:------------|
| 16 KB | 60-70% | Many tiny files |
| **64 KB** | **80-85%** | **Most use cases** ⭐ |
| 128 KB | 85-90% | Large files |
| 256 KB | 90-95% | Maximum throughput |

### 3. Reuse Your Workspace
```cpp
// ❌ Slow: Allocate every time
for (auto& file : files) {
    void* workspace;
    cudaMalloc(&workspace, size);  // Slow!
    compress(file, workspace);
    cudaFree(workspace);           // Even slower!
}

// ✅ Fast: Allocate once, reuse forever
void* workspace;
cudaMalloc(&workspace, size);  // Once!
for (auto& file : files) {
    compress(file, workspace);     // Just reuse it
}
```

---

## 🚀 Advanced Techniques

### Multi-Stream Overlap (The Pipeline Trick)

Imagine a factory where one worker loads materials, another processes them, and a third packages them—all at the same time:

```
Time →   │ Stream 1 │ Stream 2 │ Stream 3 │
─────────┼──────────┼──────────┼──────────┤
Step 1   │ Upload   │          │          │
Step 2   │ Compress │ Upload   │          │
Step 3   │ Download │ Compress │ Upload   │
Step 4   │          │ Download │ Compress │
Step 5   │          │          │ Download │
```

**Result**: 3x throughput with no extra hardware!

```cpp
// Create multiple streams
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

// Use them in rotation
for (int i = 0; i < num_chunks; i++) {
    int s = i % 4;  // Rotate through streams
    upload(streams[s]);
    compress(streams[s]);
    download(streams[s]);
}
```

### The OpenMP Secret Weapon (60+ GB/s)

This is our **highest-performing pattern**:

```cpp
#pragma omp parallel num_threads(8)
{
    // Each thread: own manager + own stream
    auto manager = cuda_zstd::create_manager(3);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    #pragma omp for
    for (int i = 0; i < num_files; i++) {
        manager->compress(files[i], stream);
    }
}
```

**Why it works**: Multiple CPU threads saturate the GPU with work, hiding all latency.

---

## 📈 Benchmark Your Setup

Run our built-in benchmark:
```bash
./benchmark_batch_throughput
```

**Expected output**:
```
=== ZSTD GPU Batch Performance ===
Parallel Batch 64KB  × 1000:  29.42 GB/s ✓
Parallel Batch 256KB ×  500:  61.91 GB/s ✓
```

If you're not hitting these numbers, check:
1. Is the GPU being throttled? (Power/thermal limits)
2. Are you using pinned memory?
3. Is another process using the GPU?

---

## 🔧 Environment Variables

| Variable | What It Does | Try This |
|:---------|:-------------|:---------|
| `CUDA_ZSTD_DEBUG_LEVEL=0` | Disable debug output | Faster in production |
| `CUDA_ZSTD_POOL_MAX_SIZE=4G` | Allow larger memory pool | For big workloads |

---

## 📊 Performance Checklist

Before going to production, verify:

- [ ] Using compression level 3 (unless you need higher ratios)
- [ ] Chunk size is 64KB or larger
- [ ] Using pinned memory for host buffers
- [ ] Workspace is pre-allocated and reused
- [ ] Using multiple streams or OpenMP pattern
- [ ] Debug logging is disabled

---

## 🎓 Learn More

- [Batch Processing](BATCH-PROCESSING.md) — Process thousands of files at once
- [Memory Pool](MEMORY-POOL-IMPLEMENTATION.md) — Optimize GPU memory usage
- [Debugging Guide](DEBUGGING-GUIDE.md) — When things don't go as planned

---

*Remember: Measure first, optimize second. Happy compressing! 🚀*
