# Batch Processing

## What is Batch Processing?

Batch processing compresses hundreds or thousands of data blocks simultaneously on the GPU. Instead of compressing one block at a time, all blocks are dispatched in a single kernel launch and compressed in parallel across thousands of CUDA cores.

```
Traditional (One at a time):        Batch (All at once):
[A] -> [A']                         [A][B][C][D][E][F][G][H] -> [A'][B'][C'][D'][E'][F'][G'][H']
[B] -> [B']                         (Happens in parallel!)
[C] -> [C']
[D] -> [D']
Time: 4 units                       Time: ~1 unit
```

## Measured Performance

All numbers measured on RTX 5080 Laptop GPU (Blackwell sm_120):

| Metric | Configuration | Throughput | Notes |
|:-------|:-------------|:-----------|:------|
| Peak batch | 250 x 256 KB blocks | **9.81 GB/s** | Per-block throughput |
| Small blocks | 2000 x 4 KB blocks | ~2.5 GB/s | Kernel launch overhead dominates |

> Use 64KB-256KB chunks for the best balance between speed and compression ratio.

---

## Why Use Batch Processing?

| Benefit | Detail |
|:--------|:-------|
| High throughput | 9.81 GB/s batch throughput (250 x 256 KB) |
| Linear scaling | More blocks = higher aggregate throughput (GPU parallelism) |
| Low latency | Thousands of small blocks processed in a single kernel launch |

---

## How to Use It

### Basic Usage

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

// Done! 1000 files compressed in milliseconds
```

### OpenMP Multi-Manager Pattern (Maximum Throughput)

For absolute maximum performance, use one manager per CPU thread:

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
// Result: >60 GB/s aggregate throughput
```

---

## When to Use Batch Processing

### Good Fit:
- **Backup systems** -- Compress thousands of files
- **Log aggregation** -- Compress server logs in real-time
- **Game assets** -- Package game files
- **Scientific data** -- Compress simulation outputs
- **Database pages** -- Compress storage pages in bulk

### Not Ideal For:
- Single large files (use [Streaming API](STREAMING-API.md) instead)
- Files that change frequently (setup overhead not worth it)

---

## How It Works

```
Your Data                        GPU (Batch Compression)
                                +---------------------------------+
Block 1 ----------------------> |  Worker 1: Compressing...       |
Block 2 ----------------------> |  Worker 2: Compressing...       |
Block 3 ----------------------> |  Worker 3: Compressing...       |
   ...                          |        ... (thousands more)     |
Block N ----------------------> |  Worker N: Compressing...       |
                                +---------------------------------+
                                            |
                                            v
                                [1'][2'][3']...[N'] All done!
```

The GPU has thousands of CUDA cores that all work simultaneously. While a CPU might have 8-16 cores, a GPU has 10,000+ parallel workers. Batch processing exploits this by assigning one block per thread group.

---

## Related Guides

- [Performance Tuning Guide](PERFORMANCE-TUNING.md) -- Optimize throughput and latency
- [Streaming API](STREAMING-API.md) -- For large files that arrive in chunks
- [Architecture Overview](ARCHITECTURE-OVERVIEW.md) -- How everything fits together
- [Quick Reference](QUICK-REFERENCE.md) -- Copy-paste code snippets
