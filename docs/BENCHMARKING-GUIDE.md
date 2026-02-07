# Benchmarking Guide

> *"If you can't measure it, you can't improve it."*

This guide explains how to verify the performance of CUDA-ZSTD on your hardware.

## Prerequisites

### 1. Build in Release Mode (Critical)

Benchmarks **must** be built in `Release` mode. Debug builds are 10-50x slower due to assertions and lack of compiler optimizations.

```bash
mkdir build_perf
cd build_perf
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 2. Lock GPU Clocks (Optional but Recommended)

For consistent results, lock your GPU clock speeds to prevent thermal throttling.

```bash
# (Requires sudo; adjust frequencies for your GPU)
sudo nvidia-smi -lgc <min_mhz>,<max_mhz>
```

---

## All 30 Benchmark Executables

The project includes 30 benchmark executables. They are built automatically when you build the project.

### Core Performance

| Executable | Description |
|:-----------|:------------|
| `benchmark_batch_throughput` | Aggregate throughput for batch compression (the "hero" benchmark) |
| `benchmark_parallel_throughput` | Multi-stream parallel throughput |
| `benchmark_streaming` | Streaming compression throughput |
| `benchmark_streaming_comparison` | Streaming vs. non-streaming comparison |
| `benchmark_pipeline` | Full pipeline: upload -> compress -> download |
| `benchmark_pipeline_streaming` | Pipeline with streaming API |
| `benchmark_zstd_comparison` | CUDA-ZSTD vs. CPU libzstd |
| `benchmark_zstd_gpu_comparison` | GPU-specific comparison paths |

### Component Benchmarks

| Executable | Description |
|:-----------|:------------|
| `benchmark_fse` | FSE (Finite State Entropy) encoding performance |
| `benchmark_fse_decode` | FSE decoding performance |
| `benchmark_fse_gpu` | GPU-specific FSE performance |
| `benchmark_fse_parallel` | Parallel FSE encoding |
| `benchmark_batch_fse` | Batched FSE operations |
| `benchmark_fse_host_fallback` | CPU fallback path for FSE |
| `benchmark_huffman` | Huffman encoding/decoding |
| `benchmark_lz77` | LZ77 pattern matching |
| `benchmark_xxhash` | XXHash checksum performance |

### Configuration and Tuning

| Executable | Description |
|:-----------|:------------|
| `benchmark_all_levels` | Throughput across all 22 compression levels |
| `benchmark_comprehensive_levels` | Detailed level-by-level analysis |
| `benchmark_adaptive` | Adaptive level selection |
| `benchmark_block_size` | Block size vs. throughput sweep |
| `benchmark_memory_alloc` | GPU memory allocation strategies |
| `benchmark_stream_pool` | StreamPool vs. cudaStreamCreate |
| `benchmark_setup_throughput` | Manager creation/teardown overhead |

### Specialized

| Executable | Description |
|:-----------|:------------|
| `benchmark_c_api` | C API overhead vs. C++ |
| `benchmark_nvcomp_interface` | NVComp v5 API performance |
| `benchmark_inference_api` | Inference workload patterns |
| `benchmark_dictionary_compression` | Dictionary-aided compression |
| `benchmark_parallel_backtracking` | Parallel backtracking search |
| `benchmark_phase3` | Phase 3 (entropy coding) performance |

### Known Issues

Three benchmarks (`benchmark_fse`, `benchmark_lz77`, `benchmark_parallel_backtracking`) may crash with a CUDA illegal memory access error when processing blocks at 1MB or larger. This is a known issue and does not affect production compression, which uses the Smart Router to handle blocks over 1MB on the CPU.

---

## Running the Benchmarks

### 1. Batch Throughput (The "Hero" Benchmark)

Measures aggregate throughput processing many files in parallel. This is the primary use case for high-throughput systems.

```bash
# Run with default settings (various sizes)
./benchmark_batch_throughput

# Custom usage:
# ./benchmark_batch_throughput <chunk_size_bytes> <batch_count>
./benchmark_batch_throughput 262144 1000
```

**Reference Output (RTX 5080 Laptop GPU, Blackwell sm_120):**

```
=== ZSTD GPU Batch Performance ===
Parallel Batch 64KB  x 1000:   9.81 GB/s
```

Your numbers will vary depending on GPU model, clock speeds, thermal conditions, and PCIe bandwidth.

### 2. Streaming (Large Files)

Measures processing a single large stream in chunks.

```bash
./benchmark_streaming --input_size 1073741824  # 1GB
```

### 3. Pipeline Benchmark (End-to-End)

Simulates a full pipeline (Upload -> Compress -> Download).

```bash
./benchmark_pipeline
```

### 4. Stream Pool

Compares pre-allocated stream pool vs. per-operation stream creation.

```bash
./benchmark_stream_pool
```

**Expected speedup:** ~4.7x for stream pool vs. cudaStreamCreate.

### 5. All Compression Levels

```bash
./benchmark_all_levels
```

---

## Measured Performance Baselines

All numbers measured on RTX 5080 Laptop GPU (Blackwell sm_120):

| Metric | Value | Configuration |
|:-------|:------|:--------------|
| Peak Compress | 800.3 MB/s | Single stream, 256 KB |
| Peak Decompress | 1.77 GB/s | Single stream, 256 KB |
| Peak Batch | 9.81 GB/s | 250 x 256 KB blocks |
| Stream Pool speedup | 4.7x | vs. cudaStreamCreate |
| GPU/CPU crossover (FSE) | ~64 KB | Below this, CPU is faster |
| Optimal block size | 64-256 KB | Best throughput per item |

---

## Interpreting Results

### Throughput (GB/s)

Higher is better.
- **Raw GB/s**: Speed of reading input data. Formula: `Input Size / Time Taken`
- **Aggregate GB/s**: Total data processed across all batch items. This is the number reported by `benchmark_batch_throughput`.

### Compression Ratio

Higher is better.
- **Formula**: `Original Size / Compressed Size`
- **Typical**: 2.0x - 4.0x (highly dependent on data content).

---

## Usage Scenarios

| Scenario | Benchmark to Run | Target Metrics |
|:---------|:-----------------|:---------------|
| **Many Small Files** | `benchmark_batch_throughput` | High aggregate GB/s |
| **Big Data / ETL** | `benchmark_streaming` | Sustained GB/s |
| **Latency Sensitive** | `benchmark_c_api` | <1ms per op |
| **Level Selection** | `benchmark_all_levels` | Speed vs. ratio curve |
| **Memory Tuning** | `benchmark_memory_alloc` | Allocation overhead |

---

## Troubleshooting Performance

### "I'm not hitting the expected numbers!"

Check these common culprits:
1. **Debug Build?** Rebuild with `-DCMAKE_BUILD_TYPE=Release`.
2. **Pinned Memory?** Are benchmarks using `cudaMallocHost`? (Included benchmarks do.)
3. **Small Batch?** GPU needs parallelism. Ensure batch count > 100.
4. **Verification Mode?** Validation adds overhead. Ensure verification is OFF for pure speed measurement.
5. **Other GPU processes?** Check `nvidia-smi` for competing workloads.

---

## Related Documentation

- [Performance Tuning](PERFORMANCE-TUNING.md) -- Optimization techniques
- [Batch Processing](BATCH-PROCESSING.md) -- Batch API details
- [Build Guide](BUILD-GUIDE.md) -- Build configuration
