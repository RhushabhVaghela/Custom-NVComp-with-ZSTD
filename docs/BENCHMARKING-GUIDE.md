# ⏱️ Benchmarking Guide

> *"If you can't measure it, you can't improve it."*

This guide explains how to verify the performance claims of CUDA-ZSTD on your hardware.

## 🛠️ Prerequisites

### 1. Build in Release Mode (CRITICAL)
Benchmarks **must** be built in `Release` mode. Debug builds are 10-50x slower due to assertions and lack of compiler optimizations.

```bash
mkdir build_perf
cd build_perf
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 2. Lock GPU Clocks (Optional but Recommended)
For consistent results, lock your GPU clock speeds to prevent thermal throttling or power saving variances.
```bash
# (Requires sudo)
sudo nvidia-smi -lgc 1410,1410 # Example for V100/A100
```

---

## 🏃 Running the Benchmarks

All benchmark executables are in the `benchmarks/` or `build/` directory after compiling.

### 1. Batch Throughput (The "Hero" Benchmark)
Measures aggregate throughput processing many files in parallel. This is the primary use case for high-throughput systems.

```bash
# Run with default settings (various sizes)
./benchmark_batch_throughput

# Custom usage:
# ./benchmark_batch_throughput <chunk_size_bytes> <batch_count>
./benchmark_batch_throughput 262144 1000
```

**Expected Output (A100):**
- 64KB chunks: ~30 GB/s
- 256KB chunks: ~60 GB/s

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

---

## 📊 Interpreting Results

### Throughput (GB/s)
Higher is better. 
- **Raw GB/s**: Speed of reading input data.
- **Formula**: `Input Size / Time Taken`

### Compression Ratio
Higher is better.
- **Formula**: `Original Size / Compressed Size`
- **Typical**: 2.0x - 4.0x (highly dependent on data).

---

## ⚙️ Usage Scenarios

| Scenario | Benchmark to Run | Target Metrics |
|:---------|:-----------------|:---------------|
| **Many Small Files** | `benchmark_batch_throughput` | 40+ GB/s (Aggregate) |
| **Big Data / ETL** | `benchmark_streaming` | 8+ GB/s (Single Stream) |
| **Latency Sensitive** | `benchmark_roundtrip` | <1ms per op |

---

## 🔧 Troubleshooting Performance

### "I'm only getting 2 GB/s on an A100!"
Check these common culprits:
1.  **Debug Build?** Rebuild with `-DCMAKE_BUILD_TYPE=Release`.
2.  **Pinned Memory?** Are benchmarks using `cudaMallocHost`? (Our included benchmarks do).
3.  **Small Batch?** GPU needs parallelism. Ensure batch count > 100.
4.  **Verification Mode?** Validation adds overhead. Ensure verification is OFF for pure speed measurement.
