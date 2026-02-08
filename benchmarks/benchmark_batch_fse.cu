// benchmark_batch_fse.cu - Batch Parallel FSE Encoder Benchmarks
// Measures throughput at various scales for 20GB+ workloads

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_safe_alloc.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

#include "benchmark_results.h"

// =============================================================================
// HELPERS
// =============================================================================

void fill_random(std::vector<byte_t> &buffer, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)dist(rng);
  }
}

struct BenchmarkResult {
  u32 num_blocks;
  u32 block_size;
  u64 total_bytes;
  double encode_ms;
  double throughput_gbps;
};

BenchmarkResult run_batch_benchmark(u32 num_blocks, u32 block_size,
                                    int warmup_iters = 2, int bench_iters = 5) {
  u64 total_bytes = (u64)num_blocks * block_size;

  printf("  Benchmarking %u blocks x %uKB = %.2f MB...\n", num_blocks,
         block_size / 1024, total_bytes / (1024.0 * 1024.0));

  // Allocate
  std::vector<std::vector<byte_t>> h_inputs(num_blocks);
  std::vector<byte_t *> d_inputs_ptrs(num_blocks);
  std::vector<byte_t *> d_outputs_ptrs(num_blocks);
  std::vector<u32> input_sizes(num_blocks);
  std::vector<u32> output_sizes(num_blocks);

  for (u32 i = 0; i < num_blocks; i++) {
    h_inputs[i].resize(block_size);
    fill_random(h_inputs[i], i);
    input_sizes[i] = block_size;
    cuda_zstd::safe_cuda_malloc(&d_inputs_ptrs[i], block_size);
    cuda_zstd::safe_cuda_malloc(&d_outputs_ptrs[i], block_size * 2);
    cudaMemcpy(d_inputs_ptrs[i], h_inputs[i].data(), block_size,
               cudaMemcpyHostToDevice);
  }

  cudaDeviceSynchronize();

  // Warmup - pass host arrays of device pointers directly
  for (int i = 0; i < warmup_iters; i++) {
    encode_fse_batch((const byte_t **)d_inputs_ptrs.data(), input_sizes.data(),
                     d_outputs_ptrs.data(), output_sizes.data(), num_blocks, 0);
    cudaDeviceSynchronize();
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; i++) {
    encode_fse_batch((const byte_t **)d_inputs_ptrs.data(), input_sizes.data(),
                     d_outputs_ptrs.data(), output_sizes.data(), num_blocks, 0);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = total_ms / bench_iters;
  double throughput_gbps = (total_bytes / 1e9) / (avg_ms / 1000.0);

  // Cleanup
  for (u32 i = 0; i < num_blocks; i++) {
    cudaFree(d_inputs_ptrs[i]);
    cudaFree(d_outputs_ptrs[i]);
  }

  BenchmarkResult result;
  result.num_blocks = num_blocks;
  result.block_size = block_size;
  result.total_bytes = total_bytes;
  result.encode_ms = avg_ms;
  result.throughput_gbps = throughput_gbps;
  return result;
}

// =============================================================================
// BENCHMARKS
// =============================================================================

void benchmark_scaling(const char *gpu_name) {
  printf("\n=== Benchmark: Scaling with Batch Size ===\n");
  printf("%-12s %-12s %-12s %-12s %-12s\n", "Blocks", "Block Size", "Total",
         "Time(ms)", "Throughput");
  printf("%-12s %-12s %-12s %-12s %-12s\n", "------", "----------", "-----",
         "--------", "----------");

  struct Config {
    u32 num_blocks;
    u32 block_size;
  };

  Config configs[] = {
      {10, 64 * 1024},     // 640KB
      {50, 256 * 1024},    // 12.5MB
      {100, 256 * 1024},   // 25MB
      {200, 512 * 1024},   // 100MB
      {500, 1024 * 1024},  // 500MB
      {1000, 1024 * 1024}, // 1GB
      // {2000, 1024 * 1024},    // 2GB - uncomment if enough GPU memory
  };

  for (auto &cfg : configs) {
    BenchmarkResult r = run_batch_benchmark(cfg.num_blocks, cfg.block_size);

    char size_str[32];
    if (r.total_bytes >= 1024 * 1024 * 1024) {
      snprintf(size_str, sizeof(size_str), "%.1fGB",
               r.total_bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (r.total_bytes >= 1024 * 1024) {
      snprintf(size_str, sizeof(size_str), "%.1fMB",
               r.total_bytes / (1024.0 * 1024.0));
    } else {
      snprintf(size_str, sizeof(size_str), "%.1fKB", r.total_bytes / 1024.0);
    }

    char block_str[32];
    snprintf(block_str, sizeof(block_str), "%uKB", r.block_size / 1024);

    printf("%-12u %-12s %-12s %-12.2f %.2f GB/s\n", r.num_blocks, block_str,
           size_str, r.encode_ms, r.throughput_gbps);

    log_benchmark_result("Batch_FSE_Scaling", gpu_name, r.block_size,
                         r.num_blocks, (double)r.total_bytes, r.encode_ms,
                         r.throughput_gbps);
  }
}

void benchmark_block_size(const char *gpu_name) {
  printf("\n=== Benchmark: Block Size Sweep (Fixed 256MB total) ===\n");
  printf("%-12s %-12s %-12s %-12s\n", "Block Size", "Blocks", "Time(ms)",
         "Throughput");
  printf("%-12s %-12s %-12s %-12s\n", "----------", "------", "--------",
         "----------");

  u64 total_target = 256 * 1024 * 1024; // 256MB

  u32 block_sizes[] = {64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024,
                       1024 * 1024};

  for (u32 bs : block_sizes) {
    u32 num_blocks = total_target / bs;
    BenchmarkResult r = run_batch_benchmark(num_blocks, bs);

    char block_str[32];
    snprintf(block_str, sizeof(block_str), "%uKB", bs / 1024);

    printf("%-12s %-12u %-12.2f %.2f GB/s\n", block_str, num_blocks,
           r.encode_ms, r.throughput_gbps);

    log_benchmark_result("Batch_FSE_BlockSize", gpu_name, r.block_size,
                         r.num_blocks, (double)num_blocks * bs, r.encode_ms,
                         r.throughput_gbps);
  }
}

void benchmark_large_scale(const char *gpu_name) {
  printf("\n=== Benchmark: Large Scale (Production Simulation) ===\n");

  // Simulate 1GB workload
  printf("\n--- 1GB Workload ---\n");
  BenchmarkResult r1gb = run_batch_benchmark(1024, 1024 * 1024, 1, 3);
  printf("  Result: %.2f ms, %.2f GB/s\n", r1gb.encode_ms,
         r1gb.throughput_gbps);
  printf("  Projected 20GB ETA: %.2f seconds\n", (20.0 / r1gb.throughput_gbps));

  log_benchmark_result("Batch_FSE_LargeScale_1GB", gpu_name, 1024 * 1024, 1024,
                       (1024.0 * 1024 * 1024), r1gb.encode_ms,
                       r1gb.throughput_gbps);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
  printf("\n========================================\n");
  printf("  Batch FSE Encoder Benchmarks\n");
  printf("========================================\n");

  // Print GPU info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("Compute: %d.%d\n", prop.major, prop.minor);
  printf("Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

  benchmark_scaling(prop.name);
  benchmark_block_size(prop.name);
  benchmark_large_scale(prop.name);

  printf("\n========================================\n");
  printf("  Benchmarks Complete\n");
  printf("========================================\n");

  return 0;
}
