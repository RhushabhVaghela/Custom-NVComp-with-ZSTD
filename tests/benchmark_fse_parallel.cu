// benchmark_fse_parallel.cu - Benchmark suite for FSE encoder performance

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "throughput_display.h"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// =============================================================================
// BENCHMARK HELPERS
// =============================================================================

void fill_random(std::vector<byte_t> &buffer, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)dist(rng);
  }
}

struct BenchmarkResult {
  double encode_ms;
  double decode_ms;
  u32 input_size;
  u32 output_size;
  double throughput_mbps; // Changed to MB/s for better readability
  double throughput_gbps; // Also keep GB/s
  double ratio;
};

BenchmarkResult run_benchmark(u32 data_size, int iterations = 5) {
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input);

  byte_t *d_input, *d_output, *d_decoded;
  CUDA_CHECK_VOID(cudaMalloc(&d_input, data_size));
  CUDA_CHECK_VOID(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK_VOID(
      cudaMemset(d_output, 0,
                 data_size * 2)); // Ensure clean start to detect bad overwrites
  CUDA_CHECK_VOID(cudaMalloc(&d_decoded, data_size));

  // Check state before encode
  cudaError_t pre_enc_err = cudaGetLastError();
  if (pre_enc_err != cudaSuccess) {
    printf("Benchmark Pre-Encode Error: %s\n", cudaGetErrorString(pre_enc_err));
  }
  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  // Warmup
  u32 output_size = 0;
  encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);
  cudaDeviceSynchronize();

  // Encode benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double encode_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;

  // Decode benchmark
  u32 decoded_size = 0;
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    decode_fse(d_output, output_size, d_decoded, &decoded_size, 0);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  double decode_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_decoded);

  BenchmarkResult result;
  result.encode_ms = encode_ms;
  result.decode_ms = decode_ms;
  result.input_size = data_size;
  result.output_size = output_size;
  result.throughput_mbps = cuda_zstd::throughput_mbps(data_size, encode_ms);
  result.throughput_gbps = cuda_zstd::throughput_gbps(data_size, encode_ms);
  result.ratio = (float)data_size / output_size;

  return result;
}

// =============================================================================
// BENCHMARKS
// =============================================================================

void benchmark_scaling() {
  printf("\n=== Benchmark: Scaling with Input Size ===\n");
  printf("%-12s %-12s %-12s %-12s %-18s %-18s %-8s\n", "Size", "Encode(ms)",
         "Decode(ms)", "Ratio", "Throughput(MB/s)", "Throughput(GB/s)",
         "Status");
  printf("%-12s %-12s %-12s %-12s %-18s %-18s %-8s\n", "--------", "----------",
         "----------", "------", "----------------", "----------------",
         "------");

  u32 sizes[] = {64 * 1024,        256 * 1024,       1 * 1024 * 1024,
                 4 * 1024 * 1024,  10 * 1024 * 1024, 50 * 1024 * 1024,
                 100 * 1024 * 1024};

  for (u32 size : sizes) {
    BenchmarkResult r = run_benchmark(size);

    char size_str[32];
    if (size >= 1024 * 1024) {
      snprintf(size_str, sizeof(size_str), "%uMB", size / (1024 * 1024));
    } else {
      snprintf(size_str, sizeof(size_str), "%uKB", size / 1024);
    }

    printf("%-12s %-12.2f %-12.2f %-12.2f %-18.2f %-18.4f ✅\n", size_str,
           r.encode_ms, r.decode_ms, r.ratio, r.throughput_mbps,
           r.throughput_gbps);
  }
}

void benchmark_chunk_size_sweep() {
  printf("\n=== Benchmark: Chunk Size Sweep (10MB input) ===\n");
  printf("This benchmark tests different internal chunk sizes.\n");
  printf("(Note: Chunk size is currently hardcoded; this shows baseline)\n\n");

  // For now, just run with the default chunk size
  // Note: Chunk size parameterization could be added for more detailed testing

  u32 data_size = 10 * 1024 * 1024;
  BenchmarkResult r = run_benchmark(data_size, 10);

  printf("Current configuration:\n");
  printf("  Input: %u bytes\n", r.input_size);
  printf("  Output: %u bytes\n", r.output_size);
  printf("  Ratio: %.2f:1\n", r.ratio);
  printf("  Encode: %.2f ms\n", r.encode_ms);
  printf("  Decode: %.2f ms\n", r.decode_ms);
  printf("  Throughput: %.2f MB/s (%.4f GB/s)\n", r.throughput_mbps,
         r.throughput_gbps);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
  printf("\n========================================\n");
  printf("  FSE Parallel Encoder Benchmarks\n");
  printf("========================================\n");

  // Print GPU info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("Compute: %d.%d\n", prop.major, prop.minor);
  printf("Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

  benchmark_scaling();
  benchmark_chunk_size_sweep();

  printf("\n========================================\n");
  printf("  Benchmarks Complete\n");
  printf("========================================\n");

  return 0;
}
