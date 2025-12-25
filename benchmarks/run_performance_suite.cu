// ==============================================================================
// run_performance_suite.cu - Performance Baseline Suite
// ==============================================================================
// Runs compression/decompression across a range of chunk sizes and reports
// throughput.

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_zstd_manager.h"

using namespace cuda_zstd;

struct Result {
  size_t size;
  double compress_mbps;
  double decompress_mbps;
};

Result benchmark_size(size_t size, int runs = 5, int warmup = 2) {
  // Generate data
  std::vector<uint8_t> h_data(size);
  // Use random-ish data but compressible
  for (size_t i = 0; i < size; i++) {
    h_data[i] = (uint8_t)((i * 12345 + i / 100) % 256);
  }

  void *d_input, *d_output, *d_decompressed;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size * 2);
  cudaMalloc(&d_decompressed, size);
  cudaMemcpy(d_input, h_data.data(), size, cudaMemcpyHostToDevice);

  // Setup Manager
  ZstdBatchManager manager(CompressionConfig::get_default());

  // Get Temp Sizes
  size_t temp_size_c = manager.get_compress_temp_size(size);
  size_t temp_size_d = manager.get_decompress_temp_size(size * 2);
  size_t temp_size = std::max(temp_size_c, temp_size_d);

  void *d_temp;
  cudaMalloc(&d_temp, temp_size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // --- Compression ---
  size_t compressed_size = size * 2;

  // Warmup
  for (int i = 0; i < warmup; i++) {
    compressed_size = size * 2;
    manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                     temp_size, nullptr, 0, stream);
    cudaStreamSynchronize(stream);
  }

  // Throughput Measurement: Pipelined
  const int ops_per_run = 50; // Queue 50 ops per run to hide latency
  // Increase runs to get meaningful time
  int actual_runs = (runs < 10 && size < 1024 * 1024) ? 50 : runs;

  auto start_c = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < actual_runs; i++) {
    for (int j = 0; j < ops_per_run; j++) {
      compressed_size = size * 2;
      manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                       temp_size, nullptr, 0, stream);
    }
    cudaStreamSynchronize(stream);
  }
  auto end_c = std::chrono::high_resolution_clock::now();
  double total_ops = (double)actual_runs * ops_per_run;
  double ms_c =
      std::chrono::duration<double, std::milli>(end_c - start_c).count();
  // Avg time per op is ms_c / total_ops, but for MB/s utilize total bytes
  double total_mb = (size * total_ops) / (1024.0 * 1024.0);
  double c_mbps = total_mb / (ms_c / 1000.0);

  // --- Decompression ---
  size_t decompressed_size = size;

  // Warmup
  for (int i = 0; i < warmup; i++) {
    decompressed_size = size;
    manager.decompress(d_output, compressed_size, d_decompressed,
                       &decompressed_size, d_temp, temp_size, stream);
    cudaStreamSynchronize(stream);
  }

  auto start_d = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < actual_runs; i++) {
    for (int j = 0; j < ops_per_run; j++) {
      decompressed_size = size;
      manager.decompress(d_output, compressed_size, d_decompressed,
                         &decompressed_size, d_temp, temp_size, stream);
    }
    cudaStreamSynchronize(stream);
  }
  auto end_d = std::chrono::high_resolution_clock::now();
  double ms_d =
      std::chrono::duration<double, std::milli>(end_d - start_d).count();
  double d_mbps = total_mb / (ms_d / 1000.0);

  cudaFree(d_temp);
  cudaStreamDestroy(stream);
  cudaFree(d_decompressed);
  cudaFree(d_output);
  cudaFree(d_input);

  return {size, c_mbps, d_mbps};
}

int main() {
  std::cout << "=== ZSTD GPU Performance Baseline Suite ===" << std::endl;
  std::cout << "Target: >10GB/s (Single stream, batched operations)"
            << std::endl
            << std::endl;

  // Check GPU
  int count;
  cudaGetDeviceCount(&count);
  if (count == 0)
    return 1;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU: " << prop.name << std::endl << std::endl;

  std::vector<size_t> sizes = {
      4096,             // 4KB
      16 * 1024,        // 16KB
      64 * 1024,        // 64KB
      256 * 1024,       // 256KB
      1024 * 1024,      // 1MB
      4 * 1024 * 1024,  // 4MB
      16 * 1024 * 1024, // 16MB
      64 * 1024 * 1024  // 64MB
  };

  std::cout << std::setw(10) << "Size" << " | " << std::setw(15)
            << "Compress (MB/s)" << " | " << std::setw(15)
            << "Decompress (MB/s)" << std::endl;
  std::cout << std::string(46, '-') << std::endl;

  for (size_t s : sizes) {
    Result r = benchmark_size(s);
    std::string unit = " B ";
    double disp_size = (double)s;
    if (s >= 1024) {
      disp_size /= 1024.0;
      unit = " KB";
    }
    if (s >= 1024 * 1024) {
      disp_size /= 1024.0;
      unit = " MB";
    } // Corrected logic: already div by 1024 once

    if (s >= 1024 * 1024)
      unit = " MB"; // Simple redisplay logic fix
    double show_val = (double)s;
    if (s >= 1024)
      show_val /= 1024;
    if (s >= 1024 * 1024)
      show_val /= 1024;

    std::cout << std::setw(7) << show_val << unit << " | " << std::setw(15)
              << std::fixed << std::setprecision(2) << r.compress_mbps << " | "
              << std::setw(15) << r.decompress_mbps << std::endl;
  }

  return 0;
}
