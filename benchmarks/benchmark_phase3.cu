#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/benchmark_results.h"
#include "../include/cuda_zstd_manager.h"

// INTERNAL: We need access to the internal FSE encoder to benchmark it directly
// skipping the LZ77 stage which currently limits block sizes.
namespace cuda_zstd {
namespace fse {
// using byte_t = uint8_t; // Rely on header
// using u32 = uint32_t; // Rely on header
// enum class Status; // Already defined in headers
Status encode_fse_advanced(const byte_t *d_input, u32 input_size,
                           byte_t *d_output, u32 *d_output_size,
                           bool gpu_optimize, cudaStream_t stream);
} // namespace fse
} // namespace cuda_zstd

// Function to generate synthetic data with specific entropy
void generate_data(std::vector<uint8_t> &data, size_t size,
                   double target_entropy) {
  data.resize(size);

  // Simple distribution generator
  // Valid for 0.0 < entropy <= 8.0
  // We'll use a Zipf-like distribution or just a limited alphabet

  int num_symbols = 256;
  if (target_entropy < 1.0)
    num_symbols = 2;
  else if (target_entropy < 3.0)
    num_symbols = 10;
  else if (target_entropy < 5.0)
    num_symbols = 50;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, num_symbols - 1);

  for (size_t i = 0; i < size; i++) {
    data[i] = (uint8_t)dist(rng);
  }
}

void benchmark_phase3_fse(const std::string &gpu_name) {
  std::cout << "============================================================"
            << std::endl;
  std::cout << "   Phase 3 Benchmark: Intra-Block Parallel FSE Throughput"
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  // Use a large file size to ensure we trigger the Chunk Parallel path (>256KB)
  // 64MB implies 64MB / 64KB = 1024 chunks
  const size_t input_size = 64 * 1024 * 1024;
  std::vector<uint8_t> h_input;
  generate_data(h_input, input_size, 4.0); // Moderate entropy

  void *d_input, *d_output;
  uint32_t *d_output_size;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, input_size * 2); // Conservative
  cudaMalloc(&d_output_size, sizeof(uint32_t));

  cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Warmup
  std::cout << "Warming up..." << std::endl;
  cuda_zstd::fse::encode_fse_advanced(
      (const uint8_t *)d_input, (uint32_t)input_size, (uint8_t *)d_output,
      d_output_size, true,
      stream // gpu_optimize=true implies parallel usage if size sufficient
  );
  cudaStreamSynchronize(stream);

  // Measure
  int iterations = 10;
  std::cout << "Running " << iterations << " iterations for "
            << (input_size / (1024.0 * 1024.0)) << " MB..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    cuda_zstd::fse::encode_fse_advanced(
        (const uint8_t *)d_input, (uint32_t)input_size, (uint8_t *)d_output,
        d_output_size, true, stream);
  }
  cudaStreamSynchronize(stream);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double total_bytes = (double)input_size * iterations;
  double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
  double throughput = total_gb / elapsed.count();

  std::cout << "Total Time: " << elapsed.count() << " s" << std::endl;
  std::cout << "Throughput: " << std::fixed << std::setprecision(2)
            << throughput << " GB/s" << std::endl;

  // Log to CSV
  // Log to CSV
  // Signature: name, gpu, block_size, num_blocks, total_bytes, time_ms,
  // throughput
  log_benchmark_result("Phase3_ChunkFSE_64MB", gpu_name.c_str(),
                       (unsigned int)input_size, 1, total_bytes,
                       elapsed.count() * 1000.0, throughput);

  // Cleanup
  cudaStreamDestroy(stream);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::string gpu_name = prop.name;

  benchmark_phase3_fse(gpu_name);
  return 0;
}
