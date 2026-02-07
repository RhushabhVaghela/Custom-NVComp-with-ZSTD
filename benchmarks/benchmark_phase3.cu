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

#include "benchmark_results.h"
#include "../include/cuda_zstd_fse.h"
#include "../include/cuda_zstd_manager.h"

// INTERNAL: We need access to the internal FSE encoder to benchmark it directly
// skipping the LZ77 stage which currently limits block sizes.
// namespace cuda_zstd {
// namespace fse {
// // using byte_t = uint8_t; // Rely on header
// // using u32 = uint32_t; // Rely on header
// // enum class Status; // Already defined in headers
// Status encode_fse_advanced(const byte_t *d_input, u32 input_size,
//                            byte_t *d_output, u32 *d_output_size,
//                            bool gpu_optimize, cudaStream_t stream,
//                            FSEContext *ctx = nullptr);
// } // namespace fse
// } // namespace cuda_zstd

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

void benchmark_phase3_fse(const std::string &gpu_name,
                          size_t custom_input_size) {
  // Ensure fresh start (Fixes sticky Pre-existing errors)
  cudaDeviceReset();

  std::cout << "============================================================"
            << std::endl;
  std::cout << "   Phase 3 Benchmark: Intra-Block Parallel FSE Throughput"
            << std::endl;
  std::cout << "   (Memory Optimized: Context Reuse Enabled)" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  const size_t input_size =
      custom_input_size > 0 ? custom_input_size : (1 * 1024 * 1024);
  std::vector<uint8_t> h_input;
  generate_data(h_input, input_size, 4.0); // Moderate entropy

  // ... (rest of function remains same until end)

  void *d_input, *d_output;
  uint32_t *d_output_size;

  cudaError_t err;
  err = cudaMalloc(&d_input, input_size);
  if (err != cudaSuccess) {
    printf("Malloc d_input failed: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMalloc(&d_output, input_size * 2);
  if (err != cudaSuccess) {
    printf("Malloc d_output failed: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMalloc(&d_output_size, sizeof(uint32_t));
  if (err != cudaSuccess) {
    printf("Malloc d_output_size failed: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("Memcpy d_input failed: %s\n", cudaGetErrorString(err));
    return;
  }

  // Warmup
  std::cout << "Warming up..." << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Create Reuse Context
  cuda_zstd::FSEContext context;
  // Initialize to nulls (constructor does this, but being explicit is safe)
  memset(&context, 0, sizeof(context));

  // Warmup (Allocates persistent memory in context)
  std::cout << "Warming up..." << std::endl;
  cudaGetLastError(); // Clear any prior errors
  auto warmup_status = cuda_zstd::fse::encode_fse_advanced(
      (const uint8_t *)d_input, (uint32_t)input_size, (uint8_t *)d_output,
      d_output_size, true, stream,
      &context // Pass context to allocate
  );
  cudaStreamSynchronize(stream);
  auto cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess || warmup_status != cuda_zstd::Status::SUCCESS) {
    std::cerr << "Warmup failed! Status=" << (int)warmup_status
              << " CUDA=" << cudaGetErrorString(cuda_err) << std::endl;
    // Continue anyway to see iteration behavior
  }

  // Measure (Reuses memory in context)
  int iterations = 10;
  std::cout << "Running " << iterations << " iterations for "
            << (input_size / (1024.0 * 1024.0)) << " MB (" << input_size
            << " bytes)..." << std::endl;

  // NOTE: Measurements here include Kernel Launch overhead and ANY remaining
  // host work but explicitly EXCLUDE cudaMalloc overhead now.

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    cuda_zstd::fse::encode_fse_advanced(
        (const uint8_t *)d_input, (uint32_t)input_size, (uint8_t *)d_output,
        d_output_size, true, stream, &context);
  }
  cudaStreamSynchronize(stream);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double total_bytes = (double)input_size * iterations;
  double throughput_gbps = (total_bytes / 1e9) / elapsed.count();
  double time_ms = (elapsed.count() * 1000.0) / iterations;

  std::cout << "Total Time: " << elapsed.count() << " s" << std::endl;
  std::cout << "Throughput: " << throughput_gbps << " GB/s" << std::endl;

  log_benchmark_result("Phase3_Parallel_FSE_Optimized", gpu_name.c_str(), 0, 0,
                       total_bytes, time_ms, throughput_gbps);

  // Cleanup Context
  if (context.d_dev_symbol_table)
    cudaFree(context.d_dev_symbol_table);
  if (context.d_dev_next_state)
    cudaFree(context.d_dev_next_state);
  if (context.d_dev_nbBits_table)
    cudaFree(context.d_dev_nbBits_table);
  if (context.d_dev_next_state_vals)
    cudaFree(context.d_dev_next_state_vals);
  if (context.d_dev_initial_states)
    cudaFree(context.d_dev_initial_states);
  if (context.d_ctable_for_encoder)
    cudaFree(context.d_ctable_for_encoder);
  if (context.d_chunk_start_states)
    cudaFree(context.d_chunk_start_states);
  if (context.d_bitstreams)
    cudaFree(context.d_bitstreams);
  if (context.d_chunk_bit_counts)
    cudaFree(context.d_chunk_bit_counts);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);
  cudaStreamDestroy(stream);
}

int main(int argc, char **argv) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::string gpu_name = prop.name;

  size_t input_size = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input_size" && i + 1 < argc) {
      input_size = std::stoull(argv[i + 1]);
      i++;
    }
  }

  benchmark_phase3_fse(gpu_name, input_size);
  return 0;
}
