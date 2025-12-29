/**
 * @file test_two_phase_unit.cu
 * @brief Unit tests for two-phase compression architecture
 */

#include "cuda_zstd_manager.h"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cuda_zstd;

/**
 * Test basic compression/decompression roundtrip
 */
int test_basic_roundtrip() {
  std::cout << "Running test_basic_roundtrip..." << std::endl;

  const size_t input_size = 64 * 1024; // 64KB block

  // Allocate input data
  std::vector<uint8_t> h_input(input_size);
  for (size_t i = 0; i < input_size; i++) {
    h_input[i] = static_cast<uint8_t>((i * 17 + 42) & 0xFF);
  }

  // Create compression manager - use ZstdBatchManager
  CompressionConfig config;
  config.level = 3;
  ZstdBatchManager manager(config);

  // Allocate output buffer
  size_t output_size = manager.get_max_compressed_size(input_size);
  uint8_t *d_output = nullptr;
  cudaError_t err = cudaMalloc(&d_output, output_size);
  if (err != cudaSuccess) {
    std::cerr << "  FAILED: cudaMalloc error" << std::endl;
    return 1;
  }

  // Allocate temp workspace
  size_t temp_size = manager.get_compress_temp_size(input_size);
  void *d_temp = nullptr;
  err = cudaMalloc(&d_temp, temp_size);
  if (err != cudaSuccess) {
    cudaFree(d_output);
    std::cerr << "  FAILED: cudaMalloc temp error" << std::endl;
    return 1;
  }

  // Compress
  size_t compressed_size = output_size;
  Status status =
      manager.compress(h_input.data(), input_size, d_output, &compressed_size,
                       d_temp, temp_size, nullptr, 0, 0);
  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: compress error, status=" << static_cast<int>(status)
              << std::endl;
    cudaFree(d_output);
    cudaFree(d_temp);
    return 1;
  }

  std::cout << "  Compressed " << input_size << " -> " << compressed_size
            << " bytes" << std::endl;

  // Cleanup
  cudaFree(d_output);
  cudaFree(d_temp);

  std::cout << "  PASSED" << std::endl;
  return 0;
}

/**
 * Test with multiple blocks
 */
int test_multi_block() {
  std::cout << "Running test_multi_block..." << std::endl;

  const size_t block_size = 64 * 1024;
  const int num_blocks = 4;
  const size_t total_size = block_size * num_blocks;

  std::vector<uint8_t> h_input(total_size);
  for (size_t i = 0; i < total_size; i++) {
    h_input[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
  }

  CompressionConfig config;
  config.level = 3;
  ZstdBatchManager manager(config);

  size_t output_size = manager.get_max_compressed_size(total_size);
  uint8_t *d_output = nullptr;
  cudaMalloc(&d_output, output_size);

  size_t temp_size = manager.get_compress_temp_size(total_size);
  void *d_temp = nullptr;
  cudaMalloc(&d_temp, temp_size);

  size_t compressed_size = output_size;
  Status status =
      manager.compress(h_input.data(), total_size, d_output, &compressed_size,
                       d_temp, temp_size, nullptr, 0, 0);

  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: compress error, status=" << static_cast<int>(status)
              << std::endl;
    cudaFree(d_output);
    cudaFree(d_temp);
    return 1;
  }

  std::cout << "  Compressed " << total_size << " -> " << compressed_size
            << " bytes (" << num_blocks << " blocks)" << std::endl;

  cudaFree(d_output);
  cudaFree(d_temp);

  std::cout << "  PASSED" << std::endl;
  return 0;
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Two-Phase Architecture Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  int failures = 0;

  failures += test_basic_roundtrip();
  failures += test_multi_block();

  std::cout << "========================================" << std::endl;
  if (failures == 0) {
    std::cout << "All tests PASSED!" << std::endl;
  } else {
    std::cout << failures << " tests FAILED!" << std::endl;
  }
  std::cout << "========================================" << std::endl;

  return failures;
}
