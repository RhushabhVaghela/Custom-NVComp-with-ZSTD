/**
 * @file test_libzstd_interop.cu
 * @brief Test GPU decoder with libzstd-compressed data
 */
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cuda_zstd;

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "libzstd Interop Test" << std::endl;
  std::cout << "========================================" << std::endl;

  // Read compressed data
  std::ifstream zst_file("test_interop.zst", std::ios::binary | std::ios::ate);
  if (!zst_file.is_open()) {
    std::cerr << "[ERROR] Cannot open test_interop.zst" << std::endl;
    return 1;
  }
  size_t comp_size = zst_file.tellg();
  zst_file.seekg(0, std::ios::beg);
  std::vector<uint8_t> compressed(comp_size);
  zst_file.read(reinterpret_cast<char *>(compressed.data()), comp_size);
  zst_file.close();

  std::cout << "Compressed size: " << comp_size << " bytes" << std::endl;

  // Read expected output
  std::ifstream exp_file("test_interop.txt", std::ios::binary | std::ios::ate);
  if (!exp_file.is_open()) {
    std::cerr << "[ERROR] Cannot open test_interop.txt" << std::endl;
    return 1;
  }
  size_t expected_size = exp_file.tellg();
  exp_file.seekg(0, std::ios::beg);
  std::vector<uint8_t> expected(expected_size);
  exp_file.read(reinterpret_cast<char *>(expected.data()), expected_size);
  exp_file.close();

  std::cout << "Expected size: " << expected_size << " bytes" << std::endl;

  // Allocate device memory
  uint8_t *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;
  cudaMalloc(&d_input, comp_size);
  cudaMalloc(&d_output, expected_size * 2);
  cudaMemcpy(d_input, compressed.data(), comp_size, cudaMemcpyHostToDevice);

  // Use manager API (same as test_correctness.cu)
  auto manager = create_manager(5);
  size_t temp_size = manager->get_compress_temp_size(expected_size);
  cudaMalloc(&d_temp, temp_size);

  // Decompress
  size_t output_size = expected_size * 2;
  Status status = manager->decompress(d_input, comp_size, d_output,
                                      &output_size, d_temp, temp_size);

  if (status != Status::SUCCESS) {
    std::cerr << "[FAIL] Decompress failed: " << (int)status << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    return 1;
  }

  std::cout << "Output size: " << output_size << " bytes" << std::endl;

  // Copy back and compare
  std::vector<uint8_t> output(output_size);
  cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

  bool match = (output_size == expected_size) &&
               (memcmp(output.data(), expected.data(), expected_size) == 0);

  if (match) {
    std::cout << "[PASS] Output matches expected!" << std::endl;
  } else {
    std::cerr << "[FAIL] Output mismatch!" << std::endl;
    if (output_size != expected_size) {
      std::cerr << "Size: got " << output_size << ", expected " << expected_size
                << std::endl;
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);

  std::cout << "\n========================================" << std::endl;
  std::cout << "Result: " << (match ? "PASS" : "FAIL") << std::endl;
  std::cout << "========================================" << std::endl;

  return match ? 0 : 1;
}
