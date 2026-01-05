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
  cudaError_t err;
  err = cudaMalloc(&d_input, comp_size);
  if (err != cudaSuccess) {
    std::cerr << "Malloc d_input failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  err = cudaMalloc(&d_output, expected_size * 2);
  if (err != cudaSuccess) {
    std::cerr << "Malloc d_output failed" << std::endl;
    return 1;
  }

  err =
      cudaMemcpy(d_input, compressed.data(), comp_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Memcpy d_input failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  // VERIFY d_input content on GPU
  uint8_t verify_buf[16];
  cudaMemcpy(verify_buf, d_input, std::min(comp_size, (size_t)16),
             cudaMemcpyDeviceToHost);
  printf("[TEST_DBG] d_input first 16 bytes: ");
  for (int i = 0; i < std::min(comp_size, (size_t)16); ++i)
    printf("%02X ", verify_buf[i]);
  printf("\n");

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
    printf("[PASS] Output matches expected!\n");
  } else {
    printf("[FAIL] Output mismatch!\n");
    if (output_size != expected_size) {
      printf("Size: got %zu, expected %zu\n", output_size, expected_size);
    }
    // Dump first 64 bytes of both to see where divergence starts
    printf("GOT (first 64):\n");
    for (int i = 0; i < std::min((int)output_size, 64); i++) {
      printf("%02X ", output[i]);
    }
    printf("\n");
    // Print as char
    for (int i = 0; i < std::min((int)output_size, 64); i++) {
      printf("%c", (output[i] >= 32 && output[i] <= 126) ? output[i] : '.');
    }
    printf("\n");

    printf("EXPECTED (first 64):\n");
    for (int i = 0; i < std::min((int)expected_size, 64); i++) {
      printf("%02X ", expected[i]);
    }
    printf("\n");
    // Print as char
    for (int i = 0; i < std::min((int)expected_size, 64); i++) {
      printf("%c",
             (expected[i] >= 32 && expected[i] <= 126) ? expected[i] : '.');
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);

  printf("\n========================================\n");
  printf("Result: %s\n", match ? "PASS" : "FAIL");
  printf("========================================\n");

  return match ? 0 : 1;
}
