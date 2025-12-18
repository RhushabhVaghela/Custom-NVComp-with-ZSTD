#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_manager.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


using namespace cuda_zstd;

int main() {
  std::cout << "Test: Dictionary compression debugging" << std::endl;

  // Simple data
  std::vector<uint8_t> h_input(1000, 0xAA);

  // Allocate device memory
  void *d_input, *d_output, *d_temp;
  cudaMalloc(&d_input, 1000);
  cudaMalloc(&d_output, 2000);
  cudaMemcpy(d_input, h_input.data(), 1000, cudaMemcpyHostToDevice);

  // Create manager
  std::cout << "Creating manager..." << std::endl;
  ZstdBatchManager manager(CompressionConfig{.level = 3});

  // Create simple dictionary
  std::cout << "Creating dictionary..." << std::endl;
  dictionary::Dictionary dict;
  dict.raw_size = 100;
  dict.raw_content = new uint8_t[100];
  memset(dict.raw_content, 0x55, 100);
  dict.header.dictionary_id = 12345;

  // Set dictionary
  std::cout << "Setting dictionary..." << std::endl;
  Status status = manager.set_dictionary(dict);
  if (status != Status::SUCCESS) {
    std::cerr << "set_dictionary failed: " << (int)status << std::endl;
    return 1;
  }
  std::cout << "✓ Dictionary set" << std::endl;

  // Get temp size and allocate
  std::cout << "Getting temp size..." << std::endl;
  size_t temp_size = manager.get_compress_temp_size(1000);
  std::cout << "Temp size: " << temp_size << std::endl;
  cudaMalloc(&d_temp, temp_size);

  // Compress
  std::cout << "Compressing..." << std::endl;
  size_t compressed_size;
  status = manager.compress(d_input, 1000, d_output, &compressed_size, d_temp,
                            temp_size, nullptr, 0);

  std::cout << "Compress status: " << (int)status << std::endl;

  if (status == Status::SUCCESS) {
    std::cout << "✓ SUCCESS! Compressed: 1000 -> " << compressed_size
              << std::endl;
  } else {
    std::cout << "✗ FAILED with status: " << (int)status << std::endl;
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);
  delete[] dict.raw_content;

  return (status == Status::SUCCESS) ? 0 : 1;
}
