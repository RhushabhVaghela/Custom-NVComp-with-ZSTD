#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>

int main() {
  using namespace cuda_zstd;

  // Reproduce Test #1: 128KB blocks on 1KB input
  const size_t input_size = 1024;    // 1 KB
  const u32 block_size = 128 * 1024; // 128 KB

  std::cout << "Reproducing Test #1 bug:\n";
  std::cout << "  Input: " << input_size << " bytes\n";
  std::cout << "  Block size: " << block_size << " bytes\n";
  std::cout << "  Block size > Input: "
            << (block_size > input_size ? "YES" : "NO") << "\n\n";

  // Create test data
  std::vector<uint8_t> input_data(input_size, 0x42);

  // Allocate device memory
  void *d_input, *d_output, *d_temp;
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, input_size * 2);

  // Create manager
  auto manager = create_manager();
  size_t temp_size = manager->get_compress_temp_size(input_size);
  cudaMalloc(&d_temp, temp_size);

  std::cout << "Workspace size: " << temp_size << " bytes\n\n";

  // Copy input
  cudaMemcpy(d_input, input_data.data(), input_size, cudaMemcpyHostToDevice);

  // Compress with config
  CompressionConfig config;
  config.block_size = block_size;
  config.level = 3;

  size_t compressed_size = input_size * 2;

  std::cout << "Calling compress...\n";
  Status status =
      manager->compress(d_input, input_size, d_output, &compressed_size, d_temp,
                        temp_size, &config, 0, 0);

  cudaError_t err = cudaGetLastError();
  std::cout << "Compress status: " << (int)status << "\n";
  std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";

  if (err != cudaSuccess) {
    std::cout << "\n❌ BUG REPRODUCED! Invalid argument error found.\n";
  } else {
    std::cout << "\n✓ No error (unexpected - bug not reproduced)\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);

  return (err != cudaSuccess) ? 1 : 0;
}
