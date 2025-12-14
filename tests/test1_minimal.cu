// Minimal Test #1 Reproduction
// Compile: nvcc -o test1_minimal test1_minimal.cu -I../include -L../build
// -lcuda_zstd -std=c++17 Run: cuda-memcheck ./test1_minimal

#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>

int main() {
  using namespace cuda_zstd;

  std::cout << "=== Reproducing Test #1 ===\n";
  std::cout << "Input: 1KB, Block size: 128KB (will be clamped to 1KB)\n\n";

  const size_t input_size = 1024;             // 1 KB
  const u32 original_block_size = 128 * 1024; // 128 KB

  // Create test data
  std::vector<uint8_t> h_input(input_size);
  for (size_t i = 0; i < input_size; i++) {
    h_input[i] = (uint8_t)(i % 256);
  }

  // Create config (same as benchmark line 105-106)
  CompressionConfig config = CompressionConfig::from_level(3);
  config.block_size = std::min(original_block_size, (u32)input_size);
  config.cpu_threshold = 0; // Force GPU (parallel mode like Test #2)

  std::cout << "Config: block_size=" << config.block_size << "\n";
  std::cout << "Creating ZstdBatchManager...\n";

  // Check for errors before manager creation
  cudaError_t pre_err = cudaGetLastError();
  if (pre_err != cudaSuccess) {
    std::cout << "ERROR before manager: " << cudaGetErrorString(pre_err)
              << "\n";
  }

  ZstdBatchManager manager(config);

  // Check for errors after manager creation
  cudaError_t post_err = cudaGetLastError();
  if (post_err != cudaSuccess) {
    std::cout << "ERROR after manager creation: "
              << cudaGetErrorString(post_err) << "\n";
    return 1;
  }
  std::cout << "Manager created successfully\n";

  // Get sizes
  std::cout << "Getting workspace sizes...\n";
  size_t max_compressed = manager.get_max_compressed_size(input_size);
  size_t temp_size = manager.get_compress_temp_size(input_size);

  cudaError_t size_err = cudaGetLastError();
  if (size_err != cudaSuccess) {
    std::cout << "ERROR after get sizes: " << cudaGetErrorString(size_err)
              << "\n";
    return 1;
  }
  std::cout << "max_compressed=" << max_compressed
            << ", temp_size=" << temp_size << "\n";

  // Allocate device memory
  std::cout << "Allocating GPU memory...\n";
  void *d_input, *d_compressed, *d_temp;
  if (cudaMalloc(&d_input, input_size) != cudaSuccess) {
    std::cout << "ERROR: cudaMalloc d_input failed\n";
    return 1;
  }
  if (cudaMalloc(&d_compressed, max_compressed) != cudaSuccess) {
    std::cout << "ERROR: cudaMalloc d_compressed failed\n";
    cudaFree(d_input);
    return 1;
  }
  if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
    std::cout << "ERROR: cudaMalloc d_temp failed\n";
    cudaFree(d_input);
    cudaFree(d_compressed);
    return 1;
  }
  std::cout << "Memory allocated\n";

  // Copy input to device
  if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "ERROR: cudaMemcpy failed\n";
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    return 1;
  }

  // Compress
  std::cout << "Calling compress...\n";
  size_t compressed_size = max_compressed;
  Status status =
      manager.compress(d_input, input_size, d_compressed, &compressed_size,
                       d_temp, temp_size, nullptr, 0, 0);

  cudaError_t compress_err = cudaGetLastError();
  std::cout << "Compress status: " << (int)status << "\n";
  std::cout << "CUDA error after compress: " << cudaGetErrorString(compress_err)
            << "\n";

  if (compress_err != cudaSuccess) {
    std::cout << "\n❌ BUG REPRODUCED!\n";
    std::cout << "Sticky error: " << cudaGetErrorString(compress_err) << "\n";
  } else {
    std::cout << "\n✓ No sticky error found\n";
  }

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_temp);

  return (compress_err != cudaSuccess) ? 1 : 0;
}
