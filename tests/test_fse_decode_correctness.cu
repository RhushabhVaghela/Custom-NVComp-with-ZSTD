#include "../include/cuda_zstd_manager.h"
#include "../include/cuda_zstd_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;     \
      exit(1);                                                                 \
    }                                                                          \
  }

// Generate reproducible test data
void generate_test_data(std::vector<byte_t> &data, size_t size, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);

  for (size_t i = 0; i < size; ++i) {
    // Mix of patterns for varied compression ratio
    if (i % 32 < 16) {
      data[i] = dist(rng) % 64; // Low entropy
    } else {
      data[i] = dist(rng); // High entropy
    }
  }
}

bool test_roundtrip(size_t size, int seed) {
  std::cout << "  Testing " << size << " bytes (seed=" << seed << ")..."
            << std::flush;

  // Prepare test data
  std::vector<byte_t> h_input(size);
  generate_test_data(h_input, size, seed);

  // Allocate device memory
  byte_t *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, size));
  CHECK_CUDA(cudaMalloc(&d_compressed, size * 2));
  CHECK_CUDA(cudaMalloc(&d_output, size));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

  // Create manager
  auto manager = create_manager(3);

  // Allocate workspace
  size_t compress_temp_size = manager->get_compress_temp_size(size);
  void *d_temp;
  CHECK_CUDA(cudaMalloc(&d_temp, compress_temp_size));
  CHECK_CUDA(cudaMemset(d_temp, 0, compress_temp_size));

  // Compress
  size_t compressed_size = size * 2;
  Status status =
      manager->compress(d_input, size, d_compressed, &compressed_size, d_temp,
                        compress_temp_size, nullptr, 0, 0);
  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (compress: " << (int)status << ")" << std::endl;
    return false;
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Reallocate workspace for decompress
  CHECK_CUDA(cudaFree(d_temp));
  size_t decompress_temp_size =
      manager->get_decompress_temp_size(compressed_size);
  CHECK_CUDA(cudaMalloc(&d_temp, decompress_temp_size));
  CHECK_CUDA(cudaMemset(d_temp, 0, decompress_temp_size));

  // Decompress
  size_t decompressed_size = size;
  status =
      manager->decompress(d_compressed, compressed_size, d_output,
                          &decompressed_size, d_temp, decompress_temp_size, 0);
  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (decompress: " << (int)status << ")" << std::endl;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
    return false;
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Verify
  std::vector<byte_t> h_output(decompressed_size);
  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, decompressed_size,
                        cudaMemcpyDeviceToHost));

  bool match = (decompressed_size == size);
  if (match) {
    for (size_t i = 0; i < size; ++i) {
      if (h_output[i] != h_input[i]) {
        std::cerr << " FAILED (mismatch at byte " << i << ")" << std::endl;
        match = false;
        break;
      }
    }
  } else {
    std::cerr << " FAILED (size mismatch: " << decompressed_size << " vs "
              << size << ")" << std::endl;
  }

  if (match) {
    double ratio = (double)size / compressed_size;
    std::cout << " PASSED (ratio: " << ratio << "x)" << std::endl;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_temp));

  return match;
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "FSE Decode Correctness Tests" << std::endl;
  std::cout << "======================================" << std::endl;

  int passed = 0;
  int failed = 0;

  // Test various sizes
  std::vector<size_t> sizes = {
      1024,   // 1 KB
      4096,   // 4 KB
      16384,  // 16 KB
      65536,  // 64 KB
      131072, // 128 KB (ZSTD_BLOCKSIZE_MAX)
      262144, // 256 KB
      1048576 // 1 MB
  };

  std::cout << "\n[TEST] Roundtrip correctness at various sizes:" << std::endl;
  for (size_t size : sizes) {
    if (test_roundtrip(size, 42)) {
      passed++;
    } else {
      failed++;
    }
  }

  std::cout << "\n[TEST] Multiple seeds for consistency:" << std::endl;
  for (int seed = 1; seed <= 5; seed++) {
    if (test_roundtrip(65536, seed)) {
      passed++;
    } else {
      failed++;
    }
  }

  std::cout << "\n======================================" << std::endl;
  std::cout << "Results: " << passed << " PASSED, " << failed << " FAILED"
            << std::endl;
  std::cout << "======================================" << std::endl;

  return failed > 0 ? 1 : 0;
}
