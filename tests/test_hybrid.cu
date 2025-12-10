#include "cuda_zstd_manager.h"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <zstd.h>

using namespace cuda_zstd;

void check(Status status, const char *msg) {
  if (status != Status::SUCCESS) {
    std::cerr << "[FAIL] " << msg << " failed with status " << (int)status
              << std::endl;
    exit(1);
  }
}

int main() {
  std::cout << "=== Hybrid CPU/GPU Integration Test + Size 1 Fix ==="
            << std::endl;

  // 1. Setup Manager
  auto manager = create_manager();
  CompressionConfig config = CompressionConfig::get_default();

  // 2. Set Dynamic Threshold (e.g., 2048 bytes)
  config.cpu_threshold = 2048;
  manager->configure(config);

  size_t small_size = 512;
  size_t large_size = 4096;

  // Generate Data
  std::vector<uint8_t> h_small(small_size);
  std::vector<uint8_t> h_large(large_size);
  for (size_t i = 0; i < small_size; i++)
    h_small[i] = (uint8_t)(i % 256);
  for (size_t i = 0; i < large_size; i++)
    h_large[i] = (uint8_t)(i % 256);

  // Device Buffers
  void *d_small, *d_large, *d_comp_small, *d_comp_large, *d_work;
  size_t comp_size_small = ZSTD_compressBound(small_size);
  size_t comp_size_large = ZSTD_compressBound(large_size);

  cudaMalloc(&d_small, small_size);
  cudaMalloc(&d_large, large_size);
  cudaMalloc(&d_comp_small, comp_size_small);
  cudaMalloc(&d_comp_large, comp_size_large);

  // Calc workspace
  size_t work_size = manager->get_compress_temp_size(large_size);
  cudaMalloc(&d_work, work_size);

  cudaMemcpy(d_small, h_small.data(), small_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_large, h_large.data(), large_size, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Test 1: Small Input (Should use CPU path internally)
  std::cout << "Testing Small Input (" << small_size << " bytes)..."
            << std::endl;
  size_t out_size_small = comp_size_small;
  check(manager->compress(d_small, small_size, d_comp_small, &out_size_small,
                          d_work, work_size, nullptr, 0, stream),
        "Compress Small");
  cudaStreamSynchronize(stream);
  std::cout << "Small Compress Success. Size: " << out_size_small << std::endl;

  // Test 2: Large Input (Should use GPU path)
  std::cout << "Testing Large Input (" << large_size << " bytes)..."
            << std::endl;
  size_t out_size_large = comp_size_large;
  check(manager->compress(d_large, large_size, d_comp_large, &out_size_large,
                          d_work, work_size, nullptr, 0, stream),
        "Compress Large");
  cudaStreamSynchronize(stream);
  std::cout << "Large Compress Success. Size: " << out_size_large << std::endl;

  // Test 3: Force GPU for small input
  std::cout << "Changing Threshold to 0 (Force GPU)..." << std::endl;
  config.cpu_threshold = 0;
  manager->configure(config);

  // Test Size 1 specifically
  std::cout << "Testing Size 1 (GPU Forced)..." << std::endl;
  // size_t size_1 = 1;  // Unused - hardcoded below
  void *d_1;
  void *d_comp_1;
  size_t comp_size_1 = ZSTD_compressBound(1);
  cudaMalloc(&d_1, 1);
  cudaMalloc(&d_comp_1, comp_size_1);
  uint8_t h_1 = 0xAA;
  cudaMemcpy(d_1, &h_1, 1, cudaMemcpyHostToDevice);

  size_t out_size_1 = comp_size_1;
  check(manager->compress(d_1, 1, d_comp_1, &out_size_1, d_work, work_size,
                          nullptr, 0, stream),
        "Compress Size 1");
  cudaStreamSynchronize(stream);

  // Verify Decompression for Size 1
  std::cout << "Decompressing Size 1..." << std::endl;
  void *d_out_1;
  cudaMalloc(&d_out_1, 1);
  // size_t decomp_size_1 = 1;  // Unused
  // size_t actual_decomp_size = 0;  // Unused

  // Note: decompress takes uncompressed_size as pointer to output capacity or
  // expected? Prototype: decompress(const void *src, size_t src_size, void
  // *dst, size_t *dst_size, ...) dst_size serves as Capacity (IN) and Actual
  // Size (OUT).
  size_t capacity_1 = 1;
  check(manager->decompress(d_comp_1, out_size_1, d_out_1, &capacity_1, d_work,
                            work_size, stream),
        "Decompress Size 1");
  cudaStreamSynchronize(stream);

  uint8_t h_out_1 = 0;
  cudaMemcpy(&h_out_1, d_out_1, 1, cudaMemcpyDeviceToHost);
  if (h_out_1 != h_1) {
    std::cerr << "Size 1 Content Mismatch! Expected " << (int)h_1 << " got "
              << (int)h_out_1 << std::endl;
    exit(1);
  }
  std::cout << "Size 1: VALID." << std::endl;

  std::cout << "=== Hybrid Test Passed ===" << std::endl;
  return 0;
}
