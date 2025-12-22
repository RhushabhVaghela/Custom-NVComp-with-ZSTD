#include "../include/cuda_zstd_manager.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple test runner for Smart Path Selection
void test_selector_logic() {
  using namespace cuda_zstd;

  std::cout << "Testing Selector Logic (Thresholds)..." << std::endl;

  // Test 1: Very Small Payload (< 64KB default threshold)
  // Expect: CPU
  auto p1 = ZstdBatchManager::select_execution_path(100);
  if (p1 != ZstdBatchManager::ExecutionPath::CPU) {
    std::cerr << "Test 1 Failed: Expected CPU for size 100" << std::endl;
    exit(1);
  }

  // Test 2: Borderline Small (63KB)
  // Expect: CPU
  auto p2 = ZstdBatchManager::select_execution_path(63 * 1024);
  if (p2 != ZstdBatchManager::ExecutionPath::CPU) {
    std::cerr << "Test 2 Failed: Expected CPU for size 63KB" << std::endl;
    exit(1);
  }

  // Test 3: Medium Payload (256KB)
  // Expect: GPU_BATCH (Standard)
  auto p3 = ZstdBatchManager::select_execution_path(256 * 1024);
  if (p3 != ZstdBatchManager::ExecutionPath::GPU_BATCH) {
    std::cerr << "Test 3 Failed: Expected GPU_BATCH for size 256KB"
              << std::endl;
    exit(1);
  }

  // Test 4: Large Payload (10MB)
  // Expect: GPU_BATCH (Fallback for now, or GPU_CHUNK later)
  // Currently implementation returns GPU_BATCH for large files too (TODO Phase
  // 3 kernels)
  auto p4 = ZstdBatchManager::select_execution_path(10 * 1024 * 1024);
  if (p4 != ZstdBatchManager::ExecutionPath::GPU_BATCH) {
    // If implemented CHUNK, this needs update
    std::cerr << "Test 4 Failed: Expected GPU_BATCH for size 10MB" << std::endl;
    exit(1);
  }

  std::cout << "Selector Logic Tests PASSED." << std::endl;
}

void test_functional_cpu_path() {
  using namespace cuda_zstd;
  std::cout << "Testing Functional CPU Path (<64KB)..." << std::endl;

  auto manager = create_manager(3);

  // Create random small data
  size_t size = 1000;
  std::vector<uint8_t> host_data(size);
  for (size_t i = 0; i < size; i++)
    host_data[i] = i % 255;

  // Compress using Manager (Auto-select CPU)
  void *d_comp;
  void *d_temp;
  size_t comp_size = size * 2;

  cudaMalloc(&d_comp, size * 2); // Max size
  cudaMalloc(&d_temp, manager->get_compress_temp_size(size));

  void *d_input;
  cudaMalloc(&d_input, size);
  cudaMemcpy(d_input, host_data.data(), size, cudaMemcpyHostToDevice);

  // This should route to CPU internally
  Status status =
      manager->compress(d_input, size, d_comp, &comp_size, d_temp,
                        manager->get_compress_temp_size(size), nullptr, 0, 0);

  if (status != Status::SUCCESS) {
    std::cerr << "Compression Failed with status " << (int)status << std::endl;
    exit(1);
  }

  std::cout << "Compressed Size: " << comp_size << std::endl;
  if (comp_size == 0) {
    std::cerr << "Compression produced 0 bytes!" << std::endl;
    exit(1);
  }

  cudaFree(d_comp);
  cudaFree(d_temp);
  cudaFree(d_input);

  std::cout << "Functional CPU Path PASSED." << std::endl;
}

int main() {
  test_selector_logic();
  test_functional_cpu_path();
  return 0;
}
