#include "../include/cuda_zstd_manager.h"
#include "../include/cuda_zstd_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

// Helper to check CUDA errors
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      exit(1);                                                                 \
    }                                                                          \
  }

void generate_random_data(std::vector<byte_t> &data, size_t size) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < size; ++i) {
    data[i] = (byte_t)dist(rng);
  }
}

int main() {
  std::cout << "Running test_fse_integration (Context Reuse)..." << std::endl;

  // Use large enough input to trigger Parallel Path (> 256KB)
  size_t input_size = 1024 * 1024; // 1MB
  std::vector<byte_t> h_input(input_size);
  generate_random_data(h_input, input_size);

  byte_t *d_input;
  byte_t *d_output;
  CHECK(cudaMalloc(&d_input, input_size));
  CHECK(cudaMalloc(&d_output, input_size * 2)); // Ample space
  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  // Create Manager
  auto manager = create_manager();

  // Get Workspace Size
  size_t workspace_size = manager->get_compress_temp_size(input_size);
  std::cout << "Workspace Size: " << workspace_size << " bytes" << std::endl;

  void *d_workspace;
  CHECK(cudaMalloc(&d_workspace, workspace_size));
  CHECK(cudaMemset(d_workspace, 0, workspace_size)); // Initialize

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Run 1
  std::cout << "Run 1..." << std::endl;
  size_t h_size1 = input_size * 2; // Initial max size
  Status status1 =
      manager->compress(d_input, input_size, d_output, &h_size1, d_workspace,
                        workspace_size, nullptr, 0, stream);
  if (status1 != Status::SUCCESS) {
    std::cerr << "Run 1 Failed: " << (int)status1 << std::endl;
    return 1;
  }
  CHECK(cudaStreamSynchronize(stream));

  std::cout << "Run 1 Output Size: " << h_size1 << std::endl;
  if (h_size1 == 0) {
    std::cerr << "Run 1 Produced 0 bytes!" << std::endl;
    return 1;
  }

  // Run 2 (Reuse Workspace)
  std::cout << "Run 2 (Reuse)..." << std::endl;
  size_t h_size2 = input_size * 2;
  Status status2 =
      manager->compress(d_input, input_size, d_output, &h_size2, d_workspace,
                        workspace_size, nullptr, 0, stream);
  if (status2 != Status::SUCCESS) {
    std::cerr << "Run 2 Failed: " << (int)status2 << std::endl;
    return 1;
  }
  CHECK(cudaStreamSynchronize(stream));

  std::cout << "Run 2 Output Size: " << h_size2 << std::endl;

  if (h_size1 != h_size2) {
    std::cerr << "Mismatch! Run 1=" << h_size1 << " Run 2=" << h_size2
              << std::endl;
    // Deterministic compression should match exactly
    return 1;
  }

  std::cout << "Integration Test PASSED" << std::endl;

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  CHECK(cudaFree(d_workspace));
  CHECK(cudaStreamDestroy(stream));

  return 0;
}
