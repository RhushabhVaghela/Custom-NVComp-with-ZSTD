// ==============================================================================
// test_cuda_zstd_utils_dedicated.cu - Dedicated tests for cuda_zstd_utils.cu
//
// Tests:
// 1. parallel_scan<u32> - Exclusive prefix sum on integers
// 2. parallel_sort_dmers - Thrust sort on Dmer structs
// 3. debug_kernel_verify - Debug utility (environment-dependent)
// ==============================================================================

#include "cuda_zstd_internal.h" // For dictionary::Dmer
#include "cuda_zstd_utils.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cuda_zstd;

#ifndef CUDA_CHECK
#define TEST_CUDA_CHECK(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)
#endif

// ==============================================================================
// Test 1: parallel_scan<u32>
// ==============================================================================
bool test_parallel_scan_u32() {
  std::cout << "[TEST] parallel_scan<u32>..." << std::flush;

  const u32 N = 1024;
  std::vector<u32> h_input(N);
  std::vector<u32> h_output(N);
  std::vector<u32> h_expected(N);

  // Fill input: 1, 2, 3, ..., N
  for (u32 i = 0; i < N; ++i)
    h_input[i] = i + 1;

  // Expected exclusive scan: 0, 1, 3, 6, 10, ...
  h_expected[0] = 0;
  for (u32 i = 1; i < N; ++i)
    h_expected[i] = h_expected[i - 1] + h_input[i - 1];

  u32 *d_input, *d_output;
  TEST_CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(u32)));
  TEST_CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(u32)));
  TEST_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(u32),
                             cudaMemcpyHostToDevice));

  cudaStream_t stream;
  TEST_CUDA_CHECK(cudaStreamCreate(&stream));

  Status status = utils::parallel_scan<u32>(d_input, d_output, N, stream);
  TEST_CUDA_CHECK(cudaStreamSynchronize(stream));

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (Status=" << (int)status << ")" << std::endl;
    return false;
  }

  TEST_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(u32),
                             cudaMemcpyDeviceToHost));

  // Verify
  bool pass = true;
  for (u32 i = 0; i < N; ++i) {
    if (h_output[i] != h_expected[i]) {
      std::cerr << " FAILED at index " << i << " (got " << h_output[i]
                << ", expected " << h_expected[i] << ")" << std::endl;
      pass = false;
      break;
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaStreamDestroy(stream);

  if (pass)
    std::cout << " PASSED" << std::endl;
  return pass;
}

// ==============================================================================
// Test 2: parallel_scan<u32> with empty input
// ==============================================================================
bool test_parallel_scan_empty() {
  std::cout << "[TEST] parallel_scan<u32> (empty)..." << std::flush;

  cudaStream_t stream;
  TEST_CUDA_CHECK(cudaStreamCreate(&stream));

  Status status = utils::parallel_scan<u32>(nullptr, nullptr, 0, stream);

  cudaStreamDestroy(stream);

  if (status == Status::SUCCESS) {
    std::cout << " PASSED" << std::endl;
    return true;
  } else {
    std::cerr << " FAILED (Status=" << (int)status << ")" << std::endl;
    return false;
  }
}

// ==============================================================================
// Test 3: parallel_sort_dmers
// ==============================================================================
bool test_parallel_sort_dmers() {
  std::cout << "[TEST] parallel_sort_dmers..." << std::flush;

  const u32 N = 512;
  std::vector<dictionary::Dmer> h_dmers(N);

  // Fill with random hashes
  for (u32 i = 0; i < N; ++i) {
    h_dmers[i].hash = (u64)(N - i) * 12345; // Reverse order
    h_dmers[i].position = i;
    h_dmers[i].length = 4;
  }

  dictionary::Dmer *d_dmers;
  TEST_CUDA_CHECK(cudaMalloc(&d_dmers, N * sizeof(dictionary::Dmer)));
  TEST_CUDA_CHECK(cudaMemcpy(d_dmers, h_dmers.data(),
                             N * sizeof(dictionary::Dmer),
                             cudaMemcpyHostToDevice));

  cudaStream_t stream;
  TEST_CUDA_CHECK(cudaStreamCreate(&stream));

  Status status = utils::parallel_sort_dmers(d_dmers, N, stream);
  TEST_CUDA_CHECK(cudaStreamSynchronize(stream));

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (Status=" << (int)status << ")" << std::endl;
    return false;
  }

  std::vector<dictionary::Dmer> h_sorted(N);
  TEST_CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_dmers,
                             N * sizeof(dictionary::Dmer),
                             cudaMemcpyDeviceToHost));

  // Verify sorted by hash
  bool pass = true;
  for (u32 i = 1; i < N; ++i) {
    if (h_sorted[i].hash < h_sorted[i - 1].hash) {
      std::cerr << " FAILED: Not sorted at index " << i << std::endl;
      pass = false;
      break;
    }
  }

  cudaFree(d_dmers);
  cudaStreamDestroy(stream);

  if (pass)
    std::cout << " PASSED" << std::endl;
  return pass;
}

// ==============================================================================
// Test 4: debug_kernel_verify (smoke test)
// ==============================================================================
bool test_debug_kernel_verify() {
  std::cout << "[TEST] debug_kernel_verify..." << std::flush;

  // Just call it, should not crash
  cudaError_t err = utils::debug_kernel_verify("test_location");

  if (err == cudaSuccess) {
    std::cout << " PASSED" << std::endl;
    return true;
  } else {
    std::cerr << " FAILED (cudaError=" << cudaGetErrorString(err) << ")"
              << std::endl;
    return false;
  }
}

// ==============================================================================
// Main
// ==============================================================================
int main() {
  std::cout << "=== cuda_zstd_utils.cu Dedicated Tests ===" << std::endl;

  int passed = 0, failed = 0;

  if (test_parallel_scan_u32())
    passed++;
  else
    failed++;
  if (test_parallel_scan_empty())
    passed++;
  else
    failed++;
  if (test_parallel_sort_dmers())
    passed++;
  else
    failed++;
  if (test_debug_kernel_verify())
    passed++;
  else
    failed++;

  std::cout << "\n=== Results: " << passed << " passed, " << failed
            << " failed ===" << std::endl;
  return failed == 0 ? 0 : 1;
}
