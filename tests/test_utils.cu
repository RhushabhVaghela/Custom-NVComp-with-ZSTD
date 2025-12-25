// test_utils.cu - Tests for cuda_zstd_utils.cu and cuda_zstd_utils.cpp
// Covers: parallel_scan, parallel_sort_dmers, radix sort kernels

#include "cuda_zstd_internal.h"
#include "cuda_zstd_types.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace cuda_zstd;

// ==============================================================================
// Test Helpers
// ==============================================================================

struct TestResult {
  const char *name;
  bool passed;
};

std::vector<TestResult> g_results;

void record_test(const char *name, bool passed) {
  g_results.push_back({name, passed});
  printf("  %s: %s\n", name, passed ? "PASS" : "FAIL");
}

// Forward declarations from cuda_zstd_utils.cu
namespace cuda_zstd {
namespace utils {
template <typename T>
__host__ Status parallel_scan(const T *d_input, u32 *d_output, u32 num_elements,
                              cudaStream_t stream);

__host__ Status parallel_sort_dmers(dictionary::Dmer *d_dmers, u32 num_dmers,
                                    cudaStream_t stream);
} // namespace utils
} // namespace cuda_zstd

// ==============================================================================
// Test: Parallel Scan - Basic
// ==============================================================================

bool test_parallel_scan_basic() {
  printf("=== Test: Parallel Scan Basic ===\n");

  const u32 num_elements = 100;

  u32 *d_input = nullptr;
  u32 *d_output = nullptr;

  cudaMalloc(&d_input, num_elements * sizeof(u32));
  cudaMalloc(&d_output, num_elements * sizeof(u32));

  // Create input: [1, 1, 1, ... 1]
  std::vector<u32> h_input(num_elements, 1);
  cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(u32),
             cudaMemcpyHostToDevice);

  Status status = utils::parallel_scan<u32>(d_input, d_output, num_elements, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Verify: exclusive scan of [1,1,1,...] should be [0,1,2,3,...]
    std::vector<u32> h_output(num_elements);
    cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(u32),
               cudaMemcpyDeviceToHost);

    for (u32 i = 0; i < num_elements; ++i) {
      if (h_output[i] != i) {
        printf("    Mismatch at %u: expected %u, got %u\n", i, i, h_output[i]);
        passed = false;
        break;
      }
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);

  record_test("Parallel Scan Basic", passed);
  return passed;
}

// ==============================================================================
// Test: Parallel Scan - Empty
// ==============================================================================

bool test_parallel_scan_empty() {
  printf("=== Test: Parallel Scan Empty ===\n");

  Status status = utils::parallel_scan<u32>(nullptr, nullptr, 0, 0);

  bool passed = (status == Status::SUCCESS);
  record_test("Parallel Scan Empty", passed);
  return passed;
}

// ==============================================================================
// Test: Parallel Scan - Large
// ==============================================================================

bool test_parallel_scan_large() {
  printf("=== Test: Parallel Scan Large ===\n");

  const u32 num_elements = 1000000; // 1M elements

  u32 *d_input = nullptr;
  u32 *d_output = nullptr;

  cudaMalloc(&d_input, num_elements * sizeof(u32));
  cudaMalloc(&d_output, num_elements * sizeof(u32));

  // Create input: [1, 2, 3, ..., n]
  std::vector<u32> h_input(num_elements);
  for (u32 i = 0; i < num_elements; ++i) {
    h_input[i] = 1;
  }
  cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(u32),
             cudaMemcpyHostToDevice);

  Status status = utils::parallel_scan<u32>(d_input, d_output, num_elements, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Spot check last element: should be sum of all previous = num_elements - 1
    std::vector<u32> h_output(num_elements);
    cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(u32),
               cudaMemcpyDeviceToHost);

    // Check a few positions
    if (h_output[0] != 0)
      passed = false;
    if (h_output[100] != 100)
      passed = false;
    if (h_output[num_elements - 1] != num_elements - 1)
      passed = false;
  }

  cudaFree(d_input);
  cudaFree(d_output);

  record_test("Parallel Scan Large (1M)", passed);
  return passed;
}

// ==============================================================================
// Test: Parallel Sort Dmers - Basic
// ==============================================================================

bool test_parallel_sort_dmers_basic() {
  printf("=== Test: Parallel Sort Dmers Basic ===\n");

  const u32 num_dmers = 100;

  dictionary::Dmer *d_dmers = nullptr;
  cudaMalloc(&d_dmers, num_dmers * sizeof(dictionary::Dmer));

  // Create unsorted dmers
  std::vector<dictionary::Dmer> h_dmers(num_dmers);
  std::mt19937 rng(42);
  for (u32 i = 0; i < num_dmers; ++i) {
    h_dmers[i].hash = rng() % 1000000;
    h_dmers[i].position = i;
  }

  cudaMemcpy(d_dmers, h_dmers.data(), num_dmers * sizeof(dictionary::Dmer),
             cudaMemcpyHostToDevice);

  Status status = utils::parallel_sort_dmers(d_dmers, num_dmers, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Verify sorted order
    std::vector<dictionary::Dmer> h_sorted(num_dmers);
    cudaMemcpy(h_sorted.data(), d_dmers, num_dmers * sizeof(dictionary::Dmer),
               cudaMemcpyDeviceToHost);

    for (u32 i = 1; i < num_dmers; ++i) {
      if (h_sorted[i].hash < h_sorted[i - 1].hash) {
        printf("    Sort order violated at index %u\n", i);
        passed = false;
        break;
      }
    }
  }

  cudaFree(d_dmers);

  record_test("Parallel Sort Dmers Basic", passed);
  return passed;
}

// ==============================================================================
// Test: Parallel Sort Dmers - Empty
// ==============================================================================

bool test_parallel_sort_dmers_empty() {
  printf("=== Test: Parallel Sort Dmers Empty ===\n");

  Status status = utils::parallel_sort_dmers(nullptr, 0, 0);

  bool passed = (status == Status::SUCCESS);
  record_test("Parallel Sort Dmers Empty", passed);
  return passed;
}

// ==============================================================================
// Test: Parallel Sort Dmers - Large
// ==============================================================================

bool test_parallel_sort_dmers_large() {
  printf("=== Test: Parallel Sort Dmers Large ===\n");

  const u32 num_dmers = 100000; // 100K dmers

  dictionary::Dmer *d_dmers = nullptr;
  cudaMalloc(&d_dmers, num_dmers * sizeof(dictionary::Dmer));

  // Create random dmers
  std::vector<dictionary::Dmer> h_dmers(num_dmers);
  std::mt19937 rng(12345);
  for (u32 i = 0; i < num_dmers; ++i) {
    h_dmers[i].hash = ((u64)rng() << 32) | rng();
    h_dmers[i].position = i;
  }

  cudaMemcpy(d_dmers, h_dmers.data(), num_dmers * sizeof(dictionary::Dmer),
             cudaMemcpyHostToDevice);

  Status status = utils::parallel_sort_dmers(d_dmers, num_dmers, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Spot check sorted order
    std::vector<dictionary::Dmer> h_sorted(num_dmers);
    cudaMemcpy(h_sorted.data(), d_dmers, num_dmers * sizeof(dictionary::Dmer),
               cudaMemcpyDeviceToHost);

    // Check first 1000 and last 1000
    for (u32 i = 1; i < 1000; ++i) {
      if (h_sorted[i].hash < h_sorted[i - 1].hash) {
        passed = false;
        break;
      }
    }
    for (u32 i = num_dmers - 999; i < num_dmers; ++i) {
      if (h_sorted[i].hash < h_sorted[i - 1].hash) {
        passed = false;
        break;
      }
    }
  }

  cudaFree(d_dmers);

  record_test("Parallel Sort Dmers Large (100K)", passed);
  return passed;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  cudaFree(0); // Initialize CUDA

  printf("========================================\n");
  printf("Utility Functions Test Suite\n");
  printf("========================================\n\n");

  test_parallel_scan_basic();
  test_parallel_scan_empty();
  test_parallel_scan_large();
  test_parallel_sort_dmers_basic();
  test_parallel_sort_dmers_empty();
  test_parallel_sort_dmers_large();

  printf("\n========================================\n");
  printf("Summary\n");
  printf("========================================\n");

  int passed = 0, failed = 0;
  for (const auto &r : g_results) {
    if (r.passed)
      passed++;
    else
      failed++;
  }

  printf("Passed: %d, Failed: %d\n", passed, failed);

  if (failed == 0) {
    printf("\n✅ ALL UTILITY FUNCTION TESTS PASSED\n");
    return 0;
  } else {
    printf("\n❌ Some tests failed\n");
    return 1;
  }
}
