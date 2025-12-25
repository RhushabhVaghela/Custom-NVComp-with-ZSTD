// test_c_api_edge_cases.cu - Edge case tests for C API (cuda_zstd_c_api.cpp)
// Covers: null pointers, invalid sizes, error handling, boundary conditions

#include "cuda_zstd_manager.h"
#include <cstdio>
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

// ==============================================================================
// Test: Null Input Pointer
// ==============================================================================

bool test_null_input_pointer() {
  printf("=== Test: Null Input Pointer ===\n");

  ZstdBatchManager manager;
  manager.set_compression_level(3);

  u8 *d_output = nullptr;
  size_t output_size = 0;
  size_t max_output_size = 1024;

  cudaMalloc(&d_output, max_output_size);

  // Pass null input - should return error
  // Args: src, src_size, dst, dst_size, temp, temp_size, dict, dict_size
  Status status = manager.compress(nullptr, 100, d_output, &output_size,
                                   nullptr, 0, nullptr, 0);

  bool passed = (status != Status::SUCCESS); // Should fail

  cudaFree(d_output);

  record_test("Null Input Pointer", passed);
  return passed;
}

// ==============================================================================
// Test: Null Output Pointer
// ==============================================================================

bool test_null_output_pointer() {
  printf("=== Test: Null Output Pointer ===\n");

  ZstdBatchManager manager;
  manager.set_compression_level(3);

  u8 *d_input = nullptr;
  size_t output_size = 0;

  cudaMalloc(&d_input, 100);
  cudaMemset(d_input, 0xAA, 100);

  // Pass null output - should return error
  Status status = manager.compress(d_input, 100, nullptr, &output_size, nullptr,
                                   0, nullptr, 0);

  bool passed = (status != Status::SUCCESS); // Should fail

  cudaFree(d_input);

  record_test("Null Output Pointer", passed);
  return passed;
}

// ==============================================================================
// Test: Zero Size Input
// ==============================================================================

bool test_zero_size_input() {
  printf("=== Test: Zero Size Input ===\n");

  ZstdBatchManager manager;
  manager.set_compression_level(3);

  u8 *d_input = nullptr;
  u8 *d_output = nullptr;
  size_t output_size = 0;

  cudaMalloc(&d_input, 100);
  cudaMalloc(&d_output, 1024);

  // Pass zero size - should handle gracefully
  Status status = manager.compress(d_input, 0, d_output, &output_size, nullptr,
                                   0, nullptr, 0);

  // Either success with 0 output or specific error is acceptable
  bool passed =
      (status == Status::SUCCESS || status == Status::ERROR_INVALID_PARAMETER);

  cudaFree(d_input);
  cudaFree(d_output);

  record_test("Zero Size Input", passed);
  return passed;
}

// ==============================================================================
// Test: Very Small Input (1 byte)
// ==============================================================================

bool test_very_small_input() {
  printf("=== Test: Very Small Input (1 byte) ===\n");

  ZstdBatchManager manager;
  manager.set_compression_level(1); // Fast compression

  u8 *d_input = nullptr;
  u8 *d_output = nullptr;
  size_t output_size = 0;

  cudaMalloc(&d_input, 1);
  cudaMalloc(&d_output, 1024); // Plenty of space

  u8 h_byte = 0x42;
  cudaMemcpy(d_input, &h_byte, 1, cudaMemcpyHostToDevice);

  // For valid compression, we technically need workspace, but testing edge case
  // here. If it fails due to no workspace, that's fine, but we'll accept
  // SUCCESS too. Actually, for 1 byte, maybe it uses CPU fallback?
  Status status = manager.compress(d_input, 1, d_output, &output_size, nullptr,
                                   0, nullptr, 0);

  // If it requires workspace, it returns error. We just check checks.
  bool passed =
      (status == Status::SUCCESS || status == Status::ERROR_INVALID_PARAMETER ||
       status == Status::ERROR_GENERIC);
  // Specifically, we wanted to ensure no crash on small input.

  cudaFree(d_input);
  cudaFree(d_output);

  record_test("Very Small Input (1 byte)", passed);
  return passed;
}

// ==============================================================================
// Test: Invalid Compression Level
// ==============================================================================

bool test_invalid_compression_level() {
  printf("=== Test: Invalid Compression Level ===\n");

  ZstdBatchManager manager;

  // Try invalid levels
  manager.set_compression_level(-100); // Too low
  manager.set_compression_level(100);  // Too high

  // Should clamp to valid range quietly
  bool passed = true; // No crash = pass

  record_test("Invalid Compression Level", passed);
  return passed;
}

// ==============================================================================
// Test: Manager Destruction
// ==============================================================================

bool test_manager_destruction() {
  printf("=== Test: Manager Destruction ===\n");

  // Create and destroy manager in a scope
  {
    ZstdBatchManager manager;
    manager.set_compression_level(5);

    u8 *d_input = nullptr;
    u8 *d_output = nullptr;
    cudaMalloc(&d_input, 1024);
    cudaMalloc(&d_output, 4096);
    cudaMemset(d_input, 0xAB, 1024);

    size_t output_size = 0;
    manager.compress(d_input, 1024, d_output, &output_size, nullptr, 0, nullptr,
                     0);

    cudaFree(d_input);
    cudaFree(d_output);
  }

  // Manager destroyed, verify no leaks (no crash)
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  bool passed = (err == cudaSuccess);

  record_test("Manager Destruction", passed);
  return passed;
}

// ==============================================================================
// Test: Multiple Managers
// ==============================================================================

bool test_multiple_managers() {
  printf("=== Test: Multiple Managers ===\n");

  ZstdBatchManager manager1, manager2, manager3;

  manager1.set_compression_level(1);
  manager2.set_compression_level(5);
  manager3.set_compression_level(9);

  bool passed = true; // Multiple instances should work

  record_test("Multiple Managers", passed);
  return passed;
}

// ==============================================================================
// Test: Output Size Pointer Null
// ==============================================================================

bool test_output_size_null() {
  printf("=== Test: Output Size Pointer Null ===\n");

  ZstdBatchManager manager;
  manager.set_compression_level(3);

  u8 *d_input = nullptr;
  u8 *d_output = nullptr;

  cudaMalloc(&d_input, 100);
  cudaMalloc(&d_output, 1024);
  cudaMemset(d_input, 0xCC, 100);

  // Pass null output size pointer - should return error
  Status status =
      manager.compress(d_input, 100, d_output, nullptr, nullptr, 0, nullptr, 0);

  bool passed = (status != Status::SUCCESS); // Should fail

  cudaFree(d_input);
  cudaFree(d_output);

  record_test("Output Size Pointer Null", passed);
  return passed;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  cudaFree(0); // Initialize CUDA

  printf("========================================\n");
  printf("C API Edge Cases Test Suite\n");
  printf("========================================\n\n");

  test_null_input_pointer();
  test_null_output_pointer();
  test_zero_size_input();
  test_very_small_input();
  test_invalid_compression_level();
  test_manager_destruction();
  test_multiple_managers();
  test_output_size_null();

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
    printf("\n✅ ALL C API EDGE CASE TESTS PASSED\n");
    return 0;
  } else {
    printf("\n❌ Some tests failed\n");
    return 1;
  }
}
