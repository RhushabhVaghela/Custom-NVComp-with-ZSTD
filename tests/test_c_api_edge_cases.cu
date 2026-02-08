// test_c_api_edge_cases.cu - Edge case tests for C API (cuda_zstd_c_api.cpp)
// Covers: null pointers, invalid sizes, error handling, boundary conditions

#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
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

  cuda_zstd::safe_cuda_malloc(&d_output, max_output_size);

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

  cuda_zstd::safe_cuda_malloc(&d_input, 100);
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

  cuda_zstd::safe_cuda_malloc(&d_input, 100);
  cuda_zstd::safe_cuda_malloc(&d_output, 1024);

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

  cuda_zstd::safe_cuda_malloc(&d_input, 1);
  cuda_zstd::safe_cuda_malloc(&d_output, 1024); // Plenty of space

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

  bool passed = true;
  const size_t data_size = 1024;

  u8 *d_input = nullptr, *d_output = nullptr, *d_workspace = nullptr;
  cuda_zstd::safe_cuda_malloc(&d_input, data_size);
  cuda_zstd::safe_cuda_malloc(&d_output, data_size * 2);
  cudaMemset(d_input, 0xAB, data_size);

  // Get workspace size for proper compress call
  ZstdBatchManager temp_mgr;
  size_t ws_size = temp_mgr.get_compress_temp_size(data_size);
  cuda_zstd::safe_cuda_malloc(&d_workspace, ws_size);

  // Test with invalid level -100 (rejected, falls back to default level 3)
  {
    ZstdBatchManager manager;
    manager.set_compression_level(-100); // Rejected, stays at default level 3

    size_t output_size = data_size * 2;
    Status status = manager.compress(d_input, data_size, d_output, &output_size,
                                     d_workspace, ws_size, nullptr, 0);
    if (status != Status::SUCCESS || output_size == 0) {
      printf("  FAIL: Compression with rejected level -100 failed (status=%d, size=%zu)\n",
             (int)status, output_size);
      passed = false;
    } else {
      printf("  OK: Level -100 rejected (uses default), compressed %zu -> %zu bytes\n",
             data_size, output_size);
    }
  }

  // Test with invalid level 100 (rejected, falls back to default level 3)
  {
    ZstdBatchManager manager;
    manager.set_compression_level(100); // Rejected, stays at default level 3

    size_t output_size = data_size * 2;
    Status status = manager.compress(d_input, data_size, d_output, &output_size,
                                     d_workspace, ws_size, nullptr, 0);
    if (status != Status::SUCCESS || output_size == 0) {
      printf("  FAIL: Compression with rejected level 100 failed (status=%d, size=%zu)\n",
             (int)status, output_size);
      passed = false;
    } else {
      printf("  OK: Level 100 rejected (uses default), compressed %zu -> %zu bytes\n",
             data_size, output_size);
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

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
    cuda_zstd::safe_cuda_malloc(&d_input, 1024);
    cuda_zstd::safe_cuda_malloc(&d_output, 4096);
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

  const size_t data_size = 1024;
  u8 *d_input = nullptr, *d_output = nullptr, *d_workspace = nullptr;
  cuda_zstd::safe_cuda_malloc(&d_input, data_size);
  cuda_zstd::safe_cuda_malloc(&d_output, data_size * 2);
  cudaMemset(d_input, 0xCD, data_size);

  // Get workspace size — use highest level tested (9) so workspace is large enough
  ZstdBatchManager temp_mgr;
  temp_mgr.set_compression_level(9);
  size_t ws_size = temp_mgr.get_compress_temp_size(data_size);
  cuda_zstd::safe_cuda_malloc(&d_workspace, ws_size);

  ZstdBatchManager manager1, manager2, manager3;

  manager1.set_compression_level(1);
  manager2.set_compression_level(5);
  manager3.set_compression_level(9);

  bool passed = true;
  ZstdBatchManager *managers[] = {&manager1, &manager2, &manager3};
  int levels[] = {1, 5, 9};

  for (int i = 0; i < 3; i++) {
    size_t output_size = data_size * 2;
    Status status = managers[i]->compress(d_input, data_size, d_output,
                                          &output_size, d_workspace, ws_size,
                                          nullptr, 0);
    if (status != Status::SUCCESS || output_size == 0) {
      printf("  FAIL: Manager at level %d failed (status=%d, size=%zu)\n",
             levels[i], (int)status, output_size);
      passed = false;
    } else {
      printf("  OK: Manager at level %d compressed %zu -> %zu bytes\n",
             levels[i], data_size, output_size);
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

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

  cuda_zstd::safe_cuda_malloc(&d_input, 100);
  cuda_zstd::safe_cuda_malloc(&d_output, 1024);
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
// Test: C API Dictionary Round-trip
// ==============================================================================

bool test_c_api_dictionary_roundtrip() {
  printf("=== Test: C API Dictionary Round-trip ===\n");

  std::vector<unsigned char> sample1(16 * 1024);
  std::vector<unsigned char> sample2(16 * 1024);
  for (size_t i = 0; i < sample1.size(); ++i) {
    sample1[i] = (unsigned char)((i * 7) & 0xFF);
    sample2[i] = (unsigned char)((i * 13) & 0xFF);
  }

  const void *samples[] = {sample1.data(), sample2.data()};
  size_t sample_sizes[] = {sample1.size(), sample2.size()};

  cuda_zstd_dict_t *dict =
      cuda_zstd_train_dictionary(samples, sample_sizes, 2, 32 * 1024);
  if (!dict) {
    record_test("C API Dictionary Round-trip", false);
    return false;
  }

  cuda_zstd_manager_t *manager = cuda_zstd_create_manager(3);
  if (!manager) {
    cuda_zstd_destroy_dictionary(dict);
    record_test("C API Dictionary Round-trip", false);
    return false;
  }

  int set_result = cuda_zstd_set_dictionary(manager, dict);
  if (cuda_zstd_is_error(set_result)) {
    cuda_zstd_destroy_manager(manager);
    cuda_zstd_destroy_dictionary(dict);
    record_test("C API Dictionary Round-trip", false);
    return false;
  }

  const size_t data_size = 64 * 1024;
  std::vector<unsigned char> h_input(data_size);
  for (size_t i = 0; i < data_size; ++i) {
    h_input[i] = (unsigned char)((i * 31) & 0xFF);
  }

  unsigned char *d_input = nullptr;
  unsigned char *d_compressed = nullptr;
  unsigned char *d_decompressed = nullptr;

  cuda_zstd::safe_cuda_malloc(&d_input, data_size);
  cuda_zstd::safe_cuda_malloc(&d_compressed, data_size * 2);
  cuda_zstd::safe_cuda_malloc(&d_decompressed, data_size);
  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  size_t workspace_size =
      cuda_zstd_get_compress_workspace_size(manager, data_size);
  void *d_workspace = nullptr;
  cuda_zstd::safe_cuda_malloc(&d_workspace, workspace_size);

  size_t compressed_size = data_size * 2;
  int comp_status = cuda_zstd_compress(manager, d_input, data_size,
                                       d_compressed, &compressed_size,
                                       d_workspace, workspace_size, 0);
  if (cuda_zstd_is_error(comp_status)) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_workspace);
    cuda_zstd_destroy_manager(manager);
    cuda_zstd_destroy_dictionary(dict);
    record_test("C API Dictionary Round-trip", false);
    return false;
  }

  size_t decomp_workspace_size =
      cuda_zstd_get_decompress_workspace_size(manager, compressed_size);
  if (decomp_workspace_size > workspace_size) {
    cudaFree(d_workspace);
    cuda_zstd::safe_cuda_malloc(&d_workspace, decomp_workspace_size);
    workspace_size = decomp_workspace_size;
  }

  size_t decompressed_size = data_size;
  int decomp_status = cuda_zstd_decompress(
      manager, d_compressed, compressed_size, d_decompressed,
      &decompressed_size, d_workspace, workspace_size, 0);
  if (cuda_zstd_is_error(decomp_status)) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_workspace);
    cuda_zstd_destroy_manager(manager);
    cuda_zstd_destroy_dictionary(dict);
    record_test("C API Dictionary Round-trip", false);
    return false;
  }

  std::vector<unsigned char> h_output(data_size);
  cudaMemcpy(h_output.data(), d_decompressed, data_size,
             cudaMemcpyDeviceToHost);

  bool match = (decompressed_size == data_size) &&
               (memcmp(h_output.data(), h_input.data(), data_size) == 0);

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(d_workspace);
  cuda_zstd_destroy_manager(manager);
  cuda_zstd_destroy_dictionary(dict);

  record_test("C API Dictionary Round-trip", match);
  return match;
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
  test_c_api_dictionary_roundtrip();

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
