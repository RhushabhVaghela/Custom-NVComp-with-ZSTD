// ============================================================================
// test_streaming_unit.cu - Unit Tests for Streaming API
// ============================================================================

#include "cuda_zstd_manager.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


using namespace cuda_zstd;

#define LOG_TEST(name) std::cout << "\n[UNIT TEST] " << name << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg)                                                    \
  std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_EQ(a, b, msg)                                                   \
  if ((a) != (b)) {                                                            \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_STATUS(status, msg)                                             \
  if ((status) != Status::SUCCESS) {                                           \
    LOG_FAIL(__func__, msg << " Status: " << (int)status);                     \
    return false;                                                              \
  }

bool test_api_validation() {
  LOG_TEST("API Parameter Validation");

  ZstdStreamingManager manager;
  Status status;
  size_t compressed_size;

  // Test 1: Null input buffer
  status =
      manager.compress_chunk(nullptr, 100, nullptr, &compressed_size, true);
  if (status == Status::SUCCESS) {
    LOG_FAIL("test_api_validation", "Null input didn't return error");
    return false;
  }

  // Test 2: Null output buffer
  void *d_dummy;
  cudaMalloc(&d_dummy, 100);
  status =
      manager.compress_chunk(d_dummy, 100, nullptr, &compressed_size, true);
  if (status == Status::SUCCESS) {
    LOG_FAIL("test_api_validation", "Null output didn't return error");
    cudaFree(d_dummy);
    return false;
  }

  // Test 3: Null size pointer
  status = manager.compress_chunk(d_dummy, 100, d_dummy, nullptr, true);
  if (status == Status::SUCCESS) {
    LOG_FAIL("test_api_validation", "Null size pointer didn't return error");
    cudaFree(d_dummy);
    return false;
  }

  cudaFree(d_dummy);
  LOG_PASS("API Parameter Validation");
  return true;
}

bool test_initialization_state() {
  LOG_TEST("Initialization State Verification");

  ZstdStreamingManager manager;

  // Should be able to re-init
  Status status = manager.init_compression();
  ASSERT_STATUS(status, "First init_compression failed");

  status = manager.init_compression();
  ASSERT_STATUS(status, "Second init_compression (reset) failed");

  // Switch to decompression
  status = manager.init_decompression();
  ASSERT_STATUS(status, "Switch to init_decompression failed");

  LOG_PASS("Initialization State Verification");
  return true;
}

bool test_basic_single_chunk() {
  LOG_TEST("Basic Single Chunk Roundtrip");

  const size_t size = 1024;
  std::vector<uint8_t> h_input(size);
  for (size_t i = 0; i < size; i++)
    h_input[i] = (uint8_t)i;

  void *d_input, *d_output, *d_compressed;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);
  cudaMalloc(&d_compressed, size * 2);

  cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

  ZstdStreamingManager manager(CompressionConfig{.level = 1});
  manager.init_compression();

  size_t compressed_size;
  Status status = manager.compress_chunk(d_input, size, d_compressed,
                                         &compressed_size, true);
  ASSERT_STATUS(status, "Compression failed");

  manager.init_decompression();
  size_t decompressed_size = size; // Initialize with capacity
  bool is_last;
  status = manager.decompress_chunk(d_compressed, compressed_size, d_output,
                                    &decompressed_size, &is_last);
  ASSERT_STATUS(status, "Decompression failed");
  ASSERT_EQ(decompressed_size, size, "Decompressed size mismatch");

  std::vector<uint8_t> h_output(size);
  cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < size; i++) {
    if (h_input[i] != h_output[i]) {
      LOG_FAIL("test_basic_single_chunk", "Data mismatch");
      return false;
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_compressed);

  LOG_PASS("Basic Single Chunk Roundtrip");
  return true;
}

int main() {
  int passed = 0;
  int total = 0;

  total++;
  if (test_api_validation())
    passed++;
  total++;
  if (test_initialization_state())
    passed++;
  total++;
  if (test_basic_single_chunk())
    passed++;

  std::cout << "\nUnit Tests: " << passed << "/" << total << " passed.\n";
  return (passed == total) ? 0 : 1;
}
