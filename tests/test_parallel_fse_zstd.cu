// test_parallel_fse_zstd.cu - Unit and Integration Tests for Parallel FSE
// Encoder

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

void fill_random(std::vector<byte_t> &buffer, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)dist(rng);
  }
}

void fill_compressible(std::vector<byte_t> &buffer) {
  // Skewed distribution - more compressible
  std::mt19937 rng(123);
  std::discrete_distribution<int> dist(
      {10, 20, 30, 20, 10, 5, 3, 2}); // Favor low values
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)(dist(rng) * 32); // 0, 32, 64, 96, ...
  }
}

bool verify_roundtrip(const byte_t *d_input, u32 input_size,
                      const byte_t *d_output, u32 output_size,
                      cudaStream_t stream) {
  // Decode the encoded output
  byte_t *d_decoded = nullptr;
  CUDA_CHECK(cudaMalloc(&d_decoded, input_size));

  u32 decoded_size = 0;
  Status status =
      decode_fse(d_output, output_size, d_decoded, &decoded_size, stream);

  if (status != Status::SUCCESS) {
    printf("  ❌ Decode failed: %d\n", (int)status);
    cudaFree(d_decoded);
    return false;
  }

  if (decoded_size != input_size) {
    printf("  ❌ Size mismatch: expected %u, got %u\n", input_size,
           decoded_size);
    cudaFree(d_decoded);
    return false;
  }

  // Compare bytes
  std::vector<byte_t> h_input(input_size);
  std::vector<byte_t> h_decoded(input_size);
  CUDA_CHECK(
      cudaMemcpy(h_input.data(), d_input, input_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_decoded.data(), d_decoded, input_size,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_decoded);

  for (u32 i = 0; i < input_size; i++) {
    if (h_input[i] != h_decoded[i]) {
      printf("  ❌ Content mismatch at index %u: expected 0x%02x, got 0x%02x\n",
             i, h_input[i], h_decoded[i]);
      return false;
    }
  }

  return true;
}

// =============================================================================
// UNIT TESTS
// =============================================================================

bool test_parallel_single_chunk() {
  printf("=== Test: Parallel Single Chunk ===\n");

  const u32 data_size = 64 * 1024; // 64KB - single chunk
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input);

  byte_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  u32 output_size = 0;
  Status status =
      encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);

  if (status != Status::SUCCESS) {
    printf("  ❌ Encode failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  bool success = verify_roundtrip(d_input, data_size, d_output, output_size, 0);

  cudaFree(d_input);
  cudaFree(d_output);

  if (success) {
    printf("  ✅ Single chunk test passed!\n\n");
  }
  return success;
}

bool test_parallel_two_chunks() {
  printf("=== Test: Parallel Two Chunks (128KB) ===\n");

  const u32 data_size = 128 * 1024; // 128KB - 2 chunks at 64KB each
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input, 99);

  byte_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  u32 output_size = 0;
  Status status =
      encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);

  if (status != Status::SUCCESS) {
    printf("  ❌ Encode failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  bool success = verify_roundtrip(d_input, data_size, d_output, output_size, 0);

  cudaFree(d_input);
  cudaFree(d_output);

  if (success) {
    printf("  ✅ Two chunk test passed!\n\n");
  }
  return success;
}

bool test_parallel_many_chunks() {
  printf("=== Test: Parallel Many Chunks (1MB = 16x64KB) ===\n");

  const u32 data_size = 1 * 1024 * 1024; // 1MB
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input, 777);

  byte_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  u32 output_size = 0;
  Status status =
      encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);

  if (status != Status::SUCCESS) {
    printf("  ❌ Encode failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  bool success = verify_roundtrip(d_input, data_size, d_output, output_size, 0);

  cudaFree(d_input);
  cudaFree(d_output);

  if (success) {
    printf("  ✅ Many chunks test passed!\n\n");
  }
  return success;
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

bool test_integration_10MB_roundtrip() {
  printf("=== Integration Test: 10MB Roundtrip ===\n");

  const u32 data_size = 10 * 1024 * 1024;
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input, 12345);

  byte_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  auto start = std::chrono::high_resolution_clock::now();

  u32 output_size = 0;
  Status status =
      encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);

  auto end = std::chrono::high_resolution_clock::now();
  double encode_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  if (status != Status::SUCCESS) {
    printf("  ❌ Encode failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  printf("  Encoded %u bytes -> %u bytes (%.2f:1) in %.2f ms\n", data_size,
         output_size, (float)data_size / output_size, encode_ms);

  bool success = verify_roundtrip(d_input, data_size, d_output, output_size, 0);

  cudaFree(d_input);
  cudaFree(d_output);

  if (success) {
    printf("  ✅ 10MB roundtrip passed!\n\n");
  }
  return success;
}

bool test_compressible_data() {
  printf("=== Test: Compressible Data (1MB) ===\n");

  const u32 data_size = 1 * 1024 * 1024;
  std::vector<byte_t> h_input(data_size);
  fill_compressible(h_input);

  byte_t *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  u32 output_size = 0;
  Status status =
      encode_fse_advanced(d_input, data_size, d_output, &output_size, true, 0);

  if (status != Status::SUCCESS) {
    printf("  ❌ Encode failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  float ratio = (float)data_size / output_size;
  printf("  Compression ratio: %.2f:1\n", ratio);

  bool success = verify_roundtrip(d_input, data_size, d_output, output_size, 0);

  cudaFree(d_input);
  cudaFree(d_output);

  if (success) {
    printf("  ✅ Compressible data test passed!\n\n");
  }
  return success;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
  printf("\n========================================\n");
  printf("  Parallel FSE Encoder Test Suite\n");
  printf("========================================\n\n");

  int passed = 0;
  int failed = 0;

  // Unit Tests - Sequential Path (≤256KB)
  if (test_parallel_single_chunk())
    passed++;
  else
    failed++;
  if (test_parallel_two_chunks())
    passed++;
  else
    failed++;

  // Unit Tests - Parallel Path (>256KB)
  if (test_parallel_many_chunks())
    passed++;
  else
    failed++;

  // Integration Tests
  if (test_integration_10MB_roundtrip())
    passed++;
  else
    failed++;
  if (test_compressible_data())
    passed++;
  else
    failed++;

  printf("========================================\n");
  printf("  Results: %d passed, %d failed\n", passed, failed);
  printf("========================================\n");

  return failed > 0 ? 1 : 0;
}
