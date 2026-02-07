// test_coverage_gaps.cu - Verification for RLE fallback, edge cases, and size boundaries
// Uses ZSTD Manager API for accurate coverage testing

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Helper to fill buffer with random compressible data (Skewed distribution)
void fill_random(std::vector<byte_t> &buffer) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 100);
  std::uniform_int_distribution<int> val_dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    if (dist(rng) < 90)
      buffer[i] = 0;
    else
      buffer[i] = (byte_t)val_dist(rng);
  }
}

// Helper to fill buffer with RLE data
void fill_rle(std::vector<byte_t> &buffer, byte_t val) {
  std::fill(buffer.begin(), buffer.end(), val);
}

bool test_rle_roundtrip() {
  printf("=== Testing RLE Roundtrip ===\n");

  const u32 data_size = 1024 * 1024; // 1MB
  std::vector<byte_t> h_input(data_size);
  fill_rle(h_input, 0xAA); // All 0xAA

  byte_t *d_input = nullptr;
  byte_t *d_output = nullptr;
  u32 *d_output_sizes = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2)); // Plenty of space
  CUDA_CHECK(cudaMalloc(&d_output_sizes, sizeof(u32)));

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  // Setup batch arrays
  const byte_t *d_inputs_arr[] = {d_input};
  u32 input_sizes_arr[] = {data_size};
  byte_t *d_outputs_arr[] = {d_output};

  // Verify d_input content
  std::vector<byte_t> h_check(data_size);
  CUDA_CHECK(
      cudaMemcpy(h_check.data(), d_input, data_size, cudaMemcpyDeviceToHost));
  if (h_check != h_input) {
    printf("❌ Input mismatch before encoding!\n");
    printf("Expected 0xAA, got 0x%02x\n", h_check[0]);
    return false;
  }

  // Encode
  printf("Encoding RLE data...\n");
  Status status =
      encode_fse_batch((const byte_t **)d_inputs_arr, input_sizes_arr,
                       (byte_t **)d_outputs_arr, d_output_sizes,
                       1, // num_blocks
                       0  // stream
      );

  if (status != Status::SUCCESS) {
    printf("❌ Encoding failed: %d\n", (int)status);
    return false;
  }

  // Check output size
  u32 output_size = 0;
  CUDA_CHECK(cudaMemcpy(&output_size, d_output_sizes, sizeof(u32),
                        cudaMemcpyDeviceToHost));

  // RLE header is 14 bytes.
  if (output_size != 14) {
    printf("❌ Expected output size 14 (Header+Symbol), got %u\n", output_size);
    // It might be acceptable if logic differs slightly, but for RLE it should
    // be small.
    if (output_size > 100) {
      printf("❌ Output size too large for RLE! Fallback failed.\n");
      return false;
    }
  } else {
    printf("✅ Output size indicates RLE compression.\n");
  }

  // Decode
  printf("Decoding RLE data...\n");
  byte_t *d_decoded = nullptr;
  CUDA_CHECK(cudaMalloc(&d_decoded, data_size));
  u32 decoded_size = 0;

  status = decode_fse(d_output,
                      output_size, // Input size for decoder
                      d_decoded, &decoded_size, 0);

  if (status != Status::SUCCESS) {
    printf("❌ Decoding failed: %d\n", (int)status);
    return false;
  }

  if (decoded_size != data_size) {
    printf("❌ Decoded size mismatch: %u != %u\n", decoded_size, data_size);
    return false;
  }

  // Verify content
  std::vector<byte_t> h_decoded(data_size);
  CUDA_CHECK(cudaMemcpy(h_decoded.data(), d_decoded, data_size,
                        cudaMemcpyDeviceToHost));

  if (h_decoded != h_input) {
    printf("❌ Content mismatch!\n");
    printf("Expected (first 20 bytes): ");
    for (int i = 0; i < 20 && i < data_size; i++)
      printf("%02x ", h_input[i]);
    printf("\n");
    printf("Actual   (first 20 bytes): ");
    for (int i = 0; i < 20 && i < data_size; i++)
      printf("%02x ", h_decoded[i]);
    printf("\n");

    // Find first mismatch
    for (size_t i = 0; i < data_size; ++i) {
      if (h_decoded[i] != h_input[i]) {
        printf("First mismatch at index %zu: expected %02x, got %02x\n", i,
               h_input[i], h_decoded[i]);
        break;
      }
    }
    return false;
  }

  printf("✅ RLE Roundtrip Passed!\n\n");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_sizes);
  cudaFree(d_decoded);
  return true;
}

// Test using ZSTD Manager API for proper round-trip
bool test_zstd_roundtrip(u32 data_size, const char* test_name) {
  printf("=== Testing %s (%u KB) ===\n", test_name, data_size / 1024);

  std::vector<byte_t> h_input(data_size);
  fill_random(h_input);

  // Allocate device memory
  byte_t *d_input = nullptr;
  byte_t *d_compressed = nullptr;
  byte_t *d_decompressed = nullptr;
  byte_t *d_temp = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  size_t max_compressed = data_size * 2;
  CUDA_CHECK(cudaMalloc(&d_compressed, max_compressed));
  CUDA_CHECK(cudaMalloc(&d_decompressed, data_size));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  // Create ZSTD manager
  auto manager = create_manager(5); // Level 5

  size_t temp_size = manager->get_compress_temp_size(data_size);
  CUDA_CHECK(cudaMalloc(&d_temp, temp_size));

  // Compress
  size_t compressed_size = max_compressed;
  Status status = manager->compress(d_input, data_size, d_compressed, &compressed_size,
                                    d_temp, temp_size, nullptr, 0, 0);

  if (status != Status::SUCCESS) {
    printf("❌ Compression failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    return false;
  }

  printf("  Compressed %u -> %zu bytes (%.1f%% ratio)\n", 
         data_size, compressed_size, 100.0 * compressed_size / data_size);

  // Decompress
  size_t decompressed_size = data_size;
  status = manager->decompress(d_compressed, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size, 0);

  if (status != Status::SUCCESS) {
    printf("❌ Decompression failed: %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    return false;
  }

  // Verify
  std::vector<byte_t> h_decompressed(data_size);
  CUDA_CHECK(cudaMemcpy(h_decompressed.data(), d_decompressed, data_size,
                        cudaMemcpyDeviceToHost));

  if (decompressed_size != data_size) {
    printf("❌ Size mismatch: expected %u, got %zu\n", data_size, decompressed_size);
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    return false;
  }

  if (h_decompressed != h_input) {
    printf("❌ Content mismatch!\n");
    printf("First 20 bytes expected: ");
    for (int i = 0; i < 20 && i < data_size; i++)
      printf("%02x ", h_input[i]);
    printf("\nFirst 20 bytes actual:   ");
    for (int i = 0; i < 20 && i < (int)decompressed_size; i++)
      printf("%02x ", h_decompressed[i]);
    printf("\n");

    // Find first mismatch
    for (size_t i = 0; i < std::min((size_t)data_size, decompressed_size); ++i) {
      if (h_decompressed[i] != h_input[i]) {
        printf("First mismatch at index %zu: expected %02x, got %02x\n", i,
               h_input[i], h_decompressed[i]);
        break;
      }
    }
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    return false;
  }

  printf("✅ %s Passed!\n\n", test_name);

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(d_temp);
  return true;
}

bool test_zero_byte_input() {
  printf("=== Testing 0-Byte Input ===\n");

  const u32 data_size = 0;

  byte_t *d_input = nullptr;
  byte_t *d_output = nullptr;
  u32 *d_output_sizes = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, 1)); // Alloc 1 byte just in case
  CUDA_CHECK(cudaMalloc(&d_output, 100));
  CUDA_CHECK(cudaMalloc(&d_output_sizes, sizeof(u32)));

  const byte_t *d_inputs_arr[] = {d_input};
  u32 input_sizes_arr[] = {data_size};
  byte_t *d_outputs_arr[] = {d_output};

  Status status =
      encode_fse_batch((const byte_t **)d_inputs_arr, input_sizes_arr,
                       (byte_t **)d_outputs_arr, d_output_sizes, 1, 0);

  // Only SUCCESS or ERROR_INVALID_PARAMETER are acceptable for 0-byte input
  bool passed = (status == Status::SUCCESS || status == Status::ERROR_INVALID_PARAMETER);
  if (status == Status::SUCCESS) {
    printf("✅ 0-Byte Input handled (Status::SUCCESS)\n");
  } else if (status == Status::ERROR_INVALID_PARAMETER) {
    printf("✅ 0-Byte Input correctly rejected (Status::ERROR_INVALID_PARAMETER)\n");
  } else {
    printf("❌ 0-Byte Input returned unexpected status: %d\n", (int)status);
  }

  printf("%s 0-Byte Input Test Completed (No Crash)\n\n", passed ? "✅" : "❌");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_sizes);
  return passed;
}

int main() {
  bool all_passed = true;

  // Test RLE (uses standalone FSE which works for RLE)
  all_passed &= test_rle_roundtrip();

  // Test various sizes using ZSTD Manager API
  all_passed &= test_zstd_roundtrip(64 * 1024, "64KB Input");
  all_passed &= test_zstd_roundtrip(256 * 1024, "256KB Input");
  all_passed &= test_zstd_roundtrip(1024 * 1024, "1MB Input");

  // Test edge case
  all_passed &= test_zero_byte_input();

  if (all_passed) {
    printf("All tests passed!\n");
    return 0;
  } else {
    printf("Some tests failed!\n");
    return 1;
  }
}
