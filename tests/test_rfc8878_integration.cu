#include "cuda_zstd_manager.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

extern "C" {
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
}

using namespace cuda_zstd;

#define INTEGRATION_CHECK(cond)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__);     \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_INT(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

bool test_gpu_compress_cpu_decompress(size_t input_size, int level) {
  printf(
      "[INT] Testing GPU Compress -> CPU Decompress (Size: %zu, Level: %d)\n",
      input_size, level);

  std::vector<uint8_t> h_input(input_size);
  for (size_t i = 0; i < input_size; i++)
    h_input[i] = (uint8_t)(i % 256);


  uint8_t *d_input, *d_output;
  CUDA_CHECK_INT(cudaMalloc(&d_input, input_size));
  CUDA_CHECK_INT(cudaMalloc(&d_output, input_size * 2));
  CUDA_CHECK_INT(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  CompressionConfig config = CompressionConfig::from_level(level);
  ZstdBatchManager manager;
  Status status = manager.configure(config);
  if (status != Status::SUCCESS) {
    printf("Manager configuration failed with status %d\n", (int)status);
    return false;
  }

  size_t temp_size = manager.get_compress_temp_size(input_size);
  void *d_temp;
  CUDA_CHECK_INT(cudaMalloc(&d_temp, temp_size));

  size_t compressed_size = input_size * 2;
  status = manager.compress(d_input, input_size, d_output, &compressed_size,
                            d_temp, temp_size, nullptr, 0, 0);
  cudaFree(d_temp);
  INTEGRATION_CHECK(status == Status::SUCCESS);

  std::vector<uint8_t> h_compressed(compressed_size);
  CUDA_CHECK_INT(cudaMemcpy(h_compressed.data(), d_output, compressed_size,
                            cudaMemcpyDeviceToHost));

  // Dump compressed data for debugging
  {
    FILE *f = fopen("debug_gpu_compressed.zst", "wb");
    if (f) {
      fwrite(h_compressed.data(), 1, compressed_size, f);
      fclose(f);
    }
    printf("  [DBG] GPU compressed %zu -> %zu bytes. First 64 bytes:\n  ", input_size, compressed_size);
    for (size_t i = 0; i < std::min(compressed_size, (size_t)64); i++) {
      printf("%02X ", h_compressed[i]);
      if ((i+1) % 16 == 0) printf("\n  ");
    }
    printf("\n");
  }

  // Decompress with libzstd (CPU)
  std::vector<uint8_t> h_decompressed(input_size);
  size_t d_size = ZSTD_decompress(h_decompressed.data(), input_size,
                                  h_compressed.data(), compressed_size);

  if (ZSTD_isError(d_size)) {
    printf("  [FAIL] CPU Decompression failed: %s\n",
           ZSTD_getErrorName(d_size));
    return false;
  }
  INTEGRATION_CHECK(d_size == input_size);
  INTEGRATION_CHECK(memcmp(h_input.data(), h_decompressed.data(), input_size) ==
                    0);

  // Write compressed data to file for debugging
  {
    FILE *f = fopen("debug_1mb.zst", "wb");
    if (f) {
      fwrite(h_compressed.data(), 1, compressed_size, f);
      fclose(f);
    }
  }

  printf("  [PASS] GPU Compress -> CPU Decompress successful\n");

  cudaFree(d_input);
  cudaFree(d_output);
  return true;
}

bool test_cpu_compress_gpu_decompress(size_t input_size, int level) {
  printf(
      "[INT] Testing CPU Compress -> GPU Decompress (Size: %zu, Level: %d)\n",
      input_size, level);

  std::vector<uint8_t> h_input(input_size);
  for (size_t i = 0; i < input_size; i++)
    h_input[i] = (uint8_t)(i % 256);


  size_t max_comp = ZSTD_compressBound(input_size);
  std::vector<uint8_t> h_compressed(max_comp);
  size_t compressed_size = ZSTD_compress(h_compressed.data(), max_comp,
                                         h_input.data(), input_size, level);
  INTEGRATION_CHECK(!ZSTD_isError(compressed_size));

  // Verify on CPU first
  std::vector<uint8_t> h_verify(input_size);
  size_t d_verify_size = ZSTD_decompress(h_verify.data(), input_size,
                                         h_compressed.data(), compressed_size);
  if (ZSTD_isError(d_verify_size)) {
    printf("  [FAIL] CPU Verification of Generated Data Failed: %s\n",
           ZSTD_getErrorName(d_verify_size));
    return false;
  }
  if (d_verify_size != input_size) {
    printf("  [FAIL] CPU Verification Size Mismatch: %zu != %zu\n",
           d_verify_size, input_size);
    return false;
  }
  if (memcmp(h_verify.data(), h_input.data(), input_size) != 0) {
    printf("  [FAIL] CPU Verification Content Mismatch\n");
    return false;
  }
  printf("  [INFO] CPU Generated Data Verified Valid (Size: %zu)\n",
         compressed_size);

  uint8_t *d_input, *d_output;
  CUDA_CHECK_INT(cudaMalloc(&d_input, compressed_size));
  CUDA_CHECK_INT(cudaMalloc(&d_output, input_size));
  // Write compressed data to file for debugging
  {
    FILE *f = fopen("debug_1024_cpu.zst", "wb");
    if (f) {
      fwrite(h_compressed.data(), 1, compressed_size, f);
      fclose(f);
    }
  }

  CUDA_CHECK_INT(cudaMemcpy(d_input, h_compressed.data(), compressed_size,
                            cudaMemcpyHostToDevice));

  ZstdBatchManager manager;
  Status status = manager.configure(CompressionConfig::from_level(level));
  if (status != Status::SUCCESS) {
    printf("Manager configuration failed with status %d\n", (int)status);
    return false;
  }

  size_t temp_size = manager.get_decompress_temp_size(compressed_size);
  void *d_temp;
  CUDA_CHECK_INT(cudaMalloc(&d_temp, temp_size));

  size_t uncompressed_size = input_size;
  status = manager.decompress(d_input, compressed_size, d_output,
                              &uncompressed_size, d_temp, temp_size, 0);
  cudaFree(d_temp);

  if (status != Status::SUCCESS) {
    printf("  [FAIL] GPU Decompression failed with status %d\n", (int)status);
    return false;
  }
  if (uncompressed_size != input_size) {
    printf("  [FAIL] Size mismatch: expected %zu, got %zu\n", input_size,
           uncompressed_size);
  }
  INTEGRATION_CHECK(uncompressed_size == input_size);

  std::vector<uint8_t> h_decompressed(input_size);
  CUDA_CHECK_INT(cudaMemcpy(h_decompressed.data(), d_output, input_size,
                            cudaMemcpyDeviceToHost));
  if (memcmp(h_input.data(), h_decompressed.data(), input_size) != 0) {
    printf("  [FAIL] Data corruption detected. First 50 mismatches:\n");
    int count = 0;
    for (size_t i = 0; i < input_size && count < 50; i++) {
      if (h_input[i] != h_decompressed[i]) {
        printf("    Byte %zu: Expected 0x%02X, Got 0x%02X\n", i, h_input[i],
               h_decompressed[i]);
        count++;
      }
    }
    for (size_t i = 256; i < input_size && i < 300; i++) {
      if (h_input[i] != h_decompressed[i]) {
        printf("    Byte %zu: Expected 0x%02X, Got 0x%02X\n", i, h_input[i],
               h_decompressed[i]);
      }
    }
    return false; // Fail gracefully
  }

  printf("  [PASS] CPU Compress -> GPU Decompress successful\n");

  cudaFree(d_input);
  cudaFree(d_output);
  return true;
}

int main() {
  printf("========================================\n");
  printf("RFC 8878 Integration Verification\n");
  printf("========================================\n\n");

  bool all_passed = true;

  // Test cases
  struct TestCase {
    size_t size;
    int level;
  };
  std::vector<TestCase> cases = {{1024, 1}};

  for (const auto &c : cases) {
    if (!test_gpu_compress_cpu_decompress(c.size, c.level))
      all_passed = false;
    if (!test_cpu_compress_gpu_decompress(c.size, c.level))
      all_passed = false;
  }

  if (all_passed) {
    printf("\n✅ ALL RFC 8878 Integration Tests PASSED\n");
  } else {
    printf("\n❌ SOME Integration Tests FAILED\n");
    return 1;
  }

  return 0;
}
