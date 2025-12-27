// ============================================================================
// test_streaming_integration.cu - Integration Tests for Streaming API
// ============================================================================

#include "cuda_zstd_manager.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cuda_zstd;

#define LOG_TEST(name) std::cout << "\n[INTEGRATION TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
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
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }

// --- Helpers ---
float calc_ratio(size_t input, size_t compressed) {
  if (compressed == 0)
    return 0.0f;
  return (float)input / compressed;
}

void generate_test_data(std::vector<uint8_t> &data, size_t size,
                        const char *pattern) {
  data.resize(size);
  if (strcmp(pattern, "repetitive") == 0) {
    for (size_t i = 0; i < size; i++)
      data[i] = (i / 16) % 4;
  } else if (strcmp(pattern, "text") == 0) {
    const char *text = "The quick brown fox jumps over the lazy dog. ";
    size_t text_len = strlen(text);
    for (size_t i = 0; i < size; i++)
      data[i] = text[i % text_len];
  } else {
    for (size_t i = 0; i < size; i++)
      data[i] = (i & 0xFF);
  }
}

bool verify_data(const uint8_t *orig, const uint8_t *decomp, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (orig[i] != decomp[i]) {
      std::cerr << "Mismatch at " << i << "\n";
      return false;
    }
  }
  return true;
}

// --- Tests ---

bool test_multi_chunk_streaming() {
  LOG_TEST("Multi-Chunk Streaming (5 chunks)");

  const size_t chunk_size = 16 * 1024; // 16KB per chunk
  const int num_chunks = 5;
  const size_t total_size = chunk_size * num_chunks;

  // Generate test data
  std::vector<uint8_t> h_input(total_size);
  generate_test_data(h_input, total_size, "text");

  // Allocate GPU memory
  void *d_input, *d_compressed, *d_output;
  cudaMalloc(&d_input, chunk_size);
  cudaMalloc(&d_compressed, chunk_size * 2);
  cudaMalloc(&d_output, chunk_size);

  ZstdStreamingManager manager(CompressionConfig{.level = 5});
  manager.init_compression();

  // Compress chunks
  std::vector<std::vector<uint8_t>> compressed_chunks;
  size_t total_compressed = 0;

  for (int i = 0; i < num_chunks; i++) {
    size_t offset = i * chunk_size;
    cudaMemcpy(d_input, h_input.data() + offset, chunk_size,
               cudaMemcpyHostToDevice);

    size_t compressed_size;
    bool is_last = (i == num_chunks - 1);
    Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                           &compressed_size, is_last);
    ASSERT_STATUS(status, "compress_chunk failed");

    std::vector<uint8_t> chunk_data(compressed_size);
    cudaMemcpy(chunk_data.data(), d_compressed, compressed_size,
               cudaMemcpyDeviceToHost);
    compressed_chunks.push_back(chunk_data);
    total_compressed += compressed_size;
  }

  LOG_INFO("Total compressed: "
           << total_compressed << " bytes, Ratio: "
           << get_compression_ratio(total_size, total_compressed) << ":1");

  // Decompress chunks
  manager.init_decompression();
  std::vector<uint8_t> h_output(total_size);

  for (int i = 0; i < num_chunks; i++) {
    cudaMemcpy(d_compressed, compressed_chunks[i].data(),
               compressed_chunks[i].size(), cudaMemcpyHostToDevice);

    size_t decompressed_size = chunk_size; // Expectation
    bool is_last;
    Status status =
        manager.decompress_chunk(d_compressed, compressed_chunks[i].size(),
                                 d_output, &decompressed_size, &is_last);
    ASSERT_STATUS(status, "decompress_chunk failed");
    ASSERT_EQ(decompressed_size, chunk_size, "Chunk size mismatch");

    cudaMemcpy(h_output.data() + (i * chunk_size), d_output, chunk_size,
               cudaMemcpyDeviceToHost);
  }

  ASSERT_TRUE(verify_data(h_input.data(), h_output.data(), total_size),
              "Verification failed");

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_output);

  LOG_PASS("Multi-Chunk Streaming");
  return true;
}

bool test_large_streaming() {
  LOG_TEST("Large File Streaming (10MB)");

  const size_t chunk_size = 128 * 1024; // 128KB
  const int num_chunks = 80;            // ~10MB
  const size_t total_size = chunk_size * num_chunks;

  void *d_input, *d_compressed;
  cudaMalloc(&d_input, chunk_size);
  cudaMalloc(&d_compressed, chunk_size * 2);

  std::vector<uint8_t> h_input(chunk_size);
  generate_test_data(h_input, chunk_size, "repetitive");

  ZstdStreamingManager manager(CompressionConfig{.level = 2});
  manager.init_compression();

  size_t total_compressed = 0;

  for (int i = 0; i < num_chunks; i++) {
    cudaMemcpy(d_input, h_input.data(), chunk_size, cudaMemcpyHostToDevice);

    size_t compressed_size;
    Status status =
        manager.compress_chunk(d_input, chunk_size, d_compressed,
                               &compressed_size, i == num_chunks - 1);
    ASSERT_STATUS(status, "Large chunk compression failed");
    total_compressed += compressed_size;
  }

  LOG_INFO("Total compressed: " << total_compressed << " bytes");

  cudaFree(d_input);
  cudaFree(d_compressed);
  LOG_PASS("Large File Streaming");
  return true;
}

int main() {
  int passed = 0;
  int total = 0;

  total++;
  if (test_multi_chunk_streaming())
    passed++;
  total++;
  if (test_large_streaming())
    passed++;

  std::cout << "\nIntegration Tests: " << passed << "/" << total
            << " passed.\n";
  return (passed == total) ? 0 : 1;
}
