/**
 * @brief Simple test for the new Huffman implementation
 */

#include "cuda_zstd_huffman.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#undef CUDA_CHECK
#define TEST_CUDA_CHECK(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace cuda_zstd;

// Note: This test exercises the production encode_huffman/decode_huffman path.
// The original _simple variants were superseded and removed from the build.

bool test_simple_roundtrip(const char *input_str) {
  size_t input_size = strlen(input_str);
  printf("Testing: \"%s\" (size=%zu)\n", input_str, input_size);

  byte_t *d_input, *d_compressed, *d_decompressed;
  TEST_CUDA_CHECK(cudaMalloc(&d_input, input_size));
  size_t max_compressed = input_size * 2 + 1024;
  TEST_CUDA_CHECK(cudaMalloc(&d_compressed, max_compressed));
  TEST_CUDA_CHECK(cudaMalloc(&d_decompressed, input_size));

  TEST_CUDA_CHECK(
      cudaMemcpy(d_input, input_str, input_size, cudaMemcpyHostToDevice));

  huffman::HuffmanTable table;
  TEST_CUDA_CHECK(cudaMalloc(&table.codes, huffman::MAX_HUFFMAN_SYMBOLS *
                                               sizeof(huffman::HuffmanCode)));

  cuda_zstd::CompressionWorkspace workspace;
  TEST_CUDA_CHECK(cudaMalloc(&workspace.d_frequencies,
                             huffman::MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
  TEST_CUDA_CHECK(
      cudaMalloc(&workspace.d_code_lengths, input_size * sizeof(u32)));
  TEST_CUDA_CHECK(
      cudaMalloc(&workspace.d_bit_offsets, input_size * sizeof(u32)));

  size_t compressed_size = 0;
  Status status =
      huffman::encode_huffman(d_input, (u32)input_size, table, d_compressed,
                              &compressed_size, &workspace, 0);

  if (status != Status::SUCCESS) {
    printf("  Encode failed: %d\n", (int)status);
    return false;
  }
  TEST_CUDA_CHECK(cudaStreamSynchronize(0));
  printf("  Encoded: %zu bytes\n", compressed_size);

  size_t decompressed_size_out = 0;
  status = huffman::decode_huffman(d_compressed, compressed_size, table,
                                   d_decompressed, &decompressed_size_out,
                                   (u32)input_size, 0);

  if (status != Status::SUCCESS) {
    printf("  Decode failed: %d\n", (int)status);
    return false;
  }
  TEST_CUDA_CHECK(cudaStreamSynchronize(0));

  byte_t *h_decompressed = new byte_t[input_size];
  TEST_CUDA_CHECK(cudaMemcpy(h_decompressed, d_decompressed, input_size,
                             cudaMemcpyDeviceToHost));

  if (memcmp(h_decompressed, input_str, input_size) == 0) {
    printf("  SUCCESS!\n");
    delete[] h_decompressed;
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(table.codes);
    cudaFree(workspace.d_frequencies);
    cudaFree(workspace.d_code_lengths);
    cudaFree(workspace.d_bit_offsets);
    return true;
  } else {
    printf("  MISMATCH!\n  Expected: %s\n  Got: ", input_str);
    for (size_t i = 0; i < input_size; i++) {
      putchar(h_decompressed[i]);
    }
    putchar('\n');
    delete[] h_decompressed;
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(table.codes);
    cudaFree(workspace.d_frequencies);
    cudaFree(workspace.d_code_lengths);
    cudaFree(workspace.d_bit_offsets);
    return false;
  }
}

int main() {
  printf("=== Simple Huffman Roundtrip Test ===\n\n");

  const char *test_strings[] = {
      "A", "AA", "AAA", "AAABBB", "Hello", "Hello World", "The quick brown fox",
  };

  int num_tests = sizeof(test_strings) / sizeof(test_strings[0]);
  int passed = 0;
  int failed = 0;

  for (int i = 0; i < num_tests; i++) {
    if (test_simple_roundtrip(test_strings[i])) {
      passed++;
    } else {
      failed++;
    }
    printf("\n");
  }

  printf("Results: %d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
