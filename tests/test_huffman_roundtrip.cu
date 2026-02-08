
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_safe_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>


// Avoid conflict with library's CUDA_CHECK which returns Status
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

void verify_roundtrip(const char *input_str) {
  size_t input_size = strlen(input_str);
  printf("Testing Roundtrip with input size: %zu\n", input_size);

  byte_t *d_input, *d_compressed, *d_decompressed;
  TEST_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  // Allocate generous space for compressed (header + data)
  size_t max_compressed_size = input_size * 2 + 1024;
  TEST_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_compressed, max_compressed_size));
  TEST_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_decompressed, input_size));

  TEST_CUDA_CHECK(
      cudaMemcpy(d_input, input_str, input_size, cudaMemcpyHostToDevice));

  huffman::HuffmanTable table;
  TEST_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&table.codes, huffman::MAX_HUFFMAN_SYMBOLS *
                                               sizeof(huffman::HuffmanCode)));

  // CompressionWorkspace is in cuda_zstd namespace (not huffman)
  cuda_zstd::CompressionWorkspace workspace;
  // Allocate workspace arrays (frequencies, lengths, offsets)
  TEST_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&workspace.d_frequencies,
                             huffman::MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
  TEST_CUDA_CHECK(
      cuda_zstd::safe_cuda_malloc(&workspace.d_code_lengths, input_size * sizeof(u32)));
  TEST_CUDA_CHECK(
      cuda_zstd::safe_cuda_malloc(&workspace.d_bit_offsets, input_size * sizeof(u32)));

  size_t compressed_size = 0;
  cudaStream_t stream = 0;

  Status status =
      huffman::encode_huffman(d_input, (u32)input_size, table, d_compressed,
                              &compressed_size, &workspace, stream);

  if (status != Status::SUCCESS) {
    fprintf(stderr, "Uncompressed encode failed: %d\n", (int)status);
    exit(1);
  }
  TEST_CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("Encoded Size: %zu\n", compressed_size);

  // 2. Decode
  size_t decompressed_size_out = 0;

  status = huffman::decode_huffman(d_compressed, compressed_size, table,
                                   d_decompressed, &decompressed_size_out,
                                   (u32)input_size, stream);

  if (status != Status::SUCCESS) {
    fprintf(stderr, "Decode failed: %d\n", (int)status);
    exit(1);
  }
  TEST_CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("Decoded Size Reported: %zu\n", decompressed_size_out);

  // 3. Verify
  byte_t *h_decompressed = new byte_t[input_size];
  TEST_CUDA_CHECK(cudaMemcpy(h_decompressed, d_decompressed, input_size,
                             cudaMemcpyDeviceToHost));

  if (memcmp(h_decompressed, input_str, input_size) != 0) {
    fprintf(stderr, "MISMATCH!\nExpected: %s\nGot: ", input_str);
    for (size_t i = 0; i < input_size; i++)
      putchar(h_decompressed[i]);
    putchar('\n');
    exit(1);
  } else {
    printf("SUCCESS! Decoded matches input.\n");
  }

  delete[] h_decompressed;
  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(table.codes);
  cudaFree(workspace.d_frequencies);
  cudaFree(workspace.d_code_lengths);
  cudaFree(workspace.d_bit_offsets);
}

int main() {
  verify_roundtrip("Hello World! This is a test string that should be "
                   "compressed and decompressed correctly.");
  return 0;
}
