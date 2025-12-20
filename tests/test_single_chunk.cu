// Single-threaded FSE Encoder Test
// Purpose: Isolate whether bug is in encoding kernel OR parallel assembly
// This test uses num_chunks=1 to bypass parallel assembly logic

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

void fill_random(std::vector<byte_t> &buffer) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)dist(rng);
  }
}

int main() {
  printf("=== Single-Threaded FSE Encoder Test ===\n\n");

  // Use small input (4KB) to fit in single chunk
  const u32 data_size = 4 * 1024;
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input);

  byte_t *d_input = nullptr;
  byte_t *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2)); // Plenty of space

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  printf("Input: %u bytes\n", data_size);
  printf("First 20 bytes: ");
  for (int i = 0; i < 20; i++)
    printf("%02x ", h_input[i]);
  printf("\n\n");

  // Save input for CPU reference comparison
  FILE *f_in = fopen("gpu_test_input.bin", "wb");
  if (f_in) {
    fwrite(h_input.data(), 1, data_size, f_in);
    fclose(f_in);
  }

  // Encode using single-buffer API (will use 1 chunk for small input)
  u32 h_output_size_val = 0;
  Status status = encode_fse_advanced(d_input, data_size, d_output,
                                      &h_output_size_val, true, 0);

  if (status != Status::SUCCESS) {
    printf("❌ Encoding failed: %d\n", (int)status);
    return 1;
  }

  printf("Encoded successfully: %u bytes\n", h_output_size_val);

  // Copy encoded data to host
  std::vector<byte_t> h_encoded(h_output_size_val);
  CUDA_CHECK(cudaMemcpy(h_encoded.data(), d_output, h_output_size_val,
                        cudaMemcpyDeviceToHost));

  printf("Encoded data (first 20 bytes): ");
  for (int i = 0; i < 20 && i < h_output_size_val; i++)
    printf("%02x ", h_encoded[i]);
  printf("\n");

  printf("Encoded data (last 20 bytes): ");
  int start = (h_output_size_val > 20) ? (h_output_size_val - 20) : 0;
  for (int i = start; i < h_output_size_val; i++)
    printf("%02x ", h_encoded[i]);
  printf("\n\n");

  // Decode
  byte_t *d_decoded = nullptr;
  CUDA_CHECK(cudaMalloc(&d_decoded, data_size));
  u32 decoded_size = 0;

  status = decode_fse(d_output, h_output_size_val, d_decoded, &decoded_size, 0);

  if (status != Status::SUCCESS) {
    printf("❌ Decoding failed: %d\n", (int)status);
    return 1;
  }

  printf("Decoded %u bytes\n", decoded_size);

  std::vector<byte_t> h_decoded(data_size);
  CUDA_CHECK(cudaMemcpy(h_decoded.data(), d_decoded, data_size,
                        cudaMemcpyDeviceToHost));

  printf("Decoded data (first 20 bytes): ");
  for (int i = 0; i < 20; i++)
    printf("%02x ", h_decoded[i]);
  printf("\n\n");

  // Verify
  if (h_decoded != h_input) {
    printf("❌ Content mismatch!\n");
    for (size_t i = 0; i < data_size; ++i) {
      if (h_decoded[i] != h_input[i]) {
        printf("First mismatch at index %zu: expected %02x, got %02x\n", i,
               h_input[i], h_decoded[i]);
        break;
      }
    }
    return 1;
  }

  printf("✅ Single-threaded test PASSED!\n");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_decoded);
  return 0;
}
