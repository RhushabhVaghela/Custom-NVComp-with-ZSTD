// Save GPU encoder output to file for comparison
#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
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
  const u32 data_size = 4 * 1024;
  std::vector<byte_t> h_input(data_size);
  fill_random(h_input);

  byte_t *d_input = nullptr;
  byte_t *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));

  u32 h_output_size_val = 0;
  Status status = encode_fse_advanced(d_input, data_size, d_output,
                                      &h_output_size_val, true, 0);

  if (status != Status::SUCCESS) {
    printf("Encoding failed\n");
    return 1;
  }

  std::vector<byte_t> h_encoded(h_output_size_val);
  CUDA_CHECK(cudaMemcpy(h_encoded.data(), d_output, h_output_size_val,
                        cudaMemcpyDeviceToHost));

  // Save full GPU output
  FILE *f = fopen("gpu_fse_output.bin", "wb");
  if (f) {
    fwrite(h_encoded.data(), 1, h_output_size_val, f);
    fclose(f);
    printf("Saved GPU output: %u bytes to gpu_fse_output.bin\n",
           h_output_size_val);
    printf("Header size: 524 bytes\n");
    printf("Payload size: %u bytes\n", h_output_size_val - 524);
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}
