#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __LINE__ << ": "                        \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  }

// Generate reproducible test data
void generate_test_data(std::vector<byte_t> &data, size_t size, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);

  for (size_t i = 0; i < size; ++i) {
    if (i % 32 < 16) {
      data[i] = dist(rng) % 64;
    } else {
      data[i] = dist(rng);
    }
  }
}

bool test_roundtrip(size_t size, int seed) {
  std::cout << "  Testing " << size << " bytes (seed=" << seed << ")..."
            << std::endl;

  std::vector<byte_t> h_input(size);
  generate_test_data(h_input, size, seed);

  // Debug: Print last 4 input bytes
  std::cout << "  Input[1020..1023]: ";
  for (size_t i = (size > 4 ? size - 4 : 0); i < size; ++i) {
    std::cout << std::hex << (int)h_input[i] << " ";
  }
  std::cout << std::dec << std::endl;

  byte_t *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, size));
  size_t max_compressed_size = size + 512;
  CHECK_CUDA(cudaMalloc(&d_compressed, max_compressed_size));
  CHECK_CUDA(cudaMalloc(&d_output, size));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

  cudaStream_t stream = 0;

  u32 *d_comp_size;
  CHECK_CUDA(cudaMalloc(&d_comp_size, sizeof(u32)));
  CHECK_CUDA(cudaMemset(d_comp_size, 0, sizeof(u32)));

  std::cout << "  Compressing..." << std::endl;
  Status status = fse::encode_fse_advanced(d_input, (u32)size, d_compressed,
                                           d_comp_size, true, stream);

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (compress: " << (int)status << ")" << std::endl;
    return false;
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  u32 h_comp_size = 0;
  CHECK_CUDA(cudaMemcpy(&h_comp_size, d_comp_size, sizeof(u32),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_comp_size));

  std::cout << "  Compressed size: " << h_comp_size << std::endl;

  std::cout << "  Decompressing..." << std::endl;
  CHECK_CUDA(cudaGetLastError()); // Clear error

  u32 decompressed_size = 0;
  status = fse::decode_fse(d_compressed, h_comp_size, d_output,
                           &decompressed_size, stream);

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (decompress: " << (int)status << ")" << std::endl;
    return false;
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());

  std::cout << "  Verifying..." << std::endl;
  std::vector<byte_t> h_output(
      decompressed_size); // Beware decompressed_size could be 0

  if (decompressed_size != size) {
    std::cerr << "SIZE MISMATCH: Expected " << size << ", Got "
              << decompressed_size << std::endl;
    // Continue to see partial output?
  }

  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, decompressed_size,
                        cudaMemcpyDeviceToHost));

  bool match = (decompressed_size == size);
  int errors = 0;
  for (size_t i = 0; i < std::min((size_t)size, (size_t)decompressed_size);
       ++i) {
    if (h_output[i] != h_input[i]) {
      if (errors < 20) {
        std::cerr << " Mismatch at " << i << ": In=" << std::hex
                  << (int)h_input[i] << " Out=" << (int)h_output[i] << std::dec
                  << std::endl;
      }
      match = false;
      errors++;
    } else {
      if (errors > 0 && errors < 1000) {
        std::cout << " First MATCH at " << i << ": Val=" << std::hex
                  << (int)h_input[i] << std::dec << std::endl;
        errors = 2000; // Stop tracking mismatch print but keep counting
      }
    }
  }
  if (errors > 0)
    std::cout << "Total errors: " << errors << std::endl;

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_output));

  return match;
}

int main() {
  setenv("CUDA_ZSTD_FSE_THRESHOLD", "0", 1); // Enable for GPU-only debugging

  std::cout << "======================================" << std::endl;
  std::cout << "FSE Decode Correctness Tests (Direct)" << std::endl;
  std::cout << "======================================" << std::endl;

  int passed = 0;
  int failed = 0;

  // Test various sizes
  std::vector<size_t> sizes = {
      1024, // 1 KB
  };

  for (size_t size : sizes) {
    if (test_roundtrip(size, 42)) {
      passed++;
    } else {
      failed++;
    }
  }

  std::cout << "\n======================================" << std::endl;
  std::cout << "Results: " << passed << " PASSED, " << failed << " FAILED"
            << std::endl;
  std::cout << "======================================" << std::endl;

  return failed > 0 ? 1 : 0;
}
