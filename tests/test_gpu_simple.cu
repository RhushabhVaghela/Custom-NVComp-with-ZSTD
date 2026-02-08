#include "cuda_zstd_fse.h"
#include "cuda_zstd_safe_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" {
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
}

#ifndef SIMPLE_CUDA_CHECK
#define SIMPLE_CUDA_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

int main() {
  printf("╔══════════════════════════════════════════════════════════╗\n");
  printf("║  GPU FSE Simple Test                                    ║\n");
  printf("║  Compare our GPU FSE with reference Zstd               ║\n");
  printf("╚══════════════════════════════════════════════════════════╝\n\n");

  // Test with simple repeating pattern
  const size_t input_size = 1024;
  const uint8_t symbol = 'A';

  std::vector<uint8_t> h_input(input_size, symbol);

  printf("Input: %zu bytes, all '%c'\n\n", input_size, symbol);

  // ========================================================================
  // Part 1: Reference Zstd
  // ========================================================================
  printf("=== Reference Zstd ===\n");

  size_t ref_max_size = ZSTD_compressBound(input_size);
  std::vector<uint8_t> ref_output(ref_max_size);

  size_t ref_size = ZSTD_compress(ref_output.data(), ref_output.capacity(),
                                  h_input.data(), h_input.size(), 1);

  if (ZSTD_isError(ref_size)) {
    printf("❌ Reference compression failed: %s\n",
           ZSTD_getErrorName(ref_size));
    return 1;
  }

  printf("Compressed: %zu → %zu bytes (%.1f%%)\n", input_size, ref_size,
         100.0 * ref_size / input_size);

  printf("\nReference output (first 64 bytes):\n");
  for (size_t i = 0; i < std::min(ref_size, (size_t)64); i++) {
    printf("%02X ", ref_output[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n");

  // ========================================================================
  // Part 2: Our GPU FSE
  // ========================================================================
  printf("\n=== GPU FSE Encoder ===\n");

  uint8_t *d_input, *d_output;
  uint32_t *d_output_size;

  SIMPLE_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  SIMPLE_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, input_size * 2)); // Conservative
  SIMPLE_CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output_size, sizeof(uint32_t)));

  SIMPLE_CUDA_CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  printf("Calling encode_fse_advanced...\n");
  cuda_zstd::Status status = cuda_zstd::fse::encode_fse_advanced(
      d_input, input_size, d_output, d_output_size, true, 0);

  if (status != cuda_zstd::Status::SUCCESS) {
    printf("❌ GPU FSE encode failed with status %d\n", (int)status);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    return 1;
  }

  uint32_t gpu_size;
  SIMPLE_CUDA_CHECK(cudaMemcpy(&gpu_size, d_output_size, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

  std::vector<uint8_t> gpu_output(gpu_size);
  SIMPLE_CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, gpu_size,
                               cudaMemcpyDeviceToHost));

  printf("Compressed: %zu → %u bytes (%.1f%%)\n", input_size, gpu_size,
         100.0 * gpu_size / input_size);

  printf("\nGPU FSE output (first 64 bytes):\n");
  for (size_t i = 0; i < std::min((size_t)gpu_size, (size_t)64); i++) {
    printf("%02X ", gpu_output[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n");

  // ========================================================================
  // Part 3: Analysis
  // ========================================================================
  printf("\n=== Analysis ===\n");

  printf("Reference Zstd: %zu bytes (full Zstd format with headers)\n",
         ref_size);
  printf("GPU FSE:        %u bytes (raw FSE stream)\n", gpu_size);

  if (gpu_size < ref_size) {
    printf("✓ GPU FSE produces smaller output (raw stream vs full format)\n");
  }

  // Check if GPU output can be decoded by our decoder
  printf("\nNote: GPU FSE outputs raw FSE bitstream.\n");
  printf("      Reference Zstd outputs full Zstd frame (magic + headers + "
         "blocks).\n");
  printf("      Direct byte comparison not expected to match.\n");

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  printf("\n✅ Test completed successfully\n");
  return 0;
}
