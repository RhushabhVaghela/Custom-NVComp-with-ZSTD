// ============================================================================
// test_c_api.c - Verify the C-API
// ============================================================================

#include "cuda_zstd_nvcomp.h" // C-API is in the nvcomp header
#include "cuda_zstd_safe_alloc.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


int main() {
  printf("\n========================================\n");
  printf("  Test: C-API Validation (NVCOMP v5)\n");
  printf("========================================\n\n");

  // 1. Create Manager
  int level = 3;
  nvcompZstdManagerHandle manager = nvcomp_zstd_create_manager_v5(level);
  if (!manager) {
    printf("  ✗ FAILED: nvcomp_zstd_create_manager_v5 returned NULL\n");
    return 1;
  }
  printf("  ✓ nvcomp_zstd_create_manager_v5 passed.\n");

  // 2. Prepare data
  size_t data_size = 128 * 1024;
  void *h_data = malloc(data_size); // Not strictly needed, but good practice
  void *d_input, *d_output, *d_temp, *d_decompressed;

  cuda_zstd::safe_cuda_malloc(&d_input, data_size);
  cuda_zstd::safe_cuda_malloc(&d_output, data_size * 2);
  for (size_t i = 0; i < data_size; ++i) {
    ((unsigned char *)h_data)[i] = (unsigned char)(i & 0xFF);
  }
  cudaMemcpy(d_input, h_data, data_size, cudaMemcpyHostToDevice);
  cuda_zstd::safe_cuda_malloc(&d_decompressed, data_size);

  size_t temp_size = nvcomp_zstd_get_compress_temp_size_v5(manager, data_size);
  cuda_zstd::safe_cuda_malloc(&d_temp, temp_size);

  printf("  ✓ Workspace allocated: %zu KB\n", temp_size / 1024);

  // 3. Compress
  // FIX: Initialize compressed_size to capacity
  size_t compressed_size = data_size * 2;
  int err =
      nvcomp_zstd_compress_async_v5(manager, d_input, data_size, d_output,
                                    &compressed_size, d_temp, temp_size, 0);
  cudaDeviceSynchronize();

  if (err != 0) {
    printf("  ✗ FAILED: nvcomp_zstd_compress_async_v5 returned error %d\n",
           err);
    return 1;
  }
  printf("  ✓ nvcomp_zstd_compress_async_v5 passed. Size: %zu\n",
         compressed_size);

  // 3b. Decompress
  size_t decompressed_size = data_size;
  err = nvcomp_zstd_decompress_async_v5(manager, d_output, compressed_size,
                                       d_decompressed, &decompressed_size,
                                       d_temp, temp_size, 0);
  cudaDeviceSynchronize();
  if (err != 0) {
    printf("  ✗ FAILED: nvcomp_zstd_decompress_async_v5 returned error %d\n",
           err);
    return 1;
  }
  if (decompressed_size != data_size) {
    printf("  ✗ FAILED: decompressed_size mismatch (%zu != %zu)\n",
           decompressed_size, data_size);
    return 1;
  }

  void *h_roundtrip = malloc(data_size);
  cudaMemcpy(h_roundtrip, d_decompressed, data_size, cudaMemcpyDeviceToHost);
  if (memcmp(h_roundtrip, h_data, data_size) != 0) {
    printf("  ✗ FAILED: roundtrip data mismatch\n");
    return 1;
  }
  free(h_roundtrip);
  printf("  ✓ nvcomp_zstd_decompress_async_v5 passed. Size: %zu\n",
         decompressed_size);

  // 4. Destroy Manager
  nvcomp_zstd_destroy_manager_v5(manager);
  printf("  ✓ nvcomp_zstd_destroy_manager_v5 passed.\n");

  // Cleanup
  free(h_data);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);
  cudaFree(d_decompressed);

  printf("\nTest complete. Result: PASSED ✓\n");
  return 0;
}
