#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

using namespace cuda_zstd;

#define TEST_CHECK(cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__);     \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_TEST(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

bool verify_rfc8878(const uint8_t *data, size_t size, size_t original_size) {
  printf("[INFO] Verifying RFC 8878 format (Compressed Size: %zu)\n", size);

  // 1. Magic Number (4 bytes)
  TEST_CHECK(size >= 4);
  uint32_t magic = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
  TEST_CHECK(magic == 0xFD2FB528);
  printf("  [PASS] Magic Number: 0x%08X\n", magic);

  // 2. Frame Header Descriptor (1 byte)
  TEST_CHECK(size >= 5);
  uint8_t fhd = data[4];
  printf("  [INFO] Frame Header Descriptor: 0x%02X\n", fhd);

  bool single_segment = (fhd & 0x20) != 0;
  uint8_t fcs_field_size = (fhd >> 6) & 0x03;
  printf("  [INFO] Single Segment: %s, FCS Size: %u\n",
         single_segment ? "Yes" : "No", fcs_field_size);

  size_t offset = 5;

  // 3. Window Descriptor (if !single_segment)
  if (!single_segment) {
    TEST_CHECK(size >= offset + 1);
    uint8_t wd = data[offset++];
    printf("  [INFO] Window Descriptor: 0x%02X\n", wd);
  }

  // 4. Dictionary ID (if present - our current impl usually doesn't write it
  // unless configured)
  if (fhd & 0x03) {
    uint32_t did_size = (1 << ((fhd & 0x03) - 1));
    if ((fhd & 0x03) == 3)
      did_size = 4;
    offset += did_size;
  }

  // 5. Frame Content Size
  if (fcs_field_size > 0) {
    size_t fcs_size = (fcs_field_size == 1) ? 2 : (fcs_field_size == 2) ? 4 : 8;
    TEST_CHECK(size >= offset + fcs_size);
    // Simplistic check: if 4 bytes, compare with u32
    if (fcs_size == 4) {
      uint32_t fcs = data[offset] | (data[offset + 1] << 8) |
                     (data[offset + 2] << 16) | (data[offset + 3] << 24);
      printf("  [PASS] Frame Content Size: %u\n", fcs);
      TEST_CHECK(fcs == (uint32_t)original_size);
    }
    offset += fcs_size;
  }

  printf("  [INFO] Blocks start at offset %zu\n", offset);

  // 6. Block Headers
  bool last_block = false;
  int block_count = 0;
  while (!last_block && offset + 3 <= size) {
    uint32_t header =
        data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16);
    last_block = (header & 0x01) != 0;
    uint32_t block_type = (header >> 1) & 0x03;
    uint32_t block_size = header >> 3;

    printf("  [BLOCK %d] Type: %u, Size: %u, Last: %s (Offset: %zu)\n",
           block_count, block_type, block_size, last_block ? "Yes" : "No",
           offset);

    TEST_CHECK(block_type != 3); // Reserved
    TEST_CHECK(offset + 3 + block_size <= size);

    offset += 3 + block_size;
    block_count++;
  }

  TEST_CHECK(last_block);
  printf("  [PASS] Successfully parsed %d blocks\n", block_count);

  // 7. Checksum (if present)
  if (fhd & 0x04) {
    printf("  [INFO] Checksum bit set, expecting 4 bytes at end\n");
    TEST_CHECK(offset + 4 <= size);
    offset += 4;
  }

  TEST_CHECK(offset == size);
  printf("  [PASS] Reached end of frame exactly at offset %zu\n", offset);

  return true;
}

int main() {
  printf("========================================\n");
  printf("RFC 8878 Compliance Unit Test\n");
  printf("========================================\n\n");

  const size_t input_size = 1024 * 1024; // 1MB
  std::vector<uint8_t> h_input(input_size);
  for (size_t i = 0; i < input_size; i++) {
    h_input[i] = (uint8_t)(i % 256); // Some pattern
  }

  uint8_t *d_input, *d_output;
  size_t *d_compressed_size;

  CUDA_CHECK_TEST(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CUDA_CHECK_TEST(cuda_zstd::safe_cuda_malloc(&d_output, input_size * 2));
  CUDA_CHECK_TEST(cuda_zstd::safe_cuda_malloc(&d_compressed_size, sizeof(size_t)));

  CUDA_CHECK_TEST(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  CompressionConfig config = CompressionConfig::from_level(3);
  ZstdBatchManager manager;
  Status status = manager.configure(config);
  if (status != Status::SUCCESS) {
    fprintf(stderr, "Manager configuration failed with status %d\n",
            (int)status);
    return 1;
  }

  size_t temp_size = manager.get_compress_temp_size(input_size);
  void *d_temp;
  CUDA_CHECK_TEST(cuda_zstd::safe_cuda_malloc(&d_temp, temp_size));

  size_t compressed_size = input_size * 2;
  status = manager.compress(d_input, input_size, d_output, &compressed_size,
                            d_temp, temp_size, nullptr, 0, 0);

  if (status != Status::SUCCESS) {
    fprintf(stderr, "Compression failed with status %d\n", (int)status);
    cudaFree(d_temp);
    return 1;
  }

  std::vector<uint8_t> h_output(compressed_size);
  CUDA_CHECK_TEST(cudaMemcpy(h_output.data(), d_output, compressed_size,
                             cudaMemcpyDeviceToHost));
  cudaFree(d_temp);

  // (NEW) Save to file for external zstd CLI verification
  FILE *f = fopen("gpu_compressed.zst", "wb");
  if (f) {
    fwrite(h_output.data(), 1, compressed_size, f);
    fclose(f);
    printf("[INFO] Saved GPU-compressed output to 'gpu_compressed.zst' (%zu "
           "bytes)\n",
           compressed_size);
  }

  if (!verify_rfc8878(h_output.data(), compressed_size, input_size)) {
    printf("\n❌ RFC 8878 Compliance Test FAILED\n");
    return 1;
  }

  printf("\n✅ RFC 8878 Compliance Test PASSED\n");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_compressed_size);

  return 0;
}
