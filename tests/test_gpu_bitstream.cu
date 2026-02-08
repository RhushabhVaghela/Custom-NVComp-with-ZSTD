// Test GPU Bitstream Utilities - Simplified
// Purpose: Verify GPU bitstream matches Zstandard reference

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_zstd_safe_alloc.h"
#include <vector>


// Inline the GPU utilities for testing
namespace test_bitstream {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

struct GPU_BitStream {
  u64 bitContainer;
  u32 bitPos;
  u8 *startPtr;
  u8 *ptr;
  u8 *endPtr;
};

__device__ inline u32 gpu_bit_init_stream(GPU_BitStream *bitC, u8 *dstBuffer,
                                          u32 dstCapacity) {
  if (dstCapacity < 1)
    return 1;
  bitC->bitContainer = 0;
  bitC->bitPos = 0;
  bitC->startPtr = dstBuffer;
  bitC->ptr = dstBuffer;
  bitC->endPtr = dstBuffer + dstCapacity;
  return 0;
}

__device__ inline void gpu_bit_add_bits(GPU_BitStream *bitC, u64 value,
                                        u32 nbBits) {
  u64 const mask = ((u64)1 << nbBits) - 1;
  value &= mask;
  bitC->bitContainer |= value << bitC->bitPos;
  bitC->bitPos += nbBits;
}

__device__ inline void gpu_bit_flush_bits(GPU_BitStream *bitC) {
  u32 const nbBytes = bitC->bitPos >> 3;
  if (bitC->ptr + nbBytes > bitC->endPtr)
    return;

  for (u32 i = 0; i < nbBytes; i++) {
    bitC->ptr[i] = (u8)(bitC->bitContainer >> (i * 8));
  }

  bitC->ptr += nbBytes;
  bitC->bitPos &= 7;
  bitC->bitContainer >>= (nbBytes * 8);
}

__device__ inline u32 gpu_bit_close_stream(GPU_BitStream *bitC) {
  // Add terminator bit
  gpu_bit_add_bits(bitC, 1, 1);
  gpu_bit_flush_bits(bitC);

  if (bitC->bitPos > 0) {
    if (bitC->ptr < bitC->endPtr) {
      *bitC->ptr = (u8)bitC->bitContainer;
      bitC->ptr++;
    }
  }

  return (u32)(bitC->ptr - bitC->startPtr);
}

} // namespace test_bitstream

using namespace test_bitstream;

__global__ void test_bitstream_kernel(u8 *d_output, u32 *d_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  GPU_BitStream bitC;
  gpu_bit_init_stream(&bitC, d_output, 1024);

  // Test: Add some bits
  gpu_bit_add_bits(&bitC, 0x5f, 8); // 8 bits: 0x5f
  gpu_bit_add_bits(&bitC, 0xcb, 8); // 8 bits: 0xcb
  gpu_bit_add_bits(&bitC, 0xf3, 8); // 8 bits: 0xf3

  gpu_bit_flush_bits(&bitC);

  // Add partial bits
  gpu_bit_add_bits(&bitC, 0x7, 3); // 3 bits
  gpu_bit_add_bits(&bitC, 0x2, 5); // 5 bits

  // Close stream (adds terminator)
  *d_size = gpu_bit_close_stream(&bitC);

  printf("[GPU] Bitstream size: %u bytes\\n", *d_size);
  printf("[GPU] First 8 bytes: ");
  for (int i = 0; i < 8 && i < *d_size; i++) {
    printf("%02x ", d_output[i]);
  }
  printf("\\n");
}

int main() {
  printf("=== GPU Bitstream Utility Test ===\\n\\n");

  u8 *d_output;
  u32 *d_size;
  cuda_zstd::safe_cuda_malloc(&d_output, 1024);
  cuda_zstd::safe_cuda_malloc(&d_size, sizeof(u32));

  // Run GPU test
  test_bitstream_kernel<<<1, 1>>>(d_output, d_size);
  cudaDeviceSynchronize();

  // Copy result
  u32 h_size;
  std::vector<u8> h_output(1024);
  cudaMemcpy(&h_size, d_size, sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output.data(), d_output, h_size, cudaMemcpyDeviceToHost);

  printf("\\n=== Expected Output ===\\n");
  printf("Bytes 0-2: 5f cb f3 (three 8-bit values)\\n");
  printf("Byte 3: 17 (3 bits: 111, then 5 bits: 00010, then terminator)\\n");
  printf("  Binary: 0b00010111 = 0x17\\n");

  printf("\\nActual GPU output (%u bytes): ", h_size);
  for (u32 i = 0; i < h_size; i++) {
    printf("%02x ", h_output[i]);
  }
  printf("\\n");

  // Verify
  bool pass = (h_size >= 4 && h_output[0] == 0x5f && h_output[1] == 0xcb &&
               h_output[2] == 0xf3 && h_output[3] == 0x17);

  if (pass) {
    printf("\\n✅ Bitstream test PASSED\\n");
  } else {
    printf("\\n❌ Bitstream test FAILED\\n");
    if (h_size < 4)
      printf("  Size too small: %u\\n", h_size);
    if (h_output[0] != 0x5f)
      printf("  Byte 0: expected 5f, got %02x\\n", h_output[0]);
    if (h_output[1] != 0xcb)
      printf("  Byte 1: expected cb, got %02x\\n", h_output[1]);
    if (h_output[2] != 0xf3)
      printf("  Byte 2: expected f3, got %02x\\n", h_output[2]);
    if (h_size >= 4 && h_output[3] != 0x17)
      printf("  Byte 3: expected 17, got %02x\\n", h_output[3]);
  }

  cudaFree(d_output);
  cudaFree(d_size);
  return pass ? 0 : 1;
}
