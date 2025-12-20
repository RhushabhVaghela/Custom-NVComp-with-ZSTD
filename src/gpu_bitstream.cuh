// GPU Bitstream Utilities - Zstandard Compatible
// Purpose: Match Zstandard's bitstream.h for FSE encoding
// Reference: build/_deps/zstd-src/lib/common/bitstream.h

#ifndef GPU_BITSTREAM_H
#define GPU_BITSTREAM_H

#include "cuda_zstd_types.h"

namespace cuda_zstd {
namespace fse {

// GPU Bitstream State (matches BIT_CStream_t)
struct GPU_BitStream {
  u64 bitContainer; // Bit accumulator (64-bit on GPU)
  u32 bitPos;       // Number of bits in bitContainer
  u8 *startPtr;     // Start of output buffer
  u8 *ptr;          // Current write position
  u8 *endPtr;       // End of output buffer
};

// Initialize bitstream for compression
// Returns 0 on success, or error code
__device__ inline u32 gpu_bit_init_stream(GPU_BitStream *bitC, u8 *dstBuffer,
                                          u32 dstCapacity) {
  if (dstCapacity < 1)
    return 1; // ZSTD_ERROR(dstSize_tooSmall)

  bitC->bitContainer = 0;
  bitC->bitPos = 0;
  bitC->startPtr = dstBuffer;
  bitC->ptr = dstBuffer;
  bitC->endPtr = dstBuffer + dstCapacity;

  return 0;
}

// Add bits to the bitstream (matches BIT_addBits)
// Note: Maximum nbBits is 31 (not 32) for compatibility
__device__ inline void gpu_bit_add_bits(GPU_BitStream *bitC, u64 value,
                                        u32 nbBits) {
  // Mask value to nbBits
  u64 const mask = ((u64)1 << nbBits) - 1;
  value &= mask;

  // Add to bit container
  bitC->bitContainer |= value << bitC->bitPos;
  bitC->bitPos += nbBits;
}

// Flush full bytes from bit container to memory
// Matches BIT_flushBits behavior
__device__ inline void gpu_bit_flush_bits(GPU_BitStream *bitC) {
  u32 const nbBytes = bitC->bitPos >> 3; // divide by 8

  if (bitC->ptr + nbBytes > bitC->endPtr) {
    // Buffer overflow - this shouldn't happen in correct usage
    // but we need to handle it gracefully
    return;
  }

  // Write bytes in little-endian order
  for (u32 i = 0; i < nbBytes; i++) {
    bitC->ptr[i] = (u8)(bitC->bitContainer >> (i * 8));
  }

  bitC->ptr += nbBytes;
  bitC->bitPos &= 7; // Keep remainder bits (modulo 8)
  bitC->bitContainer >>= (nbBytes * 8);
}

// Close bitstream and return size
// Matches BIT_closeCStream behavior
// Adds terminator bit (1) followed by padding (0s)
__device__ inline u32 gpu_bit_close_stream(GPU_BitStream *bitC) {
  // Add terminator bit
  gpu_bit_add_bits(bitC, 1, 1);

  // Flush all remaining bits
  gpu_bit_flush_bits(bitC);

  // If there are still bits in container, flush final partial byte
  if (bitC->bitPos > 0) {
    if (bitC->ptr < bitC->endPtr) {
      *bitC->ptr = (u8)bitC->bitContainer;
      bitC->ptr++;
    }
  }

  return (u32)(bitC->ptr - bitC->startPtr);
}

// Fast flush - used when buffer space is guaranteed
// Matches BIT_flushBitsFast behavior
__device__ inline void gpu_bit_flush_bits_fast(GPU_BitStream *bitC) {
  u32 const nbBits = bitC->bitPos & ~7; // Round down to multiple of 8
  u32 const nbBytes = nbBits >> 3;

  // Write bytes
  for (u32 i = 0; i < nbBytes; i++) {
    bitC->ptr[i] = (u8)(bitC->bitContainer >> (i * 8));
  }

  bitC->ptr += nbBytes;
  bitC->bitPos -= nbBits;
  bitC->bitContainer >>= nbBits;
}

} // namespace fse
} // namespace cuda_zstd

#endif // GPU_BITSTREAM_H
