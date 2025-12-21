// ==============================================================================
// NEW: ZSTANDARD-COMPATIBLE SINGLE-STATE FSE ENCODER
// ==============================================================================
// This encoder matches Zstandard's FSE logic for Single State streams.
// Implemented as Single State Forward to match Parallel Decoder Reverse Read.
// ==============================================================================

#ifndef CUDA_ZSTD_FSE_ZSTD_ENCODER_CUH
#define CUDA_ZSTD_FSE_ZSTD_ENCODER_CUH

#include "cuda_zstd_types.h"

namespace cuda_zstd {
namespace fse {

// GPU Bitstream (verified implementation)
struct GPU_BitStream {
  u64 bitContainer;
  u32 bitPos;
  byte_t *startPtr;
  byte_t *ptr;
  byte_t *endPtr;
};

__device__ inline u32 gpu_bit_init_stream(GPU_BitStream *bitC,
                                          byte_t *dstBuffer, u32 dstCapacity) {
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
    bitC->ptr[i] = (byte_t)(bitC->bitContainer >> (i * 8));
  }

  bitC->ptr += nbBytes;
  bitC->bitPos &= 7;
  bitC->bitContainer >>= (nbBytes * 8);
}

__device__ inline u32 gpu_bit_close_stream(GPU_BitStream *bitC) {
  gpu_bit_add_bits(bitC, 1, 1); // Terminator
  gpu_bit_flush_bits(bitC);

  if (bitC->bitPos > 0) {
    if (bitC->ptr < bitC->endPtr) {
      *bitC->ptr = (byte_t)bitC->bitContainer;
      bitC->ptr++;
    }
  }

  return (u32)(bitC->ptr - bitC->startPtr);
}

// GPU FSE State
struct GPU_FSE_SymbolTransform {
  i32 deltaFindState;
  u32 deltaNbBits;
};

struct GPU_FSE_CState {
  u64 value;
  const u16 *stateTable;
  const GPU_FSE_SymbolTransform *symbolTT;
  u32 stateLog;
};

__device__ inline void gpu_fse_init_state(GPU_FSE_CState *statePtr,
                                          const u16 *ctable, u32 symbol) {
  u32 const tableLog = ctable[0];
  const u16 *stateTable = ctable + 2;
  const u32 *ct_u32 = (const u32 *)ctable;
  const GPU_FSE_SymbolTransform *symbolTT =
      (const GPU_FSE_SymbolTransform *)(ct_u32 + 1 +
                                        (tableLog ? (1 << (tableLog - 1)) : 1));

  // Simple initialization to base state (Zstandard FSE_initCState)
  statePtr->value = (u64)1 << tableLog;
  statePtr->stateTable = stateTable;
  statePtr->symbolTT = symbolTT;
  statePtr->stateLog = tableLog;

  // Apply symbol-specific transformation (exact Zstandard FSE_initCState2)
  GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
  u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
  u64 tempValue = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
  u32 tableIndex = (tempValue >> nbBitsOut) + symbolTransform.deltaFindState;
  statePtr->value = stateTable[tableIndex];

  printf("[INIT_DEBUG] Sym=%u nbBits=%u tempVal=%llu idx=%u -> State=%llu\n",
         symbol, nbBitsOut, tempValue, tableIndex, statePtr->value);
}

__device__ inline void gpu_fse_encode_symbol(GPU_BitStream *bitC,
                                             GPU_FSE_CState *statePtr,
                                             u32 symbol) {
  GPU_FSE_SymbolTransform const symbolTT = statePtr->symbolTT[symbol];
  u32 const nbBitsOut = (u32)((statePtr->value + symbolTT.deltaNbBits) >> 16);

  // Write state value directly (stateTable now contains raw states 0-511)
  gpu_bit_add_bits(bitC, statePtr->value, nbBitsOut);
  statePtr->value = statePtr->stateTable[(statePtr->value >> nbBitsOut) +
                                         symbolTT.deltaFindState];
}

__device__ inline void gpu_fse_flush_state(GPU_BitStream *bitC,
                                           const GPU_FSE_CState *statePtr) {
  // Write state value directly (state now contains raw value 0-511)
  printf("[FLUSH_DEBUG] State=%llu nbBits=%u\n", statePtr->value,
         statePtr->stateLog);
  gpu_bit_add_bits(bitC, statePtr->value, statePtr->stateLog);
  gpu_bit_flush_bits(bitC);
}

/**
 * @brief Zstandard-compatible Single-State FSE encoder kernel
 * @param d_input Input data
 * @param input_size Size of input in bytes
 * @param d_ctable_u16 FSE compression table (as u16* for easy access)
 * @param d_output Output buffer
 * @param max_output_size Maximum output capacity
 * @param d_output_size Output size in bytes
 */
__global__ void fse_encode_zstd_compat_kernel(
    const byte_t *d_input, u32 input_size,
    const u16 *d_ctable_u16, // FSE_CTable as u16 array
    byte_t *d_output, u32 max_output_size, u32 *d_output_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  if (input_size <= 2) {
    *d_output_size = 0;
    return;
  }

  GPU_BitStream bitC;
  if (gpu_bit_init_stream(&bitC, d_output, max_output_size) != 0) {
    *d_output_size = 0;
    return;
  }

  // Backward Encoding (Standard Zstd / LIFO)
  GPU_FSE_CState CState;
  const byte_t *ip = d_input + input_size - 1; // Last byte
  const byte_t *const ibegin = d_input;

  // Initialize with first byte (which is last in sequence)
  if (input_size > 0) {
    gpu_fse_init_state(&CState, d_ctable_u16, *ip--);

    // Encode remaining bytes backwards
    while (ip >= ibegin) {
      gpu_fse_encode_symbol(&bitC, &CState, *ip--);
      gpu_bit_flush_bits(&bitC);
    }

    // Flush state (writes bits for the first symbol processed, transitively)
    gpu_fse_flush_state(&bitC, &CState);
  }

  *d_output_size = gpu_bit_close_stream(&bitC);
}

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_ZSTD_ENCODER_CUH
