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
// Device function to encode a single block
// UPDATED: Accepts explicit table components to avoid reconstruction
__device__ inline void fse_encode_block_device(
    const byte_t *d_input, u32 input_size, const u16 *stateTable,
    const GPU_FSE_SymbolTransform *symbolTT, u16 tableLog, byte_t *d_output,
    u32 max_output_size, u32 *d_output_size) {

  if (input_size <= 2) {
    *d_output_size = 0;
    return;
  }

  GPU_BitStream bitC;
  if (gpu_bit_init_stream(&bitC, d_output, max_output_size) != 0) {
    *d_output_size = 0; // Buffer too small
    return;
  }

  // Backward Encoding (Standard Zstd / LIFO)
  GPU_FSE_CState CState;
  const byte_t *ip = d_input + input_size - 1; // Last byte
  const byte_t *const ibegin = d_input;

  // Initialize with first byte
  if (input_size > 0) {
    CState.stateLog = tableLog;
    CState.stateTable = stateTable;
    CState.symbolTT = symbolTT;
    CState.value = (u64)1 << tableLog; // Base value

    // Apply symbol transformation
    u32 symbol = *ip--;
    GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
    u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
    u64 tempValue = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
    u32 tableIndex = (tempValue >> nbBitsOut) + symbolTransform.deltaFindState;
    CState.value = stateTable[tableIndex];

    // Encode remaining bytes backwards
    while (ip >= ibegin) {
      gpu_fse_encode_symbol(&bitC, &CState, *ip--);
      gpu_bit_flush_bits(&bitC);
    }

    // Flush state
    gpu_fse_flush_state(&bitC, &CState);
  }

  *d_output_size = gpu_bit_close_stream(&bitC);
}

// Wrapper for backward compatibility (Expects flat ctable)
static __global__ void
fse_encode_zstd_compat_kernel(const byte_t *d_input, u32 input_size,
                              const u16 *d_ctable_u16, byte_t *d_output,
                              u32 max_output_size, u32 *d_output_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Unpack flat table
  u16 tableLog = d_ctable_u16[0];
  const u16 *stateTable = d_ctable_u16 + 2;
  const u32 *ct_u32 = (const u32 *)d_ctable_u16;
  const GPU_FSE_SymbolTransform *symbolTT =
      (const GPU_FSE_SymbolTransform *)(ct_u32 + 1 +
                                        (tableLog ? (1 << (tableLog - 1)) : 1));

  fse_encode_block_device(d_input, input_size, stateTable, symbolTT, tableLog,
                          d_output, max_output_size, d_output_size);
}

// NEW: Batch Parallel Kernel
// Processes multiple blocks in parallel (one thread per block)
// Uses arrays of pointers for maximum flexibility (mixed batches)
static __global__ void fse_batch_encode_kernel(
    const byte_t *const *d_inputs, const u32 *d_input_sizes, byte_t **d_outputs,
    u32 *d_output_sizes,
    const u16 *const *d_state_tables,                      // Array of pointers
    const GPU_FSE_SymbolTransform *const *d_symbol_tables, // Array of pointers
    const u32 *d_table_logs,                               // Array of values
    u32 num_blocks) {

  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_blocks)
    return;

  u32 input_size = d_input_sizes[idx];
  u32 max_capacity = input_size * 2 + 128;

  // Use direct pointers (already resolved on host)
  u16 tableLog = (u16)d_table_logs[idx];
  const u16 *stateTable = d_state_tables[idx];
  const GPU_FSE_SymbolTransform *symbolTT = d_symbol_tables[idx];

  fse_encode_block_device(d_inputs[idx], input_size, stateTable, symbolTT,
                          tableLog, d_outputs[idx], max_capacity,
                          &d_output_sizes[idx]);
}

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_ZSTD_ENCODER_CUH
