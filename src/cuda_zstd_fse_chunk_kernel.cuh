/*
 * cuda_zstd_fse_chunk_kernel.cuh
 *
 * Intra-Block Parallel FSE Encoding Kernels (Phase 3)
 */

#ifndef CUDA_ZSTD_FSE_CHUNK_KERNEL_CUH
#define CUDA_ZSTD_FSE_CHUNK_KERNEL_CUH

#include "cuda_zstd_fse_zstd_encoder.cuh" // For GPU_FSE_SymbolTransform, GPU_BitStream
#include "cuda_zstd_types.h"
#include <cuda_runtime.h>


namespace cuda_zstd {
namespace fse {

// ==============================================================================
// 1. STATE PRE-PASS KERNEL
// ==============================================================================

__global__ void fse_compute_states_kernel(
    const byte_t *symbols, u32 num_symbols, const u16 *stateTable,
    const GPU_FSE_SymbolTransform *symbolTT, u16 tableLog, u32 chunk_size,
    u16 *out_states, // Array of start states for chunks
    u32 *bit_counts  // Output: Exact bit size of each chunk
) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  if (num_symbols == 0)
    return;

  const byte_t *ip = symbols + num_symbols - 1;
  u32 current_idx = num_symbols - 1;
  u64 state_value;
  u32 chunk_current_bits = 0;

  // --- Initialization (Last Symbol) ---
  u32 symbol = *ip--;
  current_idx--;

  GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
  u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
  u64 tempValue = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
  u32 tableIndex = (tempValue >> nbBitsOut) + symbolTransform.deltaFindState;
  state_value = stateTable[tableIndex];

  // Last Symbol bits belong to the LAST chunk
  // (Assume chunk_idx = num_chunks - 1)
  chunk_current_bits += nbBitsOut;

  // --- State Transition Scan Backwards ---
  while (current_idx > 0) {
    // Boundary Logic: Check if we just crossed a boundary
    // We are at `current_idx`. We just encoded `current_idx + 1`.
    // If `current_idx + 1` was the start of a chunk, then `state_value` is the
    // input for PREVIOUS chunk. Chunk C: [C*size ... (C+1)*size - 1] If
    // current_idx + 1 == (c+1)*size

    if ((current_idx + 1) % chunk_size == 0) {
      u32 c = (current_idx + 1) / chunk_size - 1;
      out_states[c] = (u16)state_value;

      // We just finished scanning Chunk c+1.
      // So chunk_current_bits belongs to Chunk c+1.
      // Bounds check not strictly needed if logic is correct, but safe:
      if (bit_counts != nullptr) { // Allow optional nullptr for tests
        // Map c+1 -> bit_counts index
        // bit_counts should be sized for num_chunks
        u32 max_chunks = (num_symbols + chunk_size - 1) / chunk_size;
        if (c + 1 < max_chunks) {
          bit_counts[c + 1] = chunk_current_bits;
        }
      }
      chunk_current_bits = 0; // Reset for Chunk c

      if (c == 0)
        break; // We found the start state for Chunk 0, no need to scan inside
               // Chunk 0
    }

    symbol = *ip--;
    current_idx--;

    // Transition
    GPU_FSE_SymbolTransform const sTT = symbolTT[symbol];
    u32 const nbBits = (u32)((state_value + sTT.deltaNbBits) >> 16);
    state_value = stateTable[(state_value >> nbBits) + sTT.deltaFindState];
    chunk_current_bits += nbBits;
  }

  // Remaining bits belong to Chunk 0
  if (bit_counts != nullptr) {
    bit_counts[0] =
        chunk_current_bits + tableLog; // Add Flush bits (tableLog) to Chunk 0
  }
}

// ==============================================================================
// 2. PARALLEL CHUNK ENCODING KERNEL
// ==============================================================================

__global__ void fse_encode_chunk_kernel(
    const byte_t *symbols, u32 num_symbols, const u16 *stateTable,
    const GPU_FSE_SymbolTransform *symbolTT, u16 tableLog,
    const u16 *start_states, // From pre-pass
    u32 chunk_size,
    byte_t *output_buffer, // Large global buffer (partitioned per chunk)
    u32 *chunk_offsets,    // Output: Size (in bytes, padded) used by each chunk
    u32 buffer_stride      // Stride between chunk buffers
) {
  u32 chunk_idx = threadIdx.x;

  u32 total_chunks = (num_symbols + chunk_size - 1) / chunk_size;
  if (chunk_idx >= total_chunks)
    return;

  u32 start_idx = chunk_idx * chunk_size;
  u32 end_idx = min(start_idx + chunk_size, num_symbols);

  byte_t *d_out = output_buffer + chunk_idx * buffer_stride;
  u32 max_ex = buffer_stride;

  GPU_BitStream bitC;
  if (gpu_bit_init_stream(&bitC, d_out, max_ex) != 0) {
    chunk_offsets[chunk_idx] = 0;
    return;
  }

  GPU_FSE_CState CState;
  CState.stateLog = tableLog;
  CState.stateTable = stateTable;
  CState.symbolTT = symbolTT;

  const byte_t *ip = symbols + end_idx - 1;
  const byte_t *chunk_begin = symbols + start_idx;

  // --- State Initialization ---
  // If Last Chunk (in original stream order), we do Standard ZSTD Init
  if (chunk_idx == total_chunks - 1) {
    CState.value = (u64)1 << tableLog;
    u32 symbol = *ip--;
    GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
    u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
    u64 tempValue = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
    u32 tableIndex = (tempValue >> nbBitsOut) + symbolTransform.deltaFindState;
    CState.value = stateTable[tableIndex];
  } else {
    // Use Pre-computed State
    CState.value = start_states[chunk_idx];
  }

  // --- Backward Encoding Loop ---
  while (ip >= chunk_begin) {
    gpu_fse_encode_symbol(&bitC, &CState, *ip--);
    gpu_bit_flush_bits(&bitC);
  }

  // --- Flush State ---
  // Only Chunk 0 flushes the final state bits (End of Stream)
  if (chunk_idx == 0) {
    gpu_fse_flush_state(&bitC, &CState);
  } else {
    // For other chunks, we flush bits remaining in container
    // Note: This pads to next byte.
    // Bitstream Stitching Logic will handle desync or shifting.
    gpu_bit_flush_bits(&bitC);
    // If bits remaining (<8), we force flush?
    if (bitC.bitPos > 0) {
      if (bitC.ptr < bitC.endPtr) {
        *bitC.ptr = (byte_t)bitC.bitContainer;
        bitC.ptr++;
      }
    }
  }

  chunk_offsets[chunk_idx] = (u32)(bitC.ptr - bitC.startPtr);
}

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_CHUNK_KERNEL_CUH
