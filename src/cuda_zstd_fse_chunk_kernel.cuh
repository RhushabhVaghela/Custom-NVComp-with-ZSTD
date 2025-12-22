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

static __global__ void fse_compute_states_kernel(
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

  u64 state = 1ULL << tableLog; // Base state
  GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
  u32 nbBitsOut = ((u32)state + symbolTransform.deltaNbBits) >> 16;
  u64 tempValue = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
  u32 tableIndex = (tempValue >> nbBitsOut) + symbolTransform.deltaFindState;
  state_value = stateTable[tableIndex];

  // Last Symbol bits belong to the LAST chunk
  chunk_current_bits += nbBitsOut;

  // (Fix) Write start state for the Last Chunk
  u32 max_chunks = (num_symbols + chunk_size - 1) / chunk_size;
  if (out_states && max_chunks > 0) {
    out_states[max_chunks - 1] = (u16)state_value;
  }

  // --- State Transition Scan Backwards ---
  // Scan from N-2 down to 0
  // loop runs as long as current_idx >= 0, but current_idx is u32
  while (true) {
    // Boundary Logic: Check if we just crossed a boundary
    // We are at `current_idx`. We just encoded `current_idx + 1`.
    if ((current_idx + 1) % chunk_size == 0) {
      u32 c = (current_idx + 1) / chunk_size - 1;

      // Save state at boundary (Start state for Chunk c+1? No, Chunk c?
      // Matches previous logic: c=8 writes out_states[8]. (Chunk 8).
      // Wait, consistent logic:
      // If we are at boundary 90 (Chunk 9 starts).
      // c=8. out_states[8] is start of Chunk 8?
      // out_states[c] -> Chunk c start state.
      // At boundary 90, we have state required for 89 (end of Chunk 8).
      // Wait, is state_value "state required for current_idx" or "state after
      // current_idx"? Logic: state_value updated at bottom of loop. Represents
      // state BEFORE encoding symbol at `current_idx`. At top of loop (before
      // update), state_value is state BEFORE encoding `current_idx+1`. At
      // boundary 99 (current_idx=98). (98+1)%10 != 0. At boundary 90
      // (current_idx=89). (89+1)%10 == 0. c=8. state_value is state BEFORE
      // encoding 90. 90 is start of Chunk 9. So state_value IS start state of
      // Chunk 9. So we should write out_states[9]. c+1 = 9. So out_states[c+1].

      // PREVIOUS LOGIC was: out_states[c] = state_value.
      // This implied out_states[8] gets start state of Chunk 9??
      // Or c calculation was different?
      // c = (90)/10 - 1 = 8.
      // We wrote out_states[8].
      // If out_states[8] is for Chunk 8. And it gets Start State of Chunk 9.
      // Then Chunk 8 starts with state of Chunk 9?
      // Chunk 8 (80..89) needs state at 89?
      // Reverse encoding: Chunk 8 processes 89..80.
      // Starts with state at 90 (boundary).
      // Ends with state at 80.
      // So Chunk 8 NEEDS state at 90.
      // So out_states[8] should hold state at 90.
      // Logic `out_states[c] = state_value` (state at 90) is CORRECT for
      // Chunk 8.
      out_states[c] = (u16)state_value;

      if (bit_counts != nullptr) {
        if (c + 1 < max_chunks) {
          bit_counts[c + 1] = chunk_current_bits;
        }
      }
      chunk_current_bits = 0; // Reset for Chunk c

      // (Fix) Removed early break for c==0 to scan Chunk 0
    }

    symbol = *ip--;

    // Transition
    GPU_FSE_SymbolTransform const sTT = symbolTT[symbol];
    u32 const nbBits = ((u32)state_value + sTT.deltaNbBits) >> 16;
    state_value = stateTable[(state_value >> nbBits) + sTT.deltaFindState];
    chunk_current_bits += nbBits;

    // Check termination
    if (current_idx == 0)
      break;
    current_idx--;
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

static __global__ void fse_encode_chunk_kernel(
    const byte_t *symbols, u32 num_symbols, const u16 *stateTable,
    const GPU_FSE_SymbolTransform *symbolTT, u16 tableLog,
    const u16 *start_states, // From pre-pass
    u32 chunk_size,
    byte_t *output_buffer, // Large global buffer (partitioned per chunk)
    u32 *chunk_offsets,    // Output: Size (in bytes, padded) used by each chunk
    u32 buffer_stride      // Stride between chunk buffers
) {
  u32 chunk_idx = blockIdx.x; // (FIX) Use block index, not thread index

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
    u32 nbBitsOut = ((u32)CState.value + symbolTransform.deltaNbBits) >> 16;
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

namespace cuda_zstd {
namespace fse {

// Helper for Byte Atomic OR (Simulated using 32-bit Atomic OR)
__device__ inline void atomicOr_u8(byte_t *address, byte_t val) {
  unsigned int *base_addr = (unsigned int *)((size_t)address & ~3);
  unsigned int offset = (size_t)address & 3;
  unsigned int shift = offset * 8;
  unsigned int val_u32 = (unsigned int)val << shift;

  atomicOr(base_addr, val_u32);
}

// ==============================================================================
// 3. PARALLEL BITSTREAM MERGE KERNEL
// ==============================================================================

static __global__ void fse_merge_bitstreams_kernel(
    const byte_t *input_buffers, // input chunks (stride separated)
    const u32 *chunk_bit_counts, // Exact bit size of each chunk
    const u32 *chunk_bit_starts, // Calculated prefix sum (in bits)
    byte_t *output_payload,      // Destination buffer
    u32 num_chunks, u32 buffer_stride) {
  u32 chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_idx >= num_chunks)
    return;

  u32 num_bits = chunk_bit_counts[chunk_idx];
  if (num_bits == 0)
    return; // Empty chunk

  u32 global_start_bit = chunk_bit_starts[chunk_idx];
  const byte_t *src_ptr = input_buffers + chunk_idx * buffer_stride;

  u32 start_byte = global_start_bit / 8;
  u32 bit_shift = global_start_bit % 8;

  u32 num_bytes = (num_bits + 7) / 8;

  // Simple Byte Loop
  for (u32 i = 0; i < num_bytes; i++) {
    byte_t val = src_ptr[i];

    // Mask garbage bits for last byte
    if (i == num_bytes - 1) {
      u32 remainder = num_bits % 8;
      if (remainder != 0) {
        val &= ((1 << remainder) - 1);
      }
    }

    u32 dest_idx = start_byte + i;

    // Shifted values
    byte_t val1 = (val << bit_shift);
    byte_t val2 = (val >> (8 - bit_shift));

    // Calculate global end bit to determine boundary/ownership
    u32 global_end_bit = global_start_bit + num_bits;

    // Write 1 (dest_idx)
    // Shared if this byte corresponds to the start boundary (i=0)
    // OR if it corresponds to the end boundary (shared with next chunk)
    // NOTE: dest_idx is ALWAYS == (global_start_bit / 8) + i

    bool needs_atomic = false;

    // Start Boundary condition: always atomic unless bit_shift==0 && i==0?
    // Actually, if bit_shift != 0, byte [start_byte] is shared with prev.
    if (i == 0 && bit_shift != 0)
      needs_atomic = true;

    // End Boundary condition:
    // If dest_idx == (global_end_bit / 8), it is shared with next chunk.
    if (dest_idx == (global_end_bit / 8))
      needs_atomic = true;

    if (needs_atomic) {
      atomicOr_u8(output_payload + dest_idx, val1);
    } else {
      // Safe exclusive write. We use |= just in case, or = if buffer is zeroed.
      // Assuming buffer is zeroed.
      output_payload[dest_idx] |= val1;
    }

    // Write 2 (dest_idx+1) - Only if split occurs
    if (bit_shift > 0) {
      u32 next_byte_idx = dest_idx + 1;
      bool needs_atomic2 = false;

      // Next byte is shared with next chunk if it's the boundary
      if (next_byte_idx == (global_end_bit / 8))
        needs_atomic2 = true;
      // Also if next_byte_idx is the start of next-next chunk? (Unlikely)

      if (needs_atomic2) {
        atomicOr_u8(output_payload + next_byte_idx, val2);
      } else {
        output_payload[next_byte_idx] |= val2;
      }
    }
  }
}

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_CHUNK_KERNEL_CUH
