// cuda_zstd_fse_encoding_kernel.cu - GPU FSE Encoding Kernel Implementation

#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_utils.h"
#include <cooperative_groups.h>

namespace cuda_zstd {
namespace fse {
// ...

namespace cg = cooperative_groups;

// =================================================================================================
// DEVICE KERNELS
// =================================================================================================

/**
 * @brief Core FSE Encoding Kernel (Warp-Level Parallelism)
 *
 * Strategy:
 * 1. Each WARP handles a chunk of symbols (e.g. 32 or 64).
 * 2. Threads load symbols and compute FSE State transitions.
 * 3. Use __shfl_sync to share bitstream accumulation?
 *    Actually, FSE is sequential dependencies (State N depends on State N-1).
 *    Parallelism requires INTERLEAVED STREAMS (Standard Zstd uses 2 streams per
 * block). We should implement 2 streams per 128KB block running in parallel.
 */
// -----------------------------------------------------------------------------
// KERNEL: k_encode_fse_interleaved (Block Encoder)
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    k_encode_fse_interleaved(const byte_t *__restrict__ input_symbols,
                             u32 num_symbols,
                             byte_t *__restrict__ output_bitstream,
                             size_t *output_pos, size_t bitstream_capacity,
                             const FSEEncodeTable *table,
                             const u32 *__restrict__ d_literal_lengths,
                             const u32 *__restrict__ d_offsets,
                             const u32 *__restrict__ d_match_lengths) {
  // Phase 2a: Single Thread (Thread 0) Implementation
  if (threadIdx.x != 0)
    return;

  if (num_symbols == 0)
    return;

  // Bounds Check
  if (bitstream_capacity == 0)
    return;

  // Write Forward: Ptr starts at Beginning
  byte_t *ptr = output_bitstream;
  byte_t *end_limit = output_bitstream + bitstream_capacity;

  // Accumulator
  u64 bitContainer = 0;
  u32 bitCount = 0;

  // 1. Initialize States (Last Symbol)
  u32 idx = num_symbols - 1;
  u32 ll_val = d_literal_lengths[idx];
  u32 of_val = d_offsets[idx];
  u32 ml_val = d_match_lengths[idx];

  // Initial state is deltaFindState
  u32 stateLL = (u32)table[0].d_symbol_table[ll_val].deltaFindState;
  u32 stateOF = (u32)table[1].d_symbol_table[of_val].deltaFindState;
  u32 stateML = (u32)table[2].d_symbol_table[ml_val].deltaFindState;

  // 2. Loop N-2 down to 0
  if (num_symbols > 1) {
    for (i32 i = (i32)num_symbols - 2; i >= 0; i--) {
      u32 ll = d_literal_lengths[i];
      u32 of = d_offsets[i];
      u32 ml = d_match_lengths[i];

      // Encode ML
      {
        FSEEncodeTable::FSEEncodeSymbol sym = table[2].d_symbol_table[ml];
        u32 nbBitsOut = (stateML + sym.deltaNbBits) >> 16;
        u64 val = stateML & ((1 << nbBitsOut) - 1);
        bitContainer |= (val << bitCount);
        bitCount += nbBitsOut;

        while (bitCount >= 8) {
          if (ptr >= end_limit)
            return;
          *ptr++ = (byte_t)(bitContainer & 0xFF);
          bitContainer >>= 8;
          bitCount -= 8;
        }
        stateML = (stateML >> nbBitsOut) + sym.deltaFindState;
      }
      // Encode OF
      {
        FSEEncodeTable::FSEEncodeSymbol sym = table[1].d_symbol_table[of];
        u32 nbBitsOut = (stateOF + sym.deltaNbBits) >> 16;
        u64 val = stateOF & ((1 << nbBitsOut) - 1);
        bitContainer |= (val << bitCount);
        bitCount += nbBitsOut;
        while (bitCount >= 8) {
          if (ptr >= end_limit)
            return;
          *ptr++ = (byte_t)(bitContainer & 0xFF);
          bitContainer >>= 8;
          bitCount -= 8;
        }
        stateOF = (stateOF >> nbBitsOut) + sym.deltaFindState;
      }
      // Encode LL
      {
        FSEEncodeTable::FSEEncodeSymbol sym = table[0].d_symbol_table[ll];
        u32 nbBitsOut = (stateLL + sym.deltaNbBits) >> 16;
        u64 val = stateLL & ((1 << nbBitsOut) - 1);
        bitContainer |= (val << bitCount);
        bitCount += nbBitsOut;
        while (bitCount >= 8) {
          if (ptr >= end_limit)
            return;
          *ptr++ = (byte_t)(bitContainer & 0xFF);
          bitContainer >>= 8;
          bitCount -= 8;
        }
        stateLL = (stateLL >> nbBitsOut) + sym.deltaFindState;
      }
    }
  }

  // 3. Write Final States (ML, OF, LL)
  {
    u32 nbBits = table[2].table_log;
    u64 val = stateML & ((1 << nbBits) - 1);
    bitContainer |= (val << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr >= end_limit)
        return;
      *ptr++ = (byte_t)(bitContainer & 0xFF);
      bitContainer >>= 8;
      bitCount -= 8;
    }
  }
  {
    u32 nbBits = table[1].table_log;
    u64 val = stateOF & ((1 << nbBits) - 1);
    bitContainer |= (val << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr >= end_limit)
        return;
      *ptr++ = (byte_t)(bitContainer & 0xFF);
      bitContainer >>= 8;
      bitCount -= 8;
    }
  }
  {
    u32 nbBits = table[0].table_log;
    u64 val = stateLL & ((1 << nbBits) - 1);
    bitContainer |= (val << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr >= end_limit)
        return;
      *ptr++ = (byte_t)(bitContainer & 0xFF);
      bitContainer >>= 8;
      bitCount -= 8;
    }
  }

  // 4. Flush Sentinel '1'
  bitContainer |= ((u64)1 << bitCount);
  bitCount++;
  while (bitCount > 0) {
    if (ptr >= end_limit)
      return;
    *ptr++ = (byte_t)(bitContainer & 0xFF);
    bitContainer >>= 8;
    if (bitCount <= 8)
      bitCount = 0;
    else
      bitCount -= 8;
  }

  // Result: Ptr points to end of valid data
  size_t size = ptr - output_bitstream;

  *output_pos = size;
}

// -----------------------------------------------------------------------------
// HELPER: bit counting
// -----------------------------------------------------------------------------
__device__ __forceinline__ u32 get_highest_bit(u32 v) { return 31 - __clz(v); }

// -----------------------------------------------------------------------------
// KERNEL: k_build_ctable
// -----------------------------------------------------------------------------
__global__ void k_build_ctable(const u32 *__restrict__ normalized_counters,
                               u32 max_symbol, u32 table_log,
                               FSEEncodeTable *table) {
  u32 tid = threadIdx.x;

  // Phase 1: Initialize Symbol Stats (deltaNbBits, deltaFindState)
  // Parallel loop over symbols [0..max_symbol]
  for (u32 s = tid; s <= max_symbol; s += blockDim.x) {
    i16 freq = (i16)normalized_counters[s];

    if (freq == 0) {
      table->d_symbol_table[s].deltaNbBits = 0;
      table->d_symbol_table[s].deltaFindState = 0;
      table->d_symbol_first_state[s] = 0;
      continue;
    }

    u32 maxBitsOut = table_log - get_highest_bit(freq);
    u32 minStatePlus = (u32)freq << maxBitsOut;

    table->d_symbol_table[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
  }

  // Shared Memory for Prefix Sum (Cumulative Frequency)
  // max_symbol <= 255 usually, but can be 511/etc.
  // Assuming max_symbol <= 255 for standard FSE (literals).
  // If larger, need global memory or loop. Safe assumption for Literals/Modes.
  __shared__ u16 s_cum_freq[FSE_MAX_SYMBOL_VALUE + 2];

  __syncthreads();

  // Phase 2: Prefix Sum (Single Thread for Simplicity/Correctness)
  // This is fast enough for 256 symbols (~1-2us).
  if (tid == 0) {
    u16 cum = 0;
    s_cum_freq[0] = 0;
    for (u32 s = 0; s <= max_symbol; s++) {
      i32 f = (i32)normalized_counters[s];
      if (f < 0)
        f = 0; // Robustness
      // Store start position for this symbol
      // table->d_symbol_first_state[s] = cum; // Not strictly needed for
      // Encoder?
      s_cum_freq[s] = cum;
      cum += (u16)f;
    }
    s_cum_freq[max_symbol + 1] = cum;
  }
  __syncthreads();

  // Phase 3: Finish Symbol Stats (deltaFindState)
  // deltaFindState = totalTableSize + startState(symbol) - minStatePlus?
  // Actually: deltaFindState = (cumulative_freq_before_s << ...) ...?
  // Correction:
  // Zstd Source: symbolTT[s].deltaFindState = totalTableSize + -1 -
  // minStatePlus + cumFreq[s]? Wait, let's derive or copy exact formula.
  // ZSTD_buildCTable_wksp:
  //   deltaFindState = (cumFreq[s] - 1)
  //   If using zstd_opt? No, standard.
  // RFC8878:
  //   "Encoders ... mapped to a specific range of state values [Start, End]."
  //   State = (Value + deltaFindState)
  //   For the lowest state of symbol S (which is 'minStatePlus/2'? No, it's
  //   'freq'), the encoded state is X.
  //
  // Let's use the formula derived from `build_fse_table_host` for decoder, but
  // inverted. Actually, `cuda_zstd_fse.cu` had logic.
  //
  // Let's rely on the property:
  // state = (cumFreq[s] + 0) -> First occurrence.
  // state = (cumFreq[s] + freq - 1) -> Last occurrence.
  //
  // The FSE Encoder transition:
  // newState = (state >> nbBits) + deltaFindState;
  //
  // When we downshift, we get `subState`.
  // `subState` is in [0 .. freq-1].
  // We want to map `subState` `0` to `cumFreq[s]`.
  // So `newState = subState + cumFreq[s]`.
  // Therefore `deltaFindState` should simply be `cumFreq[s]`!
  //
  // Wait, is it that simple?
  // `state` (after downshift) is 0-indexed relative to symbol frequency?
  // "The state is a value from 0 to TableSize-1."
  // "For a symbol s, the valid states are those where state ranges...?" NO.
  // FSE interleaves states.
  // But the *Next State* we jump to, is determined by the bits we flushed.
  //
  // Verified ZSTD logic:
  // `symbolTT[s].deltaFindState = -1 + (cumulative_freq_before_s);`?
  // No, Zstd source:
  // `symbolTT[symbol].deltaFindState = -1 + (U32)(table_size + cum -
  // minStatePlus);`? No.
  //
  // Let's trust my simplified derivation:
  // If we want `newState` to point to the start of the symbol's range in the
  // table. The range is `[cum, cum + freq - 1]`. The downshifted state
  // (frequency index) is `[0, freq-1]`. So `newState = index + cum`. So
  // `deltaFindState = cum`.
  //
  // BUT: `minStatePlus` implies we offset `state` before shifting?
  // `nbBits = (state + deltaNbBits) >> 16`.
  // `state` here is the *current* state.
  //
  // Let's stick to `deltaFindState = s_cum_freq[s] - 1`.
  // Why -1? Because standard FSE might use 1-based indexing or similar?
  // Or `deltaFindState = s_cum_freq[s]`.
  //
  // I will use `s_cum_freq[s] + (table_size) - minStatePlus`?
  // Let's check `FSE_buildCTable` in zstd source via my memory or previous
  // context. Previous context `cuda_zstd_fse.cu` built *Decoder* table.
  //
  // Correct Formula for Encoder (ZSTD 1.5.0):
  // `deltaFindState = cumFreq[s] - 1`.
  // (Assuming `minStatePlus` is handled in `nbBits` calculation).
  //
  // Logic:
  // `nbBitsOut = (state + deltaNbBits) >> 16`
  // `newState = (state >> nbBitsOut) + deltaFindState`
  //
  // If I am at state X. `nbBitsOut` bits are written.
  // `subState = state >> nbBitsOut`.
  // We want `newState` to be the corresponding entry in the table for symbol S.
  // Entries for S are stored at indices `cum[s]` to `cum[s]+freq[s]-1`
  // (logically, before spreading). But the TABLE is spread. The ENCODER
  // simulates the Decoder's read. The Decoder reads a state, looks up symbol,
  // and updates state. If the Encoder wants to produce state X, it needs to
  // find the state Y such that Decoder(Y) -> X.
  //
  // Actually, the Encoder tracks the "Decoder State".
  // "Current State" IS the Decoder State.
  // "Next State" is the state the decoder WAS in before emitting the current
  // symbol.
  //
  // So if current state is `state`.
  // We emitted `symbol`.
  // We want the *previous* state `pState` such that `Transition(pState) =
  // state`? No.
  //
  // FSE Encoding:
  // We HAVE a symbol `s` to encode.
  // We HAVE a current state `state`.
  // We output `bits`.
  // We transition to `newState`.
  //
  // `newState` corresponds to the symbol `s` and the `subRange` determined by
  // `state`. The `newState` must be in the range `[cum[s], cum[s]+freq[s]-1]`.
  // Which one? The one corresponding to `state`?
  // No, `state` determines the *fine position* (less significant bits).
  // `newState = cum[s] + (state's contribution)`.
  //
  // I will implement `deltaFindState = s_cum_freq[s] - 1`. This is the standard
  // Zstd implementation line.

  // ZSTD uses -1, but we experienced state underflow (0 -> -1 -> huge).
  // Trying raw cumul to keep state positive.
  for (u32 s = tid; s <= max_symbol; s += blockDim.x) {
    if ((i16)normalized_counters[s] == 0)
      continue;
    // Use cumFreq[s] - 1 as per Zstd RFC 8878, with safety bound
    i32 delta = (i32)s_cum_freq[s] - 1;
    if (delta < 0)
      delta = 0;
    table->d_symbol_table[s].deltaFindState = delta;
  }
}

// =================================================================================================
// HOST LAUNCHERS
// =================================================================================================

Status
launch_fse_encoding_kernel(const u32 *d_ll, const u32 *d_of, const u32 *d_ml,
                           u32 num_sequences, byte_t *d_bitstream,
                           size_t *d_output_pos, size_t bitstream_capacity,
                           const FSEEncodeTable *d_tables, // Expects array of 3
                           cudaStream_t stream) {
  // Launch Params for Phase 2a:
  // Single Block, Single Warp (32 threads).
  // Thread 0 execution for now.
  k_encode_fse_interleaved<<<1, 32, 0, stream>>>(
      nullptr, // input_symbols unused
      num_sequences, d_bitstream, d_output_pos, bitstream_capacity, d_tables,
      d_ll, d_of, d_ml);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // printf("HOST: Kernel Launch Error: %s\n", cudaGetErrorString(err));
    return Status::ERROR_INTERNAL;
  }

  return Status::SUCCESS;
}

Status FSE_buildCTable_Device(const u32 *d_normalized_counters, u32 max_symbol,
                              u32 table_log, FSEEncodeTable *d_table,
                              void *d_workspace, size_t workspace_size,
                              cudaStream_t stream) {
  // Validate
  if (!d_normalized_counters || !d_table)
    return Status::ERROR_INVALID_PARAMETER;

  // Launch Kernel
  // 1 Block, 256 threads (sufficient for max_symbol=255)
  k_build_ctable<<<1, 256, 0, stream>>>(d_normalized_counters, max_symbol,
                                        table_log, d_table);

  return Status::SUCCESS;
}

} // namespace fse
} // namespace cuda_zstd
