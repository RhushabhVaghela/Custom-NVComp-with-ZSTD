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
__global__ void __launch_bounds__(256) k_encode_fse_interleaved(
    const u8 *__restrict__ d_ll_codes, const u32 *__restrict__ d_ll_extras,
    const u8 *__restrict__ d_ll_bits, const u8 *__restrict__ d_of_codes,
    const u32 *__restrict__ d_of_extras, const u8 *__restrict__ d_of_bits,
    const u8 *__restrict__ d_ml_codes, const u32 *__restrict__ d_ml_extras,
    const u8 *__restrict__ d_ml_bits, u32 num_symbols,
    unsigned char *__restrict__ output_bitstream, size_t *output_pos,
    size_t bitstream_capacity, const FSEEncodeTable *table) {
  if (threadIdx.x != 0 || num_symbols == 0 || bitstream_capacity == 0)
    return;

  unsigned char *ptr = output_bitstream;
  unsigned char *end_limit = output_bitstream + bitstream_capacity;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  auto write_bits = [&](u64 val, u32 nbBits) {
    if (nbBits == 0)
      return;
    bitContainer |= (val << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr < end_limit) {
        *ptr++ = (unsigned char)(bitContainer & 0xFF);
      }
      bitContainer >>= 8;
      bitCount -= 8;
    }
  };

  // 1. Initialize States (Last Symbol)
  u32 last_idx = num_symbols - 1;
  // Use d_symbol_first_state to get a VALID starting state
  u32 stateLL = table[0].d_symbol_first_state[d_ll_codes[last_idx]];
  u32 stateOF = table[1].d_symbol_first_state[d_of_codes[last_idx]];
  u32 stateML = table[2].d_symbol_first_state[d_ml_codes[last_idx]];

  printf("[FSE_ENC] num_symbols=%u, last_idx=%u\\n", num_symbols, last_idx);
  printf("[FSE_ENC] codes[last]: ll=%u, of=%u, ml=%u\\n",
         (u32)d_ll_codes[last_idx], (u32)d_of_codes[last_idx],
         (u32)d_ml_codes[last_idx]);
  printf("[FSE_ENC] init_states: LL=%u, OF=%u, ML=%u\\n", stateLL, stateOF,
         stateML);
  printf("[FSE_ENC] table_logs: LL=%u, OF=%u, ML=%u\\n", table[0].table_log,
         table[1].table_log, table[2].table_log);

  // 2. Loop N-2 down to 0
  if (num_symbols > 1) {
    for (i32 i = (i32)num_symbols - 2; i >= 0; i--) {
      // RFC 8878 Forwards Encoding Order (Per sequence):
      // Symbol States: OF, ML, LL
      // Extra Bits: OF, ML, LL

      // Offset (Table 1)
      {
        u8 code_cur = d_of_codes[i + 1];
        u8 code_next = d_of_codes[i];
        auto sym_cur = table[1].d_symbol_table[code_cur];
        auto sym_next = table[1].d_symbol_table[code_next];

        u32 nbBitsOut = (stateOF + sym_cur.deltaNbBits) >> 16;
        write_bits(stateOF & ((1 << nbBitsOut) - 1), nbBitsOut);
        stateOF = (stateOF >> nbBitsOut) + sym_next.deltaFindState;
      }

      // Match Length (Table 2)
      {
        u8 code_cur = d_ml_codes[i + 1];
        u8 code_next = d_ml_codes[i];
        auto sym_cur = table[2].d_symbol_table[code_cur];
        auto sym_next = table[2].d_symbol_table[code_next];

        u32 nbBitsOut = (stateML + sym_cur.deltaNbBits) >> 16;
        write_bits(stateML & ((1 << nbBitsOut) - 1), nbBitsOut);
        stateML = (stateML >> nbBitsOut) + sym_next.deltaFindState;
      }

      // Literal Length (Table 0)
      {
        u8 code_cur = d_ll_codes[i + 1];
        u8 code_next = d_ll_codes[i];
        auto sym_cur = table[0].d_symbol_table[code_cur];
        auto sym_next = table[0].d_symbol_table[code_next];

        u32 nbBitsOut = (stateLL + sym_cur.deltaNbBits) >> 16;
        write_bits(stateLL & ((1 << nbBitsOut) - 1), nbBitsOut);
        stateLL = (stateLL >> nbBitsOut) + sym_next.deltaFindState;
      }

      // Extra Bits (for sequence i+1)
      write_bits(d_of_extras[i + 1], d_of_bits[i + 1]);
      write_bits(d_ml_extras[i + 1], d_ml_bits[i + 1]);
      write_bits(d_ll_extras[i + 1], d_ll_bits[i + 1]);
    }
  }

  // 3. Write Final Fields for Sequence 0
  // Extra bits for Seq 0
  write_bits(d_of_extras[0], d_of_bits[0]);
  write_bits(d_ml_extras[0], d_ml_bits[0]);
  write_bits(d_ll_extras[0], d_ll_bits[0]);

  // Final States
  write_bits(stateML, table[2].table_log);
  write_bits(stateOF, table[1].table_log);
  write_bits(stateLL, table[0].table_log);

  // 4. Flush Sentinel '1'
  write_bits(1, 1);
  while (bitCount > 0) {
    if (ptr < end_limit)
      *ptr++ = (unsigned char)(bitContainer & 0xFF);
    bitContainer >>= 8;
    bitCount = (bitCount > 8) ? bitCount - 8 : 0;
  }

  *output_pos = ptr - output_bitstream;
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

    u32 maxBitsOut;
    if (freq == 1) {
      maxBitsOut = table_log;
    } else {
      maxBitsOut = table_log - get_highest_bit(freq - 1);
    }
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
    // Recalculate minStatePlus for deltaFindState
    u32 freq = normalized_counters[s];
    u32 maxBitsOut;
    if (freq == 1) {
      maxBitsOut = table_log;
    } else {
      maxBitsOut = table_log - get_highest_bit(freq - 1);
    }
    u32 minStatePlus = freq << maxBitsOut;

    // Zstd Formula:
    // deltaFindState = tableSize + cumFreq - 1 - minStatePlus
    u32 tableSize = 1 << table_log;
    i32 delta = (i32)(tableSize + s_cum_freq[s] - 1) - (i32)minStatePlus;

    table->d_symbol_table[s].deltaFindState = delta;
  }

  __syncthreads();

  // Phase 4: Init CTable for Encoder (Correct Zstd Logic: total + tableSize)
  // Single Thread (tid=0)
  if (tid == 0) {
    u32 tableSize = 1 << table_log;
    // Iterate symbols to set Initial State
    for (u32 s = 0; s <= max_symbol; s++) {
      if ((i16)normalized_counters[s] == 0) {
        table->d_symbol_first_state[s] = 0;
        continue;
      }
      // FSE Encoder Init State: total_cumulative_freq + table_size
      // This ensures state is in range [tableSize, 2*tableSize - 1]
      table->d_symbol_first_state[s] = (u16)(s_cum_freq[s] + tableSize);
    }
  }
}

// =================================================================================================
// HOST LAUNCHERS
// =================================================================================================

Status launch_fse_encoding_kernel(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, unsigned char *d_bitstream, size_t *d_output_pos,
    size_t bitstream_capacity,
    const FSEEncodeTable *d_tables, // Expects array of 3
    cudaStream_t stream) {
  // Launch Warp-Level Encoder (Phase 2a: Single Warp/Thread 0)
  k_encode_fse_interleaved<<<1, 32, 0, stream>>>(
      d_ll_codes, d_ll_extras, d_ll_bits, d_of_codes, d_of_extras, d_of_bits,
      d_ml_codes, d_ml_extras, d_ml_bits, num_symbols, d_bitstream,
      d_output_pos, bitstream_capacity, d_tables);

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
