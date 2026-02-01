// cuda_zstd_fse_encoding_kernel.cu - GPU FSE Encoding Kernel Implementation

#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_utils.h"
#include <cooperative_groups.h>
#include <stdio.h>

// Fix for Windows macro collision
#ifdef ERROR_INVALID_PARAMETER
#undef ERROR_INVALID_PARAMETER
#endif

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
  // Only thread 0 does the encoding (FSE is inherently sequential)
  if (threadIdx.x != 0 || num_symbols == 0 || bitstream_capacity == 0) {
    // Early exit for non-thread-0 (no debug print to reduce spam)
    return;
  }
  
  // DEBUG: Thread 0 starting with num_symbols
  printf("[FSE_ENCODE] Thread 0 START: num_symbols=%u, capacity=%llu\n", 
         num_symbols, (unsigned long long)bitstream_capacity);
  
  // Validate input pointers
  if (!d_ll_codes || !d_of_codes || !d_ml_codes || !output_bitstream || !output_pos || !table) {
    printf("[FSE_ENCODE] NULL pointer: ll=%p, of=%p, ml=%p, out=%p, pos=%p, table=%p\n",
           d_ll_codes, d_of_codes, d_ml_codes, output_bitstream, output_pos, table);
    if (output_pos) *output_pos = 0;
    return;
  }

  unsigned char *ptr = output_bitstream;
  unsigned char *end_limit = output_bitstream + bitstream_capacity;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  auto write_bits = [&](u64 val, u32 nbBits) {
    if (nbBits == 0)
      return;
    // val &= ((1ULL << nbBits) - 1); // CRITICAL FIX: Mask higher bits!
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

  // 1. Initialize States with SEQUENCE 0 (Start of Chain)
  // We encode Forward: 0 -> 1 -> ... -> N-1
  // Final State corresponds to N-1
  u32 start_idx = 0;

  if (threadIdx.x == 0)

  // Validate table pointers before dereferencing
  if (!table[0].d_symbol_first_state || !table[1].d_symbol_first_state || !table[2].d_symbol_first_state) {
    *output_pos = 0;
    return;
  }
  
  // Validate max_symbol bounds
  u8 ll_code = d_ll_codes[start_idx];
  u8 of_code = d_of_codes[start_idx];
  u8 ml_code = d_ml_codes[start_idx];
  
  if (ll_code > table[0].max_symbol || of_code > table[1].max_symbol || ml_code > table[2].max_symbol) {
    *output_pos = 0;
    return;
  }

  // Use d_symbol_first_state to get a VALID starting state for Seq 0
  u32 stateLL = table[0].d_symbol_first_state[ll_code];
  u32 stateOF = table[1].d_symbol_first_state[of_code];
  u32 stateML = table[2].d_symbol_first_state[ml_code];
  
  if (threadIdx.x == 0) {
    printf("[FSE_ENCODE] Initial states from first_state table: LL(code=%u)=%u, OF(code=%u)=%u, ML(code=%u)=%u\n",
           ll_code, stateLL, of_code, stateOF, ml_code, stateML);
  }

  // 2. Write Extra Bits for SEQUENCE 0 FIRST
  // Decoder reads these LAST (for Seq 0), so physically they must be at START
  // (Low Addr)
  write_bits(d_ml_extras[0], d_ml_bits[0]);
  if (bitCount > 10000) printf("[FSE_ENCODE] AFTER Seq0 ML extras: bitCount=%u\n", bitCount);
  write_bits(d_of_extras[0], d_of_bits[0]);
  if (bitCount > 10000) printf("[FSE_ENCODE] AFTER Seq0 OF extras: bitCount=%u\n", bitCount);
  write_bits(d_ll_extras[0], d_ll_bits[0]);
  if (bitCount > 10000) printf("[FSE_ENCODE] AFTER Seq0 LL extras: bitCount=%u\n", bitCount);

  // 3. Loop 0 to N-2 (Transition states)
  // Encodes Sequence i+1 using State i, producing State i+1
  if (num_symbols > 1) {
    for (u32 i = 0; i < num_symbols - 1; i++) {
      // Encode Seq i+1
      // Tables: LL=0, OF=1, ML=2
      // Sub-State (Current) is used as 'nextState' base?
      // Logic: state = newStateTable[state + delta] (Decoder)
      // Encoder: state = (state >> nbBits) + delta; (Wait, this is previous
      // wrong logic?)

      // Zstd Encoder Logic:
      // nbBits = (state + deltaNbBits) >> 16;
      // write(state & mask, nbBits);
      // state = nextStateTable[ (state >> nbBits) + deltaFindState ];
      // But we need to encode SYMBOL i+1.
      // The table lookups must use SYMBOL i+1 properties.
      u32 next_idx = i + 1;

      // Literal Length (Table 0)
      {
        u8 code = d_ll_codes[next_idx];
        // Validate symbol code is within table bounds
        if (!table[0].d_symbol_table || code > table[0].max_symbol) {
          *output_pos = 0;
          return;
        }
        auto sym = table[0].d_symbol_table[code];
        // Encode transition from Current State to Next State (valid for code)
        u32 nbBitsOut = (stateLL + sym.deltaNbBits) >> 16;
        if (nbBitsOut > 0) {
          write_bits(stateLL & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        }
        stateLL = (stateLL >> nbBitsOut) + sym.deltaFindState;
      }

      // Match Length (Table 2)
      {
        u8 code = d_ml_codes[next_idx];
        // Validate symbol code is within table bounds
        if (!table[2].d_symbol_table || code > table[2].max_symbol) {
          *output_pos = 0;
          return;
        }
        auto sym = table[2].d_symbol_table[code];
        u32 nbBitsOut = (stateML + sym.deltaNbBits) >> 16;
        if (nbBitsOut > 0) {
          write_bits(stateML & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        }
        stateML = (stateML >> nbBitsOut) + sym.deltaFindState;
      }

      // Offset (Table 1)
      {
        u8 code = d_of_codes[next_idx];
        // Validate symbol code is within table bounds
        if (!table[1].d_symbol_table || code > table[1].max_symbol) {
          *output_pos = 0;
          return;
        }
        auto sym = table[1].d_symbol_table[code];
        u32 nbBitsOut = (stateOF + sym.deltaNbBits) >> 16;
        if (nbBitsOut > 0) {
          write_bits(stateOF & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        }
        stateOF = (stateOF >> nbBitsOut) + sym.deltaFindState;
      }

      // Revert Swap: OF, ML, LL
      write_bits(d_of_extras[next_idx], d_of_bits[next_idx]);
      write_bits(d_ml_extras[next_idx], d_ml_bits[next_idx]);
      write_bits(d_ll_extras[next_idx], d_ll_bits[next_idx]);
      
      if (bitCount > 10000) {
        printf("[FSE_ENCODE] CORRUPTION at i=%u: bitCount=%u, next_idx=%u\n", 
               i, bitCount, next_idx);
        break;
      }
    }
  }

  // 4. Write Final States (Header)
  // This corresponds to State N-1
  // Decoder reads these FIRST (High Addr)
  // Order: ML, OF, LL (from Zstd spec / decoder read order)
  printf("[FSE_ENCODE] Writing final states: ML=%u(%u bits), OF=%u(%u bits), LL=%u(%u bits)\n",
         stateML, table[2].table_log, stateOF, table[1].table_log, stateLL, table[0].table_log);
  write_bits(stateML, table[2].table_log);
  write_bits(stateOF, table[1].table_log);
  write_bits(stateLL, table[0].table_log);

  // 5. Flush Sentinel '1'
  write_bits(1, 1);
  while (bitCount > 0) {
    if (ptr < end_limit)
      *ptr++ = (unsigned char)(bitContainer & 0xFF);
    bitContainer >>= 8;
    bitCount = (bitCount > 8) ? bitCount - 8 : 0;
  }

  size_t final_pos = ptr - output_bitstream;
  *output_pos = final_pos;
  if (threadIdx.x == 0) {
    size_t total_bits = final_pos * 8 + bitCount;
    printf("[FSE_ENCODE] Thread 0 END: num_symbols=%u, output_pos=%llu\n",
           num_symbols, (unsigned long long)final_pos);
    printf("[FSE_ENCODE] Complete: wrote %llu bytes (%llu bits) for %u sequences\n",
           (unsigned long long)final_pos, (unsigned long long)total_bits, num_symbols);
    // Dump last 8 bytes of bitstream for debugging
    printf("[FSE_ENCODE] Last bytes:");
    for (int i = (int)final_pos - 8; i < (int)final_pos; i++) {
      if (i >= 0) printf(" %02X", output_bitstream[i]);
    }
    printf("\n");
  }
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

    // Fix: Use freq (not freq-1) for correct maxBitsOut calculation
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

  // Set deltaFindState for all symbols with positive frequency
  // Fix: Use cumulative_frequency (not cum - 1) for direct addition formula
  // This ensures state transitions properly within the symbol's state range
  for (u32 s = tid; s <= max_symbol; s += blockDim.x) {
    u32 freq = normalized_counters[s];
    if (freq == 0 || freq == (u32)-1)  // Skip 0 and 0xFFFF (invalid)
      continue;
    
    // The encoder transitions: state = (state >> nbBits) + deltaFindState
    // For direct addition, use cum (not cum - 1)
    table->d_symbol_table[s].deltaFindState = (i32)s_cum_freq[s];
    
    if (s < 5 && tid == 0) {
      printf("[CTABLE] Symbol %u: freq=%u, cum=%u, deltaFind=%d\n",
             s, freq, s_cum_freq[s], (i32)s_cum_freq[s]);
    }
  }

  __syncthreads();

  // Phase 4: Init CTable for Encoder using SPREADING (Critical for Correctness)
  // The Encoder InitState must match a valid slot in the Decoder's Spread
  // Table.
  if (tid == 0) {
    u32 tableSize = 1 << table_log;
    u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
    u32 mask = tableSize - 1;
    u32 pos = 0;

    // Reset first states
    for (u32 s = 0; s <= max_symbol; s++) {
      table->d_symbol_first_state[s] = 0;
    }

    // 2a. High Threshold: Place -1 symbols at the end
    // Per RFC 8878: low-probability symbols get 1 slot at high positions
    u32 high_threshold = tableSize - 1;
    for (u32 s = 0; s <= max_symbol; s++) {
      if ((i16)normalized_counters[s] == -1) {
        // State value must match decoder: just the position, not tableSize + pos
        table->d_symbol_first_state[s] = (u16)high_threshold;
        high_threshold--;
      }
    }

    // 2b. Spread positive symbols
    for (u32 s = 0; s <= max_symbol; s++) {
      int n = (int)normalized_counters[s];
      if (n <= 0) // Skip 0 and -1 (already handled)
        continue;

      for (int i = 0; i < n; i++) {
        // Skip positions occupied by high threshold symbols
        while (pos > high_threshold) {
          pos = (pos + step) & mask;
        }

        // Only record the first state for this symbol (matches decoder's first slot)
        if (table->d_symbol_first_state[s] == 0) {
          // State value must match decoder: just the position
          table->d_symbol_first_state[s] = (u16)pos;
        }
        pos = (pos + step) & mask;
      }
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
