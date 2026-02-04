// cuda_zstd_fse_encoding_kernel.cu - Optimized GPU FSE Encoding Kernel
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_internal.h"
#include <cstdio>

namespace cuda_zstd {
namespace fse {

/**
 * @brief GPU Kernel for Interleaved FSE Encoding.
 *
 * This kernel implements the RFC 8878 sequence encoding logic.
 * It processes sequences in FORWARD order (0 to N-1) to build the state chain,
 * but writes bits such that the decoder (reading backward) sees them correctly.
 *
 * Order of encoding for each sequence:
 * 1. Offset bits
 * 2. Match Length bits
 * 3. Literal Length bits
 * 4. Offset state transition
 * 5. Match Length state transition
 * 6. Literal Length state transition
 *
 * This matches the decoder's reverse order: (LL_state, ML_state, OF_state, ...)
 */
__global__ void k_encode_fse_interleaved(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, unsigned char *output_bitstream, size_t *output_pos,
    size_t bitstream_capacity, const FSEEncodeTable *table) {

  u32 tid = threadIdx.x;
  if (tid != 0)
    return;

  // Validate input pointers
  if (!d_ll_codes || !d_of_codes || !d_ml_codes || !output_bitstream ||
      !output_pos || !table) {
    if (output_pos)
      *output_pos = 0;
    return;
  }

  unsigned char *ptr = output_bitstream;
  unsigned char *end_limit = output_bitstream + bitstream_capacity;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  auto write_bits = [&](u64 val, u32 nbBits) {
    if (nbBits == 0)
      return;
    val &= ((1ULL << nbBits) - 1); // Mask higher bits
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

  // 1. Seq 0 Extras
  write_bits(d_ll_extras[0], d_ll_bits[0]);
  write_bits(d_ml_extras[0], d_ml_bits[0]);
  write_bits(d_of_extras[0], d_of_bits[0]);

  // 2. Initial States from Seq 0
  u32 stateLL = table[0].d_symbol_first_state[d_ll_codes[0]];
  u32 stateOF = table[1].d_symbol_first_state[d_of_codes[0]];
  u32 stateML = table[2].d_symbol_first_state[d_ml_codes[0]];

  // 3. Loop Transitions Seq 1..N-1
  for (u32 i = 1; i < num_symbols; i++) {
    // FSE transition order must match decoder backward read order: ML, OF, LL
    // Wait! Decoder backward update order: LL_FSE, OF_FSE, ML_FSE.
    // So Encoder forward order: ML_FSE, OF_FSE, LL_FSE.
    {
        u8 code = d_ml_codes[i];
        auto sym = table[2].d_symbol_table[code];
        u32 nbBitsOut = (stateML + sym.deltaNbBits) >> 16;
        write_bits(stateML & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        stateML = (stateML >> nbBitsOut) + sym.deltaFindState;
    }
    {
        u8 code = d_of_codes[i];
        auto sym = table[1].d_symbol_table[code];
        u32 nbBitsOut = (stateOF + sym.deltaNbBits) >> 16;
        write_bits(stateOF & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        stateOF = (stateOF >> nbBitsOut) + sym.deltaFindState;
    }
    {
        u8 code = d_ll_codes[i];
        auto sym = table[0].d_symbol_table[code];
        u32 nbBitsOut = (stateLL + sym.deltaNbBits) >> 16;
        write_bits(stateLL & ((1ULL << nbBitsOut) - 1), nbBitsOut);
        stateLL = (stateLL >> nbBitsOut) + sym.deltaFindState;
    }
    // Extras for Seq i: LL_Ex, ML_Ex, OF_Ex
    write_bits(d_ll_extras[i], d_ll_bits[i]);
    write_bits(d_ml_extras[i], d_ml_bits[i]);
    write_bits(d_of_extras[i], d_of_bits[i]);
  }

  // 4. Final States (Init for Decoder): ML, OF, LL
  write_bits(stateML, table[2].table_log);
  write_bits(stateOF, table[1].table_log);
  write_bits(stateLL, table[0].table_log);

  // 5. Flush Sentinel '1'
  write_bits(1, 1);

  // 6. Final Flush
  if (bitCount > 0) {
    if (ptr < end_limit) {
      *ptr++ = (unsigned char)(bitContainer & 0xFF);
    }
  }

  *output_pos = (size_t)(ptr - output_bitstream);
}

// -------------------------------------------------------------------------------------------------

__global__ void k_build_ctable(const u32 *__restrict__ normalized_counters,
                               u32 max_symbol, u32 table_log,
                               FSEEncodeTable *table) {
  u32 tid = threadIdx.x;

  // Phase 1: Initialize Symbol Stats (deltaNbBits, deltaFindState)
  for (u32 s = tid; s <= max_symbol; s += blockDim.x) {
    i16 freq = (i16)normalized_counters[s];

    if (freq == 0) {
      table->d_symbol_table[s].deltaNbBits = 0;
      table->d_symbol_table[s].deltaFindState = 0;
      table->d_symbol_first_state[s] = 0;
      continue;
    }

    u32 actual_freq = (freq < 0) ? 1 : (u32)freq;
    u32 maxBitsOut = table_log - (31 - cuda_zstd::utils::clz_impl(actual_freq));
    u32 minStatePlus = actual_freq << maxBitsOut;
    u32 deltaNb = (maxBitsOut << 16) - minStatePlus;

    table->d_symbol_table[s].deltaNbBits = deltaNb;
  }

  __shared__ u16 s_cum_freq[FSE_MAX_SYMBOL_VALUE + 2];
  __syncthreads();

  // Phase 2: Prefix Sum
  if (tid == 0) {
    u16 cum = 0;
    s_cum_freq[0] = 0;
    for (u32 s = 0; s <= max_symbol; s++) {
      i32 f = (i32)normalized_counters[s];
      u32 actual_f = (f < 0) ? 1 : (u32)f;
      s_cum_freq[s] = cum;
      cum += (u16)actual_f;
    }
    s_cum_freq[max_symbol + 1] = cum;
  }
  __syncthreads();

  // Phase 3: Finish Symbol Stats (deltaFindState)
  for (u32 s = tid; s <= max_symbol; s += blockDim.x) {
    i32 freq = (i32)normalized_counters[s];
    if (freq == 0)
      continue;
    table->d_symbol_table[s].deltaFindState = (i32)s_cum_freq[s];
  }

  __syncthreads();

  // Phase 4: Init CTable for Encoder using SPREADING
  if (tid == 0) {
    u32 tableSize = 1 << table_log;
    u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
    u32 mask = tableSize - 1;
    u32 pos = 0;

    u32 symbolNext[256];
    for (u32 i = 0; i <= max_symbol; i++) {
      symbolNext[i] = (i16)normalized_counters[i] < 0 ? 1 : normalized_counters[i];
      table->d_symbol_first_state[i] = 0;
    }

    for (u32 i = 0; i < tableSize; ++i)
      table->d_state_to_symbol[i] = 0;

    u32 high_threshold = tableSize - 1;
    // 2a. Low-prob symbols (-1)
    for (u32 s = 0; s <= max_symbol; s++) {
      if ((i16)normalized_counters[s] == -1) {
        u16 state_val = (u16)(high_threshold + tableSize);
        table->d_next_state[s_cum_freq[s]] = state_val;
        table->d_symbol_first_state[s] = state_val;
        table->d_state_to_symbol[high_threshold] = 1;
        high_threshold--;
      }
    }

    // 2b. Positive symbols
    for (u32 s = 0; s <= max_symbol; s++) {
      int n = (int)(i16)normalized_counters[s];
      if (n <= 0)
        continue;
      for (int i = 0; i < n; i++) {
        while (pos > high_threshold || table->d_state_to_symbol[pos] != 0) {
          pos = (pos + step) & mask;
        }
        table->d_state_to_symbol[pos] = 1;

        u32 currentSymbolNext = symbolNext[s]++;
        u32 cumul_idx = s_cum_freq[s] + (currentSymbolNext - n);
        u16 state_val = (u16)(pos + tableSize);
        table->d_next_state[cumul_idx] = state_val;
        if (i == 0)
          table->d_symbol_first_state[s] = state_val;

        pos = (pos + step) & mask;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

Status launch_fse_encoding_kernel(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, unsigned char *d_bitstream, size_t *d_output_pos,
    size_t bitstream_capacity, const FSEEncodeTable *d_tables,
    cudaStream_t stream) {
  k_encode_fse_interleaved<<<1, 32, 0, stream>>>(
      d_ll_codes, d_ll_extras, d_ll_bits, d_of_codes, d_of_extras, d_of_bits,
      d_ml_codes, d_ml_extras, d_ml_bits, num_symbols, d_bitstream, d_output_pos,
      bitstream_capacity, d_tables);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_INTERNAL;
  }

  return Status::SUCCESS;
}

Status FSE_buildCTable_Device(const u32 *d_normalized_counters, u32 max_symbol,
                              u32 table_log, FSEEncodeTable *d_table,
                              void *d_workspace, size_t workspace_size,
                              cudaStream_t stream) {
  if (!d_normalized_counters || !d_table)
    return Status::ERROR_INVALID_PARAMETER;

  k_build_ctable<<<1, 256, 0, stream>>>(d_normalized_counters, max_symbol,
                                        table_log, d_table);

  return Status::SUCCESS;
}

} // namespace fse
} // namespace cuda_zstd

