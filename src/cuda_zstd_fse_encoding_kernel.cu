// cuda_zstd_fse_encoding_kernel.cu - Optimized GPU FSE Encoding Kernel
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_internal.h"
#include <cstdio>

namespace cuda_zstd {
namespace fse {

/**
 * @brief GPU Kernel for Interleaved FSE Encoding.
 *
 * This kernel implements the RFC 8878 sequence encoding logic, matching
 * the ZSTD reference encoder (ZSTD_encodeSequences_body) exactly.
 *
 * The ZSTD bitstream is read BACKWARD by the decoder. The encoder must
 * process sequences in REVERSE order (N-1 to 0) so that the decoder
 * reads them correctly.
 *
 * Algorithm (matching zstd_compress_sequences.c):
 * 1. Initialize FSE states from seq[N-1] (last sequence)
 * 2. Write extras for seq[N-1]: LL_extra, ML_extra, OF_extra
 * 3. For i = N-2 down to 0:
 *    a. FSE_encodeSymbol(stateOF, OF_code[i])  - OF transition
 *    b. FSE_encodeSymbol(stateML, ML_code[i])  - ML transition
 *    c. FSE_encodeSymbol(stateLL, LL_code[i])  - LL transition
 *    d. BIT_addBits(LL_extra[i])
 *    e. BIT_addBits(ML_extra[i])
 *    f. BIT_addBits(OF_extra[i])
 * 4. Flush final states: ML, OF, LL (these now represent seq[0])
 * 5. Sentinel bit + final flush (BIT_closeCStream equivalent)
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

  // === Step 1: Initialize FSE states from LAST sequence (N-1) ===
  // Matches: FSE_initCState2(&stateMatchLength, CTable_ML, mlCodeTable[nbSeq-1]);
  //          FSE_initCState2(&stateOffsetBits,  CTable_OF, ofCodeTable[nbSeq-1]);
  //          FSE_initCState2(&stateLitLength,   CTable_LL, llCodeTable[nbSeq-1]);
  u32 last = num_symbols - 1;
  u32 stateLL = table[0].d_symbol_first_state[d_ll_codes[last]];
  u32 stateOF = table[1].d_symbol_first_state[d_of_codes[last]];
  u32 stateML = table[2].d_symbol_first_state[d_ml_codes[last]];

#ifdef CUDA_ZSTD_DEBUG
  printf("[FSE_ENC] num_symbols=%u, tableLogs: LL=%u OF=%u ML=%u\n",
         num_symbols, table[0].table_log, table[1].table_log, table[2].table_log);
  for (u32 s = 0; s < num_symbols; s++) {
    printf("[FSE_ENC] seq[%u]: LL_code=%u LL_extra=%u LL_bits=%u | "
           "OF_code=%u OF_extra=%u OF_bits=%u | "
           "ML_code=%u ML_extra=%u ML_bits=%u\n",
           s, d_ll_codes[s], d_ll_extras[s], d_ll_bits[s],
           d_of_codes[s], d_of_extras[s], d_of_bits[s],
           d_ml_codes[s], d_ml_extras[s], d_ml_bits[s]);
  }
  printf("[FSE_ENC] Init states from seq[%u]: stateLL=%u stateOF=%u stateML=%u\n",
         last, stateLL, stateOF, stateML);
  printf("[FSE_ENC]   LL_code=%u -> first_state=%u\n", d_ll_codes[last], stateLL);
  printf("[FSE_ENC]   OF_code=%u -> first_state=%u\n", d_of_codes[last], stateOF);
  printf("[FSE_ENC]   ML_code=%u -> first_state=%u\n", d_ml_codes[last], stateML);
#endif

  // === Step 2: Write extras for LAST sequence ===
  // Matches: BIT_addBits(&blockStream, sequences[nbSeq-1].litLength, LL_bits[...]);
  //          BIT_addBits(&blockStream, sequences[nbSeq-1].mlBase,    ML_bits[...]);
  //          BIT_addBits(&blockStream, sequences[nbSeq-1].offBase,   ofCodeTable[nbSeq-1]);
  write_bits(d_ll_extras[last], d_ll_bits[last]);
  write_bits(d_ml_extras[last], d_ml_bits[last]);
  write_bits(d_of_extras[last], d_of_bits[last]);

  // === Step 3: Loop from N-2 down to 0 ===
  // Matches: for (n=nbSeq-2 ; n<nbSeq ; n--) { ... }
  // (intentional underflow in reference — u32 wraps around)
  if (num_symbols >= 2) {
    for (u32 i = num_symbols - 2; ; i--) {
      // --- Transitions: OF, ML, LL (this exact order from reference) ---

      // FSE_encodeSymbol(&blockStream, &stateOffsetBits, ofCode);
      {
          u8 code = d_of_codes[i];
          auto sym = table[1].d_symbol_table[code];
          u32 nbBitsOut = (stateOF + sym.deltaNbBits) >> 16;
#ifdef CUDA_ZSTD_DEBUG
          printf("[FSE_ENC] Loop seq[%u] OF_tr: state=%u code=%u deltaNb=0x%08X nbBitsOut=%u write=%u\n",
                 i, stateOF, code, sym.deltaNbBits, nbBitsOut, stateOF & ((1u << nbBitsOut) - 1));
#endif
          write_bits(stateOF & ((1ULL << nbBitsOut) - 1), nbBitsOut);
          stateOF = table[1].d_next_state[(stateOF >> nbBitsOut) + sym.deltaFindState];
      }

      // FSE_encodeSymbol(&blockStream, &stateMatchLength, mlCode);
      {
          u8 code = d_ml_codes[i];
          auto sym = table[2].d_symbol_table[code];
          u32 nbBitsOut = (stateML + sym.deltaNbBits) >> 16;
#ifdef CUDA_ZSTD_DEBUG
          printf("[FSE_ENC] Loop seq[%u] ML_tr: state=%u code=%u deltaNb=0x%08X nbBitsOut=%u write=%u\n",
                 i, stateML, code, sym.deltaNbBits, nbBitsOut, stateML & ((1u << nbBitsOut) - 1));
#endif
          write_bits(stateML & ((1ULL << nbBitsOut) - 1), nbBitsOut);
          stateML = table[2].d_next_state[(stateML >> nbBitsOut) + sym.deltaFindState];
      }

      // FSE_encodeSymbol(&blockStream, &stateLitLength, llCode);
      {
          u8 code = d_ll_codes[i];
          auto sym = table[0].d_symbol_table[code];
          u32 nbBitsOut = (stateLL + sym.deltaNbBits) >> 16;
#ifdef CUDA_ZSTD_DEBUG
          printf("[FSE_ENC] Loop seq[%u] LL_tr: state=%u code=%u deltaNb=0x%08X nbBitsOut=%u write=%u\n",
                 i, stateLL, code, sym.deltaNbBits, nbBitsOut, stateLL & ((1u << nbBitsOut) - 1));
#endif
          write_bits(stateLL & ((1ULL << nbBitsOut) - 1), nbBitsOut);
          stateLL = table[0].d_next_state[(stateLL >> nbBitsOut) + sym.deltaFindState];
      }

      // --- Extra bits: LL, ML, OF ---
      // BIT_addBits(&blockStream, sequences[n].litLength, llBits);
      write_bits(d_ll_extras[i], d_ll_bits[i]);
      // BIT_addBits(&blockStream, sequences[n].mlBase, mlBits);
      write_bits(d_ml_extras[i], d_ml_bits[i]);
      // BIT_addBits(&blockStream, sequences[n].offBase, ofBits);
      write_bits(d_of_extras[i], d_of_bits[i]);

      if (i == 0) break;  // Prevent underflow for u32
    }
  }

  // === Step 4: Flush final states: ML, OF, LL ===
  // These now represent seq[0]'s initial states for the decoder.
  // Matches: FSE_flushCState(&blockStream, &stateMatchLength);
  //          FSE_flushCState(&blockStream, &stateOffsetBits);
  //          FSE_flushCState(&blockStream, &stateLitLength);
#ifdef CUDA_ZSTD_DEBUG
  printf("[FSE_ENC] Final states (for decoder init): stateLL=%u stateOF=%u stateML=%u\n",
         stateLL, stateOF, stateML);
#endif
  write_bits(stateML, table[2].table_log);
  write_bits(stateOF, table[1].table_log);
  write_bits(stateLL, table[0].table_log);

  // === Step 5: Sentinel bit (BIT_closeCStream equivalent) ===
  write_bits(1, 1);

  // === Step 6: Final flush — write remaining partial byte ===
  if (bitCount > 0) {
    if (ptr < end_limit) {
      *ptr++ = (unsigned char)(bitContainer & 0xFF);
    }
  }

  *output_pos = (size_t)(ptr - output_bitstream);

#ifdef CUDA_ZSTD_DEBUG
  printf("[FSE_ENC] Output %zu bytes:", *output_pos);
  for (size_t b = 0; b < *output_pos && b < 32; b++)
    printf(" %02X", output_bitstream[b]);
  printf("\n");
#endif
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
      // Match zstd reference: filling for compatibility with FSE_getMaxNbBits()
      table->d_symbol_table[s].deltaNbBits = ((table_log + 1) << 16) - (1 << table_log);
      table->d_symbol_table[s].deltaFindState = 0;
      table->d_symbol_first_state[s] = 0;
      continue;
    }

    u32 actual_freq = (freq < 0) ? 1 : (u32)freq;
    // Match zstd reference (fse_compress.c line 195):
    // freq == -1 or 1: highbit is 0, maxBitsOut = table_log
    // freq >= 2: highbit = highbit32(freq - 1), maxBitsOut = table_log - highbit
    u32 maxBitsOut;
    if (actual_freq >= 2) {
      u32 high_bit = 31 - cuda_zstd::utils::clz_impl(actual_freq - 1);
      maxBitsOut = table_log - high_bit;
    } else {
      maxBitsOut = table_log;  // freq==1 or -1
    }
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
      i32 f = (i32)(i16)normalized_counters[s]; // Cast via i16 to preserve -1 sign
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
    u32 actual_freq = ((i16)freq < 0) ? 1 : (u32)freq;
    // Match zstd reference (fse_compress.c line 198):
    // deltaFindState = total - normalizedCounter[s] = cumul[s] - freq
    table->d_symbol_table[s].deltaFindState = (i32)s_cum_freq[s] - (i32)actual_freq;
  }

  __syncthreads();

  // Phase 4: Init CTable for Encoder using SPREADING
  if (tid == 0) {
    u32 tableSize = 1 << table_log;
    u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
    u32 mask = tableSize - 1;
    u32 pos = 0;

    for (u32 i = 0; i <= max_symbol; i++) {
      table->d_symbol_first_state[i] = 0;
    }

    for (u32 i = 0; i < tableSize; ++i)
      table->d_state_to_symbol[i] = 0;

    u32 high_threshold = tableSize - 1;

    // 4a. Low-prob symbols (-1): place at high_threshold positions
    for (u32 s = 0; s <= max_symbol; s++) {
      if ((i16)normalized_counters[s] == -1) {
        table->d_state_to_symbol[high_threshold] = (u8)s;
        high_threshold--;
      }
    }

    // 4b. Positive-freq symbols: spread using step-and-skip
    for (u32 s = 0; s <= max_symbol; s++) {
      int n = (int)(i16)normalized_counters[s];
      if (n <= 0)
        continue;
      for (int i = 0; i < n; i++) {
        table->d_state_to_symbol[pos] = (u8)s;
        pos = (pos + step) & mask;
        while (pos > high_threshold) {
          pos = (pos + step) & mask;
        }
      }
    }

    // 4c. Build stateTable (d_next_state) by iterating positions IN ORDER.
    // Matches zstd reference fse_compress.c lines 167-173 exactly:
    //   for (u=0; u<tableSize; u++) {
    //     s = tableSymbol[u];
    //     tableU16[cumul[s]++] = tableSize + u;
    //   }
    {
      u16 cumul[256];
      for (u32 s = 0; s <= max_symbol; s++) {
        cumul[s] = s_cum_freq[s];
      }
      for (u32 u = 0; u < tableSize; u++) {
        u8 s = table->d_state_to_symbol[u];
        table->d_next_state[cumul[s]++] = (u16)(tableSize + u);
      }
    }

    // 4d. Set symbol_first_state = d_next_state[cumul_start] for each symbol
    for (u32 s = 0; s <= max_symbol; s++) {
      i16 freq = (i16)normalized_counters[s];
      if (freq == 0) continue;
      table->d_symbol_first_state[s] = table->d_next_state[s_cum_freq[s]];
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
