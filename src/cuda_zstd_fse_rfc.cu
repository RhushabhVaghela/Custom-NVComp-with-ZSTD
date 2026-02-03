/**
 * @file cuda_zstd_fse_rfc.cu
 * @brief RFC 8878 Compliant FSE Implementation
 *
 * Clean room implementation following RFC 8878 Section 4.1.
 */

#include "cuda_zstd_fse.h" // For FSEEncodeTable (old table format)
#include "cuda_zstd_fse_rfc.h"
#include "cuda_zstd_types.h"
#include <algorithm>
#include <cstdio>
#include <cub/cub.cuh>
#include <vector>

namespace cuda_zstd {
namespace fse {

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ u32 highestBit(u32 v) {
  if (v == 0)
    return 0;
  return 31 - __clz(v);
}

// =============================================================================
// TABLE BUILDING (RFC 8878 Algorithm 1)
// =============================================================================

/**
 * @brief GPU Kernel: Build FSE distribution table
 */
__global__ void k_fse_build_table(const u16 *__restrict__ normFreqs,
                                  u32 maxSymbol, u32 tableSize,
                                  FSEDistributionEntry *__restrict__ entries,
                                  u16 *__restrict__ symbolFirstState) {
  u32 tid = threadIdx.x;

  for (u32 s = tid; s <= maxSymbol; s += blockDim.x) {
    symbolFirstState[s] = 0;
  }
  __syncthreads();

  u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
  u32 mask = tableSize - 1;
  u32 pos = 0;

  __shared__ u8 s_occupied[1024]; 
  for (u32 i = tid; i < tableSize; i += blockDim.x) {
    s_occupied[i] = 0;
    entries[i].symbol = 0;
    entries[i].nbBits = 0;
    entries[i].nextState = 0;
  }
  __syncthreads();

  if (tid == 0) {
    u32 highPos = tableSize - 1;
    for (u32 s = 0; s <= maxSymbol; s++) {
      if ((i16)normFreqs[s] == -1) {
        symbolFirstState[s] = highPos;
        s_occupied[highPos] = 1;
        highPos--;
      }
    }
    for (u32 s = 0; s <= maxSymbol; s++) {
      u32 freq = normFreqs[s];
      if (freq == 0 || (i16)freq == -1)
        continue;

      for (u32 i = 0; i < freq; i++) {
        while (s_occupied[pos]) {
          pos = (pos + step) & mask;
        }
        entries[pos].symbol = (u8)s;
        s_occupied[pos] = 1;
        if (symbolFirstState[s] == 0) {
          symbolFirstState[s] = pos;
        }
        pos = (pos + step) & mask;
      }
    }
  }
  __syncthreads();

  for (u32 state = tid; state < tableSize; state += blockDim.x) {
    u8 sym = entries[state].symbol;
    u32 freq = normFreqs[sym];

    if (freq == 0) {
      entries[state].nbBits = 0;
      entries[state].nextState = 0;
    } else if ((i16)freq == -1) {
      entries[state].nbBits = highestBit(tableSize);
      entries[state].nextState = 0;
    } else {
      u32 nbBits;
      if (freq == 1) {
        nbBits = highestBit(tableSize);
      } else {
        nbBits = highestBit(tableSize) - highestBit(freq - 1);
      }
      entries[state].nbBits = (u8)nbBits;
      u32 subState = state >> nbBits;
      entries[state].nextState = (u16)(symbolFirstState[sym] + subState);
    }
  }
}

// =============================================================================
// ENCODER (RFC 8878 Section 4.1.2)
// =============================================================================

/**
 * @brief GPU Kernel: FSE Encode
 */
__global__ void k_fse_encode(const u8 *__restrict__ symbols, u32 numSymbols,
                             const FSEDistributionEntry *__restrict__ table,
                             const u16 *__restrict__ symbolFirstState,
                             u32 tableLog, u32 tableSize,
                             u8 *__restrict__ bitstream,
                             size_t bitstreamCapacity,
                             size_t *__restrict__ outputSize) {
  if (threadIdx.x != 0)
    return;

  if (numSymbols == 0) {
    *outputSize = 0;
    return;
  }

  u8 *ptr = bitstream;
  u8 *end = bitstream + bitstreamCapacity;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  auto writeBits = [&](u32 value, u32 nbBits) {
    if (nbBits == 0)
      return;
    bitContainer |= ((u64)value << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr < end) {
        *ptr++ = (u8)(bitContainer & 0xFF);
      }
      bitContainer >>= 8;
      bitCount -= 8;
    }
  };

  u32 state = symbolFirstState[symbols[0]];
  for (u32 i = 0; i < numSymbols; i++) {
    u32 nbBits = table[state].nbBits;
    if (nbBits > 0) {
      writeBits(state & ((1u << nbBits) - 1), nbBits);
    }
    u32 nextState = table[state].nextState;
    state = nextState;
  }
  writeBits(state, tableLog);
  writeBits(1, 1);

  while (bitCount > 0) {
    if (ptr < end) {
      *ptr++ = (u8)(bitContainer & 0xFF);
    }
    bitContainer >>= 8;
    bitCount = (bitCount > 8) ? bitCount - 8 : 0;
  }
  *outputSize = ptr - bitstream;
}

// =============================================================================
// DECODER (RFC 8878 Section 4.1.3)
// =============================================================================

__global__ void k_fse_decode(const u8 *__restrict__ bitstream,
                             size_t bitstreamSize,
                             const FSEDistributionEntry *__restrict__ table,
                             u32 tableLog, u32 numSymbols,
                             u8 *__restrict__ symbols) {
  if (threadIdx.x != 0)
    return;

  if (numSymbols == 0 || bitstreamSize == 0)
    return;

  u32 sentinelPos = 0;
  bool found = false;

  for (int byteIdx = (int)bitstreamSize - 1; byteIdx >= 0 && !found; byteIdx--) {
    u8 byte = bitstream[byteIdx];
    for (int bit = 0; bit < 8; bit++) {
      if ((byte >> bit) & 1) {
        sentinelPos = byteIdx * 8 + bit;
        found = true;
        break;
      }
    }
  }

  if (!found) return;

  u32 bitPos = sentinelPos - tableLog;
  u32 state = 0;
  for (u32 i = 0; i < tableLog; i++) {
    u32 byteIdx = (bitPos + i) / 8;
    u32 bitIdx = (bitPos + i) % 8;
    if (byteIdx < bitstreamSize) {
      state |= (((bitstream[byteIdx] >> bitIdx) & 1) << i);
    }
  }

  for (int i = (int)numSymbols - 1; i >= 0; i--) {
    symbols[i] = table[state].symbol;
    u32 nbBits = table[state].nbBits;
    u32 bits = 0;
    if (nbBits > 0) {
      bitPos -= nbBits;
      for (u32 j = 0; j < nbBits; j++) {
        u32 byteIdx = (bitPos + j) / 8;
        u32 bitIdx = (bitPos + j) % 8;
        if (byteIdx < bitstreamSize) {
          bits |= (((bitstream[byteIdx] >> bitIdx) & 1) << j);
        }
      }
    }
    state = table[state].nextState + bits;
  }
}

// =============================================================================
// HOST INTERFACES
// =============================================================================

__host__ Status FSE_allocTableRFC(FSETableRFC &table, u32 tableLog,
                                  u32 maxSymbol, cudaStream_t stream) {
  table.tableLog = tableLog;
  table.tableSize = 1u << tableLog;
  table.maxSymbol = maxSymbol;

  if (cudaMallocAsync(&table.d_entries, table.tableSize * sizeof(FSEDistributionEntry), stream) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  if (cudaMallocAsync(&table.d_normFreqs, (maxSymbol + 1) * sizeof(u16), stream) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  if (cudaMallocAsync(&table.d_symbolFirstState, (maxSymbol + 1) * sizeof(u16), stream) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  table.allocated = true;
  return Status::SUCCESS;
}

__host__ void FSE_freeTableRFC(FSETableRFC &table, cudaStream_t stream) {
  if (table.allocated) {
    cudaFreeAsync(table.d_entries, stream);
    cudaFreeAsync(table.d_normFreqs, stream);
    cudaFreeAsync(table.d_symbolFirstState, stream);
    table.allocated = false;
  }
}

__host__ Status FSE_buildTableRFC(const u16 *normFreqs, u32 maxSymbol,
                                  u32 tableLog, FSETableRFC &table,
                                  void *d_workspace, size_t workspaceSize,
                                  cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(table.d_normFreqs, normFreqs, (maxSymbol + 1) * sizeof(u16), cudaMemcpyHostToDevice, stream));
  u32 tableSize = 1u << tableLog;
  k_fse_build_table<<<1, 256, 0, stream>>>(table.d_normFreqs, maxSymbol, tableSize, table.d_entries, table.d_symbolFirstState);
  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

__host__ Status FSE_encodeRFC(const u8 *d_symbols, u32 numSymbols,
                              const FSETableRFC &table, u8 *d_bitstream,
                              size_t bitstreamCapacity, size_t *d_outputSize,
                              cudaStream_t stream) {
  k_fse_encode<<<1, 1, 0, stream>>>(d_symbols, numSymbols, table.d_entries, table.d_symbolFirstState, table.tableLog, table.tableSize, d_bitstream, bitstreamCapacity, d_outputSize);
  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

__host__ Status FSE_decodeRFC(const u8 *d_bitstream, size_t bitstreamSize,
                              const FSETableRFC &table, u32 numSymbols,
                              u8 *d_symbols, cudaStream_t stream) {
  k_fse_decode<<<1, 1, 0, stream>>>(d_bitstream, bitstreamSize, table.d_entries, table.tableLog, numSymbols, d_symbols);
  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

// =============================================================================
// INTERLEAVED ENCODER (RFC 8878 Compliant)
// =============================================================================

__global__ void k_fse_encode_rfc_from_old_tables(
    const u8 *__restrict__ d_ll_codes, const u32 *__restrict__ d_ll_extras,
    const u8 *__restrict__ d_ll_bits, const u8 *__restrict__ d_of_codes,
    const u32 *__restrict__ d_of_extras, const u8 *__restrict__ d_of_bits,
    const u8 *__restrict__ d_ml_codes, const u32 *__restrict__ d_ml_extras,
    const u8 *__restrict__ d_ml_bits, u32 num_symbols,
    const FSEEncodeTable *tables, unsigned char *__restrict__ output_bitstream,
    size_t bitstream_capacity, size_t *__restrict__ output_pos) {

  if (threadIdx.x != 0 || num_symbols == 0 || bitstream_capacity == 0) {
    if (threadIdx.x == 0 && output_pos) *output_pos = 0;
    return;
  }

  if (!d_ll_codes || !d_of_codes || !d_ml_codes || !output_bitstream || !output_pos || !tables) {
    if (output_pos) *output_pos = 0;
    return;
  }

  // RFC 8878 Backward Bitstack Writer
  unsigned char *ptr = output_bitstream + bitstream_capacity;
  u64 bitContainer = 0; 
  u32 bitCount = 0;

  auto write_bits = [&](u64 val, u32 nbBits) {
    if (nbBits == 0) return;
    val &= ((1ULL << nbBits) - 1);
    bitContainer = (bitContainer << nbBits) | val;
    bitCount += nbBits;
    while (bitCount >= 8) {
      if (ptr > output_bitstream) {
        *(--ptr) = (unsigned char)((bitContainer >> (bitCount - 8)) & 0xFF);
      }
      bitCount -= 8;
      bitContainer &= ((1ULL << bitCount) - 1);
    }
  };

  const u32 ll_table_log = tables[0].table_log;
  const u32 of_table_log = tables[1].table_log;
  const u32 ml_table_log = tables[2].table_log;

  // Use output buffer for temporary state storage (safe due to backward write)
  // Max sequences = 128KB block / 3 bytes = ~43k. u16 * 43k * 3 = 258KB.
  // bitstream_capacity is roughly num_seq * 8.
  // 43k * 8 = 344KB. So it fits.
  u16 *state_storage_ll = (u16*)(void*)output_bitstream;
  u16 *state_storage_of = state_storage_ll + num_symbols;
  u16 *state_storage_ml = state_storage_of + num_symbols;

  // Step 1: Forward Pass - Compute all states
  // Initial states for Symbol 0 (using d_next_state[cumul[s]])
  u32 stateLL = tables[0].d_symbol_first_state[d_ll_codes[0]];
  u32 stateOF = tables[1].d_symbol_first_state[d_of_codes[0]];
  u32 stateML = tables[2].d_symbol_first_state[d_ml_codes[0]];

  state_storage_ll[0] = (u16)stateLL;
  state_storage_of[0] = (u16)stateOF;
  state_storage_ml[0] = (u16)stateML;

  for (u32 i = 1; i < num_symbols; i++) {
    // LL
    u8 code = d_ll_codes[i];
    u32 nbBits = (stateLL + tables[0].d_symbol_table[code].deltaNbBits) >> 16;
    u32 nextIndex = (stateLL >> nbBits) + tables[0].d_symbol_table[code].deltaFindState;
    stateLL = tables[0].d_next_state[nextIndex];
    state_storage_ll[i] = (u16)stateLL;

    // ML
    code = d_ml_codes[i];
    nbBits = (stateML + tables[2].d_symbol_table[code].deltaNbBits) >> 16;
    nextIndex = (stateML >> nbBits) + tables[2].d_symbol_table[code].deltaFindState;
    stateML = tables[2].d_next_state[nextIndex];
    state_storage_ml[i] = (u16)stateML;

    // OF
    code = d_of_codes[i];
    nbBits = (stateOF + tables[1].d_symbol_table[code].deltaNbBits) >> 16;
    nextIndex = (stateOF >> nbBits) + tables[1].d_symbol_table[code].deltaFindState;
    stateOF = tables[1].d_next_state[nextIndex];
    state_storage_of[i] = (u16)stateOF;
  }

  // Step 2: Backward Push
  // 1. Sentinel
  write_bits(1, 1);

  // 2. Final States (from last symbol) - Decoder starts here
  write_bits(stateML, ml_table_log);
  write_bits(stateOF, of_table_log);
  write_bits(stateLL, ll_table_log);

  // 3. Sequences (N-1 down to 0) - Matches decoder order (N-1 down to 0)
  for (int i = (int)num_symbols - 1; i >= 0; i--) {
      // Extra bits
      write_bits(d_of_extras[i], d_of_bits[i]);
      write_bits(d_ml_extras[i], d_ml_bits[i]);
      write_bits(d_ll_extras[i], d_ll_bits[i]);

      if (i > 0) {
          // Transition bits (State[i-1] -> State[i])
          // OF (Pushed first -> Read last)
          u32 prevOF = state_storage_of[i-1];
          u8 codeOF = d_of_codes[i];
          u32 nbOF = (prevOF + tables[1].d_symbol_table[codeOF].deltaNbBits) >> 16;
          write_bits(prevOF & ((1u << nbOF) - 1), nbOF);

          // ML
          u32 prevML = state_storage_ml[i-1];
          u8 codeML = d_ml_codes[i];
          u32 nbML = (prevML + tables[2].d_symbol_table[codeML].deltaNbBits) >> 16;
          write_bits(prevML & ((1u << nbML) - 1), nbML);

          // LL (Pushed last -> Read first)
          u32 prevLL = state_storage_ll[i-1];
          u8 codeLL = d_ll_codes[i];
          u32 nbLL = (prevLL + tables[0].d_symbol_table[codeLL].deltaNbBits) >> 16;
          write_bits(prevLL & ((1u << nbLL) - 1), nbLL);
      }
  }

  // Final flush
  if (bitCount > 0) {
    if (ptr > output_bitstream) {
        *(--ptr) = (unsigned char)((bitContainer << (8 - bitCount)) & 0xFF);
    }
  }

  size_t written = (output_bitstream + bitstream_capacity) - ptr;
  *output_pos = written;
  for (size_t i = 0; i < written; i++) {
    output_bitstream[i] = ptr[i];
  }
}

__host__ Status launch_fse_encoding_kernel_rfc(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, unsigned char *d_bitstream, size_t *d_output_pos,
    size_t bitstream_capacity, const FSEEncodeTable *d_tables,
    cudaStream_t stream) {
    
  if (!d_ll_codes || !d_of_codes || !d_ml_codes || !d_bitstream || !d_output_pos || !d_tables)
    return Status::ERROR_INVALID_PARAMETER;

  if (num_symbols == 0) {
    cudaMemsetAsync(d_output_pos, 0, sizeof(size_t), stream);
    return Status::SUCCESS;
  }

  cudaMemsetAsync(d_bitstream, 0, bitstream_capacity, stream);

  k_fse_encode_rfc_from_old_tables<<<1, 1, 0, stream>>>(
      d_ll_codes, d_ll_extras, d_ll_bits, d_of_codes, d_of_extras, d_of_bits,
      d_ml_codes, d_ml_extras, d_ml_bits, num_symbols, d_tables, d_bitstream,
      bitstream_capacity, d_output_pos);

  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

__host__ Status FSE_normalizeFreqs(const u32 *rawFreqs, u32 maxSymbol,
                                   u32 numSamples, u32 tableLog,
                                   u16 *normFreqs) {
  u32 tableSize = 1u << tableLog;
  std::vector<u32> scaled(maxSymbol + 1);
  u32 totalScaled = 0;

  for (u32 s = 0; s <= maxSymbol; s++) {
    if (rawFreqs[s] > 0) {
      scaled[s] = ((u64)rawFreqs[s] * tableSize + (numSamples / 2)) / numSamples;
      if (scaled[s] == 0) scaled[s] = 1;
      totalScaled += scaled[s];
    } else {
      scaled[s] = 0;
    }
  }

  if (totalScaled > tableSize) {
    while (totalScaled > tableSize) {
      u32 maxS = 0;
      for (u32 s = 1; s <= maxSymbol; s++) {
        if (scaled[s] > scaled[maxS]) maxS = s;
      }
      if (scaled[maxS] > 1) {
        scaled[maxS]--;
        totalScaled--;
      } else break;
    }
  } else if (totalScaled < tableSize) {
    while (totalScaled < tableSize) {
      u32 minS = 0;
      for (u32 s = 1; s <= maxSymbol; s++) {
        if (scaled[s] > 0 && (scaled[s] < scaled[minS] || scaled[minS] == 0)) minS = s;
      }
      scaled[minS]++;
      totalScaled++;
    }
  }

  for (u32 s = 0; s <= maxSymbol; s++) {
    normFreqs[s] = (u16)scaled[s];
  }

  return Status::SUCCESS;
}

} // namespace fse
} // namespace cuda_zstd
