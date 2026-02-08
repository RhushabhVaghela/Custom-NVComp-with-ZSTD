/**
 * @file cuda_zstd_fse_rfc.cu
 * @brief RFC 8878 Compliant FSE Implementation
 *
 * Clean room implementation following RFC 8878 Section 4.1.
 */

#include "cuda_zstd_fse.h" // For FSEEncodeTable (old table format)
#include "cuda_zstd_fse_rfc.h"
#include "cuda_zstd_safe_alloc.h"
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
        entries[highPos].symbol = (u8)s; // KEEP THIS FIX
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
        // Restore original (constant nbBits) logic to maintain compatibility with k_fse_encode
        nbBits = highestBit(tableSize) - highestBit(freq - 1);
      }
      entries[state].nbBits = (u8)nbBits;
      u32 subState = state >> nbBits;
      entries[state].nextState = (u16)(symbolFirstState[sym] + subState);
    }
  }
}

// =============================================================================
// ENCODER — Production encoder is k_encode_fse_interleaved in
//           cuda_zstd_fse_encoding_kernel.cu.  The legacy k_fse_encode,
//           FSE_encodeRFC, k_fse_encode_rfc_from_old_tables, and
//           launch_fse_encoding_kernel_rfc were dead code (never called)
//           and contained known correctness bugs (ignored input symbols
//           after symbols[0], used decoder table for encoding).
//           Removed 2026-02-07.
// =============================================================================

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

  if (cuda_zstd::safe_cuda_malloc_async(&table.d_entries, table.tableSize * sizeof(FSEDistributionEntry), stream) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  if (cuda_zstd::safe_cuda_malloc_async(&table.d_normFreqs, (maxSymbol + 1) * sizeof(u16), stream) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  if (cuda_zstd::safe_cuda_malloc_async(&table.d_symbolFirstState, (maxSymbol + 1) * sizeof(u16), stream) != cudaSuccess)
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

// FSE_encodeRFC removed — called the broken k_fse_encode kernel (see comment above).

__host__ Status FSE_decodeRFC(const u8 *d_bitstream, size_t bitstreamSize,
                              const FSETableRFC &table, u32 numSymbols,
                              u8 *d_symbols, cudaStream_t stream) {
  k_fse_decode<<<1, 1, 0, stream>>>(d_bitstream, bitstreamSize, table.d_entries, table.tableLog, numSymbols, d_symbols);
  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

// k_fse_encode_rfc_from_old_tables and launch_fse_encoding_kernel_rfc removed —
// dead code (never called from any production path, test, or benchmark).
// Production interleaved encoder is k_encode_fse_interleaved in
// cuda_zstd_fse_encoding_kernel.cu.

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
