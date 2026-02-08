// ============================================================================
// cuda_zstd_huffman.cu - Complete Huffman Encoding/Decoding Implementation
//
// NOTE: This file is now fully parallelized.
// - 'huffman_encode_kernel' is a parallel 2-pass scan + write.
// - 'huffman_decode_sequential_kernel' is now a true parallel chunked decoder
//   (using a setup kernel to find chunk starts).
//
// (NEW) NOTE: Refactored to use cuda_zstd_utils for parallel_scan.
// ============================================================================

#include "cuda_zstd_debug.h"
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h" // <-- 1. ADDED INCLUDE
#include "cuda_zstd_safe_alloc.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>
#include <vector>

// Note: A production implementation would use CUB for these scans.
// We implement them manually to be self-contained.

namespace cuda_zstd {
namespace huffman {

// ============================================================================
// Huffman Constants
// ============================================================================

constexpr u32 HUFFMAN_ENCODE_THREADS = 256;
// constexpr u32 HUFFMAN_DECODE_THREADS_PER_CHUNK = 1; // Decode is sequential
// per chunk constexpr u32 HUFFMAN_DECODE_SYMBOLS_PER_CHUNK = 4096; // Symbols
// per chunk

// ============================================================================
// Huffman Structures (from .h file, repeated for context)
// ============================================================================

struct HuffmanEncodeTable {
  HuffmanCode *codes;
  u32 num_symbols;
  u32 max_code_length;
  u8 *h_code_lengths;
};

struct HuffmanDecodeTable {
  u16 *d_fast_lookup;
  u32 *d_decode_info;
  u32 max_length;
};

// ============================================================================
// Parallel Scan Kernels (for Encode)
// (REMOVED) - This entire section is now gone and moved to cuda_zstd_utils.cu
// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ inline u32 reverse_bits_device(u32 val, u32 bits) {
  return __brev(val) >> (32 - bits);
}

__device__ void atomicOrByte(unsigned char *address, unsigned char val) {
  unsigned int *base_addr = (unsigned int *)((size_t)address & ~3);
  unsigned int offset = (size_t)address & 3;
  unsigned int shift = offset * 8;

  unsigned int old = *base_addr;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned int new_val = assumed | ((unsigned int)val << shift);
    old = atomicCAS(base_addr, assumed, new_val);
  } while (assumed != old);
}

// ============================================================================
// Kernels
// ============================================================================
// Frequency Analysis Kernel
// ============================================================================

__global__ void analyze_frequencies_kernel(const unsigned char *input,
                                           u32 input_size,
                                           u32 *global_frequencies) {
  __shared__ u32 local_freq[MAX_HUFFMAN_SYMBOLS];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (tid < MAX_HUFFMAN_SYMBOLS) {
    local_freq[tid] = 0;
  }
  __syncthreads();

  for (int i = idx; i < input_size; i += stride) {
    u8 symbol = input[i];
    atomicAdd(&local_freq[symbol], 1);
  }
  __syncthreads();

  if (tid < MAX_HUFFMAN_SYMBOLS) {
    if (local_freq[tid] > 0) {
      atomicAdd(&global_frequencies[tid], local_freq[tid]);
    }
  }
}

__global__ void collect_chunk_offsets_kernel(const u32 *d_bit_offsets,
                                             u32 input_size,
                                             u32 *d_chunk_offsets,
                                             u32 chunk_size_symbols,
                                             u32 num_chunks) {
  u32 chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_idx >= num_chunks)
    return;

  u32 symbol_idx = chunk_idx * chunk_size_symbols;
  if (symbol_idx < input_size) {
    d_chunk_offsets[chunk_idx] = d_bit_offsets[symbol_idx];
  }
}

// ============================================================================
// Host Huffman Tree/Table Builder
// ============================================================================

class HuffmanTreeBuilder {
  struct NodeComparator {
    const HuffmanNode *nodes;
    NodeComparator(const HuffmanNode *n) : nodes(n) {}
    bool operator()(int a, int b) const {
      return nodes[a].frequency > nodes[b].frequency;
    }
  };

public:
  static void build_tree(const u32 *frequencies, u32 num_symbols,
                         HuffmanNode *nodes, u32 &num_nodes, i32 &root_idx) {
    NodeComparator comp(nodes);
    std::priority_queue<int, std::vector<int>, NodeComparator> pq(comp);
    num_nodes = 0;
    for (u32 i = 0; i < num_symbols; ++i) {
      if (frequencies[i] > 0) {
        nodes[num_nodes] = {static_cast<u16>(i), frequencies[i],
                            HUFFMAN_NULL_IDX, HUFFMAN_NULL_IDX,
                            HUFFMAN_NULL_IDX};
        pq.push(num_nodes);
        num_nodes++;
      }
    }
    if (num_nodes == 0) {
      root_idx = HUFFMAN_NULL_IDX;
      return;
    }
    if (num_nodes == 1) {
      i32 leaf = pq.top();
      nodes[num_nodes] = {0, nodes[leaf].frequency, static_cast<u16>(leaf),
                          HUFFMAN_NULL_IDX, HUFFMAN_NULL_IDX};
      nodes[leaf].parent = static_cast<u16>(num_nodes);
      root_idx = num_nodes;
      num_nodes++;
      return;
    }
    while (pq.size() > 1) {
      int left = pq.top();
      pq.pop();
      int right = pq.top();
      pq.pop();
      int parent = num_nodes;
      nodes[parent] = {0, nodes[left].frequency + nodes[right].frequency,
                       static_cast<u16>(left), static_cast<u16>(right),
                       HUFFMAN_NULL_IDX};
      nodes[left].parent = static_cast<u16>(parent);
      nodes[right].parent = static_cast<u16>(parent);
      pq.push(parent);
      num_nodes++;
    }
    root_idx = pq.top();
  }
};

__host__ Status serialize_huffman_table(const u8 *h_code_lengths,
                                        unsigned char *h_output,
                                        u32 *header_size) {
  h_output[0] = MAX_HUFFMAN_BITS;
  u32 offset = 1;
  memcpy(h_output + offset, h_code_lengths, MAX_HUFFMAN_SYMBOLS);
  offset += MAX_HUFFMAN_SYMBOLS;

  *header_size = offset;
  return Status::SUCCESS;
}

__host__ Status deserialize_huffman_table(const unsigned char *h_input,
                                          u32 input_size, u8 *h_code_lengths,
                                          u32 *header_size) {
  if (input_size < 1 + MAX_HUFFMAN_SYMBOLS) {
    return Status::ERROR_CORRUPT_DATA;
  }
  u32 max_bits = h_input[0];
  if (max_bits > MAX_HUFFMAN_BITS) {
    return Status::ERROR_CORRUPT_DATA;
  }

  memcpy(h_code_lengths, h_input + 1, MAX_HUFFMAN_SYMBOLS);
  *header_size = 1 + MAX_HUFFMAN_SYMBOLS;

  return Status::SUCCESS;
}

// ============================================================================
// RFC 8878 Huffman Weight Decoding (Standard Zstandard Format)
// ============================================================================

/**
 * @brief Decode Huffman weights from direct 4-bit representation.
 * RFC 8878 Section 4.2.1.1: When headerByte >= 128, weights are stored
 * as 4-bit nibbles (2 weights per byte).
 *
 * @param h_input Input buffer starting at first weight byte
 * @param header_byte The header byte value (determines number of symbols)
 * @param h_weights Output: decoded weights (0-11 range)
 * @param num_symbols Output: number of decoded symbols
 * @return Status::SUCCESS on success
 */
__host__ Status decode_huffman_weights_direct(const unsigned char *h_input,
                                              u32 header_byte, u8 *h_weights,
                                              u32 *num_symbols) {
  if (header_byte < 128) {
    return Status::ERROR_CORRUPT_DATA; // Not direct format
  }

  *num_symbols =
      header_byte - 127; // RFC 8878: Number_of_Symbols = headerByte - 127

  // Read 4-bit weights, 2 per byte
  // RFC 8878: Weight[0] = (Byte[0] >> 4), Weight[1] = (Byte[0] & 0xf), etc.
  // Note: num_bytes = (*num_symbols + 1) / 2 is computed implicitly below

  for (u32 i = 0; i < *num_symbols; i++) {
    u32 byte_idx = i / 2;
    if (i % 2 == 0) {
      h_weights[i] = (h_input[byte_idx] >> 4) & 0x0F;
    } else {
      h_weights[i] = h_input[byte_idx] & 0x0F;
    }
  }

  return Status::SUCCESS;
}

/**
 * @brief Decode Huffman weights from FSE-compressed format.
 * RFC 8878 Section 4.2.1.2: When headerByte < 128, weights are FSE-encoded.
 * Uses two interleaved FSE states sharing one distribution table.
 *
 * @param h_input Input buffer starting at FSE table
 * @param compressed_size Size of compressed weights (= headerByte)
 * @param h_weights Output: decoded weights (0-11 range)
 * @param num_symbols Output: number of decoded symbols
 * @return Status::SUCCESS on success
 */
__host__ Status decode_huffman_weights_fse(const unsigned char *h_input,
                                           u32 compressed_size, u8 *h_weights,
                                           u32 *num_symbols) {
  // =========================================================================
  // Rewritten to faithfully follow libzstd reference implementation:
  //   - FSE_readNCount_body  (lib/common/entropy_common.c)
  //   - FSE_buildDTable_internal (lib/common/fse_decompress.c)
  //   - FSE_decompress_usingDTable_generic (lib/common/fse_decompress.c)
  //   - BIT_DStream (lib/common/bitstream.h)
  //
  // For Huffman weights, Max_Accuracy_Log is 6.
  // FSE_MIN_TABLELOG = 5.
  // =========================================================================

  if (compressed_size < 1) return Status::ERROR_CORRUPT_DATA;

  // --------------- Helper: highbit32 (position of highest set bit) ----------
  auto highbit32 = [](u32 v) -> u32 {
    // Returns floor(log2(v)) for v > 0. Undefined for v == 0.
    u32 r = 0;
    while (v >>= 1) r++;
    return r;
  };

  // --------------- Helper: countTrailingZeros32 -----------------------------
  auto ctz32 = [](u32 v) -> u32 {
    if (v == 0) return 32;
#ifdef __GNUC__
    return (u32)__builtin_ctz(v);
#else
    u32 r = 0;
    while ((v & 1) == 0) { v >>= 1; r++; }
    return r;
#endif
  };

  // --------------- Helper: read LE32 from byte pointer ----------------------
  auto read_le32 = [](const unsigned char *p) -> u32 {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) |
           ((u32)p[3] << 24);
  };

  // =========================================================================
  // PART 1: Parse NCount header (FSE_readNCount_body)
  // =========================================================================
  constexpr u32 FSE_MIN_TABLELOG = 5;
  constexpr u32 MAX_FSE_SYMBOLS = 256; // FSE_MAX_SYMBOL_VALUE + 1

  i16 norm_counts[MAX_FSE_SYMBOLS];
  memset(norm_counts, 0, sizeof(norm_counts));

  u32 maxSymbolValue = MAX_FSE_SYMBOLS - 1; // Will be updated
  u32 tableLog;

  // Pad input to at least 8 bytes for safe LE32 reads
  unsigned char padded[64];
  memset(padded, 0, sizeof(padded));
  u32 copy_size = compressed_size < 64 ? compressed_size : 64;
  memcpy(padded, h_input, copy_size);

  const unsigned char *ip = padded;
  const unsigned char *iend = padded + compressed_size;

  u32 bitStream = read_le32(ip);
  u32 nbBits = (bitStream & 0xF) + FSE_MIN_TABLELOG;
  if (nbBits > 15) { // FSE_TABLELOG_ABSOLUTE_MAX
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] FSE tableLog too large: %u\n", nbBits);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }
  bitStream >>= 4;
  i32 bitCount = 4;
  tableLog = nbBits;
  i32 remaining = (1 << nbBits) + 1;
  i32 threshold = 1 << nbBits;
  nbBits++;

  u32 charnum = 0;
  u32 maxSV1 = maxSymbolValue + 1;
  i32 previous0 = 0;

  for (;;) {
    if (previous0) {
      // Count runs of zero-probability symbols using 2-bit repeat codes.
      // Each 0b11 pair means "3 more zeros". The final pair < 3 gives remainder.
      u32 repeats = ctz32(~bitStream | 0x80000000u) >> 1;

      // Handle long runs (repeats >= 12 means we consumed 24+ bits, need reload)
      while (repeats >= 12) {
        charnum += 3 * 12;
        if (ip <= iend - 7) {
          ip += 3;
        } else {
          bitCount -= (i32)(8 * (iend - 7 - ip));
          bitCount &= 31;
          ip = iend - 4;
        }
        bitStream = read_le32(ip) >> bitCount;
        repeats = ctz32(~bitStream | 0x80000000u) >> 1;
      }
      charnum += 3 * repeats;
      bitStream >>= 2 * repeats;
      bitCount += 2 * repeats;

      // Add the final repeat (which isn't 0b11)
      charnum += bitStream & 3;
      bitCount += 2;

      if (charnum >= maxSV1) break;

      // Reload bitstream
      if ((ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
        ip += bitCount >> 3;
        bitCount &= 7;
      } else {
        bitCount -= (i32)(8 * (iend - 4 - ip));
        bitCount &= 31;
        ip = iend - 4;
      }
      bitStream = read_le32(ip) >> bitCount;
    }

    // Decode one symbol count using threshold-based variable-length coding
    {
      i32 max_val = (2 * threshold - 1) - remaining;
      i32 count;

      if ((i32)(bitStream & (threshold - 1)) < max_val) {
        count = (i32)(bitStream & (threshold - 1));
        bitCount += (i32)nbBits - 1;
      } else {
        count = (i32)(bitStream & (2 * threshold - 1));
        if (count >= threshold) count -= max_val;
        bitCount += (i32)nbBits;
      }

      count--; // Extra accuracy: count=0 in stream means probability=-1

      if (count >= 0) {
        remaining -= count;
      } else {
        // count == -1: symbol has probability "less than 1"
        remaining += count; // remaining -= 1
      }
      norm_counts[charnum++] = (i16)count;
      previous0 = !count; // If count==0, next symbols may be zero-run

      // Update threshold when remaining shrinks
      if (remaining < threshold) {
        if (remaining <= 1) break; // Normal termination
        nbBits = highbit32((u32)remaining) + 1;
        threshold = 1 << (nbBits - 1);
      }

      if (charnum >= maxSV1) break;

      // Reload bitstream
      if ((ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
        ip += bitCount >> 3;
        bitCount &= 7;
      } else {
        bitCount -= (i32)(8 * (iend - 4 - ip));
        bitCount &= 31;
        ip = iend - 4;
      }
      bitStream = read_le32(ip) >> bitCount;
    }
  }

  if (remaining != 1) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr,
            "[ERROR] FSE NCount remaining=%d (should be 1), charnum=%u\n",
            remaining, charnum);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }
  if (charnum > maxSV1) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] FSE charnum=%u > maxSV1=%u\n", charnum, maxSV1);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }
  if (bitCount > 32) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] FSE bitCount=%d > 32\n", bitCount);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

  u32 num_fse_symbols = charnum;  // Actual number of symbols in the FSE table
  u32 header_size = (u32)((ip - padded) + ((bitCount + 7) >> 3));

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr,
          "[HUF_FSE] Parsed %u symbols, remaining=%d, expected=1, "
          "header_size=%u\n",
          num_fse_symbols, remaining, header_size);
  fprintf(stderr, "[HUF_FSE] NormCounts: ");
  for (u32 i = 0; i < num_fse_symbols && i < 32; i++) {
    fprintf(stderr, "%d ", (int)norm_counts[i]);
  }
  fprintf(stderr, "\n");
#endif

  // Max accuracy log for Huffman weights is 6
  if (tableLog > 6) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] FSE tableLog=%u > 6 for Huffman weights\n",
            tableLog);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

  // =========================================================================
  // PART 2: Build FSE decode table (FSE_buildDTable_internal)
  // =========================================================================
  u32 table_size = 1 << tableLog;
  constexpr u32 MAX_TABLE_SIZE = 64; // 2^6 = 64 max for Huffman weights

  struct FSE_Entry {
    u8 symbol;
    u8 nbBits;
    u16 newState;
  };
  FSE_Entry decode_table[MAX_TABLE_SIZE];

  u16 symbolNext[MAX_FSE_SYMBOLS];
  u32 highThreshold = table_size - 1;

  // First pass: place low-probability (-1) symbols at high positions,
  // and initialize symbolNext for all symbols
  for (u32 s = 0; s < num_fse_symbols; s++) {
    if (norm_counts[s] == -1) {
      decode_table[highThreshold].symbol = (u8)s;
      highThreshold--;
      symbolNext[s] = 1;
    } else {
      symbolNext[s] = (u16)((norm_counts[s] > 0) ? norm_counts[s] : 0);
    }
  }

  // Second pass: spread symbols using FSE_TABLESTEP
  u32 step = (table_size >> 1) + (table_size >> 3) + 3;
  u32 tableMask = table_size - 1;
  u32 position = 0;

  for (u32 s = 0; s < num_fse_symbols; s++) {
    if (norm_counts[s] <= 0) continue; // Skip -1 and 0
    for (i32 i = 0; i < norm_counts[s]; i++) {
      decode_table[position].symbol = (u8)s;
      position = (position + step) & tableMask;
      // Skip positions occupied by low-probability symbols
      while (position > highThreshold) {
        position = (position + step) & tableMask;
      }
    }
  }

  if (position != 0) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr,
            "[ERROR] FSE symbol spreading failed: position=%u (should be 0)\n",
            position);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

  // Third pass: compute nbBits and newState for each table entry
  // Reference: tableDecode[u].nbBits = tableLog - highbit32(nextState)
  //            tableDecode[u].newState = (nextState << nbBits) - tableSize
  for (u32 u = 0; u < table_size; u++) {
    u8 sym = decode_table[u].symbol;
    u32 nextState = symbolNext[sym]++;
    u32 nb = (u32)(tableLog - highbit32(nextState));
    decode_table[u].nbBits = (u8)nb;
    decode_table[u].newState = (u16)((nextState << nb) - table_size);
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[HUF_FSE] Decode table (tableLog=%u, size=%u):\n",
          tableLog, table_size);
  for (u32 i = 0; i < table_size && i < 16; i++) {
    fprintf(stderr, "  [%2u] sym=%u nb=%u ns=%u\n", i,
            decode_table[i].symbol, decode_table[i].nbBits,
            decode_table[i].newState);
  }
#endif

  // =========================================================================
  // PART 3: Decode weights using BIT_DStream (FSE_decompress_usingDTable_generic)
  // =========================================================================
  if (header_size >= compressed_size) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr,
            "[ERROR] No bitstream data: header_size=%u >= compressed_size=%u\n",
            header_size, compressed_size);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

  const unsigned char *bs_start = h_input + header_size;
  u32 bs_size = compressed_size - header_size;

  // ----- BIT_initDStream -----
  // The bitstream is read BACKWARD. We load from the end.
  u64 bit_container = 0;
  u32 bits_consumed = 0;
  const unsigned char *bs_ptr;       // Current read pointer
  const unsigned char *bs_limitPtr;  // Safe reload boundary

  bs_limitPtr = bs_start + sizeof(u64); // = bs_start + 8

  if (bs_size >= sizeof(u64)) {
    // Normal case: load last 8 bytes as LE u64
    bs_ptr = bs_start + bs_size - sizeof(u64);
    for (u32 i = 0; i < 8; i++) {
      bit_container |= ((u64)bs_ptr[i]) << (i * 8);
    }
    u8 lastByte = bs_start[bs_size - 1];
    if (lastByte == 0) return Status::ERROR_CORRUPT_DATA;
    bits_consumed = 8 - highbit32((u32)lastByte);
  } else {
    // Small bitstream: load what we have with padding
    bs_ptr = bs_start;
    bit_container = (u64)bs_start[0];
    switch (bs_size) {
    case 7:
      bit_container += ((u64)bs_start[6]) << (64 - 16);
      /* fallthrough */
    case 6:
      bit_container += ((u64)bs_start[5]) << (64 - 24);
      /* fallthrough */
    case 5:
      bit_container += ((u64)bs_start[4]) << (64 - 32);
      /* fallthrough */
    case 4:
      bit_container += ((u64)bs_start[3]) << 24;
      /* fallthrough */
    case 3:
      bit_container += ((u64)bs_start[2]) << 16;
      /* fallthrough */
    case 2:
      bit_container += ((u64)bs_start[1]) << 8;
      /* fallthrough */
    default:
      break;
    }
    u8 lastByte = bs_start[bs_size - 1];
    if (lastByte == 0) return Status::ERROR_CORRUPT_DATA;
    bits_consumed = 8 - highbit32((u32)lastByte);
    bits_consumed += (u32)(sizeof(u64) - bs_size) * 8;
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr,
          "[HUF_FSE] Bitstream: header_size=%u, bs_size=%u, "
          "bits_consumed=%u, container=0x%016llx\n",
          header_size, bs_size, bits_consumed,
          (unsigned long long)bit_container);
#endif

  // ----- BIT_lookBits / BIT_readBits / BIT_reloadDStream -----

  // BIT_DStream_status enum
  enum DStreamStatus {
    DS_unfinished = 0,
    DS_endOfBuffer = 1,
    DS_completed = 2,
    DS_overflow = 3
  };

  auto bit_look = [&](u32 nbits) -> u32 {
    // BIT_getMiddleBits: extract nbits starting at position (64-bitsConsumed-nbits)
    u32 start = (u32)(64 - bits_consumed - nbits);
    return (u32)((bit_container >> start) & ((1ULL << nbits) - 1));
  };

  auto bit_skip = [&](u32 nbits) { bits_consumed += nbits; };

  auto bit_read = [&](u32 nbits) -> u32 {
    u32 val = bit_look(nbits);
    bit_skip(nbits);
    return val;
  };

  auto bit_reload = [&]() -> DStreamStatus {
    if (bits_consumed > 64) {
      return DS_overflow;
    }
    if (bs_ptr >= bs_limitPtr) {
      // Normal reload: move ptr back, reload 8 bytes
      bs_ptr -= (bits_consumed >> 3);
      bits_consumed &= 7;
      bit_container = 0;
      for (u32 i = 0; i < 8; i++) {
        bit_container |= ((u64)bs_ptr[i]) << (i * 8);
      }
      return DS_unfinished;
    }
    if (bs_ptr == bs_start) {
      if (bits_consumed < 64)
        return DS_endOfBuffer;
      return DS_completed;
    }
    // Cautious reload: partial
    u32 nbBytes = bits_consumed >> 3;
    DStreamStatus result = DS_unfinished;
    if (bs_ptr - nbBytes < bs_start) {
      nbBytes = (u32)(bs_ptr - bs_start);
      result = DS_endOfBuffer;
    }
    bs_ptr -= nbBytes;
    bits_consumed -= nbBytes * 8;
    bit_container = 0;
    // Read up to 8 bytes from bs_ptr (may be less than 8 if near start)
    u32 avail = (u32)(bs_start + bs_size - bs_ptr);
    u32 to_read = avail < 8 ? avail : 8;
    for (u32 i = 0; i < to_read; i++) {
      bit_container |= ((u64)bs_ptr[i]) << (i * 8);
    }
    return result;
  };

  // ----- FSE_initDState: read initial states -----
  u32 state1 = bit_read(tableLog);
  u32 state2 = bit_read(tableLog);

  DStreamStatus reload_status = bit_reload();
  if (reload_status == DS_overflow) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] BIT_DStream overflow after state init\n");
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[HUF_FSE] Initial states: s1=%u, s2=%u\n", state1, state2);
#endif

  // ----- FSE_decodeSymbol -----
  auto fse_decode = [&](u32 &state) -> u8 {
    FSE_Entry &e = decode_table[state];
    u8 sym = e.symbol;
    u32 nb = e.nbBits;
    u32 lowBits = bit_read(nb);
    state = (u32)e.newState + lowBits;
    return sym;
  };

  // ----- Main decode loop -----
  u32 out_idx = 0;
  constexpr u32 MAX_WEIGHTS = 255;

  // Fast loop: runs while bitstream is fully available (unfinished)
  // Decode 4 symbols per iteration like the reference
  while ((reload_status == DS_unfinished) && (out_idx + 3 < MAX_WEIGHTS)) {
    h_weights[out_idx++] = fse_decode(state1);

    // Reload if needed (for tableLog*2+7 > 64, always reload - but with
    // tableLog<=6, max bits per symbol = 6, so 4*6=24 < 57, no mid-reload needed)
    h_weights[out_idx++] = fse_decode(state2);
    h_weights[out_idx++] = fse_decode(state1);
    h_weights[out_idx++] = fse_decode(state2);

    reload_status = bit_reload();
  }

  // Tail loop: handle remaining symbols carefully
  // Reference pattern: decode state1, check reload, decode state2, check reload
  while (out_idx < MAX_WEIGHTS) {
    if (out_idx >= MAX_WEIGHTS) break;
    h_weights[out_idx++] = fse_decode(state1);
    reload_status = bit_reload();
    if (reload_status == DS_overflow) {
      // Overflow after state1: emit state2's current symbol and done
      if (out_idx < MAX_WEIGHTS) {
        h_weights[out_idx++] = decode_table[state2].symbol;
      }
      break;
    }

    if (out_idx >= MAX_WEIGHTS) break;
    h_weights[out_idx++] = fse_decode(state2);
    reload_status = bit_reload();
    if (reload_status == DS_overflow) {
      // Overflow after state2: emit state1's current symbol and done
      if (out_idx < MAX_WEIGHTS) {
        h_weights[out_idx++] = decode_table[state1].symbol;
      }
      break;
    }
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[HUF_FSE] Decoded %u weights: ", out_idx);
  for (u32 i = 0; i < out_idx && i < 32; i++) {
    fprintf(stderr, "%u ", (unsigned)h_weights[i]);
  }
  if (out_idx > 32) fprintf(stderr, "...");
  fprintf(stderr, "\n");
#endif

  *num_symbols = out_idx;
  return Status::SUCCESS;
}

/**
 * @brief Convert weights to code lengths according to RFC 8878 Section 4.2.1.
 */
__host__ Status weights_to_code_lengths(const u8 *h_weights, u32 num_weights,
                                        u8 *h_code_lengths, u32 *max_bits) {
  memset(h_code_lengths, 0, MAX_HUFFMAN_SYMBOLS);

  // First pass: find sum of 2^(Weight-1) to determine Max_Number_of_Bits
  // RFC 8878: The last weight is adjusted so the sum is an exact power of 2.
  u32 weight_sum = 0;
  for (u32 i = 0; i < num_weights; i++) {
    if (h_weights[i] > 0) {
      weight_sum += 1U << (h_weights[i] - 1);
    }
  }

  if (weight_sum == 0) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[W2CL] ERROR: weight_sum==0, num_weights=%u\n", num_weights);
#endif
    return Status::ERROR_CORRUPT_DATA;
  }
#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[W2CL] weight_sum=%u, num_weights=%u\n", weight_sum, num_weights);
#endif

  // Determine tableLog following libzstd convention:
  //   tableLog = highbit32(weight_sum) + 1
  // This always gives a tableLog such that 2^tableLog > weight_sum,
  // ensuring there's always room for the last (implicit) symbol.
  // Our old code used "smallest k such that 2^k >= weight_sum" which
  // produced k == highbit32(weight_sum) when weight_sum is a power of 2,
  // resulting in last_weight_val = 0 and NO last symbol — WRONG.
  u32 max_num_bits = 0;
  {
    u32 temp = weight_sum;
    while (temp > 0) {
      temp >>= 1;
      max_num_bits++;
    }
    // max_num_bits = floor(log2(weight_sum)) + 1 = highbit32(weight_sum) + 1
  }

  // RFC 8878: The last weight's value fills the gap to 2^tableLog
  u32 next_power = 1U << max_num_bits;
  u32 last_weight_val = next_power - weight_sum; // Always > 0 with this tableLog
  u32 last_weight = 0;
  if (last_weight_val > 0) {
    // The last weight is such that 2^(last_weight-1) = last_weight_val
    // => last_weight = log2(last_weight_val) + 1
    u32 log_val = 0;
    u32 temp = last_weight_val;
    while (temp > 1) {
      temp >>= 1;
      log_val++;
    }
    // Verify it's exactly a power of 2
    if ((1U << log_val) == last_weight_val) {
      last_weight = log_val + 1;
    } else {
      // Not a power of 2 — corrupt data per RFC
#ifdef CUDA_ZSTD_DEBUG
      fprintf(stderr, "[W2CL] ERROR: last_weight_val=%u is not a power of 2\n", last_weight_val);
#endif
      return Status::ERROR_CORRUPT_DATA;
    }
  }

  *max_bits = max_num_bits;

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[W2CL] max_num_bits=%u, next_power=%u, last_weight_val=%u, last_weight=%u\n",
          max_num_bits, next_power, last_weight_val, last_weight);
#endif

  // Convert weights to code lengths: bits = MaxBits + 1 - Weight
  // RFC 8878: C[w] = (MaxBits + 1 - w) for w in weights
  for (u32 i = 0; i < num_weights; i++) {
    if (h_weights[i] > 0) {
      // Code length = max_bits + 1 - weight
      u8 code_len = (u8)(max_num_bits + 1 - h_weights[i]);
      h_code_lengths[i] = code_len > 0 ? code_len : 0;
    } else {
      h_code_lengths[i] = 0;
    }
  }

  // Add reconstructed last symbol if not 0
  if (num_weights < MAX_HUFFMAN_SYMBOLS && last_weight > 0) {
    u8 code_len = (u8)(max_num_bits + 1 - last_weight);
    h_code_lengths[num_weights] = code_len > 0 ? code_len : 0;
  }

  return Status::SUCCESS;
}

/**
 * @brief RFC 8878-compliant Huffman table decoder entry point.
 * Parses Huffman tree description from Zstandard-formatted input.
 *
 * @param h_input Input buffer (header byte + weights data)
 * @param input_size Size of input buffer
 * @param h_code_lengths Output: code lengths for all 256 symbols
 * @param header_size Output: bytes consumed from input
 * @return Status::SUCCESS on success
 */
__host__ Status deserialize_huffman_table_rfc8878(const unsigned char *h_input,
                                                  u32 input_size,
                                                  u8 *h_code_lengths,
                                                  u32 *header_size) {
  if (input_size < 1) {
    return Status::ERROR_CORRUPT_DATA;
  }

  u8 header_byte = h_input[0];

  // Check for custom format: [MaxBits(1)][CodeLengths(256)]
  // serialize_huffman_table writes MAX_HUFFMAN_BITS (=24) as first byte
  if (header_byte == MAX_HUFFMAN_BITS &&
      input_size >= 1 + MAX_HUFFMAN_SYMBOLS) {
    memcpy(h_code_lengths, h_input + 1, MAX_HUFFMAN_SYMBOLS);
    *header_size = 1 + MAX_HUFFMAN_SYMBOLS;
    return Status::SUCCESS;
  }

  u8 h_weights[MAX_HUFFMAN_SYMBOLS] = {0};
  u32 num_symbols = 0;
  Status status;

  if (header_byte >= 128) {
    // Direct representation (4-bit weights)
    *header_size = 1 + ((header_byte - 127 + 1) / 2); // header + weight bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_direct(h_input + 1, header_byte, h_weights,
                                           &num_symbols);
  } else {
    // FSE-compressed representation
    *header_size = 1 + header_byte; // header + compressed bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_fse(h_input + 1, header_byte, h_weights,
                                        &num_symbols);
  }

  if (status != Status::SUCCESS) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[ERROR] decode_huffman_weights_fse status=%d\n",
            (int)status);
#endif
    return status;
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[HUF_TABLE] FSE decode OK, num_symbols=%u\n", num_symbols);
#endif

  // Convert weights to code lengths
  u32 max_bits;
  status = weights_to_code_lengths(h_weights, num_symbols, h_code_lengths,
                                   &max_bits);

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[HUF_TABLE] weights_to_code_lengths status=%d, max_bits=%u\n",
          (int)status, max_bits);
  if (status == Status::SUCCESS) {
    // Print first 20 non-zero code lengths
    int printed = 0;
    for (int i = 0; i < MAX_HUFFMAN_SYMBOLS && printed < 20; i++) {
      if (h_code_lengths[i] > 0) {
        fprintf(stderr, "[HUF_TABLE]   sym[%d]=%u\n", i, h_code_lengths[i]);
        printed++;
      }
    }
  }
#endif

  return status;
}

// ============================================================================
// Huffman Encoding (Parallel)
// ============================================================================

/**
 * @brief (NEW) Kernel to get the code length for each symbol in parallel.
 */
__global__ void get_symbol_lengths_kernel(const unsigned char *input,
                                          u32 input_size,
                                          const HuffmanCode *codes,
                                          u32 *d_code_lengths // Output
) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size)
    return;

  u8 symbol = input[idx];
  d_code_lengths[idx] = codes[symbol].length;
}

/**
 * @brief Phase 1: Store encoded symbols in thread-local format
 * Each thread stores its code and position info for later merging.
 */
__global__ void
huffman_encode_phase1_kernel(const unsigned char *input, u32 input_size,
                             const HuffmanCode *codes, const u32 *d_bit_offsets,
                             u32 *d_codes_out,    // Output: code values
                             u32 *d_lengths_out,  // Output: code lengths
                             u32 *d_positions_out // Output: bit positions
) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size)
    return;

  u8 symbol = input[idx];
  const HuffmanCode &c = codes[symbol];

  // Store code info for phase 2
  d_codes_out[idx] = c.code;
  d_lengths_out[idx] = c.length;
  d_positions_out[idx] = d_bit_offsets[idx];

  // Debug output for first few symbols
#ifdef CUDA_ZSTD_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0 && idx < 10) {
    printf("[ENCODE-PHASE1] idx=%u symbol=%u('%c') code=0x%X len=%u pos=%u\n",
           idx, symbol, symbol >= 32 ? symbol : '?', c.code, c.length,
           d_bit_offsets[idx]);
  }
#endif
}

/**
 * @brief Phase 2: Merge encoded symbols into final bitstream (ATOMIC-FREE)
 *
 * Uses block-level cooperation: each block processes a contiguous range
 * of symbols, building output in shared memory, then writing coalesced
 * to global memory. No atomics needed.
 */
__global__ void
huffman_encode_phase2_kernel(const u32 *d_codes, const u32 *d_lengths,
                             const u32 *d_positions, u32 input_size,
                             unsigned char *output, u32 header_size_bits) {
  const u32 BUFFER_SIZE = 512; // Bytes per block buffer
  static_assert((BUFFER_SIZE % 4) == 0,
                "BUFFER_SIZE must be a multiple of 4 for 32-bit atomics");
  const u32 BUFFER_WORDS = BUFFER_SIZE / 4;
  __shared__ u32 shared_words[BUFFER_WORDS];

  // Each block processes THREADS_PER_BLOCK symbols
  u32 block_start_idx = blockIdx.x * blockDim.x;
  u32 block_end_idx = min(block_start_idx + blockDim.x, input_size);

  if (block_start_idx >= input_size)
    return;

  // Clear shared buffer in 32-bit words; ensures 4-byte alignment for atomics
  for (u32 i = threadIdx.x; i < BUFFER_WORDS; i += blockDim.x) {
    shared_words[i] = 0u;
  }
  __syncthreads();

  // Find block's bit range
  u32 block_first_bit =
      (block_start_idx == 0) ? 0 : d_positions[block_start_idx];
  u32 block_last_bit =
      (block_end_idx == 0)
          ? 0
          : (d_positions[block_end_idx - 1] + d_lengths[block_end_idx - 1]);

  u32 global_start_bit = header_size_bits + block_first_bit;
  u32 global_bit_offset = global_start_bit & 7;

  // Each thread encodes its symbol into shared buffer
  u32 idx = block_start_idx + threadIdx.x;
  if (idx < block_end_idx) {
    u32 code = d_codes[idx];
    u32 length = d_lengths[idx];
    u32 bit_pos = d_positions[idx];

    if (idx < 5) {
#ifdef CUDA_ZSTD_DEBUG
      printf("[ENCODE-PHASE2] idx=%u block=%u code=%u len=%u pos=%u\n", idx,
             blockIdx.x, code, length, bit_pos);
#endif
    }

    if (length > 0) {
      // Position relative to block start, PLUS global offset
      u32 local_bit_pos = (bit_pos - block_first_bit) + global_bit_offset;
      u32 local_byte_pos = local_bit_pos >> 3;
      u32 local_bit_offset = local_bit_pos & 7;

      // Write to shared memory (within block, positions don't overlap in
      // bits)
      u64 shifted_code = static_cast<u64>(code) << local_bit_offset;

      // Write up to 3 bytes (max for 15-bit code + 7-bit offset)
      for (u32 i = 0; i < 3 && local_byte_pos + i < BUFFER_SIZE; i++) {
        u8 byte_val = (shifted_code >> (i * 8)) & 0xFF;
        if (byte_val != 0) {
          // Use 32-bit atomic OR on 4-byte aligned words in shared memory.
          // atomicOr on a byte pointer is invalid if the address is not
          // 4-byte aligned; compute the aligned word index and shift
          // the byte into the correct position.
          u32 byte_offset = local_byte_pos + i;
          u32 aligned_word_idx = byte_offset >> 2; // /4
          u32 byte_shift = (byte_offset & 3) * 8;  // 0..24
          // shared_words is 4-byte aligned by construction
          atomicOr(&shared_words[aligned_word_idx],
                   (u32)byte_val << byte_shift);
#ifdef CUDA_ZSTD_DEBUG
          if (idx < 5) {
            printf("[ENCODE-WRITE] idx=%u byte_offset=%u word=%u shift=%u "
                   "val=0x%02X\n",
                   idx, byte_offset, aligned_word_idx, byte_shift, byte_val);
          }
#endif
        }
      }
    }
  }
  __syncthreads();

  // Cooperatively write shared buffer to global memory (coalesced)
  u32 global_byte_start = global_start_bit >> 3;
  u32 total_bits = (block_last_bit - block_first_bit) + global_bit_offset;
  u32 bytes_to_write = (total_bits + 7) / 8;

  // Write from the word-aligned shared array back to bytes for global output
  // Use atomic OR for ALL bytes to ensure bits from multiple blocks don't race
  const unsigned char *shared_bytes =
      reinterpret_cast<const unsigned char *>(shared_words);
  for (u32 i = threadIdx.x; i < bytes_to_write && i < BUFFER_SIZE;
       i += blockDim.x) {
    // Always use atomic OR to handle potential overlaps with other blocks
    atomicOrByte(output + global_byte_start + i, shared_bytes[i]);
  }
}

/**
 * @brief Optimized Parallel Huffman encoding kernel (REDUCED ATOMICS).
 *
 * This kernel uses warp-level aggregation to reduce atomic operations
 * by 32x. Each warp of 32 threads shares atomic operations, significantly
 * reducing global memory contention.
 */
__global__ void parallel_huffman_encode_kernel(
    const unsigned char *input, u32 input_size, const HuffmanCode *codes,
    const u32 *d_bit_offsets, // Input from prefix sum
    unsigned char *output, u32 header_size_bits) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size)
    return;

  u8 symbol = input[idx];
  const HuffmanCode &c = codes[symbol];

  if (c.length == 0)
    return;

  u32 bit_pos = header_size_bits + d_bit_offsets[idx];
  u32 byte_pos = bit_pos >> 3;
  u32 bit_offset = bit_pos & 7;

  // Warp-level optimization: Use warp shuffle to reduce atomic operations
  const u32 lane_id = threadIdx.x & 31;

  // Each thread prepares its write
  [[maybe_unused]] u64 shifted_code = static_cast<u64>(c.code) << bit_offset;

  // Determine which 64-bit word this thread writes to
  u32 word_idx = byte_pos >> 3;

  // Use warp vote to see if multiple threads write to same word
  u32 my_word = word_idx;
  u32 same_word_mask = __match_any_sync(0xFFFFFFFF, my_word);

  // If I'm the first thread in my word group, I do the write
  bool should_write = (__ffs(same_word_mask) - 1) == lane_id;

  if (should_write) {
    // Aggregate writes from all threads in this word group
    u64 aggregated_value = 0;

    for (u32 i = 0; i < 32; i++) {
      if (same_word_mask & (1u << i)) {
        u32 other_code = __shfl_sync(same_word_mask, c.code, i);
        u32 other_len = __shfl_sync(same_word_mask, c.length, i);
        u32 other_offset = __shfl_sync(same_word_mask, bit_offset, i);

        u64 other_shifted = static_cast<u64>(other_code) << other_offset;
        aggregated_value |= other_shifted;
      }
    }

    // Single atomic write for the entire warp group.
    // If the 64-bit target is misaligned we split into two 32-bit atomics
    // to avoid invalid 64-bit atomic on platforms that require 8-byte
    // alignment.
    unsigned char *out_ptr = output + (word_idx << 3);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_ptr);
    if ((out_addr & 7) == 0) {
      atomicOr(reinterpret_cast<unsigned long long *>(out_ptr),
               aggregated_value);
    } else {
      // Fall back to two 32-bit atomicOr operations. This assumes
      // the device global allocation is at least 4-byte aligned.
      u32 *out_words = reinterpret_cast<u32 *>(out_ptr);
      atomicOr(&out_words[0], (u32)aggregated_value);
      atomicOr(&out_words[1], (u32)(aggregated_value >> 32));
    }
  }
}

// ============================================================================
// Huffman Decoding (REPLACED with Parallel)
// ============================================================================

/**
 * @brief Sequential kernel to build the decode table.
 * This is fast and small, no need to parallelize.
 */
__global__ void
build_decode_table_kernel(const u8 *code_lengths, u32 num_symbols,
                          u32 *d_first_code,   // [MAX_HUFFMAN_BITS + 2]
                          u16 *d_symbol_index, // [MAX_HUFFMAN_BITS + 1]
                          u8 *d_symbols        // [num_symbols]
) {
  // This kernel is run with one thread (1,1)
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Count symbols per length
  u32 length_count[MAX_HUFFMAN_BITS + 1] = {0};
  u32 max_len = 0;
  for (u32 i = 0; i < num_symbols; ++i) {
    if (code_lengths[i] > 0 && code_lengths[i] <= MAX_HUFFMAN_BITS) {
      length_count[code_lengths[i]]++;
      max_len = max(max_len, (u32)code_lengths[i]);
    }
  }



  // Build first_code table (canonical Huffman codes)
  // RFC 8878 Section 4.1.1: Canonical Huffman codes
  // first_code[len] = (first_code[len-1] + length_count[len-1]) << 1
  u32 code = 0;
  d_first_code[0] = 0;
  for (u32 len = 1; len <= max_len; ++len) {
    d_first_code[len] = code;
    code = (code + length_count[len])
           << 1; // FIXED: Must left-shift for next length
  }
  d_first_code[max_len + 1] = 0xFFFFFFFF; // Sentinel

  // Build symbol index table
  // d_symbol_index[len] = sum of length_count[1..len-1]
  // i.e., starting index in d_symbols for symbols of length len
  u32 idx = 0;
  for (u32 len = 1; len <= max_len; ++len) {
    d_symbol_index[len] = static_cast<u16>(idx);
    idx += length_count[len];
  }
  // Store total count at max_len + 1 for count_at_len calculation
  d_symbol_index[max_len + 1] = static_cast<u16>(idx);

  // Fill symbols array in canonical order
  idx = 0;
  for (u32 len = 1; len <= max_len; ++len) {
    for (u32 sym = 0; sym < num_symbols; ++sym) {
      if (code_lengths[sym] == len) {
        d_symbols[idx] = static_cast<u8>(sym);

        idx++;
      }
    }
  }

  // Store max_length
  d_symbol_index[0] = max_len;
}

// ============================================================================
// DTable-based Huffman Decode (zstd reference-compatible)
// ============================================================================

/**
 * @brief Build a flat DTable for O(1) Huffman symbol lookup.
 *
 * This matches the reference zstd HUF_readDTableX1_wksp approach:
 *   - Table has (1 << tableLog) entries
 *   - Each entry stores {nbBits, symbol}
 *   - To decode: peek tableLog bits -> entry = dtable[val] -> symbol = entry.symbol,
 *     advance bits_consumed by entry.nbBits
 *
 * @param code_lengths  Array of code lengths per symbol (0 = unused)
 * @param num_symbols   Number of symbols (typically 256)
 * @param d_dtable      Output: flat decode table of HuffmanDecoderEntry
 * @param table_log     The table log (= max code length from the Huffman table)
 */
__global__ void build_huffman_dtable_kernel(
    const u8 *code_lengths, u32 num_symbols,
    HuffmanDecoderEntry *d_dtable, u32 table_log) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  u32 table_size = 1U << table_log;

  // Zero-init the table
  for (u32 i = 0; i < table_size; i++) {
    d_dtable[i].num_bits = 0;
    d_dtable[i].symbol = 0;
  }

  // Fill DTable entries matching libzstd's HUF_readDTableX1_wksp order:
  //   - Iterate by WEIGHT from 1 to tableLog (i.e., by code length from
  //     tableLog DOWN to 1 — longest codes first, shortest codes last).
  //   - Weight w  →  nbBits = tableLog + 1 - w
  //   - Each symbol with weight w gets 2^(w-1) = (1 << (tableLog - nbBits)) entries.
  //   - Within each weight/length, symbols appear in ascending order (canonical).
  //
  // This puts longest codes (weight 1, nbBits=tableLog) at position 0 with
  // 1 entry each, and shortest codes (weight tableLog, nbBits=1) at the end
  // with 2^(tableLog-1) entries each.
  //
  // The zstd compressor assigns the smallest code VALUES to the longest codes,
  // so the MSB-first DTable lookup (val = top tableLog bits) must map small
  // values → longest codes.  Our old code filled shortest codes first at
  // position 0, which was INVERTED relative to the compressor's code assignment.
  u32 pos = 0; // Current position in DTable
  for (i32 nbBits = (i32)table_log; nbBits >= 1; nbBits--) {
    u32 num_entries = 1U << (table_log - (u32)nbBits); // Entries per symbol at this length
    for (u32 sym = 0; sym < num_symbols; sym++) {
      if (code_lengths[sym] == (u32)nbBits) {
        for (u32 j = 0; j < num_entries; j++) {
          if (pos < table_size) {
            d_dtable[pos].num_bits = (u8)nbBits;
            d_dtable[pos].symbol = (u8)sym;
            pos++;
          }
        }
      }
    }
  }

#ifdef CUDA_ZSTD_DEBUG
  printf("[DTABLE] Built DTable: tableLog=%u, tableSize=%u, filled %u entries\n",
         table_log, table_size, pos);
  // Print first 16 entries for verification
  for (u32 i = 0; i < 16 && i < table_size; i++) {
    printf("[DTABLE] entry[%u] = {nbBits=%u, symbol=%u('%c')}\n",
           i, d_dtable[i].num_bits, d_dtable[i].symbol,
           d_dtable[i].symbol >= 32 ? d_dtable[i].symbol : '?');
  }
#endif
}

/**
 * @brief DTable-based 4-stream Huffman decode kernel (zstd reference-compatible).
 *
 * Reads the Huffman bitstream BACKWARD per the Zstandard spec using exact
 * BIT_DStream semantics from the reference implementation:
 *   - Init: load LE u64 from end of stream, find sentinel bit
 *   - Decode: peek tableLog MSB bits -> O(1) DTable lookup -> symbol + nbBits
 *   - Refill: after EVERY symbol (not just when bits_consumed >= 32)
 *   - Output: symbols in reverse order (backward bitstream = last symbol first)
 *
 * This replaces the old canonical-code-iteration kernel that had subtle bugs.
 */
__global__ void huffman_decode_dtable_kernel(
    const unsigned char *input, u32 stream_bytes,
    const HuffmanDecoderEntry *d_dtable, u32 table_log,
    unsigned char *output, u32 total_regen_size,
    u32 output_start_offset, u32 stream_id_debug,
    u32 num_symbols_to_decode) {

  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  if (stream_bytes == 0 || num_symbols_to_decode == 0)
    return;

  // --- Step 1: Find sentinel bit in last byte ---
  u8 last_byte = input[stream_bytes - 1];
  if (last_byte == 0) return; // No sentinel — corrupt

  // highbit32: position of highest set bit (0-indexed from LSB)
  u32 high_bit = 0;
  for (i32 i = 7; i >= 0; i--) {
    if (last_byte & (1 << i)) {
      high_bit = (u32)i;
      break;
    }
  }

  // --- Step 2: Initialize BIT_DStream ---
  // Reference: BIT_initDStream
  // Load up to 8 bytes LE from the end of the stream
  const unsigned char *stream_start = input;
  u32 load_start = (stream_bytes >= 8) ? (stream_bytes - 8) : 0;
  u32 load_bytes = stream_bytes - load_start;

  u64 bit_container = 0;
  for (u32 i = 0; i < load_bytes; i++) {
    bit_container |= ((u64)input[load_start + i]) << (i * 8);
  }

  // bits_consumed = (64 - load_bytes*8) [empty top bits] + (8 - high_bit) [padding + sentinel]
  u32 bits_consumed = 64 - load_bytes * 8 + (8 - high_bit);

  // ptr points to current window start; refills move it backward
  const unsigned char *ptr = input + load_start;

  // limitPtr: we can do fast reloads as long as ptr >= limitPtr
  // Reference: sizeof(bitD->bitContainer) = 8, so limitPtr = start + 8
  const unsigned char *limitPtr = stream_start + 8;

#ifdef CUDA_ZSTD_DEBUG
  printf("[DTABLE-KERNEL] stream=%u bytes=%u last_byte=0x%02X high_bit=%u "
         "bits_consumed=%u load_start=%u load_bytes=%u tableLog=%u\n",
         stream_id_debug, stream_bytes, last_byte, high_bit,
         bits_consumed, load_start, load_bytes, table_log);
#endif

  // --- BIT_reloadDStream implementation ---
  // Returns: 0=unfinished, 1=endOfBuffer, 2=completed, 3=overflow
  // We use an inline lambda-like approach via a local variable

  // --- Step 3: Decode symbols ---
  u32 num_decoded = 0;

  // Fast loop: while we have enough margin for fast reloads
  // Reference: BIT_reloadDStreamFast requires ptr >= limitPtr (i.e., at least 8 bytes before ptr)
  while (num_decoded + 3 < num_symbols_to_decode && ptr >= limitPtr) {
    // Decode 4 symbols per iteration (like the reference HUF_decodeStreamX1)
    for (u32 k = 0; k < 4 && num_decoded < num_symbols_to_decode; k++) {
      // BIT_lookBitsFast: peek tableLog bits from MSB
      u32 val = (u32)((bit_container << bits_consumed) >> (64 - table_log));
      u8 symbol = d_dtable[val].symbol;
      u8 nbBits = d_dtable[val].num_bits;
      bits_consumed += nbBits;

      // Output in reverse order (backward bitstream)
      u32 out_idx = output_start_offset + (num_symbols_to_decode - 1 - num_decoded);
      if (out_idx < total_regen_size) {
        output[out_idx] = symbol;
      }

#ifdef CUDA_ZSTD_DEBUG
      if (num_decoded < 8) {
        printf("[DTABLE-KERNEL] stream=%u sym[%u]=%u('%c') val=0x%X nbBits=%u "
               "out_idx=%u bits_consumed=%u\n",
               stream_id_debug, num_decoded, symbol,
               symbol >= 32 ? symbol : '?', val, nbBits, out_idx, bits_consumed);
      }
#endif
      num_decoded++;
    }

    // BIT_reloadDStreamFast: fast reload (no bounds check needed since ptr >= limitPtr)
    {
      u32 nbBytes = bits_consumed >> 3;
      ptr -= nbBytes;
      bits_consumed -= nbBytes * 8;
      // Reload 8 bytes LE from ptr
      bit_container = 0;
      for (u32 i = 0; i < 8; i++) {
        bit_container |= ((u64)ptr[i]) << (i * 8);
      }
    }
  }

  // Tail loop: careful decoding with bounds-checked reloads
  while (num_decoded < num_symbols_to_decode) {
    // Check if we have enough bits
    u32 remaining = 64 - bits_consumed;
    if (remaining < table_log) {
      // Try to reload
      if (bits_consumed > 64) break; // overflow

      if (ptr >= limitPtr) {
        // Normal reload
        u32 nbBytes = bits_consumed >> 3;
        ptr -= nbBytes;
        bits_consumed -= nbBytes * 8;
        bit_container = 0;
        for (u32 i = 0; i < 8; i++) {
          bit_container |= ((u64)ptr[i]) << (i * 8);
        }
      } else if (ptr > stream_start) {
        // Cautious reload
        u32 nbBytes = bits_consumed >> 3;
        if (ptr - nbBytes < stream_start) {
          nbBytes = (u32)(ptr - stream_start);
        }
        ptr -= nbBytes;
        bits_consumed -= nbBytes * 8;
        // Read available bytes from ptr (may be < 8)
        u32 avail = (u32)((stream_start + stream_bytes) - ptr);
        if (avail > 8) avail = 8;
        bit_container = 0;
        for (u32 i = 0; i < avail; i++) {
          bit_container |= ((u64)ptr[i]) << (i * 8);
        }
      } else {
        // At stream_start, no more bytes to load
        // Check if we have enough bits remaining
        remaining = 64 - bits_consumed;
        if (remaining < table_log) {
          // Pad with zeros for final lookup — remaining bits are valid
          // The DTable lookup with fewer bits may still work if the code is short enough
          if (remaining == 0) break;
        }
      }
      remaining = 64 - bits_consumed;
      if (remaining < 1) break; // Truly exhausted
    }

    // Decode one symbol
    u32 peek_bits = (remaining >= table_log) ? table_log : remaining;
    u32 val = (u32)((bit_container << bits_consumed) >> (64 - table_log));
    // If we have fewer bits than tableLog, the lookup is still valid because
    // the low bits are zero-padded, and the DTable entry will have nbBits <= remaining
    u8 symbol = d_dtable[val].symbol;
    u8 nbBits = d_dtable[val].num_bits;

    if (nbBits > remaining) {
      // Not enough bits for this symbol — we're done
#ifdef CUDA_ZSTD_DEBUG
      printf("[DTABLE-KERNEL] stream=%u STOPPED at sym %u/%u: nbBits=%u > remaining=%u\n",
             stream_id_debug, num_decoded, num_symbols_to_decode, nbBits, remaining);
#endif
      break;
    }

    bits_consumed += nbBits;

    u32 out_idx = output_start_offset + (num_symbols_to_decode - 1 - num_decoded);
    if (out_idx < total_regen_size) {
      output[out_idx] = symbol;
    }

    num_decoded++;

    // Reload after every symbol in tail loop (reference: BIT_reloadDStream)
    if (bits_consumed > 64) break;

    if (ptr >= limitPtr) {
      u32 nbBytes = bits_consumed >> 3;
      ptr -= nbBytes;
      bits_consumed -= nbBytes * 8;
      bit_container = 0;
      for (u32 i = 0; i < 8; i++) {
        bit_container |= ((u64)ptr[i]) << (i * 8);
      }
    } else if (ptr > stream_start) {
      u32 nbBytes = bits_consumed >> 3;
      if (ptr - nbBytes < stream_start) {
        nbBytes = (u32)(ptr - stream_start);
      }
      if (nbBytes > 0) {
        ptr -= nbBytes;
        bits_consumed -= nbBytes * 8;
        u32 avail = (u32)((stream_start + stream_bytes) - ptr);
        if (avail > 8) avail = 8;
        bit_container = 0;
        for (u32 i = 0; i < avail; i++) {
          bit_container |= ((u64)ptr[i]) << (i * 8);
        }
      }
    }
    // else: at stream_start, can't reload, but may still have bits
  }

#ifdef CUDA_ZSTD_DEBUG
  printf("[DTABLE-KERNEL] stream=%u done: decoded %u/%u symbols\n",
         stream_id_debug, num_decoded, num_symbols_to_decode);
#endif
}

/**
 * @brief (NEW) Pass 1 for Parallel Decode: Find chunk start bits.
 * This kernel is SEQUENTIAL (<<<1, 1>>>) and scans the bitstream
 * to find the starting bit_pos for each chunk.
 */
__global__ void find_chunk_start_bits_kernel(
    const unsigned char *input, u32 header_size_bytes, u32 input_size_bytes,
    const u32 *d_first_code, const u16 *d_symbol_index, const u8 *d_symbols,
    u32 decompressed_size, u32 num_chunks, u32 symbols_per_chunk,
    u32 *d_chunk_start_bits // Output
) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  u32 max_len = d_symbol_index[0]; // Read max_length

  u32 bit_pos = header_size_bytes * 8;
  const u32 end_bit_pos = input_size_bytes * 8;
  u32 out_idx = 0;

  d_chunk_start_bits[0] = bit_pos; // First chunk starts after header

  for (u32 chunk = 1; chunk < num_chunks; ++chunk) {
    u32 symbols_to_decode = symbols_per_chunk;

    while (symbols_to_decode > 0 && out_idx < decompressed_size) {
      if (bit_pos + max_len > end_bit_pos) {
        if (bit_pos >= end_bit_pos)
          break;
      }

      u32 byte_pos = bit_pos >> 3;
      u32 bit_offset = bit_pos & 7;

      u64 value = 0;
      memcpy(&value, input + byte_pos, min(8u, input_size_bytes - byte_pos));
      value >>= bit_offset;

      u32 code = 0;
      u32 len = 1;
      for (; len <= max_len; ++len) {
        code = value & ((1U << len) - 1);
        u32 normal_code = reverse_bits_device(code, len);
        if (normal_code < (d_first_code[len + 1] >> 1)) {
          break;
        }
      }

      if (len > max_len)
        return; // Corrupt

      bit_pos += len;
      out_idx++;
      symbols_to_decode--;
    }

    d_chunk_start_bits[chunk] = bit_pos;
  }

  // Last chunk start is used to find total size
  d_chunk_start_bits[num_chunks] = bit_pos;
}

/**
 * @brief Decode one Huffman symbol from the MSB of a bit container.
 *
 * @param bit_container  The 64-bit container loaded in LE order
 * @param bits_consumed  How many MSB bits have been consumed so far
 * @param d_first_code   First canonical code at each length
 * @param d_symbol_index Starting index in d_symbols for each length
 * @param d_symbols      Symbols array in canonical order
 * @param max_len        Maximum code length
 * @param consumed       [out] How many bits this symbol used (0 = failure)
 */
__device__ u8 decode_huff_symbol(u64 bit_container, u32 bits_consumed,
                                 const u32 *d_first_code,
                                 const u16 *d_symbol_index, const u8 *d_symbols,
                                 u32 max_len, u32 &consumed) {
  u32 remaining = 64 - bits_consumed;
  for (u32 len = 1; len <= max_len; ++len) {
    if (len > remaining)
      break;
    // Extract top `len` bits from the unconsumed portion (MSB-first)
    u32 code = (u32)((bit_container << bits_consumed) >> (64 - len));
    u32 first_code = d_first_code[len];
    u32 count_at_len = d_symbol_index[len + 1] - d_symbol_index[len];
    if (count_at_len > 0 && code >= first_code &&
        code < first_code + count_at_len) {
      consumed = len;
      u32 idx = d_symbol_index[len] + (code - first_code);
      return d_symbols[idx];
    }
  }

  consumed = 0;
  return 0;
}

/**
 * @brief Parallel Huffman 4-stream decoder kernel (RFC 8878 compliant).
 *
 * Reads the Huffman bitstream BACKWARD per the Zstandard spec:
 *   - Bytes are loaded as little-endian u64 from the current pointer
 *   - Bits are extracted from the MSB of the container (top-down)
 *   - The pointer moves backward through the stream for refills
 *   - Symbols are emitted in reverse order (last symbol first)
 *
 * This matches the reference zstd BIT_DStream implementation.
 */
__global__ void huffman_decode_rfc8878_kernel(
    const unsigned char *input, u32 input_size, const u32 *d_first_code,
    const u16 *d_symbol_index, const u8 *d_symbols, unsigned char *output,
    u32 total_regen_size, u32 stream_start_bits, u32 stream_end_bits,
    u32 output_start_offset, u32 stream_id_debug, u32 num_symbols_to_decode) {

  u32 max_len = d_symbol_index[0];
  u32 num_decoded = 0;

  // --- Step 1: Determine sub-stream byte range ---
  u32 stream_bytes = (stream_end_bits + 7) / 8;
  if (stream_bytes == 0) return;

  // --- Step 2: Find sentinel bit in the last byte ---
  // Per RFC 8878 §4.2.1: The last byte has a sentinel '1' bit followed by
  // padding '0' bits. We need to find the highest set bit in the last byte.
  u8 last_byte = input[stream_bytes - 1];
  if (last_byte == 0) return; // No sentinel — corrupt

  u32 high_bit = 0; // Bit position of sentinel within the last byte (0=LSB, 7=MSB)
  for (i32 i = 7; i >= 0; i--) {
    if (last_byte & (1 << i)) {
      high_bit = (u32)i;
      break;
    }
  }
  // padding_bits = number of zero bits above the sentinel = 7 - high_bit
  // Total overhead in last byte = padding_bits + 1 (sentinel) = 8 - high_bit

  // --- Step 3: Load initial bit container (LE u64 from end of stream) ---
  // We want to load bytes ending at the end of the stream
  u32 load_start = (stream_bytes >= 8) ? (stream_bytes - 8) : 0;
  u32 load_bytes = stream_bytes - load_start;

  u64 bit_container = 0;
  for (u32 i = 0; i < load_bytes; i++) {
    bit_container |= ((u64)input[load_start + i]) << (i * 8);
  }

  // bits_consumed: how many MSB bits of the 64-bit container we've consumed.
  // The container has `load_bytes * 8` meaningful bits in positions [0, load_bytes*8).
  // The last byte of the stream is at container position [(load_bytes-1)*8, load_bytes*8).
  // The sentinel is at container bit: (load_bytes - 1) * 8 + high_bit.
  // Everything at or above that bit is overhead (sentinel + padding).
  // From the MSB (bit 63): bits_consumed = 64 - ((load_bytes-1)*8 + high_bit)
  //                                       = 64 - load_bytes*8 + 8 - high_bit
  u32 bits_consumed = 64 - load_bytes * 8 + (8 - high_bit);
  // This consumes: (64-load_bytes*8) empty top bits + (7-high_bit) padding + 1 sentinel

  // Pointer for refills — next bytes are below load_start
  const unsigned char *ptr = input + load_start;
  const unsigned char *stream_start = input; // Don't read before stream start

#ifdef CUDA_ZSTD_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[HUFF-KERNEL] stream=%u bytes=%u last_byte=0x%02X high_bit=%u "
           "bits_consumed=%u load_start=%u load_bytes=%u\n",
           stream_id_debug, stream_bytes, last_byte, high_bit,
           bits_consumed, load_start, load_bytes);
  }
#endif

  // --- Step 4: Decode symbols ---
  while (num_decoded < num_symbols_to_decode) {
    // Refill: when bits_consumed >= 32 and we have bytes left to load
    if (bits_consumed >= 32 && ptr > stream_start) {
      u32 bytes_back = bits_consumed >> 3; // How many full bytes consumed
      // Don't go below stream start
      if (ptr - bytes_back < stream_start) {
        bytes_back = (u32)(ptr - stream_start);
      }
      ptr -= bytes_back;
      bits_consumed -= bytes_back * 8;

      // Reload container from new ptr position
      u32 avail = (u32)((input + stream_bytes) - ptr);
      if (avail > 8) avail = 8;
      bit_container = 0;
      for (u32 i = 0; i < avail; i++) {
        bit_container |= ((u64)ptr[i]) << (i * 8);
      }
    }

    u32 remaining = 64 - bits_consumed;
    if (remaining == 0) break;

    // Try to decode a symbol: extract top `len` bits from the unconsumed portion
    u32 found = 0;
    for (u32 len = 1; len <= max_len; len++) {
      if (len > remaining) break;

      // Extract top `len` bits of the unconsumed data.
      // Unconsumed data starts at bit position (63 - bits_consumed) counting from bit 0.
      // Shift container left by bits_consumed to put unconsumed at MSB,
      // then shift right by (64 - len) to extract top `len` bits.
      u32 code = (u32)((bit_container << bits_consumed) >> (64 - len));

      // Canonical Huffman: check if code matches [first_code, first_code+count)
      u32 count_at_len = d_symbol_index[len + 1] - d_symbol_index[len];
      if (count_at_len > 0 && code >= d_first_code[len] &&
          code < d_first_code[len] + count_at_len) {
        u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];

        // Symbols are decoded in reverse order (backward bitstream).
        // The first decoded symbol is the LAST symbol of this stream segment.
        u32 symbol_idx_within_stream = num_symbols_to_decode - 1 - num_decoded;
        u32 out_idx = output_start_offset + symbol_idx_within_stream;

        if (out_idx < total_regen_size) {
          output[out_idx] = symbol;
        }

#ifdef CUDA_ZSTD_DEBUG
        if (threadIdx.x == 0 && blockIdx.x == 0 && num_decoded < 8) {
          printf("[HUFF-KERNEL] stream=%u sym[%u]=%u('%c') code=0x%X len=%u "
                 "out_idx=%u bits_consumed=%u\n",
                 stream_id_debug, num_decoded, symbol,
                 symbol >= 32 ? symbol : '?', code, len, out_idx,
                 bits_consumed);
        }
#endif

        num_decoded++;
        bits_consumed += len;
        found = 1;
        break;
      }
    }

    if (!found) {
#ifdef CUDA_ZSTD_DEBUG
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        u32 remaining_bits = 64 - bits_consumed;
        u32 peek = 0;
        if (remaining_bits >= max_len)
          peek = (u32)((bit_container << bits_consumed) >> (64 - max_len));
        printf("[HUFF-KERNEL] stream=%u FAILED at sym %u/%u, remaining=%u "
               "peek=0x%X max_len=%u\n",
               stream_id_debug, num_decoded, num_symbols_to_decode,
               remaining_bits, peek, max_len);
      }
#endif
      break;
    }
  }

#ifdef CUDA_ZSTD_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[HUFF-KERNEL] stream=%u done: decoded %u/%u symbols\n",
           stream_id_debug, num_decoded, num_symbols_to_decode);
  }
#endif
}

// ============================================================================
// Host API Functions
// ============================================================================

Status encode_huffman(const unsigned char *d_input, u32 input_size,
                      const HuffmanTable &table, unsigned char *d_output,
                      size_t *output_size, // Host pointer
                      CompressionWorkspace *workspace, cudaStream_t stream) {
  // Clear any pre-existing sticky CUDA error (Blackwell/sm_120 cudaMalloc quirk)
  cudaGetLastError();

  if (!d_input || !d_output || !output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // --- 1. Analyze frequencies - USE WORKSPACE BUFFER ---
  u32 *d_frequencies = workspace ? workspace->d_frequencies : nullptr;
  bool allocated_temp = false;

  if (!d_frequencies) {
    // Fallback: allocate if no workspace provided (backward compatibility)
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
    allocated_temp = true;
  }
  CUDA_CHECK(cudaMemsetAsync(d_frequencies, 0,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u32), stream));

  int threads = HUFFMAN_ENCODE_THREADS;
  int blocks = (input_size + threads - 1) / threads;

  analyze_frequencies_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, d_frequencies);

  // Use pinned memory for async transfer
  u32 *h_frequencies = nullptr;
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc_host(&h_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));

  CUDA_CHECK(cudaMemcpyAsync(h_frequencies, d_frequencies,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u32),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(
      stream)); // Required: need frequencies for tree building

  // Only free if we allocated it ourselves
  if (allocated_temp) {
    CUDA_CHECK(cudaFree(d_frequencies));
  }

  // --- 2. Build Tables on Host ---
  HuffmanNode *h_nodes = new HuffmanNode[MAX_HUFFMAN_SYMBOLS * 2];
  u32 num_nodes = 0;
  i32 root_idx = -1;
  HuffmanTreeBuilder::build_tree(h_frequencies, MAX_HUFFMAN_SYMBOLS, h_nodes,
                                  num_nodes, root_idx);

  u8 *h_code_lengths = new u8[MAX_HUFFMAN_SYMBOLS];
  // (FIX) Need to actually generate the code lengths from the tree
  memset(h_code_lengths, 0, MAX_HUFFMAN_SYMBOLS);
  std::function<void(int, u8)> find_lengths = [&](int node_idx, u8 depth) {
    if (node_idx == HUFFMAN_NULL_IDX)
      return;

    // Leaf node
    if (h_nodes[node_idx].left_child == HUFFMAN_NULL_IDX) {
      h_code_lengths[h_nodes[node_idx].symbol] = depth;
      return;
    }

    if (depth < MAX_HUFFMAN_BITS) {
      find_lengths(h_nodes[node_idx].left_child, depth + 1);
      find_lengths(h_nodes[node_idx].right_child, depth + 1);
    }
  };
  find_lengths(root_idx, 0);

  // Use canonical Huffman codes (RFC 8878 compliant)
  HuffmanCode *h_codes = new HuffmanCode[MAX_HUFFMAN_SYMBOLS];
  
  Status status =
      huffman::generate_canonical_codes(h_code_lengths, MAX_HUFFMAN_SYMBOLS,
                                        h_codes // Generate into host buffer
      );
  if (status != Status::SUCCESS) {
    cudaFreeHost(h_frequencies); // FIX: Use cudaFreeHost for pinned memory
    delete[] h_nodes;
    delete[] h_code_lengths;
    delete[] h_codes;
    return status;
  }

  // --- 3. Serialize table header ---
  // Format:
  // [MaxBits(1)][CodeLengths(256)][ChunkOffsets(NumChunks*4)][Bitstream...]
  // We need to calculate NumChunks first
  u32 chunk_size_symbols = 4096; // HUFFMAN_DECODE_SYMBOLS_PER_CHUNK
  u32 num_chunks = (input_size + chunk_size_symbols - 1) / chunk_size_symbols;

  unsigned char *h_header = new unsigned char[1 + MAX_HUFFMAN_SYMBOLS];
  u32 header_size = 0;
  serialize_huffman_table(h_code_lengths, h_header, &header_size);

  // Write basic header
  CUDA_CHECK(cudaMemcpyAsync(d_output, h_header, header_size,
                             cudaMemcpyHostToDevice, stream));

  // We will write offsets later, after calculating them.
  // But we need to reserve space.
  u32 offsets_size = num_chunks * sizeof(u32);
  u32 total_header_size = header_size + offsets_size;
  u32 header_size_bits = total_header_size * 8;

  // --- 4. Parallel Encode ---

  // Copy codes to device (use the table's device pointer)
  CUDA_CHECK(cudaMemcpyAsync(table.codes, h_codes,
                             MAX_HUFFMAN_SYMBOLS * sizeof(HuffmanCode),
                             cudaMemcpyHostToDevice, stream));

  // Allocate temp buffers for scan - USE WORKSPACE BUFFERS
  u32 *d_code_lengths = workspace ? workspace->d_code_lengths : nullptr;
  u32 *d_bit_offsets = workspace ? workspace->d_bit_offsets : nullptr;
  bool allocated_scan_buffers = false;

  if (!d_code_lengths || !d_bit_offsets) {
    // Fallback: allocate if no workspace
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_code_lengths, input_size * sizeof(u32)));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_bit_offsets, input_size * sizeof(u32)));
    allocated_scan_buffers = true;
  }
  // Clear output area (bitstream only, not header/offsets)
  // We clear less conservatively to avoid going out of bounds
  size_t bitstream_max_size = input_size * 2;
  CUDA_CHECK(cudaMemsetAsync(d_output + total_header_size, 0,
                              bitstream_max_size, stream));

  // Allocate buffers for two-phase atomic-free encoding
  u32 *d_codes_temp = nullptr;
  u32 *d_positions_temp = nullptr;
  bool allocated_phase_buffers = false;

  // Allocate temporary buffers (workspace doesn't have temp_storage members)
  {
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_codes_temp, input_size * sizeof(u32)));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_positions_temp, input_size * sizeof(u32)));
    allocated_phase_buffers = true;
  }

  // Pass 1a: Get length of each symbol
  get_symbol_lengths_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, table.codes, d_code_lengths);

  // Pass 1b: Parallel prefix sum to compute bit offsets
  status = cuda_zstd::utils::parallel_scan(d_code_lengths, d_bit_offsets,
                                            input_size, stream);
  if (status != Status::SUCCESS) {
    if (allocated_phase_buffers) {
      cudaFree(d_codes_temp);
      cudaFree(d_positions_temp);
    }
    if (allocated_scan_buffers) {
      cudaFree(d_code_lengths);
      cudaFree(d_bit_offsets);
    }
    cudaFreeHost(h_frequencies);
    delete[] h_nodes;
    delete[] h_code_lengths;
    delete[] h_header;
    delete[] h_codes;
    return status;
  }

  // --- Collect Chunk Offsets ---
  u32 *d_chunk_offsets = nullptr;
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_chunk_offsets, offsets_size));

  u32 collect_threads = 256;
  u32 collect_blocks = (num_chunks + collect_threads - 1) / collect_threads;
  collect_chunk_offsets_kernel<<<collect_blocks, collect_threads, 0, stream>>>(
      d_bit_offsets, input_size, d_chunk_offsets, chunk_size_symbols,
      num_chunks);

  // Write offsets to output (after table)
  CUDA_CHECK(cudaMemcpyAsync(d_output + header_size, d_chunk_offsets,
                             offsets_size, cudaMemcpyDeviceToDevice, stream));
  cudaFree(d_chunk_offsets);

  // Pass 2a: Store codes and positions (Phase 1 of atomic-free encoding)
  huffman_encode_phase1_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, table.codes, d_bit_offsets, d_codes_temp,
      d_code_lengths, d_positions_temp);

  // Pass 2b: Merge into final bitstream using block-cooperative approach
  // (ATOMIC-FREE) Each block builds its segment in shared memory, then writes
  // coalesced to global
  huffman_encode_phase2_kernel<<<blocks, threads, 0, stream>>>(
      d_codes_temp, d_code_lengths, d_positions_temp, input_size, d_output,
      header_size_bits // Global bit offset (includes offsets array)
  );

  // Cleanup phase buffers
  if (allocated_phase_buffers) {
    cudaFree(d_codes_temp);
    cudaFree(d_positions_temp);
  }

  // --- 5. Get final size ---
  u32 h_last_offset = 0, h_last_length = 0;
  if (input_size > 0) {
    CUDA_CHECK(cudaMemcpy(&h_last_offset, d_bit_offsets + (input_size - 1),
                          sizeof(u32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_last_length, d_code_lengths + (input_size - 1),
                          sizeof(u32), cudaMemcpyDeviceToHost));
  }
  u32 total_bits = h_last_offset + h_last_length;
  *output_size = total_header_size + ((total_bits + 7) / 8);

  // Cleanup - only free if we allocated
  if (allocated_scan_buffers) {
    cudaFree(d_code_lengths);
    cudaFree(d_bit_offsets);
  }
  cudaFreeHost(h_frequencies); // Free pinned memory
  delete[] h_nodes;
  delete[] h_code_lengths;
  delete[] h_header;
  delete[] h_codes;

  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

Status decode_huffman(const unsigned char *d_input,
                      size_t input_size,         // Full size of compressed data
                      const HuffmanTable &table, // Not used
                      unsigned char *d_output,
                      size_t *d_output_size, // This is a host pointer
                      u32 decompressed_size, // We know this
                      cudaStream_t stream) {
  // Use RFC 8878 compliant decoder (assume single stream if called via legacy
  // API)
  return decode_huffman_rfc8878(d_input, input_size, d_output, d_output_size,
                                decompressed_size, false, stream);
}

/**
 * @brief RFC 8878-compliant Huffman decode for standard Zstandard format.
 * Parses Huffman tree from weights (FSE or direct encoded), then decodes
 * literals using backward bitstream reading per RFC 8878 Section 4.2.
 *
 * @param d_input Device pointer to compressed Huffman data (starts with
 * header)
 * @param input_size Size of compressed data in bytes
 * @param d_output Device pointer to output buffer
 * @param d_output_size Host pointer for output size
 * @param decompressed_size Expected decompressed size
 * @param four_streams True if 4-stream format (size_format >= 1)
 * @param stream CUDA stream
 * @return Status::SUCCESS on success
 */

/**
 * @brief Forward-reading Huffman decoder kernel for single-stream mode.
 * Reads the bitstream forward (as written by the encoder).
 * Canonical codes are written LSB-first, so we read LSB-first without reversal.
 */
__global__ void huffman_decode_forward_kernel(
    const unsigned char *input, u32 input_size, const u32 *d_first_code,
    const u16 *d_symbol_index, const u8 *d_symbols, unsigned char *output,
    u32 total_regen_size, u32 bitstream_start_bits) {

  u32 max_len = d_symbol_index[0];
  u32 bit_pos = bitstream_start_bits;
  const u32 end_bit_pos = input_size * 8;
  u32 num_decoded = 0;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef CUDA_ZSTD_DEBUG
    printf("[DECODE-FWD] max_len=%u, input_size=%u, total=%u, start_bits=%u\n",
           max_len, input_size, total_regen_size, bitstream_start_bits);
    printf(
        "[DECODE-FWD] first_code[1]=%u, first_code[2]=%u, first_code[3]=%u\n",
        d_first_code[1], d_first_code[2], d_first_code[3]);
    printf("[DECODE-FWD] symbol_index[1]=%u, symbol_index[2]=%u, "
           "symbol_index[3]=%u\n",
           d_symbol_index[1], d_symbol_index[2], d_symbol_index[3]);
#endif
  }

  // Bit container for forward reading
  u64 bit_container = 0;
  u32 bits_available = 0;

  while (num_decoded < total_regen_size) {
    // Refill bit container when low
    while (bits_available <= 56 && bit_pos < end_bit_pos) {
      u32 byte_idx = bit_pos >> 3;
      if (byte_idx >= input_size)
        break;
      u32 bits_in_byte = min(8u, end_bit_pos - bit_pos);
      u8 next_byte = input[byte_idx] & ((1u << bits_in_byte) - 1);
      bit_container |= ((u64)next_byte) << bits_available;
      bits_available += bits_in_byte;
      bit_pos += bits_in_byte;
    }

    if (bits_available == 0)
      break;

    // Decode symbol - read LSB-first (codes are stored LSB-first in bitstream)
    // Read from bottom of container, consuming bits as we go

    u32 code = 0; // Declare outside loop
    u32 len = 0;
    for (u32 l = 1; l <= max_len; l++) {
      if (l > bits_available)
        break;
      // Extract bottom 'l' bits (LSB-first reading from bitstream)
      u32 raw_code = (u32)(bit_container & ((1U << l) - 1));
      // Reverse to get canonical code (canonical codes are MSB-first)
      code = reverse_bits_device(raw_code, l);

      u32 count_at_len = d_symbol_index[l + 1] - d_symbol_index[l];
#ifdef CUDA_ZSTD_DEBUG
      if (threadIdx.x == 0 && blockIdx.x == 0 && num_decoded < 5) {
        printf(
            "[DECODE-FWD] l=%u raw=0x%X code=0x%X first=%u count=%u match=%d\n",
            l, raw_code, code, d_first_code[l], count_at_len,
            (count_at_len > 0 && code >= d_first_code[l] &&
             code < d_first_code[l] + count_at_len)
                ? 1
                : 0);
      }
#endif

      if (count_at_len > 0 && code >= d_first_code[l] &&
          code < d_first_code[l] + count_at_len) {
        len = l;
        break;
      }
    }

    if (len == 0) {
#ifdef CUDA_ZSTD_DEBUG
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[DECODE-FWD] Failed to decode at symbol %u, bits_avail=%u\n",
               num_decoded, bits_available);
      }
#endif
      break;
    }

    u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];
    output[num_decoded] = symbol;

#ifdef CUDA_ZSTD_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0 && num_decoded < 10) {
      printf("[DECODE-FWD] sym[%u]=%u('%c') code=0x%X len=%u bits_avail=%u\n",
             num_decoded, symbol, symbol >= 32 ? symbol : '?', code, len,
             bits_available);
    }
#endif

    num_decoded++;
    bit_container >>= len;
    bits_available -= len;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef CUDA_ZSTD_DEBUG
    printf("[DECODE-FWD] Done: decoded %u symbols\n", num_decoded);
#endif
  }
}

Status decode_huffman_rfc8878(const unsigned char *d_input, size_t input_size,
                              unsigned char *d_output, size_t *d_output_size,
                              u32 decompressed_size, bool four_streams,
                              cudaStream_t stream) {
  if (!d_input || !d_output || !d_output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // --- 1. Read and Build Huffman Tree ---
  u8 h_code_lengths[MAX_HUFFMAN_SYMBOLS] = {0};
  u32 huf_header_size = 0;

  // Weights are at the beginning of d_input
  unsigned char *h_weights_data = new unsigned char[input_size];
  CUDA_CHECK(cudaMemcpyAsync(h_weights_data, d_input, input_size,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Status status = deserialize_huffman_table_rfc8878(
      h_weights_data, input_size, h_code_lengths, &huf_header_size);
  delete[] h_weights_data;
#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[DECODE_HUF] deserialize status=%d, huf_header_size=%u, input_size=%zu\n",
          (int)status, huf_header_size, input_size);
#endif
  if (status != Status::SUCCESS)
    return status;

  // Compute max code length (= tableLog for DTable)
  u32 max_bits = 0;
  for (u32 i = 0; i < MAX_HUFFMAN_SYMBOLS; i++) {
    if (h_code_lengths[i] > max_bits)
      max_bits = h_code_lengths[i];
  }
  if (max_bits == 0) {
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[DECODE_HUF] ERROR: max_bits==0, no Huffman codes\n");
#endif
    return Status::ERROR_CORRUPT_DATA;
  }

#ifdef CUDA_ZSTD_DEBUG
  fprintf(stderr, "[DECODE_HUF] max_bits (tableLog) = %u\n", max_bits);
#endif

  // --- 2. Allocate and upload code_lengths to GPU ---
  u8 *d_code_lengths;
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_code_lengths, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));
  CUDA_CHECK(cudaMemcpyAsync(d_code_lengths, h_code_lengths,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u8),
                             cudaMemcpyHostToDevice, stream));

  const unsigned char *d_bitstream_base = d_input + huf_header_size;
  u32 bitstream_size_base = (u32)(input_size - huf_header_size);

  if (four_streams) {
    // --- 3a. Build DTable for O(1) Huffman decode ---
    u32 table_log = max_bits;
    u32 dtable_size = 1U << table_log;
    HuffmanDecoderEntry *d_dtable;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_dtable, dtable_size * sizeof(HuffmanDecoderEntry)));

    build_huffman_dtable_kernel<<<1, 1, 0, stream>>>(
        d_code_lengths, MAX_HUFFMAN_SYMBOLS, d_dtable, table_log);

#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[DECODE_HUF] four_streams: bitstream_size_base=%u, huf_header_size=%u, tableLog=%u, dtable_size=%u\n",
            bitstream_size_base, huf_header_size, table_log, dtable_size);
#endif
    if (bitstream_size_base < 6) {
#ifdef CUDA_ZSTD_DEBUG
      fprintf(stderr, "[DECODE_HUF] ERROR: bitstream_size_base(%u) < 6\n", bitstream_size_base);
#endif
      cudaFree(d_dtable);
      cudaFree(d_code_lengths);
      return Status::ERROR_CORRUPT_DATA;
    }

    u16 stream_sizes[3];
    CUDA_CHECK(cudaMemcpyAsync(stream_sizes, d_bitstream_base, 6,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    u32 L1 = stream_sizes[0];
    u32 L2 = stream_sizes[1];
    u32 L3 = stream_sizes[2];
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[DECODE_HUF] L1=%u, L2=%u, L3=%u, 6+L1+L2+L3=%u, bitstream_size_base=%u\n",
            L1, L2, L3, 6+L1+L2+L3, bitstream_size_base);
#endif
    if (6 + L1 + L2 + L3 > bitstream_size_base) {
#ifdef CUDA_ZSTD_DEBUG
      fprintf(stderr, "[DECODE_HUF] ERROR: stream sizes overflow: 6+%u+%u+%u=%u > %u\n",
              L1, L2, L3, 6+L1+L2+L3, bitstream_size_base);
#endif
      cudaFree(d_dtable);
      cudaFree(d_code_lengths);
      return Status::ERROR_CORRUPT_DATA;
    }
    u32 L4 = bitstream_size_base - 6 - L1 - L2 - L3;

    // RFC 8878 / zstd reference: segmentSize = (dstSize + 3) / 4
    // N1 = N2 = N3 = segmentSize, N4 = dstSize - 3*segmentSize
    u32 segmentSize = (decompressed_size + 3) / 4;
    u32 N1 = segmentSize;
    u32 N2 = segmentSize;
    u32 N3 = segmentSize;
    u32 N4 = decompressed_size - 3 * segmentSize;

    const unsigned char *d_data = d_bitstream_base + 6;
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[DECODE_HUF] Launching 4-stream DTable kernels: L1=%u L2=%u L3=%u L4=%u, N1=%u N2=%u N3=%u N4=%u, decompressed_size=%u\n",
            L1, L2, L3, L4, N1, N2, N3, N4, decompressed_size);
#endif
    huffman_decode_dtable_kernel<<<1, 1, 0, stream>>>(
        d_data, L1, d_dtable, table_log, d_output,
        decompressed_size, 0, 0, N1);
    huffman_decode_dtable_kernel<<<1, 1, 0, stream>>>(
        d_data + L1, L2, d_dtable, table_log, d_output,
        decompressed_size, N1, 1, N2);
    huffman_decode_dtable_kernel<<<1, 1, 0, stream>>>(
        d_data + L1 + L2, L3, d_dtable, table_log, d_output,
        decompressed_size, N1 + N2, 2, N3);
    huffman_decode_dtable_kernel<<<1, 1, 0, stream>>>(
        d_data + L1 + L2 + L3, L4, d_dtable, table_log, d_output,
        decompressed_size, N1 + N2 + N3, 3, N4);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_dtable);
  } else {
    // --- 3b. Single-stream: use old canonical code tables + forward decoder ---
    u32 *d_first_code;
    u16 *d_symbol_index;
    u8 *d_symbols;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_first_code, (MAX_HUFFMAN_BITS + 2) * sizeof(u32)));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_symbol_index, (MAX_HUFFMAN_BITS + 2) * sizeof(u16)));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_symbols, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));

    build_decode_table_kernel<<<1, 1, 0, stream>>>(
        d_code_lengths, MAX_HUFFMAN_SYMBOLS, d_first_code, d_symbol_index,
        d_symbols);

    // Calculate chunk offsets size and skip past them
    u32 chunk_size_symbols = 4096;
    u32 num_chunks =
        (decompressed_size + chunk_size_symbols - 1) / chunk_size_symbols;
    u32 offsets_size = num_chunks * sizeof(u32);

    // Calculate actual bitstream size: total compressed size minus header and
    // offsets
    u32 bitstream_actual_size =
        (u32)input_size - huf_header_size - offsets_size;

    // The decoder reads from bitstream_base, which is d_input + huf_header_size
    // But we need to skip the chunk offsets within that region
    const unsigned char *d_bitstream_with_offsets =
        d_input + huf_header_size + offsets_size;

    // Start from bit 0 of the adjusted bitstream (we've already skipped
    // header+offsets)
    huffman_decode_forward_kernel<<<1, 1, 0, stream>>>(
        d_bitstream_with_offsets, bitstream_actual_size, d_first_code,
        d_symbol_index, d_symbols, d_output, decompressed_size, 0);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_first_code);
    cudaFree(d_symbol_index);
    cudaFree(d_symbols);
  }

  cudaFree(d_code_lengths);

  *d_output_size = decompressed_size;
  return Status::SUCCESS;
}

/**
 * @brief (HELPER) A simple __device__ bitstream reader for Huffman.
 *
 * Reads the stream FORWARD. Not optimized for parallel reads.
 */
struct HuffmanBitStreamReader {
  const unsigned char *stream_ptr;
  const unsigned char *stream_end;
  u64 bit_container;
  i32 bits_remaining;

  /**
   * @brief Initializes the reader.
   *
   * @param stream_start Points to the beginning of the bitstream.
   * @param stream_size The total size in bytes.
   */
  __device__ void init(const unsigned char *stream_start, size_t stream_size) {
    stream_ptr = stream_start;
    stream_end = stream_start + stream_size;
    bit_container = 0;
    bits_remaining = 0;

    // Pre-load the bit container
    reload();
    reload(); // Load up to 64 bits
  }

  /**
   * @brief Ensures the bit container has at least 32 bits, if possible.
   */
  __device__ void reload() {
    if (bits_remaining <= 32 && stream_ptr <= stream_end - 4) {
      u64 next_bits = *reinterpret_cast<const u32 *>(stream_ptr);
      stream_ptr += 4;
      bit_container |= (next_bits << bits_remaining);
      bits_remaining += 32;
    }
  }

  /**
   * @brief Peeks at `num_bits` without consuming them.
   */
  __device__ u32 peek(u32 num_bits) {
    return bit_container & ((1ULL << num_bits) - 1);
  }

  /**
   * @brief Consumes `num_bits` from the stream.
   */
  __device__ void consume(u32 num_bits) {
    bit_container >>= num_bits;
    bits_remaining -= num_bits;

    // Reload if we are running low
    reload();
  }
};

Status free_huffman_decoder_table(HuffmanDecoderTable *table,
                                  cudaStream_t stream) {
  if (table->d_table) {
    CUDA_CHECK(cudaFreeAsync(table->d_table, stream));
    table->d_table = nullptr;
  }
  return Status::SUCCESS;
}

} // namespace huffman
} // namespace cuda_zstd
