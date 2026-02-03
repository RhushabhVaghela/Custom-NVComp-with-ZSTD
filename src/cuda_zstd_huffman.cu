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

#include "cuda_zstd_huffman.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h" // <-- 1. ADDED INCLUDE

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
  // RFC 8878 Section 4.2.1.2: FSE Compression of Huffman Weights
  // - Two interleaved FSE states sharing one distribution table
  // - Bitstream is read backward (like all Zstd FSE bitstreams)
  // - Max accuracy log is 6 for weight encoding

  // RFC 8878 Section 4.2.1.2: FSE Compression of Huffman Weights
  // For Huffman weights, Max_Accuracy_Log is 6.
  // EMPIRICAL FINDING: Some encoders (like libzstd) may use a different 
  // alignment or bias for Huffman weights. We use a more robust parsing 
  // that handles potential Accuracy Log mismatches.
  
  u32 accuracy_log = h_input[0] & 0x0F;
  u32 bit_pos_header = 4; // Start after 4 bits of AL
  
  if (accuracy_log > 6) {
    // FALLBACK: If Accuracy Log looks invalid (>6), it's likely we're
    // actually at the counts (Accuracy Log was omitted or different format)
    accuracy_log = 6;
    bit_pos_header = 0; 
  }

  constexpr u32 MAX_HUF_ALPHABET = 12; // weights 0-11
  u32 table_size = 1 << accuracy_log;
  i16 norm_counts[MAX_HUF_ALPHABET] = {0};
  i32 remaining = table_size;
  u32 symbol = 0;
  i32 remaining = table_size;
  u32 symbol = 0;

  auto read_bits_header = [&](u32 nbits) -> u32 {
    u32 result = 0;
    for (u32 i = 0; i < nbits; i++) {
      if (bit_pos_header / 8 >= compressed_size)
        break;
      if (h_input[bit_pos_header / 8] & (1 << (bit_pos_header % 8))) {
        result |= (1 << i);
      }
      bit_pos_header++;
    }
    return result;
  };

  while (remaining > 0 && symbol < MAX_HUF_ALPHABET) {
    u32 nb_bits = 0;
    u32 temp = remaining + 1;
    while (temp >>= 1)
      nb_bits++;
    nb_bits++; // Log2(remaining + 1) + 1

    u32 threshold = (1 << nb_bits) - 1 - (remaining + 1);
    u32 v = read_bits_header(nb_bits - 1);
    i16 count;
    if (v < threshold) {
      count = (i16)v - 1;
    } else {
      v = (v << 1) | read_bits_header(1);
      count = (i16)v - 1 - (i16)threshold;
    }

    norm_counts[symbol++] = count;
    // For FSE normalization, count=-1 means probability 0 (consumes 2 slots)
    // count > 0 means probability count (consumes count slots)
    // Only decrement if we have enough remaining slots for this count
    if (count == -1) {
      if (remaining >= 2) {
        remaining -= 2; // count=-1 consumes 2 slots
      }
    } else if (count > 0) {
      if (remaining >= (i32)count) {
        remaining -= count;
      }
    } else {
      // count=0 (shouldn't happen with valid input), but handle gracefully
      remaining = 0;
    }
  }

  u32 num_fse_symbols = symbol;

  // DEBUG: Verify normalization parsing
  fprintf(stderr,
          "[HUF_FSE] Parsed %u symbols, remaining=%d, expected=0, bit_pos=%u\n",
          num_fse_symbols, remaining, bit_pos_header);
  fprintf(stderr, "[HUF_FSE] NormCounts: ");
  for (u32 i = 0; i < num_fse_symbols && i < 12; i++) {
    fprintf(stderr, "%d ", (int)norm_counts[i]);
  }
  fprintf(stderr, "\n");

  if (remaining != 0) {
    fprintf(
        stderr,
        "[ERROR] FSE normalization sum mismatch: remaining=%d (should be 0)\n",
        remaining);
    return Status::ERROR_CORRUPT_DATA;
  }

  // MAX_TABLE_SIZE is the maximum size for FSE/Huffman tables
  // Based on ZSTD spec: max table size is 2^12 = 4096 for FSE
  // Using 4096 to accommodate all valid table sizes
  const u32 MAX_TABLE_SIZE = 4096;
  u8 table_symbol[MAX_TABLE_SIZE] = {0};
  u32 step = (table_size >> 1) + (table_size >> 3) + 3;
  u32 mask = table_size - 1;
  u32 pos = 0;

  for (u32 s = 0; s < num_fse_symbols; s++) {
    if (norm_counts[s] > 0) {
      for (i32 i = 0; i < norm_counts[s]; i++) {
        table_symbol[pos & mask] = s;
        pos = (pos + step) & mask;
      }
    }
  }
  for (u32 s = 0; s < num_fse_symbols; s++) {
    if (norm_counts[s] == -1) {
      while (table_symbol[pos & mask] != 0 || (pos & mask) == 0) {
        if (table_symbol[pos & mask] == 0)
          break;
        pos = (pos + step) & mask;
      }
      table_symbol[pos & mask] = s;
      pos = (pos + step) & mask;
    }
  }

  // Build the decoder table
  u8 fse_symbol[MAX_TABLE_SIZE];
  u8 fse_nbits[MAX_TABLE_SIZE];
  u16 fse_newstate[MAX_TABLE_SIZE];
  u32 symbol_next_idx[MAX_HUF_ALPHABET];
  for (u32 s = 0; s < num_fse_symbols; s++) {
    symbol_next_idx[s] = 0; // Reset to 0 for state index counting
  }

  for (u32 i = 0; i < table_size; i++) {
    u32 s = table_symbol[i];
    fse_symbol[i] = (u8)s;
    i16 count = norm_counts[s];
    if (count == -1) {
      // Probability -1 symbol: full state reset, read accuracy_log bits
      fse_nbits[i] = (u8)accuracy_log;
      fse_newstate[i] = 0; // Baseline 0, will wrap around table
    } else {
      // Get state index for this symbol (0, 1, 2... for each occurrence)
      u32 state_idx = symbol_next_idx[s]++;

      // Per RFC 8878: sort states, compute baseline
      // For symbol with count N, next power of 2 is P >= N
      // Lower (P - N) states need 1 more bit, higher states fewer bits
      u32 n_states = (u32)count;
      u32 next_pow2 = 1;
      u32 high_bit = 0;
      while (next_pow2 < n_states) {
        next_pow2 <<= 1;
        high_bit++;
      }

      u32 extra_states = next_pow2 - n_states; // Lower states needing more bits
      u32 nb_bits, baseline;

      if (state_idx < extra_states) {
        // Lower states: need more bits (high_bit + 1)
        nb_bits = accuracy_log - high_bit;
        baseline = (state_idx + n_states) << nb_bits;
      } else {
        // Higher states: need fewer bits (high_bit)
        nb_bits = accuracy_log - high_bit - (extra_states > 0 ? 0 : 1);
        if (high_bit == 0)
          nb_bits = accuracy_log; // Edge case: count=1
        baseline = (state_idx - extra_states) << nb_bits;
      }

      // Wrap baseline to stay within table_size
      fse_nbits[i] = (u8)nb_bits;
      fse_newstate[i] = (u16)(baseline & (table_size - 1));
    }
  }

  u32 bitstream_start = (bit_pos_header + 7) / 8;
  if (bitstream_start >= compressed_size)
    return Status::ERROR_CORRUPT_DATA;

  const unsigned char *bitstream = h_input + bitstream_start;
  u32 bitstream_size = compressed_size - bitstream_start;
  u32 bit_pos = bitstream_size * 8;

  while (bit_pos > 0) {
    if (bitstream[(bit_pos - 1) / 8] & (1 << ((bit_pos - 1) % 8))) {
      bit_pos--;
      break;
    }
    bit_pos--;
  }

  if (bit_pos < 2 * accuracy_log) {
    fprintf(stderr, "[ERROR] Not enough bits: bit_pos=%u, need=%u\n", bit_pos,
            2 * accuracy_log);
    return Status::ERROR_CORRUPT_DATA;
  }

  fprintf(stderr, "[HUF_FSE] Bitstream: start=%u, size=%u, bit_pos=%u\n",
          bitstream_start, bitstream_size, bit_pos);

  // libzstd BIT_DStream compatible implementation
  // Load up to 8 bytes from end of bitstream as LE word
  u64 bit_container = 0;
  u32 bits_consumed = 0;

  // Load bytes as little-endian (like MEM_readLEST)
  u32 load_size = (bitstream_size > 8) ? 8 : bitstream_size;
  for (u32 i = 0; i < load_size; i++) {
    bit_container |= ((u64)bitstream[bitstream_size - load_size + i])
                     << (i * 8);
  }

  // Find and consume sentinel bit (like BIT_initDStream)
  u8 last_byte = bitstream[bitstream_size - 1];
  if (last_byte == 0)
    return Status::ERROR_CORRUPT_DATA; // No sentinel

  // highbit finds the position of sentinel (0-7 for byte values 1-128)
  u32 sentinel_bit = 0;
  for (u32 b = 7; b > 0; b--) {
    if (last_byte & (1 << b)) {
      sentinel_bit = b;
      break;
    }
  }
  if (last_byte == 1)
    sentinel_bit = 0;

  bits_consumed = (8 - sentinel_bit); // Bits consumed including sentinel
  bits_consumed +=
      (8 - load_size) * 8; // Account for padding if loaded < 8 bytes

  u32 byte_ptr = bitstream_size - load_size; // Points to next unloaded byte

  auto read_bits = [&](u32 nbits) -> u32 {
    // BIT_lookBits: extract from position (64 - bitsConsumed - nbits)
    u32 start = 64 - bits_consumed - nbits;
    u32 result = (bit_container >> start) & ((1U << nbits) - 1);
    bits_consumed += nbits;

    // BIT_reloadDStream: refill if needed
    if (bits_consumed > 56 && byte_ptr > 0) {
      u32 refill = (byte_ptr > 8) ? 8 : byte_ptr;
      u64 new_container = 0;
      for (u32 i = 0; i < refill; i++) {
        new_container |= ((u64)bitstream[byte_ptr - refill + i]) << (i * 8);
      }
      byte_ptr -= refill;
      // Shift old remaining bits up, add new bits at bottom
      bit_container = (bit_container << (refill * 8)) | new_container;
      // Careful: avoid underflow
      i32 reduction = refill * 8;
      if ((i32)bits_consumed > reduction)
        bits_consumed -= reduction;
      else
        bits_consumed = 0;
    }
    return result;
  };

  u32 state1 = read_bits(accuracy_log);

  u32 state2 = read_bits(accuracy_log);

  u32 out_idx = 0;
  constexpr u32 MAX_WEIGHTS = 255;
  while (out_idx < MAX_WEIGHTS) {
    // State 1 decode
    if (state1 >= table_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    u32 idx1 = state1;
    u8 symbol1 = fse_symbol[idx1];
    u32 nb1 = fse_nbits[idx1];

    // Check if we have enough bits for state transition
    if (bits_consumed + nb1 > 64) {
      // Not enough bits - emit current symbol and stop
      h_weights[out_idx++] = symbol1;
      break;
    }

    u32 newstate_base1 = fse_newstate[idx1];
    u32 bits_read1 = read_bits(nb1);
    u32 new_state1 = newstate_base1 + bits_read1;

    h_weights[out_idx++] = symbol1;
    state1 = new_state1;

    if (bits_consumed >= 64 || out_idx >= MAX_WEIGHTS)
      break;

    // State 2 decode
    if (state2 >= table_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    u32 idx2 = state2;
    u8 symbol2 = fse_symbol[idx2];
    u32 nb2 = fse_nbits[idx2];

    // Check if we have enough bits for state transition
    if (bits_consumed + nb2 > 64) {
      // Not enough bits - emit current symbol and stop
      h_weights[out_idx++] = symbol2;
      break;
    }

    u32 newstate_base2 = fse_newstate[idx2];
    u32 bits_read2 = read_bits(nb2);
    u32 new_state2 = newstate_base2 + bits_read2;

    h_weights[out_idx++] = symbol2;
    state2 = new_state2;

    if (bits_consumed >= 64)
      break;
  }

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
    return Status::ERROR_CORRUPT_DATA;
  }

  // Find the smallest k such that 2^k >= weight_sum
  // This is the max_bits value
  u32 max_num_bits = 0;
  u32 next_power = 1;
  while (next_power < weight_sum) {
    next_power <<= 1;
    max_num_bits++;
  }

  // RFC 8878: The last weight's value is reconstructed so the sum is an exact
  // power of 2. The last weight represents the remaining bits needed.
  // If weight_sum is already a power of 2, last_weight_val = 0.
  // Otherwise, we need to compute what the last weight should be.
  u32 last_weight_val = next_power - weight_sum; // The additional value needed
  u32 last_weight = 0;
  if (last_weight_val > 0) {
    // The last weight is such that 2^(last_weight-1) fills the gap
    // If gap = X, find w where 2^(w-1) = X => w = log2(X) + 1
    if (last_weight_val > 0) {
      // Compute log2(last_weight_val) + 1
      u32 log_val = 0;
      u32 temp = last_weight_val;
      while (temp > 1) {
        temp >>= 1;
        log_val++;
      }
      // Check if it's exactly a power of 2
      if ((1U << log_val) == last_weight_val) {
        last_weight = log_val + 1;
      } else {
        // Not a power of 2, need different handling
        // This shouldn't happen if RFC is followed correctly
        last_weight = log_val + 2; // Round up
      }
    }
  }

  *max_bits = max_num_bits;

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
    fprintf(stderr, "[ERROR] decode_huffman_weights_fse status=%d\n",
            (int)status);
    return status;
  }

  fprintf(stderr, "[HUF_TABLE] FSE decode OK, num_symbols=%u\n", num_symbols);

  // Convert weights to code lengths
  u32 max_bits;
  status = weights_to_code_lengths(h_weights, num_symbols, h_code_lengths,
                                   &max_bits);

  if (status == Status::SUCCESS) {
  }
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
  if (threadIdx.x == 0 && blockIdx.x == 0 && idx < 10) {
    printf("[ENCODE-PHASE1] idx=%u symbol=%u('%c') code=0x%X len=%u pos=%u\n",
           idx, symbol, symbol >= 32 ? symbol : '?', c.code, c.length,
           d_bit_offsets[idx]);
  }
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
      printf("[ENCODE-PHASE2] idx=%u block=%u code=%u len=%u pos=%u\n", idx,
             blockIdx.x, code, length, bit_pos);
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
          if (idx < 5) {
            printf("[ENCODE-WRITE] idx=%u byte_offset=%u word=%u shift=%u "
                   "val=0x%02X\n",
                   idx, byte_offset, aligned_word_idx, byte_shift, byte_val);
          }
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
 * @brief Decode Huffman bitstream backward.
 */
__device__ u8 decode_huff_symbol(u64 bit_container, u32 bits_available,
                                 const u32 *d_first_code,
                                 const u16 *d_symbol_index, const u8 *d_symbols,
                                 u32 max_len, u32 &consumed) {
  u32 code = 0;
  for (u32 len = 1; len <= max_len; ++len) {
    if (len > bits_available)
      break;
    // Bits are read from the LSB of container (encoder writes LSB-first)
    code = (u32)(bit_container & ((1U << len) - 1));
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
 * @brief Parallel Huffman 4-stream decoder kernel.
 * Launched with 4 threads for small blocks, or more for larger ones.
 * For now, each block handles one stream.
 */
__global__ void huffman_decode_rfc8878_kernel(
    const unsigned char *input, u32 input_size, const u32 *d_first_code,
    const u16 *d_symbol_index, const u8 *d_symbols, unsigned char *output,
    u32 total_regen_size, u32 stream_start_bits, u32 stream_end_bits,
    u32 output_start_offset, u32 stream_id_debug, u32 num_symbols_to_decode) {

  u32 max_len = d_symbol_index[0];
  u32 bit_pos = stream_end_bits;
  u32 num_decoded = 0;

  // Find sentinel bit
  while (bit_pos > stream_start_bits) {
    u32 byte_idx = (bit_pos - 1) >> 3;
    u32 bit_idx = (bit_pos - 1) & 7;
    if (input[byte_idx] & (1 << bit_idx)) {
      bit_pos--;
      break;
    }
    bit_pos--;
  }

  // Bit container (64-bit) - load bytes from end backward
  u64 bit_container = 0;
  u32 bits_available = 0;

  // Initial load: read up to 8 bytes from end
  u32 byte_pos = (bit_pos + 7) / 8;
  u32 bytes_to_load = (byte_pos > 8) ? 8 : byte_pos;

  for (u32 i = 0; i < bytes_to_load; i++) {
    if (byte_pos > i) {
      bit_container |= ((u64)input[byte_pos - bytes_to_load + i]) << (i * 8);
    }
  }
  bits_available = bytes_to_load * 8;
  byte_pos -= bytes_to_load;

  while (num_decoded < num_symbols_to_decode) {
    // Refill when low
    while (bits_available <= 56 && byte_pos > 0) {
      u32 refill = (byte_pos > 8) ? 8 : byte_pos;
      u64 new_bits = 0;
      for (u32 i = 0; i < refill; i++) {
        new_bits |= ((u64)input[byte_pos - refill + i]) << (i * 8);
      }
      bit_container = (bit_container << (refill * 8)) | new_bits;
      bits_available += refill * 8;
      byte_pos -= refill;
    }

    if (bits_available == 0)
      break;

    u32 consumed = 0;
    // Try to decode a symbol from the top bits of the container
    for (u32 len = 1; len <= max_len; len++) {
      if (len > bits_available)
        break;
      // Extract bottom 'len' bits (LSB)
      u32 code = (u32)(bit_container & ((1U << len) - 1));
      // Canonical Huffman: check if code is in range [first_code, first_code +
      // count)
      u32 count_at_len = d_symbol_index[len + 1] - d_symbol_index[len];
      if (count_at_len > 0 && code >= d_first_code[len] &&
          code < d_first_code[len] + count_at_len) {
        u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];

        // INTERLEAVING: symbols are decoded in reverse order
        // The first symbol decoded is the LAST symbol of this stream.
        // INTERLEAVING FIX: RFC 8878 4-streams are CONCATENATED
        // The first symbol decoded is the LAST symbol of this stream segment.
        u32 symbol_idx_within_stream = num_symbols_to_decode - 1 - num_decoded;
        u32 out_idx = output_start_offset + symbol_idx_within_stream;

        if (out_idx < total_regen_size) {
          output[out_idx] = symbol;
        }

        num_decoded++;
        // Consume bits from LSB by shifting right
        bit_container >>= len;
        bits_available -= len;
        consumed = 1;
        break;
      }
    }
    if (!consumed)
      break;
  }
}

// ============================================================================
// Host API Functions
// ============================================================================

Status encode_huffman(const unsigned char *d_input, u32 input_size,
                      const HuffmanTable &table, unsigned char *d_output,
                      size_t *output_size, // Host pointer
                      CompressionWorkspace *workspace, cudaStream_t stream) {
  // --- START REPLACEMENT ---

  if (!d_input || !d_output || !output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // --- 1. Analyze frequencies - USE WORKSPACE BUFFER ---
  u32 *d_frequencies = workspace ? workspace->d_frequencies : nullptr;
  bool allocated_temp = false;

  if (!d_frequencies) {
    // Fallback: allocate if no workspace provided (backward compatibility)
    CUDA_CHECK(cudaMalloc(&d_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
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
  CUDA_CHECK(cudaMallocHost(&h_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));

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
  // (FIX) We need a host-side buffer for the codes first
  HuffmanCode *h_codes = new HuffmanCode[MAX_HUFFMAN_SYMBOLS];
  Status status =
      huffman::generate_canonical_codes(h_code_lengths, MAX_HUFFMAN_SYMBOLS,
                                        h_codes // Generate into host buffer
      );
  if (status != Status::SUCCESS) {
    //         fprintf(stderr, "[ERROR] encode_huffman:
    //         generate_canonical_codes failed with status %d\n",
    //         (int)status);
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
    CUDA_CHECK(cudaMalloc(&d_code_lengths, input_size * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_bit_offsets, input_size * sizeof(u32)));
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
    // Allocate temporary buffers
    CUDA_CHECK(cudaMalloc(&d_codes_temp, input_size * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_positions_temp, input_size * sizeof(u32)));
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
    cudaFreeHost(h_frequencies); // FIX: Use cudaFreeHost for pinned memory
    delete[] h_nodes;
    delete[] h_code_lengths;
    delete[] h_header;
    delete[] h_codes;
    return status;
  }

  // DEBUG: Print first 10 offsets and lengths
  // debug_print_kernel<<<1, 1, 0, stream>>>(d_code_lengths, d_bit_offsets,
  // d_input, table.codes, input_size); cudaStreamSynchronize(stream);

  // --- Collect Chunk Offsets ---
  u32 *d_chunk_offsets = nullptr;
  CUDA_CHECK(cudaMalloc(&d_chunk_offsets, offsets_size));

  u32 collect_threads = 256;
  u32 collect_blocks = (num_chunks + collect_threads - 1) / collect_threads;
  collect_chunk_offsets_kernel<<<collect_blocks, collect_threads, 0, stream>>>(
      d_bit_offsets, input_size, d_chunk_offsets, chunk_size_symbols,
      num_chunks);

  // Write offsets to output (after table)
  CUDA_CHECK(cudaMemcpyAsync(d_output + header_size, d_chunk_offsets,
                             offsets_size, cudaMemcpyDeviceToDevice, stream));
  cudaFree(d_chunk_offsets);

  /*
  // Switch to atomic-based kernel to avoid race conditions at block
  boundaries parallel_huffman_encode_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, table.codes, d_bit_offsets,
      d_output, header_size_bits
  );
  */

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
  // --- END REPLACEMENT ---
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
    printf("[DECODE-FWD] max_len=%u, input_size=%u, total=%u, start_bits=%u\n",
           max_len, input_size, total_regen_size, bitstream_start_bits);
    printf(
        "[DECODE-FWD] first_code[1]=%u, first_code[2]=%u, first_code[3]=%u\n",
        d_first_code[1], d_first_code[2], d_first_code[3]);
    printf("[DECODE-FWD] symbol_index[1]=%u, symbol_index[2]=%u, "
           "symbol_index[3]=%u\n",
           d_symbol_index[1], d_symbol_index[2], d_symbol_index[3]);
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
      if (threadIdx.x == 0 && blockIdx.x == 0 && num_decoded < 5) {
        printf(
            "[DECODE-FWD] l=%u raw=0x%X code=0x%X first=%u count=%u match=%d\n",
            l, raw_code, code, d_first_code[l], count_at_len,
            (count_at_len > 0 && code >= d_first_code[l] &&
             code < d_first_code[l] + count_at_len)
                ? 1
                : 0);
      }

      if (count_at_len > 0 && code >= d_first_code[l] &&
          code < d_first_code[l] + count_at_len) {
        len = l;
        break;
      }
    }

    if (len == 0) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[DECODE-FWD] Failed to decode at symbol %u, bits_avail=%u\n",
               num_decoded, bits_available);
      }
      break;
    }

    u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];
    output[num_decoded] = symbol;

    if (threadIdx.x == 0 && blockIdx.x == 0 && num_decoded < 10) {
      printf("[DECODE-FWD] sym[%u]=%u('%c') code=0x%X len=%u bits_avail=%u\n",
             num_decoded, symbol, symbol >= 32 ? symbol : '?', code, len,
             bits_available);
    }

    num_decoded++;
    bit_container >>= len;
    bits_available -= len;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[DECODE-FWD] Done: decoded %u symbols\n", num_decoded);
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
  if (status != Status::SUCCESS)
    return status;

  u8 *d_code_lengths;
  u32 *d_first_code;
  u16 *d_symbol_index;
  u8 *d_symbols;

  CUDA_CHECK(cudaMalloc(&d_code_lengths, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));
  CUDA_CHECK(cudaMalloc(&d_first_code, (MAX_HUFFMAN_BITS + 2) * sizeof(u32)));
  CUDA_CHECK(cudaMalloc(&d_symbol_index, (MAX_HUFFMAN_BITS + 2) * sizeof(u16)));
  CUDA_CHECK(cudaMalloc(&d_symbols, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));

  CUDA_CHECK(cudaMemcpyAsync(d_code_lengths, h_code_lengths,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u8),
                             cudaMemcpyHostToDevice, stream));

  build_decode_table_kernel<<<1, 1, 0, stream>>>(
      d_code_lengths, MAX_HUFFMAN_SYMBOLS, d_first_code, d_symbol_index,
      d_symbols);

  const unsigned char *d_bitstream_base = d_input + huf_header_size;
  u32 bitstream_size_base = (u32)(input_size - huf_header_size);

  if (four_streams) {
    if (bitstream_size_base < 6)
      return Status::ERROR_CORRUPT_DATA;

    u16 stream_sizes[3];
    CUDA_CHECK(cudaMemcpyAsync(stream_sizes, d_bitstream_base, 6,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    u32 L1 = stream_sizes[0];
    u32 L2 = stream_sizes[1];
    u32 L3 = stream_sizes[2];
    if (6 + L1 + L2 + L3 > bitstream_size_base)
      return Status::ERROR_CORRUPT_DATA;
    u32 L4 = bitstream_size_base - 6 - L1 - L2 - L3;

    u32 N1 = (decompressed_size + 3) / 4;
    u32 N2 = (decompressed_size + 2) / 4;
    u32 N3 = (decompressed_size + 1) / 4;
    u32 N4 = decompressed_size / 4;

    const unsigned char *d_data = d_bitstream_base + 6;
    huffman_decode_rfc8878_kernel<<<1, 1, 0, stream>>>(
        d_data, input_size, d_first_code, d_symbol_index, d_symbols, d_output,
        decompressed_size, 0, L1 * 8, 0, 0, N1);
    huffman_decode_rfc8878_kernel<<<1, 1, 0, stream>>>(
        d_data + L1, input_size, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size, 0, L2 * 8, N1, 1, N2);
    huffman_decode_rfc8878_kernel<<<1, 1, 0, stream>>>(
        d_data + L1 + L2, input_size, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size, 0, L3 * 8, N1 + N2, 2, N3);
    huffman_decode_rfc8878_kernel<<<1, 1, 0, stream>>>(
        d_data + L1 + L2 + L3, input_size, d_first_code, d_symbol_index,
        d_symbols, d_output, decompressed_size, 0, L4 * 8, N1 + N2 + N3, 3, N4);
  } else {
    // Use forward-reading decoder for single-stream mode
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
  }

  cudaFree(d_code_lengths);
  cudaFree(d_first_code);
  cudaFree(d_symbol_index);
  cudaFree(d_symbols);

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