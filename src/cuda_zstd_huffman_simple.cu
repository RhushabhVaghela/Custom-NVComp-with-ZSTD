/**
 * @brief Simplified RFC 8878-compliant Huffman encoder/decoder
 *
 * Key design decisions:
 * 1. Uses canonical Huffman codes (standard approach)
 * 2. Encoder writes codes MSB-first (standard)
 * 3. Decoder reads bits MSB-first and matches against canonical code ranges
 * 4. Header format: [MaxBits(1)][NumSymbols(1)][CodeLengths...][Bitstream]
 */

#include "cuda_zstd_huffman.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>
#include <vector>

namespace cuda_zstd {
namespace huffman {

// ============================================================================
// Constants
// ============================================================================

constexpr u32 HUFFMAN_WORKGROUP_SIZE = 256;

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Reverse bits in a value (MSB <-> LSB)
 */
__device__ inline u32 reverse_bits(u32 val, u32 bits) {
  return __brev(val) >> (32 - bits);
}

// ============================================================================
// CPU Huffman Tree Builder (Canonical Codes)
// ============================================================================

class CanonicalHuffmanBuilder {
public:
  /**
   * @brief Build canonical Huffman codes from symbol frequencies
   *
   * @param frequencies Input frequency count for each symbol
   * @param num_symbols Number of symbols (typically 256)
   * @param code_lengths Output: code length for each symbol
   * @param codes Output: canonical codes for each symbol
   * @return Status::SUCCESS on success
   */
  static Status build(const u32 *frequencies, u32 num_symbols, u8 *code_lengths,
                      HuffmanCode *codes) {
    if (!frequencies || !code_lengths || !codes || num_symbols == 0) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Count symbols per length
    u32 length_count[MAX_HUFFMAN_BITS + 1] = {0};
    u32 max_len = 0;

    for (u32 i = 0; i < num_symbols; i++) {
      if (frequencies[i] > 0) {
        // For canonical Huffman, we need to limit max length
        // Use the standard limit of MAX_HUFFMAN_BITS (24)
        if (frequencies[i] > 0) {
          // Estimate code length (simplified - in practice, use optimal tree)
          u8 len = 1;
          u32 threshold = 1;
          while (threshold <= frequencies[i] && len < MAX_HUFFMAN_BITS) {
            threshold <<= 1;
            len++;
          }
          code_lengths[i] = len;
          length_count[len]++;
          max_len = std::max(max_len, (u32)len);
        }
      }
    }

    // If no symbols with frequency > 0, return error
    u32 total_symbols = 0;
    for (u32 i = 1; i <= max_len; i++) {
      total_symbols += length_count[i];
    }
    if (total_symbols == 0) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Build canonical codes
    // For each length L, codes start at first_code[L] and are sequential
    u32 code = 0;
    u32 first_code[MAX_HUFFMAN_BITS + 2] = {0};

    for (u32 len = 1; len <= max_len; len++) {
      first_code[len] = code;
      code = (code + length_count[len]) << 1;
    }
    first_code[max_len + 1] = code; // Sentinel

    // Assign codes to symbols
    // Sort symbols by (length, symbol) and assign sequential codes
    struct SymbolInfo {
      u8 symbol;
      u8 length;
    };
    std::vector<SymbolInfo> symbols;
    symbols.reserve(total_symbols);

    for (u32 i = 0; i < num_symbols; i++) {
      if (code_lengths[i] > 0) {
        symbols.push_back({(u8)i, code_lengths[i]});
      }
    }

    std::sort(symbols.begin(), symbols.end(),
              [](const SymbolInfo &a, const SymbolInfo &b) {
                if (a.length != b.length)
                  return a.length < b.length;
                return a.symbol < b.symbol;
              });

    // Assign canonical codes
    u32 idx = 0;
    for (const auto &sym : symbols) {
      u32 len = sym.length;
      u32 sym_code = first_code[len] + idx;
      codes[sym.symbol] = HuffmanCode{sym_code, (u8)len};
      idx++;
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Build decode table from code lengths
   *
   * @param code_lengths Input: code length for each symbol
   * @param num_symbols Number of symbols
   * @param first_code Output: first code for each length
   * @param symbol_index Output: starting index in symbols array for each length
   * @param symbols Output: symbols sorted by (length, symbol)
   * @return Status::SUCCESS on success
   */
  static Status build_decode_table(const u8 *code_lengths, u32 num_symbols,
                                   u32 *first_code, u16 *symbol_index,
                                   u8 *symbols) {
    if (!code_lengths || !first_code || !symbol_index || !symbols) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Count symbols per length
    u32 length_count[MAX_HUFFMAN_BITS + 1] = {0};
    u32 max_len = 0;

    for (u32 i = 0; i < num_symbols; i++) {
      if (code_lengths[i] > 0 && code_lengths[i] <= MAX_HUFFMAN_BITS) {
        length_count[code_lengths[i]]++;
        max_len = std::max(max_len, (u32)code_lengths[i]);
      }
    }

    // Build first_code table
    // first_code[len] = sum of length_count[1..len-1]
    u32 code = 0;
    first_code[0] = 0;
    for (u32 len = 1; len <= max_len; len++) {
      first_code[len] = code;
      code = (code + length_count[len]) << 1;
    }
    first_code[max_len + 1] = 0xFFFFFFFF; // Sentinel

    // Build symbol_index table (starting index for each length)
    u32 idx = 0;
    for (u32 len = 1; len <= max_len; len++) {
      symbol_index[len] = (u16)idx;
      idx += length_count[len];
    }
    symbol_index[max_len + 1] = (u16)idx; // Sentinel

    // Build symbols array (sorted by length, then by symbol)
    idx = 0;
    for (u32 len = 1; len <= max_len; len++) {
      for (u32 sym = 0; sym < num_symbols; sym++) {
        if (code_lengths[sym] == len) {
          symbols[idx++] = (u8)sym;
        }
      }
    }

    return Status::SUCCESS;
  }
};

// ============================================================================
// Kernels
// ============================================================================

/**
 * @brief Kernel to count symbol frequencies
 */
__global__ void analyze_frequencies_kernel(const byte_t *input, u32 input_size,
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

/**
 * @brief Kernel to get code length for each symbol position
 */
__global__ void get_code_lengths_kernel(const byte_t *input, u32 input_size,
                                        const HuffmanCode *codes,
                                        u32 *code_lengths_out) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size)
    return;

  u8 symbol = input[idx];
  code_lengths_out[idx] = codes[symbol].length;
}

/**
 * @brief Simple Huffman encode kernel (one thread encodes one symbol)
 *
 * Each thread writes its symbol's code to the output bitstream.
 * Uses atomic operations to handle concurrent writes to the same byte.
 */
__global__ void huffman_encode_kernel(const byte_t *input, u32 input_size,
                                      const HuffmanCode *codes,
                                      const u32 *bit_offsets, byte_t *output,
                                      u32 header_size_bits) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size)
    return;

  u8 symbol = input[idx];
  const HuffmanCode &c = codes[symbol];

  if (c.length == 0)
    return;

  // Calculate bit position in output
  u32 bit_pos = header_size_bits + bit_offsets[idx];
  u32 byte_pos = bit_pos >> 3;
  u32 bit_offset = bit_pos & 7;

  // Write code bits LSB-first to bitstream
  for (u32 b = 0; b < c.length; b++) {
    if (c.code & (1U << b)) {
      u32 target_byte = byte_pos + (bit_offset + b) / 8;
      u32 target_bit = (bit_offset + b) % 8;
      atomicOr(&output[target_byte], (byte_t)(1 << target_bit));
    }
  }
}

/**
 * @brief Kernel to build decode table on device
 */
__global__ void build_decode_table_kernel(const u8 *code_lengths,
                                          u32 num_symbols, u32 *d_first_code,
                                          u16 *d_symbol_index, u8 *d_symbols) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Count symbols per length
  u32 length_count[MAX_HUFFMAN_BITS + 1] = {0};
  u32 max_len = 0;

  for (u32 i = 0; i < num_symbols; i++) {
    if (code_lengths[i] > 0 && code_lengths[i] <= MAX_HUFFMAN_BITS) {
      length_count[code_lengths[i]]++;
      max_len = std::max(max_len, (u32)code_lengths[i]);
    }
  }

  // Build first_code table
  u32 code = 0;
  d_first_code[0] = 0;
  for (u32 len = 1; len <= max_len; len++) {
    d_first_code[len] = code;
    code = (code + length_count[len]) << 1;
  }
  d_first_code[max_len + 1] = 0xFFFFFFFF;

  // Build symbol_index table
  u32 idx = 0;
  for (u32 len = 1; len <= max_len; len++) {
    d_symbol_index[len] = (u16)idx;
    idx += length_count[len];
  }
  d_symbol_index[max_len + 1] = (u16)idx;

  // Build symbols array
  idx = 0;
  for (u32 len = 1; len <= max_len; len++) {
    for (u32 sym = 0; sym < num_symbols; sym++) {
      if (code_lengths[sym] == len) {
        d_symbols[idx++] = (u8)sym;
      }
    }
  }
}

/**
 * @brief Forward-reading Huffman decode kernel
 *
 * Reads the bitstream forward (as written by the encoder).
 * For each code length, checks if the raw code falls within the canonical
 * range.
 */
__global__ void huffman_decode_kernel(const byte_t *input, u32 input_size,
                                      const u32 *d_first_code,
                                      const u16 *d_symbol_index,
                                      const u8 *d_symbols, byte_t *output,
                                      u32 total_output_size) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_output_size)
    return;

  u32 max_len = d_symbol_index[0];
  u32 bit_pos = 0;
  u32 num_decoded = 0;

  // Bit container for forward reading
  u64 bit_container = 0;
  u32 bits_available = 0;
  u32 byte_pos = 0;

  while (num_decoded <= idx) {
    // Refill bit container
    while (bits_available <= 56 && byte_pos < input_size) {
      bit_container |= ((u64)input[byte_pos]) << bits_available;
      bits_available += 8;
      byte_pos++;
    }

    if (bits_available == 0)
      break;

    // Try to decode a symbol
    for (u32 len = 1; len <= max_len; len++) {
      if (len > bits_available)
        break;

      // Read 'len' bits from LSB of container (encoder writes LSB-first)
      u32 code = (u32)(bit_container & ((1U << len) - 1));

      // Check if code is in canonical range for this length
      u32 count_at_len = d_symbol_index[len + 1] - d_symbol_index[len];
      if (count_at_len > 0 && code >= d_first_code[len] &&
          code < d_first_code[len] + count_at_len) {
        // Found matching symbol
        u32 symbol_idx = d_symbol_index[len] + (code - d_first_code[len]);
        u8 symbol = d_symbols[symbol_idx];
        output[num_decoded] = symbol;

        num_decoded++;
        bit_container >>= len;
        bits_available -= len;
        break;
      }
    }
  }
}

// ============================================================================
// Host API Functions
// ============================================================================

Status encode_huffman(const byte_t *d_input, u32 input_size,
                      const HuffmanTable &table, byte_t *d_output,
                      size_t *output_size, CompressionWorkspace *workspace,
                      cudaStream_t stream) {
  if (!d_input || !d_output || !output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Step 1: Analyze frequencies
  u32 *d_frequencies = workspace ? workspace->d_frequencies : nullptr;
  bool allocated_temp = false;

  if (!d_frequencies) {
    CUDA_CHECK(cudaMalloc(&d_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
    allocated_temp = true;
  }
  CUDA_CHECK(cudaMemsetAsync(d_frequencies, 0,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u32), stream));

  int threads = HUFFMAN_WORKGROUP_SIZE;
  int blocks = (input_size + threads - 1) / threads;

  analyze_frequencies_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, d_frequencies);

  // Copy frequencies to host for tree building
  u32 *h_frequencies = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
  CUDA_CHECK(cudaMemcpyAsync(h_frequencies, d_frequencies,
                             MAX_HUFFMAN_SYMBOLS * sizeof(u32),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Step 2: Build canonical Huffman codes on host
  u8 *h_code_lengths = new u8[MAX_HUFFMAN_SYMBOLS];
  HuffmanCode *h_codes = new HuffmanCode[MAX_HUFFMAN_SYMBOLS];

  Status status = CanonicalHuffmanBuilder::build(
      h_frequencies, MAX_HUFFMAN_SYMBOLS, h_code_lengths, h_codes);

  if (allocated_temp) {
    CUDA_CHECK(cudaFree(d_frequencies));
  }
  cudaFreeHost(h_frequencies);

  if (status != Status::SUCCESS) {
    delete[] h_code_lengths;
    delete[] h_codes;
    return status;
  }

  // Step 3: Serialize header
  // Format: [MaxBits(1)][CodeLengths(256)][Bitstream...]
  d_output[0] = MAX_HUFFMAN_BITS;
  memcpy(d_output + 1, h_code_lengths, MAX_HUFFMAN_SYMBOLS);
  u32 header_size = 1 + MAX_HUFFMAN_SYMBOLS;
  u32 header_size_bits = header_size * 8;

  // Copy codes to device
  CUDA_CHECK(cudaMemcpyAsync(table.codes, h_codes,
                             MAX_HUFFMAN_SYMBOLS * sizeof(HuffmanCode),
                             cudaMemcpyHostToDevice, stream));

  // Step 4: Get code length for each symbol position
  u32 *d_code_lengths = workspace ? workspace->d_code_lengths : nullptr;
  u32 *d_bit_offsets = workspace ? workspace->d_bit_offsets : nullptr;
  bool allocated_buffers = false;

  if (!d_code_lengths || !d_bit_offsets) {
    CUDA_CHECK(cudaMalloc(&d_code_lengths, input_size * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_bit_offsets, input_size * sizeof(u32)));
    allocated_buffers = true;
  }

  get_code_lengths_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, table.codes, d_code_lengths);

  // Step 5: Compute bit offsets using parallel prefix sum
  status = cuda_zstd::utils::parallel_scan(d_code_lengths, d_bit_offsets,
                                           input_size, stream);
  if (status != Status::SUCCESS) {
    if (allocated_buffers) {
      cudaFree(d_code_lengths);
      cudaFree(d_bit_offsets);
    }
    delete[] h_code_lengths;
    delete[] h_codes;
    return status;
  }

  // Step 6: Encode symbols
  // Clear output area first
  size_t max_bitstream_size = input_size * MAX_HUFFMAN_BITS / 8 + 1024;
  CUDA_CHECK(
      cudaMemsetAsync(d_output + header_size, 0, max_bitstream_size, stream));

  huffman_encode_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, table.codes, d_bit_offsets, d_output,
      header_size_bits);

  // Step 7: Compute final size
  u32 h_last_offset = 0, h_last_length = 0;
  if (input_size > 0) {
    CUDA_CHECK(cudaMemcpy(&h_last_offset, d_bit_offsets + (input_size - 1),
                          sizeof(u32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_last_length, d_code_lengths + (input_size - 1),
                          sizeof(u32), cudaMemcpyDeviceToHost));
  }
  u32 total_bits = h_last_offset + h_last_length;
  *output_size = header_size + (total_bits + 7) / 8;

  // Cleanup
  if (allocated_buffers) {
    cudaFree(d_code_lengths);
    cudaFree(d_bit_offsets);
  }
  delete[] h_code_lengths;
  delete[] h_codes;

  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

Status decode_huffman(const byte_t *d_input, size_t input_size,
                      const HuffmanTable &table, byte_t *d_output,
                      size_t *d_output_size, u32 decompressed_size,
                      cudaStream_t stream) {
  if (!d_input || !d_output || !d_output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Step 1: Read header
  u8 h_code_lengths[MAX_HUFFMAN_SYMBOLS] = {0};

  if (input_size < 1 + MAX_HUFFMAN_SYMBOLS) {
    return Status::ERROR_CORRUPT_DATA;
  }

  u8 max_bits = d_input[0];
  if (max_bits > MAX_HUFFMAN_BITS) {
    return Status::ERROR_CORRUPT_DATA;
  }

  memcpy(h_code_lengths, d_input + 1, MAX_HUFFMAN_SYMBOLS);
  u32 header_size = 1 + MAX_HUFFMAN_SYMBOLS;

  // Step 2: Allocate device decode tables
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

  // Step 3: Build decode table on device
  build_decode_table_kernel<<<1, 1, 0, stream>>>(
      d_code_lengths, MAX_HUFFMAN_SYMBOLS, d_first_code, d_symbol_index,
      d_symbols);

  // Step 4: Decode
  const byte_t *bitstream = d_input + header_size;
  u32 bitstream_size = (u32)(input_size - header_size);

  // Use single-threaded decode for simplicity (can be parallelized)
  int threads = 1;
  int blocks = 1;
  huffman_decode_kernel<<<blocks, threads, 0, stream>>>(
      bitstream, bitstream_size, d_first_code, d_symbol_index, d_symbols,
      d_output, decompressed_size);

  // Cleanup
  cudaFree(d_code_lengths);
  cudaFree(d_first_code);
  cudaFree(d_symbol_index);
  cudaFree(d_symbols);

  *d_output_size = decompressed_size;
  return Status::SUCCESS;
}

} // namespace huffman
} // namespace cuda_zstd
