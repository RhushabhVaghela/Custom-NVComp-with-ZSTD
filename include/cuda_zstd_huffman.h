// ============================================================================
// cuda_zstd_huffman.h - Huffman Encoding / Decoding Interface
// ============================================================================

#ifndef CUDA_ZSTD_HUFFMAN_H_
#define CUDA_ZSTD_HUFFMAN_H_

#ifdef __cplusplus
#include <algorithm>
#include <vector>
#endif
#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"

#ifdef __cplusplus
namespace cuda_zstd {
namespace huffman {

// ============================================================================
// Huffman Constants
// ============================================================================

constexpr u32 MAX_HUFFMAN_SYMBOLS = 256;
constexpr u32 MAX_HUFFMAN_BITS = 24; // Increased to 24 to handle skewed
                                     // distributions without length limiting

// ============================================================================
// Huffman Structures
// ============================================================================

/**
 * @brief Represents a single Huffman code.
 * This is the correct type to be used by other modules (like the dictionary).
 */
struct HuffmanCode {
  u32 code;  // Huffman code (bit pattern)
  u8 length; // Code length in bits
  __device__ __host__ HuffmanCode() : code(0), length(0) {}
  __device__ __host__ HuffmanCode(u32 c, u8 l) : code(c), length(l) {}
};

/**
 * @brief Host-side context for managing a Huffman table.
 * The manager will own this and pass it to the encode/decode functions.
 */
struct HuffmanTable {
  HuffmanCode *codes; // Device pointer to 256 HuffmanCode entries
  HuffmanTable() : codes(nullptr) {}
};

// ============================================================================
// Helper Code Functions
// ============================================================================

struct CanonicalHuffmanCode {
  u8 symbol;
  u8 length;
  bool operator<(const CanonicalHuffmanCode &other) const {
    if (length != other.length)
      return length < other.length;
    return symbol < other.symbol;
  }
};

inline u32 reverse_bits(u32 val, u8 bits) {
  u32 r = 0;
  for (u8 i = 0; i < bits; ++i) {
    if ((val >> i) & 1) {
      r |= (1 << (bits - 1 - i));
    }
  }
  return r;
}

inline Status generate_canonical_codes(const u8 *code_lengths, u32 num_symbols,
                                       HuffmanCode *out_codes) {
  if (!code_lengths || !out_codes || num_symbols == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }
  std::vector<CanonicalHuffmanCode> sorted_codes;
  sorted_codes.reserve(num_symbols);
  for (u32 i = 0; i < num_symbols; ++i) {
    if (code_lengths[i] > 0) {
      sorted_codes.push_back({static_cast<u8>(i), code_lengths[i]});
    }
  }
  std::sort(sorted_codes.begin(), sorted_codes.end());
  u32 code = 0;
  u8 prev_length = 0;
  for (const auto &cc : sorted_codes) {
    if (cc.length > prev_length) {
      code <<= (cc.length - prev_length);
      prev_length = cc.length;
    }
    // Reverse bits for Zstd/Deflate compatibility (MSB is first bit of stream)
    // Decoder reads LSB-first, so we reverse after reading to get canonical code
    u32 reversed_code = reverse_bits(code, cc.length);
    out_codes[cc.symbol] = HuffmanCode{reversed_code, cc.length};
    code++;
  }
  return Status::SUCCESS;
}

// ============================================================================
// Host API Functions
// ============================================================================

/**
 * @brief Encodes input data using Huffman coding.
 * This function builds the Huffman tree, serializes the table into the
 * output, and then encodes the data.
 *
 * @param d_input      Device pointer to the uncompressed input data.
 * @param input_size   Size of the input data in bytes.
 * @param table        A HuffmanTable context (the 'codes' buffer will be used).
 * @param d_output     Device pointer to the output buffer for compressed data.
 * @param output_size  Host pointer to store the final compressed size.
 * @param workspace    Workspace for temporary buffers (frequencies,
 * code_lengths, bit_offsets)
 * @param stream       CUDA stream.
 */
Status encode_huffman(const unsigned char *d_input, u32 input_size,
                      const HuffmanTable &table, unsigned char *d_output,
                      size_t *output_size,
                      CompressionWorkspace *workspace = nullptr,
                      cudaStream_t stream = 0);

/**
 * @brief Decodes Huffman-coded data.
 * This function reads the serialized table from the input stream, builds the
 * decode table, and decodes the data.
 *
 * @param d_input           Device pointer to the compressed input data.
 * @param input_size        Size of the compressed data in bytes.
 * @param table             (Unused) The table is read from the stream.
 * @param d_output          Device pointer to the output buffer for decompressed
 * data.
 * @param d_output_size     Host pointer to store the final decompressed size.
 * @param decompressed_size The *expected* decompressed size (required).
 * @param stream            CUDA stream.
 */
Status decode_huffman(const unsigned char *d_input, size_t input_size,
                      [[maybe_unused]] const HuffmanTable &table,
                      unsigned char *d_output, size_t *d_output_size,
                      u32 decompressed_size, cudaStream_t stream = 0);

/**
 * @brief RFC 8878-compliant Huffman decode for standard Zstandard format.
 * Parses Huffman tree from weights (FSE or direct encoded), then decodes
 * literals using standard Zstandard bitstream format.
 *
 * @param d_input Device pointer to compressed Huffman data (starts with weight
 * header)
 * @param input_size Size of compressed data in bytes
 * @param d_output Device pointer to output buffer
 * @param d_output_size Host pointer for output size
 * @param decompressed_size Expected decompressed size
 * @param four_streams True if 4-stream format (size_format >= 1)
 * @param stream CUDA stream
 * @return Status::SUCCESS on success
 */
Status decode_huffman_rfc8878(const unsigned char *d_input, size_t input_size,
                              unsigned char *d_output, size_t *d_output_size,
                              u32 decompressed_size, bool four_streams = false,
                              cudaStream_t stream = 0);

/**
 * @brief Internal function exposed for unit testing.
 * Decodes FSE-compressed Huffman weights.
 */
Status decode_huffman_weights_fse(const unsigned char *h_input, u32 compressed_size,
                                  u8 *h_weights, u32 *num_symbols);

// ============================================================================
// Huffman Decompression Structures (NEW)
// ============================================================================

/**
 * @brief Represents an entry in the fast Huffman decoding table.
 */
struct HuffmanDecoderEntry {
  u8 num_bits; // Number of bits for this code
  u8 symbol;   // The decoded symbol
};

/**
 * @brief GPU-resident Huffman decoding table.
 *
 * This table is built from the weights read from the Zstd block.
 * It's designed for fast lookups.
 */
struct HuffmanDecoderTable {
  // The table_log determines the size (1 << table_log)
  HuffmanDecoderEntry *d_table;
  u32 table_log;
  u32 max_symbol_value;
};

/**
 * @brief Frees the memory associated with a HuffmanDecoderTable.
 */
Status free_huffman_decoder_table(HuffmanDecoderTable *table,
                                  cudaStream_t stream);

} // namespace huffman
} // namespace cuda_zstd

#endif // __cplusplus

#endif // CUDA_ZSTD_HUFFMAN_H
