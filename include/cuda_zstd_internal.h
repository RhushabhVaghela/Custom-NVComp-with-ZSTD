#ifndef CUDA_ZSTD_INTERNAL_H_
#define CUDA_ZSTD_INTERNAL_H_

#include <iostream>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h> // For _BitScanReverse
#endif
#include "cuda_zstd_debug.h" // Debug configuration
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h"

namespace cuda_zstd {

// Dictionary segment structure for block processing
struct DictSegment {
  u32 start_offset; // Starting offset in the input data
  u32 length;       // Length of this segment
  u32 block_index;  // Block index this segment belongs to
};

// Make DictSegment available at global scope for compatibility
using DictSegment = DictSegment;

// Dictionary D-mer structure for compression
namespace dictionary {
struct Dmer {
  u64 hash;     // Hash value for the D-mer
  u32 position; // Position in the input data
  u32 length;   // Length of the D-mer

  Dmer() : hash(0), position(0), length(0) {}
  Dmer(u64 h, u32 p, u32 l) : hash(h), position(p), length(l) {}

  __device__ __host__ bool operator<(const Dmer &other) const {
    return hash < other.hash;
  }
};
} // namespace dictionary

namespace fse {

/**
 * @brief Reads an FSE header from the input stream.
 *
 * Scans the FSE header (Accuracy Log + Normalized Counts) from the bitstream
 * as per RFC 8878. Populates the `normalized_counts` vector, which can then
 * be used to build a decoding table.
 *
 * @param input Pointer to the start of the FSE header.
 * @param input_size Remaining size of the input buffer.
 * @param normalized_counts Output vector for normalized counts.
 * @param max_symbol Pointer to output the maximum symbol found.
 * @param table_log Pointer to output the accuracy log (table log).
 * @param bytes_read Pointer to output the number of bytes consumed.
 * @return Status SUCCESS on valid header, or error code.
 */
__host__ Status read_fse_header(const unsigned char *input, u32 input_size,
                                std::vector<u16> &normalized_counts,
                                u32 *max_symbol, u32 *table_log,
                                u32 *bytes_read);

} // namespace fse

namespace huffman {

constexpr u16 HUFFMAN_NULL_IDX = 0xFFFF;

struct HuffmanNode {
  u16 symbol;
  u32 frequency;
  u16 left_child;
  u16 right_child;
  u16 parent;
};

} // namespace huffman

namespace sequence {

// (We need the bitstream reader from FSE)
struct FSEBitStreamReader {
  const unsigned char *stream_start;
  const unsigned char *stream_ptr;
  u64 bit_container;
  i32 bits_remaining;
  u32 bit_pos;     // For tracking total bits read
  u32 stream_size; // TRACKING BOUNDS

  __device__ __host__ static __forceinline__ u32 bswap32(u32 x) {
#ifdef __CUDA_ARCH__
    return __byte_perm(x, 0, 0x0123);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) |
           ((x >> 24) & 0xFF);
#endif
  }

  __device__ __host__ FSEBitStreamReader()
      : stream_start(nullptr), stream_ptr(nullptr), bit_container(0),
        bits_remaining(0), bit_pos(0), stream_size(0) {}

  __device__ __host__ FSEBitStreamReader(const unsigned char *input,
                                         u32 start_bit_pos, u32 size) {
    bit_pos = start_bit_pos;
    stream_start = input;
    stream_size = size;
    // Align to byte boundary
    // Align to byte boundary
    // u32 byte_off = bit_pos / 8; // Unused
    // u32 bit_off = bit_pos % 8;  // Unused

    // In Zstd, we read backwards. For simplicity in the reader, let's just use
    // read_bits logic if needed, but the FSE reader normally pulls 32-64 bits
    // at a time. For debugging, let's just initialize it such that 'read'
    // works.
    init_at_bit(input, start_bit_pos);
  }

  __device__ __host__ void init_at_bit(const unsigned char *input,
                                       u32 start_bit_pos) {
    bit_pos = start_bit_pos;
    stream_start = input;
  }

  __device__ __host__ void set_stream_size(u32 size) { stream_size = size; }

  __device__ __host__ void init(const unsigned char *stream_end,
                                size_t stream_size) {
    stream_ptr = stream_end - 4;
    stream_start = stream_end - stream_size;
    bit_container = bswap32(*reinterpret_cast<const u32 *>(stream_ptr));

#ifdef __CUDA_ARCH__
    int clz = __clz(bit_container);
#elif defined(__GNUC__) || defined(__clang__)
    int clz = __builtin_clz(bit_container);
#else
    int clz = 0;
    u32 temp = bit_container;
    while (temp && !(temp & 0x80000000)) {
      temp <<= 1;
      clz++;
    }
    if (!temp)
      clz = 32;
#endif

    bits_remaining = 31 - clz;
    this->stream_size = (u32)stream_size;
  }

  __device__ __host__ u32 read(u8 num_bits) {
    if (num_bits == 0)
      return 0;
    if (bit_pos < num_bits)
      return 0; // Should not happen in valid stream

    // Encoder uses MSB-first bit packing within a big-endian byte stream.
    // bit_pos is the position of the NEXT bit to read (counting from bit 0 = LSB of byte 0).
    // We read `num_bits` bits starting just BELOW bit_pos.
    // 
    // Example: bit_pos=18, num_bits=6
    //   We read bits 17,16,15,14,13,12 (MSB to LSB of the value)
    //   These are at positions 12-17 in the stream
    
    bit_pos -= num_bits;
    
    // Now bit_pos points to the LSB of the value we want to read.
    // The value spans bits [bit_pos, bit_pos + num_bits - 1]
    
    u32 byte_start = bit_pos / 8;
    u32 bit_start = bit_pos % 8;
    
    // Load up to 8 bytes to cover all bits we need
    u64 data = 0;
    for (int i = 0; i < 8 && (byte_start + i) < stream_size; ++i) {
      data |= ((u64)stream_start[byte_start + i] << (i * 8));
    }
    
    // Extract bits starting at bit_start
    u32 val = (u32)((data >> bit_start) & ((1ULL << num_bits) - 1));
    return val;
  }

  __device__ __host__ u32 peek(u8 num_bits) {
    if (num_bits == 0)
      return 0;
    if (bit_pos < num_bits)
      return 0;

    // Peek at bits without modifying bit_pos
    u32 temp_pos = bit_pos - num_bits;
    u32 byte_start = temp_pos / 8;
    u32 bit_start = temp_pos % 8;

    u64 data = 0;
    for (int i = 0; i < 8 && (byte_start + i) < stream_size; ++i) {
      data |= ((u64)stream_start[byte_start + i] << (i * 8));
    }

    u32 val = (u32)((data >> bit_start) & ((1ULL << num_bits) - 1));
    return val;
  }
};

struct ZstdSequence {
  u32 lit_len;
  u32 offset;
  u32 match_len;

  // --- (NEW) Code helpers (definitions missing in dictionary.cu) ---
  // Helper function for count leading zeros - use appropriate intrinsic
  __device__ __host__ static __forceinline__ u32 clz_impl(u32 x) {
    return cuda_zstd::utils::clz_impl(x);
  }

  // Note: These functions work in both device and host code
  // --- (NEW) Code helpers with CORRECT ZSTD logic (RFC 8878) ---

  __device__ __host__ static __forceinline__ u32 get_lit_len_code(u32 length) {
    if (length < 16)
      return length;
    if (length < 18)
      return 16;
    if (length < 20)
      return 17;
    if (length < 22)
      return 18;
    if (length < 24)
      return 19;
    if (length < 28)
      return 20;
    if (length < 32)
      return 21;
    if (length < 40)
      return 22;
    if (length < 48)
      return 23;
    if (length < 64)
      return 24;
    if (length < 80)
      return 25;
    if (length < 112)
      return 26;
    if (length < 144)
      return 27;
    if (length < 208)
      return 28;
    if (length < 272)
      return 29;
    if (length < 400)
      return 30;
    if (length < 528)
      return 31;
    if (length < 784)
      return 32;
    if (length < 1040)
      return 33;
    if (length < 1552)
      return 34;
    return 35; // Max for Predefined
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_code(u32 length) {
    if (length < 3)
      return 0;
    if (length < 35)
      return (length - 3);

    // RFC 8878 Table 17 mapping
    if (length <= 36) return 32;
    if (length <= 38) return 33;
    if (length <= 40) return 34;
    if (length <= 42) return 35;
    if (length <= 46) return 36;
    if (length <= 50) return 37;
    if (length <= 58) return 38;
    if (length <= 66) return 39;
    if (length <= 82) return 40;
    if (length <= 98) return 41;
    if (length <= 130) return 42;
    if (length <= 162) return 43;
    if (length <= 226) return 44;
    if (length <= 290) return 45;
    if (length <= 418) return 46;
    if (length <= 546) return 47;
    if (length <= 802) return 48;
    if (length <= 1058) return 49;
    if (length <= 1570) return 50;
    if (length <= 2082) return 51;
    return 52;
  }

  __device__ __host__ static __forceinline__ u32 get_offset_code(u32 offset) {
    // RFC 8878 Section 3.1.1.1.3: Offset_Value = Offset + 3
    // Offset_Code = floor(log2(Offset_Value))
    u32 offset_val = offset + 3;
    if (offset_val == 0) return 0;
    return 31 - clz_impl(offset_val);
  }

  // --- Base Values (RFC 8878 Tables 16, 17, 18) ---

  __device__ __host__ static __forceinline__ u32 get_lit_len(u32 code) {
    if (code < 16)
      return code;
    // RFC 8878 Table 16 - Base Values
    static const u32 LL_base[36] = {
        0,   1,   2,   3,    4,    5,    6,    7,     8,     9,
        10,  11,  12,  13,   14,   15, // 0-15
        16,  18,  20,  22,   24,   28,   32,   40,    48,    64,
        128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    return (code < 36) ? LL_base[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_ll_base_predefined(u32 code) {
    return get_lit_len(code);
  }

  __device__ __host__ static __forceinline__ u32 get_match_len(u32 code) {
    if (code < 32)
      return code + 3;
    // RFC 8878 Table 17 - Base Values
    static const u32 ML_base[53] = {
        3,  4,   5,   6,   7,   8,   9,   10,  11,   12,   13,  14, 15, 16,
        17, 18,  19,  20,  21,  22,  23,  24,  25,   26,   27,  28, 29, 30,
        31, 32,  33,  34,  // 0-31
        35, 37,  39,  41,  // 32-35
        43, 47,  // 36-37
        51, 59,  // 38-39
        67, 83,  99,  131, 163, 227, 291, 419, 547, 803, 1059, 1571, 2083};
    return (code < 53) ? ML_base[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_ml_base_predefined(u32 code) {
    return get_match_len(code);
  }

  // --- Extra Bits (RFC 8878 Tables 10, 12, 14) ---

  __device__ __host__ static __forceinline__ u32
  get_lit_len_extra_bits(u32 code) {
    if (code < 16)
      return 0;
    // RFC 8878 Table 16 - Bits
    static const u8 LL_bits[36] = {
        0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, // 0-15
        1, 1, 1, 1,                                           // 16-19
        2, 2,                                                 // 20-21
        3, 3,                                                 // 22-23
        4, 6,                                                 // 24-25
        7, 8, 9, 10, 11, 12, 13, 14, 15, 16                   // 26-35
    };
    return (code < 36) ? LL_bits[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_extra_bits(u32 code) {
    if (code < 32)
      return 0;
    // RFC 8878 Table 17 - Bits
    static const u8 ML_bits[53] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-15
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16-31
        1, 1, 1, 1,                                     // 32-35
        2, 2,                                           // 36-37
        3, 3,                                           // 38-39
        4, 4,                                           // 40-41
        5, 5,                                           // 42-43
        6, 6,                                           // 44-45
        7, 7,                                           // 46-47
        8, 8,                                           // 48-49
        9, 9,                                           // 50-51
        16                                              // 52
    };
    return (code < 53) ? ML_bits[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_code_extra_bits(u32 code) {
    return code; // Extra bits = Offset_Code
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_extra_bits(u32 offset, u32 code) {
    u32 offset_val = offset + 3;
    return offset_val - (1u << code);
  }

  __device__ __host__ static __forceinline__ u32 get_offset(u32 code) {
    // Returns Offset_Value baseline (1 << code)
    return 1u << code;
  }

  __device__ __host__ static __forceinline__ u32
  get_lit_len_bits(u32 symbol, FSEBitStreamReader &reader) {
    u32 num_bits = get_lit_len_extra_bits(symbol);
    return (num_bits > 0) ? reader.read(num_bits) : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_bits(u32 symbol, FSEBitStreamReader &reader) {
    u32 num_bits = get_match_len_extra_bits(symbol);
    return (num_bits > 0) ? reader.read(num_bits) : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_bits(u32 symbol, FSEBitStreamReader &reader) {
    u32 num_bits = get_offset_code_extra_bits(symbol);
    return (num_bits > 0) ? reader.read(num_bits) : 0;
  }

};

} // namespace sequence

namespace dictionary {

// Forward declaration of the kernel template
template <u32 BLOCK_SIZE>
__global__ void
block_scan_prefix_sum_kernel_t(const DictSegment *__restrict__ segments,
                               u32 *__restrict__ segment_offsets,
                               u32 *__restrict__ d_block_sums, u32 num_segments,
                               const u32 *d_input_lengths);

// New flexible kernel launcher
Status launch_block_scan_kernel(const DictSegment *d_segments,
                                u32 *d_segment_offsets, u32 *d_block_sums,
                                u32 num_segments, const u32 *d_input_lengths,
                                int device_id, cudaStream_t stream);

// GPU architecture detection
u32 get_optimal_block_size(int device_id);

} // namespace dictionary

} // namespace cuda_zstd
#endif // CUDA_ZSTD_INTERNAL_H_
