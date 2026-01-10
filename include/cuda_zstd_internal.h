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
__host__ Status read_fse_header(const byte_t *input, u32 input_size,
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
  const byte_t *stream_start;
  const byte_t *stream_ptr;
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

  __device__ __host__ FSEBitStreamReader(const byte_t *input, u32 start_bit_pos,
                                         u32 size) {
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

  __device__ __host__ void init_at_bit(const byte_t *input, u32 start_bit_pos) {
    bit_pos = start_bit_pos;
    stream_start = input;
  }

  __device__ __host__ void set_stream_size(u32 size) { stream_size = size; }

  __device__ __host__ void init(const byte_t *stream_end, size_t stream_size) {
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

    bit_pos -= num_bits;

    // RFC 8878: Bits are read backwards from the END of the stream
    // bit_pos counts down from sentinel position towards 0
    // We need to convert bit_pos to actual byte/bit offsets from stream start

    // bit_pos is the bit index from the START of the stream
    // But we need to read bytes in little-endian order from that position
    u32 byte_offset = bit_pos / 8;
    u32 bit_offset = bit_pos % 8;

    u64 data = 0;
    // Load up to 8 bytes starting at byte_offset (forward from stream start)
    for (int i = 0; i < 8; ++i) {
      if (byte_offset + i < stream_size) {
        data |= ((u64)stream_start[byte_offset + i] << (i * 8));
      }
    }

    u32 val = (u32)((data >> bit_offset) & ((1ULL << num_bits) - 1));
    return val;
  }

  __device__ __host__ u32 peek(u8 num_bits) {
    if (num_bits == 0)
      return 0;
    if (bit_pos < num_bits)
      return 0;

    // Peek uses the position that read() WOULD move to
    u32 temp_pos = bit_pos - num_bits;
    u32 byte_offset = temp_pos / 8;
    u32 bit_offset = temp_pos % 8;

    u64 data = 0;
    for (int i = 0; i < 8; ++i) {
      if (byte_offset + i < stream_size) {
        data |= ((u64)stream_start[byte_offset + i] << (i * 8));
      }
    }

    u32 val = (u32)((data >> bit_offset) & ((1ULL << num_bits) - 1));
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
    if (length <= 34)
      return (length - 3);
    if (length <= 36)
      return 32;
    if (length <= 38)
      return 33;
    if (length <= 40)
      return 34;
    if (length <= 42)
      return 35;
    if (length <= 44)
      return 36;
    if (length <= 48)
      return 37;
    if (length <= 52)
      return 38;
    if (length <= 60)
      return 39;
    if (length <= 68)
      return 40;
    if (length <= 84)
      return 41;
    if (length <= 100)
      return 42;
    if (length <= 132)
      return 43;
    if (length <= 164)
      return 44;
    if (length <= 228)
      return 45;
    if (length <= 292)
      return 46;
    if (length <= 420)
      return 47;
    if (length <= 548)
      return 48;
    if (length <= 804)
      return 49;
    if (length <= 1060)
      return 50;
    if (length <= 1572)
      return 51;
    return 52;
  }

  __device__ __host__ static __forceinline__ u32 get_offset_code(u32 offset) {
    // RFC 8878: Code = floor(log2(offset)) + 3 for Raw Offsets (Code >= 3)
    if (offset == 0)
      return 0;

    // Repcodes should be handled by caller context (checking history).
    // If strict raw offset is needed for small values:
    // Dist 1 (Code 3)
    // Dist 2 (Code 4)
    // Dist 3 (Code 4)

    // Note: If offset is 1, clz(1)=31 -> 31-31=0 -> 0+3=3. Correct.
    return (31 - clz_impl(offset)) + 3;
  }

  // --- Extra Bits ---

  // --- Extra Bits (RFC 8878 Tables 10, 12, 14) ---

  __device__ __host__ static __forceinline__ u32
  get_lit_len_extra_bits(u32 code) {
    if (code < 16)
      return 0;
    // RFC 8878 Table 9
    static const u8 LL_bits[36] = {
        0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-15
        1, 1,  1,  1,                                      // 16-19
        2, 2,                                              // 20-21
        3, 3,                                              // 22-23
        4, 4,                                              // 24-25
        5, 5,                                              // 26-27
        6,                                                 // 28
        7,                                                 // 29
        8,                                                 // 30
        9, 10, 11, 12, 13 // 31-35 (Max Code 35)
    };
    return (code < 36) ? LL_bits[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_extra_bits(u32 code) {
    if (code < 32)
      return 0;
    // RFC 8878 Table 11 - Correct extra bits for ML codes
    static const u8 ML_bits[53] = {
        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-15
        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16-31
        1,  1, 1, 1,                                     // 32-35: 1 extra bit
        2,  2,                                           // 36-37: 2 extra bits
        3,  3,                                           // 38-39: 3 extra bits
        4,  4,                                           // 40-41: 4 extra bits
        5,  5,                                           // 42-43: 5 extra bits
        6,                                               // 44: 6 extra bits
        7,                                               // 45: 7 extra bits
        8,                                               // 46: 8 extra bits
        9,                                               // 47: 9 extra bits
        10,                                              // 48: 10 extra bits
        11,                                              // 49: 11 extra bits
        12,                                              // 50: 12 extra bits
        13,                                              // 51: 13 extra bits
        16                                               // 52: 16 extra bits
    };
    return (code < 53) ? ML_bits[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_code_extra_bits(u32 code) {
    // Table 14: Code 3->0, 4->1, 5->2... => Code - 3
    if (code < 3)
      return 0;
    return code - 3;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_extra_bits(u32 offset, u32 code) {
    // encoding: extra = offset - base
    // base = 1 << (code - 3)
    if (code < 3)
      return 0;
    return offset - (1u << (code - 3));
  }

  // --- Base Values (RFC 8878 Tables 9, 11, 13) ---

  __device__ __host__ static __forceinline__ u32 get_lit_len(u32 code) {
    if (code < 16)
      return code;
    // RFC 8878 Table 9
    static const u32 LL_base[36] = {
        0,   1,    2,    3,    4,   5,  6,  7,
        8,   9,    10,   11,   12,  13, 14, 15, // 0-15
        16,  18,   20,   22,                    // 16-19
        24,  28,                                // 20-21
        32,  40,                                // 22-23
        48,  64,                                // 24-25
        80,  112,                               // 26-27 (RFC Standard)
        144,                                    // 28
        208,                                    // 29
        336,                                    // 30
        592, 1104, 2128, 4176, 8272             // 31-35
    };
    return (code < 36) ? LL_base[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_ll_base_predefined(u32 code) {
    static const u32 LL_base_predefined[36] = {
        0,   1,    2,    3,    4,   5,  6,  7,  8,  // 0-8
        9,   10,   11,   12,   13,  14, 15, 16, 18, // 9-17
        20,  22,   24,   28,   32,  44, 48, 64, 80, // 18-26 (Index 23: 40->44)
        248, 144,  208,  336,       // 27-30 (Modified for Predefined)
        592, 1104, 2128, 4176, 8272 // 31-35
    };
    return (code < 36) ? LL_base_predefined[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32 get_match_len(u32 code) {
    if (code < 32)
      return code + 3;
    // RFC 8878 Table 11 - Correct baselines for ML codes
    static const u32 ML_base[53] = {
        3,    4,   5,  6,  7,  8,  9,  10, 11, // 0-8: Base values 3-11
        12,   13,  14, 15, 16, 17, 18, 19, 20, // 9-17: Base values 12-20
        21,   22,  23, 24, 25, 26, 27, 28, 29, // 18-26: Base values 21-29
        30,   31,  32, 33, 34,                 // 27-31: Base values 30-34
        35,   37,  39, 41, // 32-35: Base values for 1 extra bit
        43,   47,          // 36-37: Base values for 2 extra bits
        51,   59,          // 38-39: Base values for 3 extra bits
        67,   83,          // 40-41: Base values for 4 extra bits
        99,   131,         // 42-43: Base values for 5 extra bits
        163,               // 44: Base value for 6 extra bits
        227,               // 45: Base value for 7 extra bits (RFC Standard)
        355,               // 46: Base value for 8 extra bits
        483,               // 47: Base value for 9 extra bits
        739,               // 48: Base value for 10 extra bits
        1251,              // 49: Base value for 11 extra bits
        2275,              // 50: Base value for 12 extra bits
        4323,              // 51: Base value for 13 extra bits
        65539              // 52: Base value for 16 extra bits
    };
    return (code < 53) ? ML_base[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_ml_base_predefined(u32 code) {
    if (code < 32)
      return code + 3;
    static const u32 ML_base_predefined[53] = {
        3,   4,   5,   6,    7,    8,     9,    10,  11,  12, 13, 14,
        15,  16,  17,  18,   19,   20,    21,   22,  23,  24, 25, 26,
        27,  28,  29,  30,   31,   32,    33,   34,  35,  37, 39, 41,
        43,  47,  51,  59,   67,   83,    99,   131, 163,
        705,                                      // 45: Modified for Predefined
        355, 483, 739, 1251, 2275, 57311, 65539}; // 51: Patched for libzstd
                                                  // 64KB
    return (code < 53) ? ML_base_predefined[code] : 0;
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

  __device__ __host__ static __forceinline__ u32 get_offset(u32 code) {
    // RFC 8878 Table 14: For OF_Code N (where N >= 1):
    //   Baseline = 1 << N
    //   Extra bits = N
    //   Offset range = [1<<N, (1<<(N+1))-1]
    // Example: OF_Code 8 -> baseline 256, extra 8 bits -> offset 256-511
    //
    // OF_Codes 0,1,2 can be rep codes depending on literal_length (handled by
    // caller) For OF_Code 0: offset = 1 (no extra bits)
    if (code == 0)
      return 1;
    // For OF_Code >= 1: baseline = 1 << code
    return 1u << code;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_bits(u32 symbol, FSEBitStreamReader &reader) {
    if (symbol == 0)
      return 0;
    return reader.read(symbol);
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
