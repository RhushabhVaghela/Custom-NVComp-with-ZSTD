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

    // In Zstd, we read backwards from the sentinel.
    // The 'bit_pos' here is the index of the sentinel bit.
    // We want 'read(nb)' to take the next 'nb' bits below 'bit_pos'.
  }

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

    // Direct read using bit_pos
    u32 byte_offset = bit_pos / 8;
    u32 bit_offset = bit_pos % 8;

    u64 data = 0;
    // Load up to 8 bytes, staying within the buffer [stream_start, stream_end)
    // We assume stream_ptr + 8 is safe if we have enough padding,
    // but let's be explicit and use the known stream count.
    for (int i = 0; i < 8; ++i) {
      // Simple safety: if we go beyond the provided section, stop loading
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
#ifdef __CUDA_ARCH__
    // Device code: use CUDA intrinsic
    return __clz(static_cast<int>(x));
#elif defined(__GNUC__) || defined(__clang__)
    // Host code with GCC/Clang: use builtin
    return __builtin_clz(x);
#elif defined(_MSC_VER)
    // MSVC: use _BitScanReverse
    if (x == 0)
      return 32;
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - index;
#else
    // Fallback: software implementation
    if (x == 0)
      return 32;
    u32 n = 0;
    if (x <= 0x0000FFFF) {
      n += 16;
      x <<= 16;
    }
    if (x <= 0x00FFFFFF) {
      n += 8;
      x <<= 8;
    }
    if (x <= 0x0FFFFFFF) {
      n += 4;
      x <<= 4;
    }
    if (x <= 0x3FFFFFFF) {
      n += 2;
      x <<= 2;
    }
    if (x <= 0x7FFFFFFF) {
      n += 1;
    }
    return n;
#endif
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
    // CRITICAL FIX: Literal-only sequences (ML=0) should never call this
    // But if they do, return 0 instead of wrapping around
    if (length < 3)
      return 0; // Minimum valid match length is 3
    if (length < 32)
      return (length - 3);
    if (length < 35)
      return 32;
    if (length < 37)
      return 33;
    if (length < 39)
      return 34;
    if (length < 41)
      return 35;
    if (length < 43)
      return 36;
    if (length < 47)
      return 37;
    if (length < 51)
      return 38;
    if (length < 59)
      return 39;
    if (length < 67)
      return 40;
    if (length < 83)
      return 41;
    if (length < 99)
      return 42;
    if (length < 131)
      return 43;
    if (length < 163)
      return 44;
    if (length < 227)
      return 45;
    if (length < 291)
      return 46;
    if (length < 419)
      return 47;
    if (length < 547)
      return 48;
    if (length < 803)
      return 49;
    if (length < 1059)
      return 50;
    if (length < 1571)
      return 51;
    return 52;
  }

  __device__ __host__ static __forceinline__ u32 get_offset_code(u32 offset) {
    // ZSTD RFC: offsetCode = position of highest bit set
    // For offset=1024 (0x400 = bit 10), code should be 10
    if (offset == 0)
      return 0;
    if (offset <= 1)
      return offset;                // Special case: offset=1 → code=1
    u32 fl = 31 - clz_impl(offset); // Position of highest bit
    return fl;                      // Return bit position directly, NOT fl+1
  }

  // --- Extra Bits ---

  __device__ __host__ static __forceinline__ u32
  get_lit_len_extra_bits(u32 code) {
    if (code < 16)
      return 0;
    static const u8 LL_bits[36] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, // 0-15
        1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    return LL_bits[code];
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_extra_bits(u32 code) {
    if (code < 32)
      return 0;
    // RFC 8878 Table 12: Extra bits for match length codes
    // Codes 32-35: 1 bit, 36-37: 2 bits, 38-39: 3 bits, 40-41: 4 bits
    // 42-43: 5 bits, 44-45: 6 bits, 46-47: 7 bits, 48-49: 8 bits
    // 50-51: 9 bits, 52: 16 bits
    static const u8 ML_bits[53] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4,
                                   5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 16};
    return (code < 53) ? ML_bits[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_code_extra_bits(u32 code) {
    return code;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_extra_bits(u32 offset, u32 code) {
    if (code <= 1)
      return 0;
    u32 base = (1u << code);
    return offset - base;
  }

  // --- Base Values ---

  __device__ __host__ static __forceinline__ u32 get_lit_len(u32 code) {
    if (code < 16)
      return code;
    // RFC 8878 Table 9
    static const u32 LL_base[36] = {
        0,  1,  2,  3,   4,   5,   6,   7,   8,   9,   10,   11,
        12, 13, 14, 15,  16,  18,  20,  22,  24,  28,  32,   40,
        48, 64, 80, 112, 144, 208, 272, 400, 528, 784, 1040, 65536};
    return (code < 36) ? LL_base[code] : 0;
  }

  __device__ __host__ static __forceinline__ u32 get_match_len(u32 code) {
    if (code < 32)
      return code + 3;
    // RFC 8878 Table 11: Match length baselines
    // Codes 32-35: 35, 37, 39, 41 (1 extra bit each)
    // Codes 36-37: 43, 47 (2 extra bits each)
    // Codes 38-39: 51, 59 (3 extra bits each)
    // Codes 40-41: 67, 83 (4 extra bits each)
    // Codes 42-43: 99, 131 (5 extra bits each)
    // Codes 44-45: 163, 227 (6 extra bits each)
    // Codes 46-47: 291, 419 (7 extra bits each)
    // Codes 48-49: 547, 803 (8 extra bits each)
    // Codes 50-51: 1059, 1571 (9 extra bits each)
    // Code 52: 2083 (16 extra bits)
    static const u32 ML_base[53] = {
        3,  4,   5,   6,   7,   8,   9,   10,  11,   12,   13,  14, 15, 16,
        17, 18,  19,  20,  21,  22,  23,  24,  25,   26,   27,  28, 29, 30,
        31, 32,  33,  34,  35,  37,  39,  41,  43,   47,   51,  59, 67, 83,
        99, 131, 163, 227, 291, 419, 547, 803, 1059, 1571, 2083};
    return (code < 53) ? ML_base[code] : 0;
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
    return (1u << code);
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_bits(u32 symbol, FSEBitStreamReader &reader) {
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
