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

  __device__ void init(const byte_t *stream_end, size_t stream_size) {
    stream_ptr = stream_end - 4;
    stream_start = stream_end - stream_size;
    // Read and swap bytes to match reverse writer order
    bit_container = bswap32(*reinterpret_cast<const u32 *>(stream_ptr));

// Find the highest set bit (terminator bit)
#ifdef __CUDA_ARCH__
    int clz = __clz(bit_container);
#else
    int clz = __builtin_clz(bit_container);
#endif

    bits_remaining = 31 - clz;
  }

  __device__ u32 read(u8 num_bits) {
    if (bits_remaining < num_bits) {
      stream_ptr -= 4;
      // Read and swap bytes
      u64 next_bits = bswap32(*reinterpret_cast<const u32 *>(stream_ptr));
      bit_container |= (next_bits << bits_remaining);
      bits_remaining += 32;
    }
    u32 val = bit_container & ((1U << num_bits) - 1);
    bit_container >>= num_bits;
    bits_remaining -= num_bits;
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
    if (code < 20)
      return 1;
    if (code < 24)
      return 2;
    if (code < 28)
      return 3;
    if (code < 32)
      return 4;
    return 5;
  }

  __device__ __host__ static __forceinline__ u32
  get_match_len_extra_bits(u32 code) {
    if (code < 32)
      return 0;
    if (code < 36)
      return 1;
    if (code < 38)
      return 2;
    if (code < 40)
      return 3;
    if (code < 42)
      return 4;
    if (code < 44)
      return 5;
    if (code < 46)
      return 6;
    if (code < 48)
      return 7;
    if (code < 50)
      return 8;
    if (code < 52)
      return 9;
    if (code == 52)
      return 16;
    return 0;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_code_extra_bits(u32 code) {
    if (code <= 3)
      return 0;
    return code - 3;
  }

  __device__ __host__ static __forceinline__ u32
  get_offset_extra_bits(u32 offset, u32 code) {
    // ZSTD RFC: extra bits are the remainder after subtracting base
    // Base = (1 << code), so extra = offset - base
    if (code <= 1)
      return 0;
    u32 base = (1u << code);
    return offset - base;
  }

  // --- Base Values (Decoder) ---

  __device__ __host__ static __forceinline__ u32 get_lit_len(u32 code) {
    if (code < 16)
      return code;
    switch (code) {
    case 16:
      return 16;
    case 17:
      return 18;
    case 18:
      return 20;
    case 19:
      return 22;
    case 20:
      return 24;
    case 21:
      return 28;
    case 22:
      return 32;
    case 23:
      return 40;
    case 24:
      return 48;
    case 25:
      return 64;
    case 26:
      return 80;
    case 27:
      return 112;
    case 28:
      return 144;
    case 29:
      return 208;
    case 30:
      return 272;
    case 31:
      return 400;
    case 32:
      return 528;
    case 33:
      return 784;
    case 34:
      return 1040;
    case 35:
      return 1552;
    default:
      return 1552;
    }
  }

  __device__ static __forceinline__ u32
  get_lit_len_bits(u32 symbol, FSEBitStreamReader &reader) {
    u32 num_bits = get_lit_len_extra_bits(symbol);
    return (num_bits > 0) ? reader.read(num_bits) : 0;
  }

  // --- (NEW) Match Length __device__ Helpers ---

  __device__ __host__ static __forceinline__ u32 get_match_len(u32 symbol) {
    if (symbol < 32)
      return symbol + 3;
    switch (symbol) {
    case 32:
      return 35;
    case 33:
      return 37;
    case 34:
      return 39;
    case 35:
      return 41;
    case 36:
      return 43;
    case 37:
      return 47;
    case 38:
      return 51;
    case 39:
      return 59;
    case 40:
      return 67;
    case 41:
      return 83;
    case 42:
      return 99;
    case 43:
      return 131;
    case 44:
      return 163;
    case 45:
      return 227;
    case 46:
      return 291;
    case 47:
      return 419;
    case 48:
      return 547;
    case 49:
      return 803;
    case 50:
      return 1059;
    case 51:
      return 1571;
    case 52:
      return 2083;
    default:
      return 2083;
    }
  }

  __device__ static __forceinline__ u32
  get_match_len_bits(u32 symbol, FSEBitStreamReader &reader) {
    // (Implementation of Zstd ML_defaultExtra)
    u32 num_bits = 0;
    if (symbol >= 32)
      num_bits = 1 + (symbol - 32);
    if (symbol >= 36)
      num_bits = 3 + (symbol - 36);
    if (symbol >= 40)
      num_bits = 5 + (symbol - 40);
    if (symbol >= 44)
      num_bits = 7 + (symbol - 44);
    if (symbol >= 48)
      num_bits = 9 + (symbol - 48);
    return reader.read(num_bits);
  }

  // --- (NEW) Offset __device__ Helpers ---

  __device__ __host__ static __forceinline__ u32 get_offset(u32 code) {
    // ZSTD RFC 8878: Offset_Value = (1 << (offsetCode-1)) + extraBits
    if (code <= 1)
      return 1;
    return (1 << (code - 1));
  }

  __device__ static __forceinline__ u32
  get_offset_bits(u32 symbol, FSEBitStreamReader &reader) {
    // (Implementation of Zstd OF_defaultExtra)
    u32 num_bits = (symbol == 0) ? 0 : symbol - 1;
    return reader.read(num_bits);
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
