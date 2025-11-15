#ifndef CUDA_ZSTD_INTERNAL_H_
#define CUDA_ZSTD_INTERNAL_H_

#include <iostream>
#ifdef _MSC_VER
#include <intrin.h>  // For _BitScanReverse
#endif
#include "cuda_zstd_utils.h"
#include "cuda_zstd_nvcomp.h"        // For Status
#include "cuda_zstd_huffman.h"       // Adds HuffmanCode, CanonicalHuffmanCode, MAX_HUFFMAN_SYMBOLS
#include "cuda_zstd_sequence.h"      // Adds Sequence
#include "cuda_zstd_types.h"         // Added for u32 and DictSegment

namespace cuda_zstd {

namespace huffman {

constexpr u16 HUFFMAN_NULL_IDX = 0xFFFF;

struct HuffmanNode {
    u16 symbol;
    u16 frequency;
    u16 left_child;
    u16 right_child;
    u16 parent;
};

} // namespace huffman

namespace sequence {

// (We need the bitstream reader from FSE)
struct FSEBitStreamReader {
    const byte_t* stream_start;
    const byte_t* stream_ptr;
    u64 bit_container;
    i32 bits_remaining;

    __device__ void init(const byte_t* stream_end, size_t stream_size) {
        stream_ptr = stream_end - 4;
        stream_start = stream_end - stream_size;
        bit_container = *reinterpret_cast<const u32*>(stream_ptr);
        bits_remaining = 32;
    }

    __device__ u32 read(u8 num_bits) {
        if (bits_remaining < num_bits) {
            stream_ptr -= 4;
            u64 next_bits = *reinterpret_cast<const u32*>(stream_ptr);
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
            if (x == 0) return 32;
            unsigned long index;
            _BitScanReverse(&index, x);
            return 31 - index;
        #else
            // Fallback: software implementation
            if (x == 0) return 32;
            u32 n = 0;
            if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
            if (x <= 0x00FFFFFF) { n += 8;  x <<= 8;  }
            if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4;  }
            if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2;  }
            if (x <= 0x7FFFFFFF) { n += 1; }
            return n;
        #endif
    }
    
    // Note: These functions work in both device and host code
    __device__ __host__ static __forceinline__ u32 get_lit_len_code(u32 length) {
        if (length < 65536) {
            u32 const fl = 31 - clz_impl(static_cast<u32>(length));
            return (length > 15) ? fl + 12 : length;
        }
        return 35;
    }
    __device__ __host__ static __forceinline__ u32 get_match_len_code(u32 length) {
        if (length < (3 + 131072)) {
            u32 const fl = 31 - clz_impl(static_cast<u32>(length));
            return (length > 15) ? fl + 28 : length - 3;
        }
        return 52;
    }
    __device__ __host__ static __forceinline__ u32 get_offset_code(u32 offset) {
        if (offset == 0) {
            return 0;
        }
        u32 const fl = 31 - clz_impl(static_cast<u32>(offset));
        return fl + 1;
    }
    // --- (END NEW) ---
    
    // --- (NEW) Literal Length __device__ Helpers ---

    __device__ static __forceinline__ u32 get_lit_len(u32 symbol) {
        // (Implementation of Zstd LL_defaultBase)
        if (symbol < 16) return symbol;
        if (symbol < 20) return (1 << (symbol - 14)) + 16;
        if (symbol < 24) return (1 << (symbol - 17)) + 80;
        if (symbol < 28) return (1 << (symbol - 20)) + 336;
        if (symbol < 32) return (1 << (symbol - 23)) + 2384;
        return (1 << (symbol - 26)) + 18656;
    }
    
    __device__ static __forceinline__ u32 get_lit_len_bits(u32 symbol, FSEBitStreamReader& reader) {
        // (Implementation of Zstd LL_defaultExtra)
        u32 num_bits = 0;
        if (symbol >= 16) num_bits = 1 + (symbol - 16);
        if (symbol >= 20) num_bits = 3 + (symbol - 20);
        if (symbol >= 24) num_bits = 5 + (symbol - 24);
        if (symbol >= 28) num_bits = 7 + (symbol - 28);
        if (symbol >= 32) num_bits = 9 + (symbol - 32);
        return reader.read(num_bits);
    }

    // --- (NEW) Match Length __device__ Helpers ---

    __device__ static __forceinline__ u32 get_match_len(u32 symbol) {
        // (Implementation of Zstd ML_defaultBase)
        if (symbol < 32) return symbol + 3;
        if (symbol < 36) return (1 << (symbol - 30)) + 35;
        if (symbol < 40) return (1 << (symbol - 33)) + 163;
        if (symbol < 44) return (1 << (symbol - 36)) + 707;
        if (symbol < 48) return (1 << (symbol - 39)) + 3395;
        return (1 << (symbol - 42)) + 13379;
    }

    __device__ static __forceinline__ u32 get_match_len_bits(u32 symbol, FSEBitStreamReader& reader) {
        // (Implementation of Zstd ML_defaultExtra)
        u32 num_bits = 0;
        if (symbol >= 32) num_bits = 1 + (symbol - 32);
        if (symbol >= 36) num_bits = 3 + (symbol - 36);
        if (symbol >= 40) num_bits = 5 + (symbol - 40);
        if (symbol >= 44) num_bits = 7 + (symbol - 44);
        if (symbol >= 48) num_bits = 9 + (symbol - 48);
        return reader.read(num_bits);
    }
    
    // --- (NEW) Offset __device__ Helpers ---

    __device__ static __forceinline__ u32 get_offset(u32 symbol) {
        // (Implementation of Zstd OF_defaultBase)
        if (symbol == 0) return 0;
        if (symbol == 1) return 1;
        return (1 << (symbol - 1)) + 1;
    }

    __device__ static __forceinline__ u32 get_offset_bits(u32 symbol, FSEBitStreamReader& reader) {
        // (Implementation of Zstd OF_defaultExtra)
        u32 num_bits = (symbol == 0) ? 0 : symbol - 1;
        return reader.read(num_bits);
    }
};

} // namespace sequence


namespace dictionary {

// Forward declaration of the kernel template
template<u32 BLOCK_SIZE>
__global__ void block_scan_prefix_sum_kernel_t(
    const DictSegment* __restrict__ segments,
    u32* __restrict__ segment_offsets,
    u32* __restrict__ d_block_sums,
    u32 num_segments,
    const u32* d_input_lengths
);

// New flexible kernel launcher
Status launch_block_scan_kernel(
    const DictSegment* d_segments,
    u32* d_segment_offsets,
    u32* d_block_sums,
    u32 num_segments,
    const u32* d_input_lengths,
    int device_id,
    cudaStream_t stream
);

// GPU architecture detection
u32 get_optimal_block_size(int device_id);

} // namespace dictionary

} // namespace cuda_zstd
#endif // CUDA_ZSTD_INTERNAL_H_