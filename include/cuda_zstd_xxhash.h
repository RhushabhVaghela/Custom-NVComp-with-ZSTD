// ============================================================================
// cuda_zstd_xxhash.h - XXHash Checksum Functions
// ============================================================================

#ifndef CUDA_ZSTD_XXHASH_H
#define CUDA_ZSTD_XXHASH_H

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>

namespace cuda_zstd {
namespace xxhash {

// ============================================================================
// XXHash Constants
// ============================================================================

// XXHash64 primes
constexpr u64 PRIME64_1 = 0x9E3779B185EBCA87ULL;
constexpr u64 PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr u64 PRIME64_3 = 0x165667B19E3779F9ULL;
constexpr u64 PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr u64 PRIME64_5 = 0x27D4EB2F165667C5ULL;

// XXHash32 primes
constexpr u32 PRIME32_1 = 0x9E3779B1U;
constexpr u32 PRIME32_2 = 0x85EBCA77U;
constexpr u32 PRIME32_3 = 0xC2B2AE3DU;
constexpr u32 PRIME32_4 = 0x27D4EB2FU;
constexpr u32 PRIME32_5 = 0x165667B1U;

// ============================================================================
// Device/Host Helper Functions (PATCHED)
//
// These must be __device__ __host__ so that the __host__ CPU
// implementation (xxhash_64_cpu) can call them.
// ============================================================================

__device__ __host__ __forceinline__ u64 rotl64(u64 x, int r) {
    return (x << r) | (x >> (64 - r));
}

__device__ __host__ __forceinline__ u32 rotl32(u32 x, int r) {
    return (x << r) | (x >> (32 - r));
}

__device__ __host__ __forceinline__ u64 xxh_read64(const unsigned char* ptr) {
    return *reinterpret_cast<const u64*>(ptr);
}

__device__ __host__ __forceinline__ u32 xxh_read32(const unsigned char* ptr) {
    return *reinterpret_cast<const u32*>(ptr);
}

__device__ __host__ __forceinline__ u64 xxh64_round(u64 acc, u64 input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

__device__ __host__ __forceinline__ u32 xxh32_round(u32 acc, u32 input) {
    acc += input * PRIME32_2;
    acc = rotl32(acc, 13);
    acc *= PRIME32_1;
    return acc;
}

__device__ __host__ __forceinline__ u64 xxh64_avalanche(u64 h64) {
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

__device__ __host__ __forceinline__ u32 xxh32_avalanche(u32 h32) {
    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;
    return h32;
}

// XXHash64 merge round for finalization
__device__ __host__ __forceinline__ u64 xxh64_merge_round(u64 acc, u64 val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

// Stripe length constants
constexpr u32 XXH64_STRIPE_LEN = 32;

// ============================================================================
// Kernel Implementations (Inline to avoid device linking issues)
// ============================================================================

// XXHash32 constants
constexpr u32 XXH32_STRIPE_LEN = 16;

__global__ void xxhash32_kernel(
    const unsigned char* input,
    u32 input_size,
    u32 seed,
    u32* hash_out
);

__global__ void xxhash64_kernel(
    const unsigned char* input,
    u32 input_size,
    u64 seed,
    u64* hash_out
);

// ============================================================================
// Streaming XXHash State & Inline Functions
// ============================================================================

struct XXH64_State {
    u64 total_len;
    u64 v1;
    u64 v2;
    u64 v3;
    u64 v4;
    unsigned char buffer[XXH64_STRIPE_LEN];
    u32 buffer_size;
    
    __device__ __host__ void reset(u64 seed) {
        total_len = 0;
        v1 = seed + PRIME64_1 + PRIME64_2;
        v2 = seed + PRIME64_2;
        v3 = seed + 0;
        v4 = seed - PRIME64_1;
        buffer_size = 0;
    }
};

__device__ __host__ inline void xxh64_update(
    XXH64_State& state,
    const unsigned char* input,
    u32 input_size
) {
    const unsigned char* p = input;
    const unsigned char* const b_end = p + input_size;
    state.total_len += input_size;
    
    // Fill buffer if partially filled
    if (state.buffer_size + input_size < XXH64_STRIPE_LEN) {
        for (u32 i = 0; i < input_size; ++i) {
            state.buffer[state.buffer_size++] = p[i];
        }
        return;
    }
    
    // Process buffered data if any
    if (state.buffer_size > 0) {
        u32 to_copy = XXH64_STRIPE_LEN - state.buffer_size;
        for (u32 i = 0; i < to_copy; ++i) {
            state.buffer[state.buffer_size++] = p[i];
        }
        p += to_copy;
        
        const unsigned char* buf_p = state.buffer;
        state.v1 = xxh64_round(state.v1, xxh_read64(buf_p)); buf_p += 8;
        state.v2 = xxh64_round(state.v2, xxh_read64(buf_p)); buf_p += 8;
        state.v3 = xxh64_round(state.v3, xxh_read64(buf_p)); buf_p += 8;
        state.v4 = xxh64_round(state.v4, xxh_read64(buf_p));
        state.buffer_size = 0;
    }
    
    // Process stripes
    if (p <= b_end - XXH64_STRIPE_LEN) {
        const unsigned char* const limit = b_end - XXH64_STRIPE_LEN;
        do {
            state.v1 = xxh64_round(state.v1, xxh_read64(p)); p += 8;
            state.v2 = xxh64_round(state.v2, xxh_read64(p)); p += 8;
            state.v3 = xxh64_round(state.v3, xxh_read64(p)); p += 8;
            state.v4 = xxh64_round(state.v4, xxh_read64(p)); p += 8;
        } while (p <= limit);
    }
    
    // Buffer remaining bytes
    while (p < b_end) {
        state.buffer[state.buffer_size++] = *p++;
    }
}

__device__ __host__ inline u64 xxh64_finalize(XXH64_State& state) {
    u64 h64;
    
    if (state.total_len >= XXH64_STRIPE_LEN) {
        h64 = rotl64(state.v1, 1) + rotl64(state.v2, 7) +
              rotl64(state.v3, 12) + rotl64(state.v4, 18);
        h64 = xxh64_merge_round(h64, state.v1);
        h64 = xxh64_merge_round(h64, state.v2);
        h64 = xxh64_merge_round(h64, state.v3);
        h64 = xxh64_merge_round(h64, state.v4);
    } else {
        h64 = state.v3 + PRIME64_5;  // v3 contains seed
    }
    
    h64 += state.total_len;
    
    // Process remaining buffered bytes
    const unsigned char* p = state.buffer;
    const unsigned char* const b_end = p + state.buffer_size;
    
    while (p + 8 <= b_end) {
        u64 const k1 = xxh64_round(0, xxh_read64(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    
    if (p + 4 <= b_end) {
        h64 ^= static_cast<u64>(xxh_read32(p)) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    
    while (p < b_end) {
        h64 ^= static_cast<u64>(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }
    
    return xxh64_avalanche(h64);
}

// Streaming Kernels
__global__ void xxhash64_init_kernel(XXH64_State* state, u64 seed);
__global__ void xxhash64_update_kernel(XXH64_State* state, const unsigned char* input, size_t size);
__global__ void xxhash64_digest_kernel(XXH64_State* state, u64* hash_out);

/**
 * @brief Compute XXHash64 checksum on GPU
 * * @param d_input Input data on device
 * @param input_size Size of input in bytes
 * @param seed Seed value
 * @param d_output Output hash on device (u64*)
 * @param stream CUDA stream
 * @return Status
 */
Status compute_xxhash64(
    const void* d_input,
    size_t input_size,
    u64 seed,
    u64* d_output,
    cudaStream_t stream = 0
);

/**
 * @brief Compute XXHash32 checksum on GPU
 * * @param d_input Input data on device
 * @param input_size Size of input in bytes
 * @param seed Seed value
 * @param d_output Output hash on device (u32*)
 * @param stream CUDA stream
 * @return Status
 */
Status compute_xxhash32(
    const void* d_input,
    size_t input_size,
    u32 seed,
    u32* d_output,
    cudaStream_t stream = 0
);

/**
 * @brief Verify XXHash64 checksum
 * * @param d_input Input data
 * @param input_size Size of input
 * @param expected_hash Expected hash value
 * @param seed Seed used for hashing
 * @param stream CUDA stream
 * @return Status::SUCCESS if matches, Status::ERROR_CHECKSUM_FAILED if not
 */
Status verify_xxhash64(
    const void* d_input,
    size_t input_size,
    u64 expected_hash,
    u64 seed,
    cudaStream_t stream = 0
);

/**
 * @brief Verify XXHash32 checksum
 * * @param d_input Input data
 * @param input_size Size of input
 * @param expected_hash Expected hash value
 * @param seed Seed used for hashing
 * @param stream CUDA stream
 * @return Status::SUCCESS if matches, Status::ERROR_CHECKSUM_FAILED if not
 */
Status verify_xxhash32(
    const void* d_input,
    size_t input_size,
    u32 expected_hash,
    u32 seed,
    cudaStream_t stream = 0
);

/**
 * @brief Host-side (CPU) XXHash64 implementation
 */
__host__ u64 xxhash_64_cpu(
    const unsigned char* input,
    size_t input_size,
    u64 seed
);

/**
 * @brief Host-side (CPU) XXHash32 implementation
 */
__host__ u32 xxhash_32_cpu(
    const unsigned char* input,
    size_t input_size,
    u32 seed
);

} // namespace xxhash
} // namespace cuda_zstd

#endif // CUDA_ZSTD_XXHASH_H