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

__device__ __host__ __forceinline__ u64 xxh_read64(const byte_t* ptr) {
    return *reinterpret_cast<const u64*>(ptr);
}

__device__ __host__ __forceinline__ u32 xxh_read32(const byte_t* ptr) {
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

// ============================================================================
// Kernel Declarations
// ============================================================================

__global__ void xxhash64_kernel(
    const byte_t* input,
    size_t input_size,
    u64 seed,
    u64* output
);

__global__ void xxhash32_kernel(
    const byte_t* input,
    size_t input_size,
    u32 seed,
    u32* output
);

// ============================================================================
// Host Functions
// ============================================================================

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
    const byte_t* input,
    size_t input_size,
    u64 seed
);

/**
 * @brief Host-side (CPU) XXHash32 implementation
 */
__host__ u32 xxhash_32_cpu(
    const byte_t* input,
    size_t input_size,
    u32 seed
);

} // namespace xxhash
} // namespace cuda_zstd

#endif // CUDA_ZSTD_XXHASH_H