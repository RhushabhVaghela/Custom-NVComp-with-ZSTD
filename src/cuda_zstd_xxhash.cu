// ============================================================================
// cuda_zstd_xxhash.cu - Complete XXHash64 Implementation
// ============================================================================
// Based on: XXHash64 algorithm by Yann Collet
// Reference: https://github.com/Cyan4973/xxHash
// Extremely fast non-cryptographic 64-bit hash function
// ============================================================================

#include "cuda_zstd_types.h"
#include <cstdio>  // ✅ ADDED for printf
#include "cuda_zstd_xxhash.h"

namespace cuda_zstd {
namespace xxhash {

// ============================================================================
// XXHash Constants (magic primes)
// ============================================================================

// XXHash Constants (magic primes) - Defined in header


// ============================================================================
// ✅ FIXED: Changed __device__ to __device__ __host__ for all helper functions
// ============================================================================





// ============================================================================
// Single Block Kernels (Moved from header to avoid multiple definitions)
// ============================================================================

__global__ void xxhash32_kernel(
    const byte_t* input,
    u32 input_size,
    u32 seed,
    u32* hash_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    const byte_t* p = input;
    const byte_t* const b_end = p + input_size;
    u32 h32;

    if (input_size >= XXH32_STRIPE_LEN) {
        u32 v1 = seed + PRIME32_1 + PRIME32_2;
        u32 v2 = seed + PRIME32_2;
        u32 v3 = seed + 0;
        u32 v4 = seed - PRIME32_1;

        const byte_t* const limit = b_end - XXH32_STRIPE_LEN;
        do {
            v1 = xxh32_round(v1, xxh_read32(p)); p += 4;
            v2 = xxh32_round(v2, xxh_read32(p)); p += 4;
            v3 = xxh32_round(v3, xxh_read32(p)); p += 4;
            v4 = xxh32_round(v4, xxh_read32(p)); p += 4;
        } while (p <= limit);

        h32 = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
    } else {
        h32 = seed + PRIME32_5;
    }

    h32 += input_size;

    while (p + 4 <= b_end) {
        h32 += xxh_read32(p) * PRIME32_3;
        h32 = rotl32(h32, 17) * PRIME32_4;
        p += 4;
    }

    while (p < b_end) {
        h32 += (*p) * PRIME32_5;
        h32 = rotl32(h32, 11) * PRIME32_1;
        p++;
    }

    h32 = xxh32_avalanche(h32);
    *hash_out = h32;
}

__global__ void xxhash64_kernel(
    const byte_t* input,
    u32 input_size,
    u64 seed,
    u64* hash_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;  // Only one thread computes hash
    
    const byte_t* p = input;
    const byte_t* const b_end = p + input_size;
    u64 h64;
    
    if (input_size >= XXH64_STRIPE_LEN) {
        // Initialize accumulators
        u64 v1 = seed + PRIME64_1 + PRIME64_2;
        u64 v2 = seed + PRIME64_2;
        u64 v3 = seed + 0;
        u64 v4 = seed - PRIME64_1;
        
        // Process 32-byte stripes
        const byte_t* const limit = b_end - XXH64_STRIPE_LEN;
        do {
            v1 = xxh64_round(v1, xxh_read64(p)); p += 8;
            v2 = xxh64_round(v2, xxh_read64(p)); p += 8;
            v3 = xxh64_round(v3, xxh_read64(p)); p += 8;
            v4 = xxh64_round(v4, xxh_read64(p)); p += 8;
        } while (p <= limit);
        
        // Merge accumulators
        h64 = rotl64(v1, 1) + rotl64(v2, 7) +
              rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        // Small input
        h64 = seed + PRIME64_5;
    }
    
    h64 += static_cast<u64>(input_size);
    
    // Process remaining bytes
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
    
    // Avalanche
    h64 = xxh64_avalanche(h64);
    *hash_out = h64;
}

// ============================================================================
// Parallel Block Hashing Kernel
// ============================================================================

__global__ void xxhash64_blocks_kernel(
    const byte_t* input,
    const u32* block_offsets,
    u32 num_blocks,
    u64 seed,
    u64* hashes_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    u32 block_start = block_offsets[idx];
    u32 block_end = block_offsets[idx + 1];
    u32 block_size = block_end - block_start;
    
    const byte_t* p = input + block_start;
    const byte_t* const b_end = p + block_size;
    u64 h64;
    
    if (block_size >= XXH64_STRIPE_LEN) {
        u64 v1 = seed + PRIME64_1 + PRIME64_2;
        u64 v2 = seed + PRIME64_2;
        u64 v3 = seed + 0;
        u64 v4 = seed - PRIME64_1;
        
        const byte_t* const limit = b_end - XXH64_STRIPE_LEN;
        do {
            v1 = xxh64_round(v1, xxh_read64(p)); p += 8;
            v2 = xxh64_round(v2, xxh_read64(p)); p += 8;
            v3 = xxh64_round(v3, xxh_read64(p)); p += 8;
            v4 = xxh64_round(v4, xxh_read64(p)); p += 8;
        } while (p <= limit);
        
        h64 = rotl64(v1, 1) + rotl64(v2, 7) +
              rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }
    
    h64 += static_cast<u64>(block_size);
    
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
    
    h64 = xxh64_avalanche(h64);
    hashes_out[idx] = h64;
}

// ============================================================================
// Streaming XXHash State
// ============================================================================

// ============================================================================
// Streaming Kernels Implementation
// ============================================================================

__global__ void xxhash64_init_kernel(XXH64_State* state, u64 seed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state->reset(seed);
    }
}

__global__ void xxhash64_update_kernel(XXH64_State* state, const byte_t* input, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        xxh64_update(*state, input, (u32)size);
    }
}

__global__ void xxhash64_digest_kernel(XXH64_State* state, u64* hash_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *hash_out = xxh64_finalize(*state);
    }
}

// ============================================================================
// Host API Functions
// ============================================================================

Status compute_xxhash64(
    const void* d_input,
    size_t input_size,
    u64 seed,
    u64* d_output,
    cudaStream_t stream
) {
    if (!d_input || !d_output || input_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    xxhash64_kernel<<<1, 1, 0, stream>>>(
        static_cast<const byte_t*>(d_input), input_size, seed, d_output
    );
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
}

Status compute_xxhash32(
    const void* d_input,
    size_t input_size,
    u32 seed,
    u32* d_output,
    cudaStream_t stream
) {
    if (!d_input || !d_output || input_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    xxhash32_kernel<<<1, 1, 0, stream>>>(
        static_cast<const byte_t*>(d_input), input_size, seed, d_output
    );
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
}

Status xxhash64(
    const byte_t* d_input,
    u32 input_size,
    u64 seed,
    u64* h_hash_out,
    cudaStream_t stream
) {
    if (!d_input || !h_hash_out || input_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    u64* d_hash;
    CUDA_CHECK(cudaMalloc(&d_hash, sizeof(u64)));
    
    xxhash64_kernel<<<1, 1, 0, stream>>>(
        d_input, input_size, seed, d_hash
    );
    
    // h_hash_out is a host-side pointer; use synchronous copy to avoid
    // requiring pinned memory for the host buffer.
    CUDA_CHECK(cudaMemcpy(h_hash_out, d_hash, sizeof(u64), cudaMemcpyDeviceToHost));
    
    cudaFree(d_hash);
    CUDA_CHECK(cudaGetLastError());
    
    return Status::SUCCESS;
}

Status xxhash64_blocks(
    const byte_t* d_input,
    const u32* d_block_offsets,
    u32 num_blocks,
    u64 seed,
    u64* d_hashes_out,
    cudaStream_t stream
) {
    if (!d_input || !d_block_offsets || !d_hashes_out || num_blocks == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    
    xxhash64_blocks_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_block_offsets, num_blocks, seed, d_hashes_out
    );
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
}

// ✅ FIXED: CPU reference implementation - now all helpers are __device__ __host__
__host__ u32 xxhash_32_cpu(
    const byte_t* input,
    size_t input_size,
    u32 seed
) {
    const byte_t* p = input;
    const byte_t* const b_end = p + input_size;
    u32 h32;

    if (input_size >= XXH32_STRIPE_LEN) {
        u32 v1 = seed + PRIME32_1 + PRIME32_2;
        u32 v2 = seed + PRIME32_2;
        u32 v3 = seed + 0;
        u32 v4 = seed - PRIME32_1;

        const byte_t* const limit = b_end - XXH32_STRIPE_LEN;
        do {
            v1 = xxh32_round(v1, xxh_read32(p)); p += 4;
            v2 = xxh32_round(v2, xxh_read32(p)); p += 4;
            v3 = xxh32_round(v3, xxh_read32(p)); p += 4;
            v4 = xxh32_round(v4, xxh_read32(p)); p += 4;
        } while (p <= limit);

        h32 = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
    } else {
        h32 = seed + PRIME32_5;
    }

    h32 += input_size;

    while (p + 4 <= b_end) {
        h32 += xxh_read32(p) * PRIME32_3;
        h32 = rotl32(h32, 17) * PRIME32_4;
        p += 4;
    }

    while (p < b_end) {
        h32 += (*p) * PRIME32_5;
        h32 = rotl32(h32, 11) * PRIME32_1;
        p++;
    }

    return xxh32_avalanche(h32);
}

__host__ u64 xxhash_64_cpu(
    const byte_t* input,
    size_t input_size,  // ✅ FIXED: Changed from u32 to size_t
    u64 seed
) {
    const byte_t* p = input;
    const byte_t* const b_end = p + input_size;
    u64 h64;
    
    if (input_size >= XXH64_STRIPE_LEN) {
        u64 v1 = seed + PRIME64_1 + PRIME64_2;
        u64 v2 = seed + PRIME64_2;
        u64 v3 = seed + 0;
        u64 v4 = seed - PRIME64_1;
        
        const byte_t* const limit = b_end - XXH64_STRIPE_LEN;
        do {
            v1 = xxh64_round(v1, xxh_read64(p)); p += 8;
            v2 = xxh64_round(v2, xxh_read64(p)); p += 8;
            v3 = xxh64_round(v3, xxh_read64(p)); p += 8;
            v4 = xxh64_round(v4, xxh_read64(p)); p += 8;
        } while (p <= limit);
        
        h64 = rotl64(v1, 1) + rotl64(v2, 7) +
              rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }
    
    h64 += static_cast<u64>(input_size);
    
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

// ============================================================================
// Utility Functions
// ============================================================================

[[maybe_unused]] static void print_xxhash64(u64 hash) {
    printf("XXH64: 0x%016lX\n", hash);
}

bool verify_xxhash64(
    const byte_t* data,
    u32 size,
    u64 expected_hash,
    u64 seed
) {
    u64 computed_hash = xxhash_64_cpu(data, size, seed);
    return computed_hash == expected_hash;
}

}  // namespace xxhash
}  // namespace cuda_zstd