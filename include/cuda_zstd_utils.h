// ============================================================================
// cuda_zstd_utils.h - Shared Utility Functions
//
// This file contains common, reusable components like parallel scan
// operations, moved from other modules to avoid code duplication.
// ============================================================================

#ifndef CUDA_ZSTD_UTILS_H_
#define CUDA_ZSTD_UTILS_H_

#include "cuda_zstd_types.h"

namespace cuda_zstd {

// Forward declare Dmer from the dictionary namespace
namespace dictionary {
struct Dmer;
}

namespace utils {

/**
 * @brief Performs a parallel inclusive prefix sum (scan) on a device array.
 *
 * @param d_input      Device pointer to the input array of u32 values.
 * @param d_output     Device pointer to the output array (can be same as input).
 * @param num_elements The number of elements in the array.
 * @param stream       The CUDA stream to execute on.
 * @return Status      SUCCESS or an error code.
 */
template <typename T>
__host__ Status parallel_scan(
    const T* d_input,
    u32* d_output,
    u32 num_elements,
    cudaStream_t stream
);

// Explicit template declaration for u32 (most common use case)
extern template __host__ Status parallel_scan<u32>(const u32*, u32*, u32, cudaStream_t);

/**
 * @brief Performs a parallel radix sort on an array of Dmers on the GPU.
 *
 * @param d_dmers      Device pointer to the array of Dmers to be sorted.
 * @param num_dmers    The number of dmers in the array.
 * @param stream       The CUDA stream to execute on.
 * @return Status      SUCCESS or an error code.
 */
__host__ Status parallel_sort_dmers(
    dictionary::Dmer* d_dmers,
    u32 num_dmers,
    cudaStream_t stream
);

inline u32 get_literal_length_code(u32 ll) {
    if (ll < 32) return ll;
    if (ll < 64) return 32;
    if (ll < 128) return 33;
    if (ll < 256) return 34;
    return 35;
}

// Hash function for dmers
__device__ __forceinline__ u64 hash_dmer(const byte_t* data, u32 d) {
    constexpr u64 prime = 0x9E3779B185EBCA8DULL;
    u64 hash = 0;
    for (u32 i = 0; i < d; ++i) {
        hash = hash * prime + data[i];
    }
    return hash;
}

} // namespace utils
} // namespace cuda_zstd

#endif // CUDA_ZSTD_UTILS_H