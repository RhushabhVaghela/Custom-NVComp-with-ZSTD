#ifndef CUDA_ZSTD_HASH_H_
#define CUDA_ZSTD_HASH_H_

#include "cuda_zstd_primitives.h"
#include "cuda_zstd_types.h"

namespace cuda_zstd {
namespace hash {

// ============================================================================
// Hashing Constants
// ============================================================================

constexpr u32 HASH_LOG_MIN = 12;
constexpr u32 HASH_LOG_MAX = 26;
constexpr u32 HASH_PRIME_32 = 2654435761U;

// ============================================================================
// Hashing Functions (Device)
// ============================================================================

/**
 * @brief Computes a 32-bit hash for a sequence of bytes.
 * This is a fast hash used for finding potential LZ77 matches.
 */
__device__ __forceinline__ u32 hash_bytes(const byte_t *data, u32 len,
                                          u32 hash_log) {
  u32 hash = 0;
  if (len >= 4) {
    hash = (*reinterpret_cast<const u32 *>(data) * HASH_PRIME_32) >>
           (32 - hash_log);
  } else {
    // Handle smaller lengths if necessary, though min_match is usually >= 3
    u32 val = data[0];
    if (len > 1)
      val |= (data[1] << 8);
    if (len > 2)
      val |= (data[2] << 16);
    hash = (val * HASH_PRIME_32) >> (32 - hash_log);
  }
  return hash;
}

/**
 * @brief Looks up a value in the hash table.
 */
__device__ __forceinline__ u32 hash_table_lookup(const HashTable &table,
                                                 u32 hash) {
  return table.table[hash];
}

/**
 * @brief Looks up a value in the chain table.
 */
__device__ __forceinline__ u32 chain_table_lookup(const ChainTable &table,
                                                  u32 index) {
  return table.prev[index];
}

/**
 * @brief Initializes a hash table with a specific value (e.g., invalid
 * position). Kernel declaration - implementation is in cuda_zstd_hash.cu
 */
__global__ void init_hash_table_kernel(u32 *table, u32 size, u32 value);

// ============================================================================
// Hashing Functions (Host)
// ============================================================================

/**
 * @brief Host wrapper to initialize a hash table on the device.
 * Implementation is in cuda_zstd_hash.cu
 */
void init_hash_table(u32 *hash_table, int size, u32 value, cudaStream_t stream);

} // namespace hash
} // namespace cuda_zstd

#endif // CUDA_ZSTD_HASH_H_