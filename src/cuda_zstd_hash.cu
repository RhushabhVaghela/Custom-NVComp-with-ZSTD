#include "cuda_zstd_hash.h"
#include "cuda_zstd_utils.h"

namespace cuda_zstd {
namespace hash {

/**
 * @brief CUDA kernel to initialize a hash table with a specific value.
 * @param hash_table Pointer to hash table in device memory.
 * @param size Number of entries in the hash table.
 * @param value Value to initialize each entry with.
 */
__global__ void init_hash_table_kernel(u32* hash_table, int size, u32 value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        hash_table[idx] = value;
    }
}

/**
 * @brief Host wrapper to initialize a hash table on the device.
 * @param hash_table Pointer to hash table in device memory.
 * @param size Number of entries in the hash table.
 * @param value Value to initialize each entry with.
 * @param stream CUDA stream for asynchronous execution.
 */
void init_hash_table(u32* hash_table, int size, u32 value, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    init_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(hash_table, size, value);
    {
        cudaError_t __err = cudaGetLastError();
        if (__err != cudaSuccess) {
            ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__, __FUNCTION__);
            __ctx.cuda_error = __err;
            log_error(__ctx);
        }
        // Runtime toggle: perform synchronous verification if enabled
        cuda_zstd::utils::debug_kernel_verify("hash::init_hash_table: after init_hash_table_kernel");
    }
}

/**
 * @brief CUDA kernel to insert/update hash table entries atomically.
 * Used for LZ77 and dictionary training.
 * @param hash_table Pointer to hash table in device memory.
 * @param positions Array of positions to insert.
 * @param hashes Array of hash values corresponding to positions.
 * @param count Number of entries to insert.
 */
__global__ void insert_hash_table_kernel(u32* hash_table, const u32* positions, const u32* hashes, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        atomicExch(&hash_table[hashes[idx]], positions[idx]);
    }
}

/**
 * @brief Host wrapper for atomic hash table insertion.
 * @param hash_table Pointer to hash table in device memory.
 * @param positions Array of positions to insert.
 * @param hashes Array of hash values.
 * @param count Number of entries.
 * @param stream CUDA stream for asynchronous execution.
 * @return Status code.
 */
Status insert_hash_table(u32* hash_table, const u32* positions, const u32* hashes, int count, cudaStream_t stream)
{
    VALIDATE_NOT_NULL(hash_table, "hash_table");
    VALIDATE_NOT_NULL(positions, "positions");
    VALIDATE_NOT_NULL(hashes, "hashes");
    if (count <= 0) return Status::ERROR_INVALID_PARAMETER;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    insert_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(hash_table, positions, hashes, count);
    CUDA_CHECK(cudaGetLastError());
    cuda_zstd::utils::debug_kernel_verify("hash::insert_hash_table: after insert_hash_table_kernel");
    return Status::SUCCESS;
}

/**
 * @brief CUDA kernel for parallel hash table lookup.
 * @param hash_table Pointer to hash table in device memory.
 * @param hashes Array of hash values to lookup.
 * @param results Output array for found positions.
 * @param count Number of lookups.
 */
__global__ void lookup_hash_table_kernel(const u32* hash_table, const u32* hashes, u32* results, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        results[idx] = hash_table[hashes[idx]];
    }
}

/**
 * @brief Host wrapper for parallel hash table lookup.
 * @param hash_table Pointer to hash table in device memory.
 * @param hashes Array of hash values.
 * @param results Output array for found positions.
 * @param count Number of lookups.
 * @param stream CUDA stream for asynchronous execution.
 * @return Status code.
 */
Status lookup_hash_table(const u32* hash_table, const u32* hashes, u32* results, int count, cudaStream_t stream)
{
    VALIDATE_NOT_NULL(hash_table, "hash_table");
    VALIDATE_NOT_NULL(hashes, "hashes");
    VALIDATE_NOT_NULL(results, "results");
    if (count <= 0) return Status::ERROR_INVALID_PARAMETER;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    lookup_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(hash_table, hashes, results, count);
    CUDA_CHECK(cudaGetLastError());
    cuda_zstd::utils::debug_kernel_verify("hash::lookup_hash_table: after lookup_hash_table_kernel");
    return Status::SUCCESS;
}

/**
 * @brief CUDA kernel to update chain table entries.
 * Used for LZ77 match chaining.
 * @param chain_table Pointer to chain table in device memory.
 * @param positions Array of positions to update.
 * @param prev_positions Array of previous positions.
 * @param count Number of entries.
 */
__global__ void update_chain_table_kernel(u32* chain_table, const u32* positions, const u32* prev_positions, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        chain_table[positions[idx]] = prev_positions[idx];
    }
}

/**
 * @brief Host wrapper for chain table update.
 * @param chain_table Pointer to chain table in device memory.
 * @param positions Array of positions to update.
 * @param prev_positions Array of previous positions.
 * @param count Number of entries.
 * @param stream CUDA stream for asynchronous execution.
 * @return Status code.
 */
Status update_chain_table(u32* chain_table, const u32* positions, const u32* prev_positions, int count, cudaStream_t stream)
{
    VALIDATE_NOT_NULL(chain_table, "chain_table");
    VALIDATE_NOT_NULL(positions, "positions");
    VALIDATE_NOT_NULL(prev_positions, "prev_positions");
    if (count <= 0) return Status::ERROR_INVALID_PARAMETER;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    update_chain_table_kernel<<<grid_size, block_size, 0, stream>>>(chain_table, positions, prev_positions, count);
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
}

} // namespace hash
} // namespace cuda_zstd