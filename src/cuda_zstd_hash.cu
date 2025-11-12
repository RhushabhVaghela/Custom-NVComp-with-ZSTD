#include "cuda_zstd_hash.h"
#include "cuda_zstd_utils.h"

namespace cuda_zstd {
namespace hash {

__global__ void init_hash_table_kernel(u32* hash_table, int size, u32 value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        hash_table[idx] = value;
    }
}

void init_hash_table(u32* hash_table, int size, u32 value, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    init_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(hash_table, size, value);
}

} // namespace hash
} // namespace cuda_zstd