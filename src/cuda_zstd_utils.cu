// ============================================================================
// cuda_zstd_utils.cu - Shared Utility Function Implementations
//
// This file contains the implementation for common, reusable components
// like parallel scan, removing them from other modules.
// ============================================================================

#include "cuda_zstd_utils.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_internal.h"
#include <cuda_runtime.h>
#include <algorithm> // for std::min
#include <cstdlib>
#include <iostream>

namespace cuda_zstd {
namespace utils {

constexpr u32 SCAN_THREADS = 256;

// ============================================================================
// Parallel Scan Kernels (Moved from huffman.cu / fse.cu)
// ============================================================================

/**
 * @brief Pass 1a: Parallel Scan (Prefix Sum) within each block.
 */
template <typename T>
__global__ void block_scan_prefix_sum_kernel(
    const T* d_input,
    u32* d_output,
    u32* d_block_sums,
    u32 num_elements
) {
    __shared__ u32 s_data[SCAN_THREADS];
    
    u32 tid = threadIdx.x;
    u32 g_idx = blockIdx.x * blockDim.x + tid;

    if (g_idx < num_elements) {
        if constexpr (std::is_same<T, u32>::value) {
            s_data[tid] = d_input[g_idx];
        } else if constexpr (std::is_same<T, DictSegment>::value) {
            s_data[tid] = d_input[g_idx].length;
        }
    } else {
        s_data[tid] = 0;
    }
    __syncthreads();

    // Parallel scan (reduction phase)
    for (u32 d = 1; d < blockDim.x; d *= 2) {
        u32 k = (tid + 1) * d * 2 - 1;
        if (k < blockDim.x) {
            s_data[k] += s_data[k - d];
        }
        __syncthreads();
    }

    // Post-scan (down-sweep) phase
    if (tid == 0) {
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = s_data[blockDim.x - 1];
        }
        s_data[blockDim.x - 1] = 0; // Clear last element
    }
    __syncthreads();

    for (u32 d = blockDim.x / 2; d > 0; d /= 2) {
        u32 k = (tid + 1) * d * 2 - 1;
        if (k < blockDim.x) {
            u32 t = s_data[k - d];
            s_data[k - d] = s_data[k];
            s_data[k] += t;
        }
        __syncthreads();
    }

    // Write results
    if (g_idx < num_elements) {
        d_output[g_idx] = s_data[tid];
    }
}

/**
 * @brief Pass 1b: Add the block sums to each block's partial sums.
 */
__global__ void add_block_offsets_kernel(
    u32* d_data,
    const u32* d_block_sums, // This is the *scanned* block_sums
    u32 num_elements
) {
    u32 g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_elements) return;
    
    u32 block_sum = d_block_sums[blockIdx.x];
    d_data[g_idx] += block_sum;
}

// ============================================================================
// Host Function (Moved from huffman.cu / fse.cu)
// ============================================================================
template <typename T>
__host__ Status parallel_scan(
    const T* d_input,
    u32* d_output,
    u32 num_elements,
    cudaStream_t stream
) {
    if (num_elements == 0) return Status::SUCCESS;
    
    const u32 threads = SCAN_THREADS;
    u32 blocks = (num_elements + threads - 1) / threads;
    
    u32* d_block_sums;
    u32* d_scanned_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, blocks * sizeof(u32)));

    // Pass 1a: Compute prefix sum within each block
    block_scan_prefix_sum_kernel<T><<<blocks, threads, 0, stream>>>(
        d_input, d_output, d_block_sums, num_elements
    );
    
    if (blocks > 1) {
        // Pass 1b: Compute prefix sum *of the block sums*
        u32 scan_threads = std::min(blocks, (u32)SCAN_THREADS);
        u32 scan_blocks = (blocks + scan_threads - 1) / scan_threads;

        block_scan_prefix_sum_kernel<u32><<<scan_blocks, scan_threads, 0, stream>>>(
            d_block_sums, d_scanned_block_sums, nullptr, blocks
        );
        
        // Pass 1c: Add the scanned block sums back to the main array
        add_block_offsets_kernel<<<blocks, threads, 0, stream>>>(
            d_output, d_scanned_block_sums, num_elements
        );
    }
    
    cudaFree(d_block_sums);
    cudaFree(d_scanned_block_sums);
    CUDA_CHECK(cudaGetLastError());
    cuda_zstd::utils::debug_kernel_verify("utils::parallel_scan: after block_scan_prefix_sum_kernel");
    return Status::SUCCESS;
}

// Explicit template instantiation
template __host__ Status parallel_scan<u32>(const u32*, u32*, u32, cudaStream_t);
template __host__ Status parallel_scan<DictSegment>(const DictSegment*, u32*, u32, cudaStream_t);

__host__ cudaError_t debug_kernel_verify(const char* where) {
    const char* env = std::getenv("CUDA_ZSTD_DEBUG_KERNEL_VERIFY");
    if (!env || env[0] == '\0') {
        return cudaSuccess;
    }

    // Force synchronous behavior and print an informative message
    cudaError_t err = cudaDeviceSynchronize();
    if (where) {
        std::cerr << "[DEBUG] debug_kernel_verify: " << where << " cudaDeviceSynchronize()=" << err << " (" << cudaGetErrorString(err) << ")\n";
    } else {
        std::cerr << "[DEBUG] debug_kernel_verify: cudaDeviceSynchronize()=" << err << " (" << cudaGetErrorString(err) << ")\n";
    }

    return err;
}

/**
 * @brief (NEW) Radix Sort Pass 1: Histogram Kernel
 * Each block computes a local histogram for the current N bits
 * of the 64-bit hash.
 */
__global__ void radix_sort_histogram_kernel(
    const dictionary::Dmer* __restrict__ dmers_in,
    u32 num_dmers,
    u32* d_histogram, // Global histogram, size [num_blocks * num_buckets]
    u32 bit_shift,     // 0, 8, 16, ...
    u32 num_buckets,   // 256 (for 8 bits)
    u32 num_blocks
) {
    __shared__ u32 s_local_histogram[256]; // Shared mem for 8 bits (256 buckets)

    u32 tid = threadIdx.x;
    u32 g_idx = blockIdx.x * blockDim.x + tid;
    u32 stride = gridDim.x * blockDim.x;
    u32 bucket_mask = num_buckets - 1;

    // 1. Clear shared memory
    if (tid < num_buckets) {
        s_local_histogram[tid] = 0;
    }
    __syncthreads();

    // 2. Compute local histogram
    for (u32 i = g_idx; i < num_dmers; i += stride) {
        u32 bucket = (dmers_in[i].hash >> bit_shift) & bucket_mask;
        atomicAdd(&s_local_histogram[bucket], 1);
    }
    __syncthreads();

    // 3. Write local histogram to global memory
    if (tid < num_buckets) {
        d_histogram[blockIdx.x * num_buckets + tid] = s_local_histogram[tid];
    }
}

/**
 * @brief (NEW) Radix Sort Pass 2: Reorder Kernel
 * Each block re-scatters its share of elements into the sorted
 * output buffer using the global scanned offsets.
 */
__global__ void radix_sort_reorder_kernel(
    const dictionary::Dmer* __restrict__ dmers_in,
    u32 num_dmers,
    const u32* __restrict__ d_scanned_offsets, // Global offsets [num_blocks * num_buckets]
    dictionary::Dmer* __restrict__ dmers_out,               // Sorted output
    u32 bit_shift,
    u32 num_buckets,
    u32 num_blocks
) {
    __shared__ u32 s_bucket_offsets[256]; // Shared mem for 8 bits

    u32 tid = threadIdx.x;
    u32 g_idx = blockIdx.x * blockDim.x + tid;
    u32 stride = gridDim.x * blockDim.x;
    u32 bucket_mask = num_buckets - 1;

    // 1. Load this block's starting offsets into shared memory
    if (tid < num_buckets) {
        s_bucket_offsets[tid] = d_scanned_offsets[blockIdx.x * num_buckets + tid];
    }
    __syncthreads();

    // 2. Scatter elements
    for (u32 i = g_idx; i < num_dmers; i += stride) {
        u32 bucket = (dmers_in[i].hash >> bit_shift) & bucket_mask;
        
        // Get the correct output position atomically
        u32 out_idx = atomicAdd(&s_bucket_offsets[bucket], 1);
        dmers_out[out_idx] = dmers_in[i];
    }
}

/**
 * @brief Host-side launcher for the parallel RADIX SORT.
 * This function orchestrates the multi-pass radix sort
 * completely on the GPU.
 */
__host__ Status parallel_sort_dmers(dictionary::Dmer* d_dmers, u32 num_dmers, cudaStream_t stream) {
    if (num_dmers == 0) return Status::SUCCESS;

    // --- 1. Setup ---
    const u32 bits_per_pass = 8;
    const u32 num_buckets = 1 << bits_per_pass; // 256
    
    int threads = 256;
    int blocks = std::min((num_dmers + threads - 1) / threads, 1024u);

    // Temp storage for ping-pong buffers
    dictionary::Dmer* d_dmers_temp;
    CUDA_CHECK(cudaMalloc(&d_dmers_temp, num_dmers * sizeof(dictionary::Dmer)));
    
    // Temp storage for histograms and their scanned offsets
    u32* d_histogram;
    u32* d_scanned_offsets;
    u32 histogram_size = blocks * num_buckets;
    CUDA_CHECK(cudaMalloc(&d_histogram, histogram_size * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_scanned_offsets, histogram_size * sizeof(u32)));

    dictionary::Dmer* d_dmers_in = d_dmers;
    dictionary::Dmer* d_dmers_out = d_dmers_temp;

    // --- 2. Run Radix Sort Passes ---
    // 8 passes for a 64-bit hash (8 bits per pass)
    for (u32 bit_shift = 0; bit_shift < 64; bit_shift += bits_per_pass) {
        
        // Pass 1: Compute histogram
        CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, histogram_size * sizeof(u32), stream));
        radix_sort_histogram_kernel<<<blocks, threads, 0, stream>>>(
            d_dmers_in, num_dmers, d_histogram,
            bit_shift, num_buckets, blocks
        );

        // Pass 2: Scan histogram
        Status status = cuda_zstd::utils::parallel_scan(
            d_histogram, d_scanned_offsets, histogram_size, stream
        );
        if (status != Status::SUCCESS) return status;

        // Pass 3: Reorder
        radix_sort_reorder_kernel<<<blocks, threads, 0, stream>>>(
            d_dmers_in, num_dmers, d_scanned_offsets,
            d_dmers_out, bit_shift, num_buckets, blocks
        );

        // Swap input/output pointers for next pass (ping-pong)
        std::swap(d_dmers_in, d_dmers_out);
    }
    
    // if the number of passes is odd, the final sorted data is in d_dmers_temp
    // we need to copy it back to the original buffer
    if ((64 / bits_per_pass) % 2 != 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_dmers, d_dmers_in, num_dmers * sizeof(dictionary::Dmer), cudaMemcpyDeviceToDevice, stream));
    }

    // Cleanup
    cudaFree(d_dmers_temp);
    cudaFree(d_histogram);
    cudaFree(d_scanned_offsets);

    CUDA_CHECK(cudaGetLastError());
    cuda_zstd::utils::debug_kernel_verify("utils::parallel_scan: after add_block_offsets_kernel");
    return Status::SUCCESS;
}

} // namespace utils
} // namespace cuda_zstd
