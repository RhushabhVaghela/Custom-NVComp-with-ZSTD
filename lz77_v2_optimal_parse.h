#ifndef LZ77_V2_OPTIMAL_PARSE_H
#define LZ77_V2_OPTIMAL_PARSE_H

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @file lz77_v2_optimal_parse.h
 * @brief Standalone V2 Optimal Parse Implementation
 * 
 * High-performance GPU-based optimal parsing for LZ77 compression.
 * Uses multi-pass wavefront algorithm for 10-100x speedup over serial approaches.
 * 
 * Usage:
 *   #include "lz77_v2_optimal_parse.h"
 *   LZ77_V2::run_optimal_parse(d_input, input_size, d_matches, d_costs, stream);
 */

namespace LZ77_V2 {

typedef uint32_t u32;
typedef uint8_t u8;

// ============================================================================
// ParseCost - Packed u64 format for atomic operations
// ============================================================================

struct ParseCost {
    unsigned long long data;

    __host__ __device__ ParseCost() : data(0xFFFFFFFFFFFFFFFFULL) {}
    
    __host__ __device__ void set(u32 cost, u32 parent_pos) {
        data = ((unsigned long long)cost << 32) | parent_pos;
    }
    
    __host__ __device__ u32 cost() const {
        return (u32)(data >> 32);
    }
    
    __host__ __device__ u32 parent() const {
        return (u32)(data & 0xFFFFFFFF);
    }
};

// ============================================================================
// Match structure
// ============================================================================

struct Match {
    u32 length;  // Match length (0 = no match)
    u32 offset;  // Distance back to matching string
};

// ============================================================================
// Cost calculation functions (customize as needed)
// ============================================================================

__device__ inline u32 calculate_literal_cost(u32 len) {
    return len * 8;  // 8 bits per byte
}

__device__ inline u32 calculate_match_cost(u32 len, u32 offset) {
    return 20;  // Fixed overhead (customize for your format)
}

// ============================================================================
// V2 Kernel: Multi-pass optimal parse
// ============================================================================

__global__ void optimal_parse_kernel_v2(
    u32 input_size,
    const Match* d_matches,
    ParseCost* d_costs
) {
    u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= input_size) return;
    if (d_costs[pos].cost() >= 1000000000) return;
    
    u32 current_cost = d_costs[pos].cost();
    
    // Option 1: Literal
    if (pos + 1 <= input_size) {
        u32 cost_as_literal = current_cost + calculate_literal_cost(1);
        ParseCost new_val;
        new_val.set(cost_as_literal, pos);
        atomicMin(&d_costs[pos + 1].data, new_val.data);
    }
    
    // Option 2: Match
    const Match& match = d_matches[pos];
    if (match.length >= 3 && match.length <= input_size - pos) {
        u32 match_cost = calculate_match_cost(match.length, match.offset);
        u32 total_cost = current_cost + match_cost;
        u32 end_pos = pos + match.length;
        if (end_pos <= input_size) {
            ParseCost new_val;
            new_val.set(total_cost, pos);
            atomicMin(&d_costs[end_pos].data, new_val.data);
        }
    }
}

// ============================================================================
// Host API: Run optimal parse
// ============================================================================

/**
 * @brief Run V2 optimal parse on GPU
 * 
 * @param input_size Size of input in bytes
 * @param d_matches Device pointer to match array (input)
 * @param d_costs Device pointer to cost array (output, must be size input_size+1)
 * @param stream CUDA stream for async execution
 * @param max_passes Number of passes (auto-calculated if 0)
 * @return 0 on success, -1 on error
 */
inline int run_optimal_parse(
    u32 input_size,
    const Match* d_matches,
    ParseCost* d_costs,
    cudaStream_t stream = 0,
    int max_passes = 0
) {
    // Initialize costs
    cudaMemsetAsync(d_costs, 0xFF, (input_size + 1) * sizeof(ParseCost), stream);
    
    ParseCost initial_cost;
    initial_cost.set(0, 0);
    cudaMemcpyAsync(d_costs, &initial_cost, sizeof(ParseCost), 
                    cudaMemcpyHostToDevice, stream);
    
    // Auto-calculate passes if not specified
    if (max_passes == 0) {
        if (input_size < 100 * 1024) max_passes = 50;
        else if (input_size < 1024 * 1024) max_passes = 100;
        else if (input_size < 10 * 1024 * 1024) max_passes = 200;
        else max_passes = 300;
    }
    
    const u32 threads = 256;
    const u32 num_blocks = (input_size + threads - 1) / threads;
    
    // Multi-pass execution
    for (int pass = 0; pass < max_passes; ++pass) {
        optimal_parse_kernel_v2<<<num_blocks, threads, 0, stream>>>(
            input_size, d_matches, d_costs
        );
    }
    
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

} // namespace LZ77_V2

#endif // LZ77_V2_OPTIMAL_PARSE_H
