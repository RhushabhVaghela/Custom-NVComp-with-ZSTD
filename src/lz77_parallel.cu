// ==============================================================================
// lz77_parallel.cu - Three-pass parallel LZ77 implementation
// ==============================================================================

#include "lz77_parallel.h"
#include <iostream>

namespace compression {
namespace lz77 {

__global__ void init_hash_table_kernel(
    u32* hash_table,
    u32* chain_table,
    u32 hash_size,
    u32 chain_size
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (u32 i = idx; i < hash_size; i += stride) {
        hash_table[i] = 0xFFFFFFFF;
    }

    for (u32 i = idx; i < chain_size; i += stride) {
        chain_table[i] = 0xFFFFFFFF;
    }
}

__global__ void find_matches_kernel(
    const u8* input,
    u32 input_size,
    u32* hash_table,
    u32* chain_table,
    Match* matches,
    const LZ77Config config,
    u32 hash_log
) {
    u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= input_size - config.min_match) return;

    u32 hash = compute_hash(input, pos, hash_log);
    u32 hash_idx = hash % (1 << hash_log);

    u32 prev_pos = atomicExch(&hash_table[hash_idx], pos);
    chain_table[pos] = prev_pos;

    Match best_match(pos, 0, 0, 0);
    u32 best_cost = 0xFFFFFFFF;

    u32 search_pos = prev_pos;
    for (u32 depth = 0; depth < config.search_depth && search_pos != 0xFFFFFFFF; depth++) {
        if (search_pos >= pos) break;

        u32 offset = pos - search_pos;
        u32 len = match_length(input, pos, search_pos, config.nice_length, input_size);

        if (len >= config.min_match) {
            u32 cost = calculate_match_cost(len, offset);
            if (cost < best_cost) {
                best_cost = cost;
                best_match.offset = offset;
                best_match.length = len;
            }

            if (len >= config.good_length) break;
        }

        search_pos = chain_table[search_pos];
    }

    matches[pos] = best_match;
}

__global__ void compute_costs_kernel(
    const u8* input,
    u32 input_size,
    const Match* matches,
    ParseCost* costs,
    const LZ77Config config
) {
    u32 pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (pos >= input_size) return;

    ParseCost literal_cost;
    literal_cost.cost = costs[pos - 1].cost + calculate_literal_cost(1);
    literal_cost.len = 1;
    literal_cost.is_match = false;

    ParseCost best_cost = literal_cost;

    Match m = matches[pos];
    if (m.length >= config.min_match) {
        u32 match_cost = costs[pos - 1].cost + calculate_match_cost(m.length, m.offset);

        if (match_cost < best_cost.cost) {
            best_cost.cost = match_cost;
            best_cost.len = m.length;
            best_cost.offset = m.offset;
            best_cost.is_match = true;
        }
    }

    costs[pos] = best_cost;
}

__global__ void backtrack_kernel(
    const ParseCost* costs,
    u32 input_size,
    u32* literal_lengths,
    u32* match_lengths,
    u32* offsets,
    u32* num_sequences
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    u32 pos = input_size - 1;
    u32 seq_idx = 0;
    u32 literal_count = 0;

    while (pos > 0) {
        ParseCost c = costs[pos];

        if (c.is_match) {
            literal_lengths[seq_idx] = literal_count;
            match_lengths[seq_idx] = c.len;
            offsets[seq_idx] = c.offset;
            seq_idx++;
            literal_count = 0;
            pos -= c.len;
        } else {
            literal_count++;
            pos--;
        }
    }

    if (literal_count > 0) {
        literal_lengths[seq_idx] = literal_count;
        match_lengths[seq_idx] = 0;
        offsets[seq_idx] = 0;
        seq_idx++;
    }

    *num_sequences = seq_idx;
}

Status find_matches_parallel(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
) {
//     std::cout << "[LZ77] Pass 1: Finding matches (parallel)" << std::endl;

    u32 hash_size = workspace.hash_table_size;
    u32 chain_size = workspace.chain_table_size;

    const u32 init_threads = 256;
    const u32 init_blocks = (std::max(hash_size, chain_size) + init_threads - 1) / init_threads;

    init_hash_table_kernel<<<init_blocks, init_threads, 0, stream>>>(
        workspace.d_hash_table,
        workspace.d_chain_table,
        hash_size,
        chain_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
//         std::cerr << "[LZ77] init_hash_table_kernel launch failed: " 
//                   << cudaGetErrorString(err) << std::endl;
        return Status::ERROR_CUDA_ERROR;
    }

    const u32 threads = 256;
    const u32 blocks = (input_size + threads - 1) / threads;

    find_matches_kernel<<<blocks, threads, 0, stream>>>(
        d_input,
        input_size,
        workspace.d_hash_table,
        workspace.d_chain_table,
        workspace.d_matches,
        config,
        config.hash_log
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
//         std::cerr << "[LZ77] find_matches_kernel launch failed: " 
//                   << cudaGetErrorString(err) << std::endl;
        return Status::ERROR_CUDA_ERROR;
    }

    return Status::SUCCESS;
}

Status compute_optimal_parse(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
) {
//     std::cout << "[LZ77] Pass 2: Computing optimal parse (parallel)" << std::endl;

    ParseCost initial_cost;
    initial_cost.cost = 0;
    initial_cost.len = 0;
    initial_cost.is_match = false;

    cudaMemcpyAsync(workspace.d_costs, &initial_cost, sizeof(ParseCost),
                    cudaMemcpyHostToDevice, stream);

    const u32 threads = 256;
    const u32 blocks = (input_size + threads - 1) / threads;

    compute_costs_kernel<<<blocks, threads, 0, stream>>>(
        d_input,
        input_size,
        workspace.d_matches,
        workspace.d_costs,
        config
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
//         std::cerr << "[LZ77] compute_costs_kernel launch failed: " 
//                   << cudaGetErrorString(err) << std::endl;
        return Status::ERROR_CUDA_ERROR;
    }

    return Status::SUCCESS;
}


// CPU-based backtracking (Option B: CPU Offloading)
// This runs on the host CPU instead of GPU to avoid sequential kernel bottleneck
void backtrack_sequences_cpu(
    const ParseCost* h_costs,
    u32 input_size,
    u32* h_literal_lengths,
    u32* h_match_lengths,
    u32* h_offsets,
    u32* h_num_sequences
) {
    u32 pos = input_size - 1;
    u32 seq_idx = 0;
    u32 literal_count = 0;

    while (pos > 0) {
        ParseCost c = h_costs[pos];

        if (c.is_match) {
            h_literal_lengths[seq_idx] = literal_count;
            h_match_lengths[seq_idx] = c.len;
            h_offsets[seq_idx] = c.offset;
            seq_idx++;
            literal_count = 0;
            pos -= c.len;
        } else {
            literal_count++;
            pos--;
        }
    }

    if (literal_count > 0) {
        h_literal_lengths[seq_idx] = literal_count;
        h_match_lengths[seq_idx] = 0;
        h_offsets[seq_idx] = 0;
        seq_idx++;
    }

    *h_num_sequences = seq_idx;
}

Status backtrack_sequences(
    u32 input_size,
    CompressionWorkspace& workspace,
    u32* h_num_sequences,
    cudaStream_t stream
) {
//     std::cout << "[LZ77] Pass 3: Backtracking sequences (CPU offload)" << std::endl;

    // Allocate host memory for costs array and sequence outputs
    ParseCost* h_costs = new ParseCost[input_size + 1];
    u32* h_literal_lengths = new u32[workspace.max_sequences];
    u32* h_match_lengths = new u32[workspace.max_sequences];
    u32* h_offsets = new u32[workspace.max_sequences];

    // Copy costs from device to host (async for potential overlap)
    cudaError_t err = cudaMemcpyAsync(h_costs, workspace.d_costs, 
                                      (input_size + 1) * sizeof(ParseCost),
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

    // Wait for D2H copy to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

    // Perform backtracking on CPU
    backtrack_sequences_cpu(h_costs, input_size, h_literal_lengths, 
                           h_match_lengths, h_offsets, h_num_sequences);

    // Copy results back to device (async)
    err = cudaMemcpyAsync(workspace.d_literal_lengths_reverse, h_literal_lengths,
                         *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

    err = cudaMemcpyAsync(workspace.d_match_lengths_reverse, h_match_lengths,
                         *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

    err = cudaMemcpyAsync(workspace.d_offsets_reverse, h_offsets,
                         *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

    // Wait for H2D copies to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return Status::ERROR_CUDA_ERROR;
    }

//     std::cout << "[LZ77] Found " << *h_num_sequences << " sequences (CPU)" << std::endl;

    // Cleanup
    delete[] h_costs;
    delete[] h_literal_lengths;
    delete[] h_match_lengths;
    delete[] h_offsets;

    return Status::SUCCESS;
}

} // namespace lz77
} // namespace compression
