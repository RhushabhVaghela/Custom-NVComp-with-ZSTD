// ==============================================================================
// lz77_parallel.cu - Three-pass parallel LZ77 implementation
// ==============================================================================

#include "lz77_parallel.h"
#include "cuda_zstd_lz77.h"  // For V2 kernel
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

    u32 prev_cost = costs[pos - 1].cost();
    u32 literal_cost_val = prev_cost + calculate_literal_cost(1);
    
    ParseCost literal_cost;
    literal_cost.set(literal_cost_val, pos - 1);  // parent = pos-1

    ParseCost best_cost = literal_cost;

    Match m = matches[pos];
    if (m.length >= config.min_match) {
        u32 match_cost_val = prev_cost + calculate_match_cost(m.length, m.offset);

        if (match_cost_val < literal_cost_val) {
            best_cost.set(match_cost_val, pos - 1);  // parent = pos-1
        }
    }

    costs[pos] = best_cost;
}

__global__ void backtrack_kernel(
    const ParseCost* costs,
    u32 input_size,
    u32* literal_lengths,
// LEGACY_BACKTRACK:     u32* match_lengths,
// LEGACY_BACKTRACK:     u32* offsets,
// LEGACY_BACKTRACK:     u32* num_sequences
// LEGACY_BACKTRACK: ) {
// LEGACY_BACKTRACK:     if (threadIdx.x != 0 || blockIdx.x != 0) return;
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK:     u32 pos = input_size - 1;
// LEGACY_BACKTRACK:     u32 seq_idx = 0;
// LEGACY_BACKTRACK:     u32 literal_count = 0;
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK:     while (pos > 0) {
// LEGACY_BACKTRACK:         ParseCost c = costs[pos];
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK:         if (c.is_match) {
// LEGACY_BACKTRACK:             literal_lengths[seq_idx] = literal_count;
// LEGACY_BACKTRACK:             match_lengths[seq_idx] = c.len;
// LEGACY_BACKTRACK:             offsets[seq_idx] = c.offset;
// LEGACY_BACKTRACK:             seq_idx++;
// LEGACY_BACKTRACK:             literal_count = 0;
// LEGACY_BACKTRACK:             pos -= c.len;
// LEGACY_BACKTRACK:         } else {
// LEGACY_BACKTRACK:             literal_count++;
// LEGACY_BACKTRACK:             pos--;
// LEGACY_BACKTRACK:         }
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK:     if (literal_count > 0) {
// LEGACY_BACKTRACK:         literal_lengths[seq_idx] = literal_count;
// LEGACY_BACKTRACK:         match_lengths[seq_idx] = 0;
// LEGACY_BACKTRACK:         offsets[seq_idx] = 0;
// LEGACY_BACKTRACK:         seq_idx++;
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK:     *num_sequences = seq_idx;
// LEGACY_BACKTRACK: }
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK: // ==============================================================================
// LEGACY_BACKTRACK: // Parallel Backtracking Kernel (Phase 2)
// LEGACY_BACKTRACK: // ==============================================================================
// LEGACY_BACKTRACK: 
// LEGACY_BACKTRACK: // Parallel segment-based backtracking kernel
// LEGACY_BACKTRACK: // Each block processes one segment independently
// LEGACY_BACKTRACK: __global__ void backtrack_segments_parallel_kernel(
// LEGACY_BACKTRACK:     const ParseCost* costs,
// LEGACY_BACKTRACK:     const SegmentInfo* segments,
// LEGACY_BACKTRACK:     u32* d_literal_lengths,     // Global sequence buffer
// LEGACY_BACKTRACK:     u32* d_match_lengths,
// LEGACY_BACKTRACK:     u32* d_offsets,
// LEGACY_BACKTRACK:     u32* d_sequence_counts,     // Per-segment counts
// LEGACY_BACKTRACK:     u32 num_segments,
// LEGACY_BACKTRACK:     u32 max_seq_per_segment
// LEGACY_BACKTRACK: ) {
// LEGACY_BACKTRACK:     u32 seg_id = blockIdx.x;
// LEGACY_BACKTRACK:     if (seg_id >= num_segments) return;
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     // Each block processes one segment
// LEGACY_BACKTRACK:     if (threadIdx.x != 0) return;  // Only first thread in block does work
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     SegmentInfo seg = segments[seg_id];
// LEGACY_BACKTRACK:     u32 start = seg.start_pos;
// LEGACY_BACKTRACK:     u32 end = seg.end_pos;
// LEGACY_BACKTRACK:     u32 offset = seg.sequence_offset;
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     // Backtrack from end to start of this segment
// LEGACY_BACKTRACK:     u32 pos = end;
// LEGACY_BACKTRACK:     u32 seq_idx = 0;
// LEGACY_BACKTRACK:     u32 literal_count = 0;
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     while (pos > start && seq_idx < max_seq_per_segment) {
// LEGACY_BACKTRACK:         ParseCost c = costs[pos];
// LEGACY_BACKTRACK:         
// LEGACY_BACKTRACK:         if (c.is_match) {
// LEGACY_BACKTRACK:             // Record match sequence
// LEGACY_BACKTRACK:             d_literal_lengths[offset + seq_idx] = literal_count;
// LEGACY_BACKTRACK:             d_match_lengths[offset + seq_idx] = c.len;
// LEGACY_BACKTRACK:             d_offsets[offset + seq_idx] = c.offset;
// LEGACY_BACKTRACK:             seq_idx++;
// LEGACY_BACKTRACK:             literal_count = 0;
// LEGACY_BACKTRACK:             pos -= c.len;
// LEGACY_BACKTRACK:         } else {
// LEGACY_BACKTRACK:             // Count literal
// LEGACY_BACKTRACK:             literal_count++;
// LEGACY_BACKTRACK:             pos--;
// LEGACY_BACKTRACK:         }
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     // Handle trailing literals
// LEGACY_BACKTRACK:     if (literal_count > 0 && seq_idx < max_seq_per_segment) {
// LEGACY_BACKTRACK:         d_literal_lengths[offset + seq_idx] = literal_count;
// LEGACY_BACKTRACK:         d_match_lengths[offset + seq_idx] = 0;
// LEGACY_BACKTRACK:         d_offsets[offset + seq_idx] = 0;
// LEGACY_BACKTRACK:         seq_idx++;
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK:     
// LEGACY_BACKTRACK:     // Store sequence count for this segment
// LEGACY_BACKTRACK:     d_sequence_counts[seg_id] = seq_idx;
// LEGACY_BACKTRACK: }

// LEGACY_BACKTRACK_IMPL: Status find_matches_parallel(
// LEGACY_BACKTRACK_IMPL:     const u8* d_input,
// LEGACY_BACKTRACK_IMPL:     u32 input_size,
// LEGACY_BACKTRACK_IMPL:     CompressionWorkspace& workspace,
// LEGACY_BACKTRACK_IMPL:     const LZ77Config& config,
// LEGACY_BACKTRACK_IMPL:     cudaStream_t stream
// LEGACY_BACKTRACK_IMPL: ) {
// LEGACY_BACKTRACK_IMPL: //     std::cout << "[LZ77] Pass 1: Finding matches (parallel)" << std::endl;
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     u32 hash_size = workspace.hash_table_size;
// LEGACY_BACKTRACK_IMPL:     u32 chain_size = workspace.chain_table_size;
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     const u32 init_threads = 256;
// LEGACY_BACKTRACK_IMPL:     const u32 init_blocks = (std::max(hash_size, chain_size) + init_threads - 1) / init_threads;
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     init_hash_table_kernel<<<init_blocks, init_threads, 0, stream>>>(
// LEGACY_BACKTRACK_IMPL:         workspace.d_hash_table,
// LEGACY_BACKTRACK_IMPL:         workspace.d_chain_table,
// LEGACY_BACKTRACK_IMPL:         hash_size,
// LEGACY_BACKTRACK_IMPL:         chain_size
// LEGACY_BACKTRACK_IMPL:     );
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     cudaError_t err = cudaGetLastError();
// LEGACY_BACKTRACK_IMPL:     if (err != cudaSuccess) {
// LEGACY_BACKTRACK_IMPL: //         std::cerr << "[LZ77] init_hash_table_kernel launch failed: " 
// LEGACY_BACKTRACK_IMPL: //                   << cudaGetErrorString(err) << std::endl;
// LEGACY_BACKTRACK_IMPL:         return Status::ERROR_CUDA_ERROR;
// LEGACY_BACKTRACK_IMPL:     }
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     const u32 threads = 256;
// LEGACY_BACKTRACK_IMPL:     const u32 blocks = (input_size + threads - 1) / threads;
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     find_matches_kernel<<<blocks, threads, 0, stream>>>(
// LEGACY_BACKTRACK_IMPL:         d_input,
// LEGACY_BACKTRACK_IMPL:         input_size,
// LEGACY_BACKTRACK_IMPL:         workspace.d_hash_table,
// LEGACY_BACKTRACK_IMPL:         workspace.d_chain_table,
// LEGACY_BACKTRACK_IMPL:         workspace.d_matches,
// LEGACY_BACKTRACK_IMPL:         config,
// LEGACY_BACKTRACK_IMPL:         config.hash_log
// LEGACY_BACKTRACK_IMPL:     );
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     err = cudaGetLastError();
// LEGACY_BACKTRACK_IMPL:     if (err != cudaSuccess) {
// LEGACY_BACKTRACK_IMPL: //         std::cerr << "[LZ77] find_matches_kernel launch failed: " 
// LEGACY_BACKTRACK_IMPL: //                   << cudaGetErrorString(err) << std::endl;
// LEGACY_BACKTRACK_IMPL:         return Status::ERROR_CUDA_ERROR;
// LEGACY_BACKTRACK_IMPL:     }
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL:     return Status::SUCCESS;
// LEGACY_BACKTRACK_IMPL: }
// LEGACY_BACKTRACK_IMPL: 
// LEGACY_BACKTRACK_IMPL: Status compute_optimal_parse(
// LEGACY_BACKTRACK_IMPL:     const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
) {
//     std::cout << "[LZ77] Pass 2: Computing optimal parse (parallel)" << std::endl;

    ParseCost initial_cost;
    initial_cost.set(0, 0);  // cost=0, parent=0

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


// ==============================================================================
// Parallel Backtracking Implementation (Phase 2)
// ==============================================================================

// Create backtracking configuration based on input size
BacktrackConfig create_backtrack_config(u32 input_size) {
    BacktrackConfig config;
    
    // Threshold: use parallel for inputs >= 1MB
    config.parallel_threshold = 1024 * 1024;  // 1MB
    
    if (input_size < config.parallel_threshold) {
        // Small input: use CPU (parallel overhead not worth it)
        config.use_parallel = false;
        config.segment_size = input_size;
        config.num_segments = 1;
    } else {
        // Large input: use parallel GPU backtracking
        config.use_parallel = true;
        
        // Adaptive segment sizing based on input size
        if (input_size < 10 * 1024 * 1024) {
            // 1-10MB: use 256KB segments
            config.segment_size = 256 * 1024;
        } else {
            // >10MB: use 1MB segments
            config.segment_size = 1024 * 1024;
        }
        
        config.num_segments = (input_size + config.segment_size - 1) / config.segment_size;
    }
    
    return config;
}

// ============================================================================
// V2: Optimized Multi-Pass Optimal Parser (10-100x faster!)
// ============================================================================

Status compute_optimal_parse_v2(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
) {
    cudaMemsetAsync(workspace.d_costs, 0xFF, (input_size + 1) * sizeof(ParseCost), stream);

    ParseCost initial_cost;
    initial_cost.set(0, 0);
    cudaMemcpyAsync(workspace.d_costs, &initial_cost, sizeof(ParseCost),
                    cudaMemcpyHostToDevice, stream);

    const u32 threads = 256;
    const u32 num_blocks = (input_size + threads - 1) / threads;
    
    int max_passes;
    if (input_size < 100 * 1024) max_passes = 50;
    else if (input_size < 1024 * 1024) max_passes = 100;
    else if (input_size < 10 * 1024 * 1024) max_passes = 200;
    else max_passes = 300;
    
    for (int pass = 0; pass < max_passes; ++pass) {
        cuda_zstd::lz77::optimal_parse_kernel_v2<<<num_blocks, threads, 0, stream>>>(
            input_size,
            reinterpret_cast<cuda_zstd::lz77::Match*>(workspace.d_matches),
            reinterpret_cast<cuda_zstd::lz77::ParseCost*>(workspace.d_costs)
        );
        cudaStreamSynchronize(stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return Status::ERROR_CUDA_ERROR;
    }
    return Status::SUCCESS;
}

// DEPRECATED: CPU backtracking uses old ParseCost format - not used (we use parallel GPU)
/*
Status backtrack_sequences_cpu(
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
    return Status::SUCCESS;
}
*/

// Parallel GPU backtracking implementation
Status backtrack_sequences_parallel(
    const ParseCost* d_costs,
    u32 input_size,
    CompressionWorkspace& workspace,
    u32* h_num_sequences,
    cudaStream_t stream
) {
    // Get configuration for this input size
    BacktrackConfig config = create_backtrack_config(input_size);
    
    if (!config.use_parallel) {
        // Shouldn't happen, but handle gracefully
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    u32 num_segments = config.num_segments;
    u32 segment_size = config.segment_size;
    
    // Estimate max sequences per segment
    // Typical: 1 sequence per 100 bytes is more realistic
    // Add buffer for worst case (many small matches)
    u32 max_seq_per_segment = (segment_size / 50) + 1000;  // Much more conservative
    u32 total_max_sequences = num_segments * max_seq_per_segment;
    
    // Allocate host memory for segments and results
    SegmentInfo* h_segments = new SegmentInfo[num_segments];
    u32* h_sequence_counts = new u32[num_segments];
    
    // Initialize segment boundaries
    for (u32 i = 0; i < num_segments; ++i) {
        h_segments[i].start_pos = i * segment_size;
        h_segments[i].end_pos = std::min((i + 1) * segment_size, input_size) - 1;
        h_segments[i].sequence_offset = i * max_seq_per_segment;
        h_segments[i].num_sequences = 0;
    }
    
    // Allocate device memory for segments and temporary sequence buffers
    SegmentInfo* d_segments = nullptr;
    u32* d_sequence_counts = nullptr;
    u32* d_literal_lengths_temp = nullptr;
    u32* d_match_lengths_temp = nullptr;
    u32* d_offsets_temp = nullptr;
    
    cudaError_t err;
    err = cudaMalloc(&d_segments, num_segments * sizeof(SegmentInfo));
    if (err != cudaSuccess) {
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    err = cudaMalloc(&d_sequence_counts, num_segments * sizeof(u32));
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    err = cudaMalloc(&d_literal_lengths_temp, total_max_sequences * sizeof(u32));
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    err = cudaMalloc(&d_match_lengths_temp, total_max_sequences * sizeof(u32));
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    err = cudaMalloc(&d_offsets_temp, total_max_sequences * sizeof(u32));
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Copy segment info to device
    err = cudaMemcpyAsync(d_segments, h_segments, num_segments * sizeof(SegmentInfo),
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to copy segments to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_CUDA_ERROR;
    }
    
    // Debug: Print configuration
    printf("[DEBUG] Parallel backtracking config:\n");
    printf("  Input size: %u bytes (%.2f MB)\n", input_size, input_size / (1024.0f * 1024.0f));
    printf("  Num segments: %u\n", num_segments);
    printf("  Segment size: %u bytes (%.2f KB)\n", segment_size, segment_size / 1024.0f);
    printf("  Max seq/segment: %u\n", max_seq_per_segment);
    printf("  Total temp memory: %.2f MB\n", (total_max_sequences * 12) / (1024.0f * 1024.0f));
    
    // Debug: Validate pointers
    printf("[DEBUG] Validating pointers:\n");
    printf("  d_costs: %p\n", (void*)d_costs);
    printf("  d_segments: %p\n", (void*)d_segments);
    printf("  d_literal_lengths_temp: %p\n", (void*)d_literal_lengths_temp);
    printf("  d_match_lengths_temp: %p\n", (void*)d_match_lengths_temp);
    printf("  d_offsets_temp: %p\n", (void*)d_offsets_temp);
    printf("  d_sequence_counts: %p\n", (void*)d_sequence_counts);
    
    if (d_costs == nullptr) {
        printf("[ERROR] d_costs is NULL!\n");
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // Launch parallel backtracking kernel (one block per segment)
    // Use 32 threads per block (warp size) even though only first thread does work
    printf("[DEBUG] Launching kernel with %u blocks, 32 threads/block...\n", num_segments);
    backtrack_segments_parallel_kernel<<<num_segments, 32>>>(
        d_costs,
        d_segments,
        d_literal_lengths_temp,
        d_match_lengths_temp,
        d_offsets_temp,
        d_sequence_counts,
        num_segments,
        max_seq_per_segment
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_CUDA_ERROR;
    }
    
    // Wait for kernel to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_CUDA_ERROR;
    }
    
    printf("[DEBUG] Kernel completed successfully\n");
    
    // Copy sequence counts back to host for merging
    err = cudaMemcpyAsync(h_sequence_counts, d_sequence_counts, 
                         num_segments * sizeof(u32),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_CUDA_ERROR;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_segments);
        cudaFree(d_sequence_counts);
        cudaFree(d_literal_lengths_temp);
        cudaFree(d_match_lengths_temp);
        cudaFree(d_offsets_temp);
        delete[] h_segments;
        delete[] h_sequence_counts;
        return Status::ERROR_CUDA_ERROR;
    }
    
    // Merge segments: simply concatenate (no boundary stitching needed if segments are independent)
    u32 total_sequences = 0;
    for (u32 i = 0; i < num_segments; ++i) {
        total_sequences += h_sequence_counts[i];
    }
    
    // Allocate host memory for merged sequences
    u32* h_literal_lengths = new u32[total_sequences];
    u32* h_match_lengths = new u32[total_sequences];
    u32* h_offsets = new u32[total_sequences];
    
    // Copy all sequences from device
    u32* h_literal_lengths_temp = new u32[total_max_sequences];
    u32* h_match_lengths_temp = new u32[total_max_sequences];
    u32* h_offsets_temp = new u32[total_max_sequences];
    
    err = cudaMemcpyAsync(h_literal_lengths_temp, d_literal_lengths_temp,
                         total_max_sequences * sizeof(u32),
                         cudaMemcpyDeviceToHost, stream);
    err = cudaMemcpyAsync(h_match_lengths_temp, d_match_lengths_temp,
                         total_max_sequences * sizeof(u32),
                         cudaMemcpyDeviceToHost, stream);
    err = cudaMemcpyAsync(h_offsets_temp, d_offsets_temp,
                         total_max_sequences * sizeof(u32),
                         cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Merge: copy sequences from each segment to final buffer
    u32 out_idx = 0;
    for (u32 i = 0; i < num_segments; ++i) {
        u32 seg_offset = h_segments[i].sequence_offset;
        u32 seg_count = h_sequence_counts[i];
        
        for (u32 j = 0; j < seg_count; ++j) {
            h_literal_lengths[out_idx] = h_literal_lengths_temp[seg_offset + j];
            h_match_lengths[out_idx] = h_match_lengths_temp[seg_offset + j];
            h_offsets[out_idx] = h_offsets_temp[seg_offset + j];
            out_idx++;
        }
    }
    
    *h_num_sequences = total_sequences;
    
    // Copy merged results to device workspace
    err = cudaMemcpyAsync(workspace.d_literal_lengths_reverse, h_literal_lengths,
                         total_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    err = cudaMemcpyAsync(workspace.d_match_lengths_reverse, h_match_lengths,
                         total_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    err = cudaMemcpyAsync(workspace.d_offsets_reverse, h_offsets,
                         total_sequences * sizeof(u32), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(d_segments);
    cudaFree(d_sequence_counts);
    cudaFree(d_literal_lengths_temp);
    cudaFree(d_match_lengths_temp);
    cudaFree(d_offsets_temp);
    delete[] h_segments;
    delete[] h_sequence_counts;
    delete[] h_literal_lengths;
    delete[] h_match_lengths;
    delete[] h_offsets;
    delete[] h_literal_lengths_temp;
    delete[] h_match_lengths_temp;
    delete[] h_offsets_temp;
    
    return Status::SUCCESS;
}


// Adaptive backtracking: chooses parallel or CPU based on input size
Status backtrack_sequences(
    u32 input_size,
    CompressionWorkspace& workspace,
    u32* h_num_sequences,
    cudaStream_t stream
) {
    // Get adaptive configuration
    BacktrackConfig config = create_backtrack_config(input_size);
    
    if (config.use_parallel) {
        // Use parallel GPU backtracking for large inputs
//        std::cout << "[LZ77] Pass 3: Backtracking sequences (parallel GPU, " 
//                  << config.num_segments << " segments)" << std::endl;
        return backtrack_sequences_parallel(workspace.d_costs, input_size, 
                                           workspace, h_num_sequences, stream);
    } else {
        // Use CPU backtracking for small inputs
//        std::cout << "[LZ77] Pass 3: Backtracking sequences (CPU offload)" << std::endl;
        
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
        
//        std::cout << "[LZ77] Found " << *h_num_sequences << " sequences (CPU)" << std::endl;
        
        // Cleanup
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        
        return Status::SUCCESS;
    }
}

} // namespace lz77
} // namespace compression
