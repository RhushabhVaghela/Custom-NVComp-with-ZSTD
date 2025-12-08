// ==============================================================================
// lz77_parallel.cu - Three-pass parallel LZ77 implementation
// ==============================================================================

#include "cuda_zstd_lz77.h" // For V2 kernel
#include "lz77_parallel.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace cuda_zstd {
namespace lz77 {

__global__ void init_hash_table_kernel(u32 *hash_table, u32 *chain_table,
                                       u32 hash_size, u32 chain_size) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < hash_size; i += stride) {
    hash_table[i] = 0xFFFFFFFF;
  }

  for (u32 i = idx; i < chain_size; i += stride) {
    chain_table[i] = 0xFFFFFFFF;
  }
}

__global__ void find_matches_kernel(const u8 *input, u32 input_size,
                                    u32 *hash_table, u32 *chain_table,
                                    Match *matches, const LZ77Config config,
                                    u32 hash_log) {
  u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= input_size - config.min_match)
    return;

  u32 hash = compute_hash(input, pos, hash_log);
  u32 hash_idx = hash % (1 << hash_log);

  u32 chain_mask = (1 << config.chain_log) - 1;
  u32 prev_pos = atomicExch(&hash_table[hash_idx], pos);
  chain_table[pos & chain_mask] = prev_pos;

  Match best_match(pos, 0, 0, 0);
  u32 best_cost = 0xFFFFFFFF;

  u32 search_pos = prev_pos;
  for (u32 depth = 0; depth < config.search_depth && search_pos != 0xFFFFFFFF;
       depth++) {
    if (search_pos >= pos)
      break;

    u32 offset = pos - search_pos;
    u32 len =
        match_length(input, pos, search_pos, config.nice_length, input_size);

    if (len >= config.min_match) {
      u32 cost = calculate_match_cost(len, offset);
      if (cost < best_cost) {
        best_cost = cost;
        best_match.offset = offset;
        best_match.length = len;
      }

      if (len >= config.good_length)
        break;
    }

    search_pos = chain_table[search_pos & chain_mask];
  }

  matches[pos] = best_match;
}

__global__ void compute_costs_kernel(const u8 *input, u32 input_size,
                                     const Match *matches, ParseCost *costs,
                                     const LZ77Config config) {
  u32 pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (pos >= input_size)
    return;

  u32 prev_cost = costs[pos - 1].cost();
  u32 literal_cost_val = prev_cost + calculate_literal_cost(1);

  ParseCost literal_cost;
  literal_cost.set(literal_cost_val, pos - 1); // parent = pos-1

  ParseCost best_cost = literal_cost;

  Match m = matches[pos];
  if (m.length >= config.min_match) {
    u32 match_cost_val = prev_cost + calculate_match_cost(m.length, m.offset);

    if (match_cost_val < literal_cost_val) {
      best_cost.set(match_cost_val, pos - 1); // parent = pos-1
    }
  }

  costs[pos] = best_cost;
}

// LEGACY_BACKTRACK: __global__ void backtrack_kernel(
// LEGACY_BACKTRACK:     const ParseCost* costs,
// LEGACY_BACKTRACK:     u32 input_size,
// LEGACY_BACKTRACK:     u32* literal_lengths,
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
// LEGACY_BACKTRACK: //
// ==============================================================================
// LEGACY_BACKTRACK: // Parallel Backtracking Kernel (Phase 2)
// LEGACY_BACKTRACK: //
// ==============================================================================
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
// LEGACY_BACKTRACK:     if (threadIdx.x != 0) return;  // Only first thread in
// block does work LEGACY_BACKTRACK: LEGACY_BACKTRACK:     SegmentInfo seg =
// segments[seg_id]; LEGACY_BACKTRACK:     u32 start = seg.start_pos;
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
// LEGACY_BACKTRACK:             d_literal_lengths[offset + seq_idx] =
// literal_count; LEGACY_BACKTRACK:             d_match_lengths[offset +
// seq_idx] = c.len; LEGACY_BACKTRACK:             d_offsets[offset + seq_idx] =
// c.offset; LEGACY_BACKTRACK:             seq_idx++; LEGACY_BACKTRACK:
// literal_count = 0; LEGACY_BACKTRACK:             pos -= c.len;
// LEGACY_BACKTRACK:         } else {
// LEGACY_BACKTRACK:             // Count literal
// LEGACY_BACKTRACK:             literal_count++;
// LEGACY_BACKTRACK:             pos--;
// LEGACY_BACKTRACK:         }
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK:
// LEGACY_BACKTRACK:     // Handle trailing literals
// LEGACY_BACKTRACK:     if (literal_count > 0 && seq_idx < max_seq_per_segment)
// { LEGACY_BACKTRACK:         d_literal_lengths[offset + seq_idx] =
// literal_count; LEGACY_BACKTRACK:         d_match_lengths[offset + seq_idx] =
// 0; LEGACY_BACKTRACK:         d_offsets[offset + seq_idx] = 0;
// LEGACY_BACKTRACK:         seq_idx++;
// LEGACY_BACKTRACK:     }
// LEGACY_BACKTRACK:
// LEGACY_BACKTRACK:     // Store sequence count for this segment
// LEGACY_BACKTRACK:     d_sequence_counts[seg_id] = seq_idx;
// LEGACY_BACKTRACK: }

Status find_matches_parallel(const u8 *d_input, u32 input_size,
                             CompressionWorkspace *workspace,
                             const LZ77Config &config, cudaStream_t stream) {
  if (!workspace) {
    printf("Error: workspace is null\n");
    return Status::ERROR_INVALID_PARAMETER;
  }

  u32 hash_size = workspace->hash_table_size;
  u32 chain_size = workspace->chain_table_size;

  const u32 init_threads = 256;
  const u32 init_blocks =
      (std::max(hash_size, chain_size) + init_threads - 1) / init_threads;

  init_hash_table_kernel<<<init_blocks, init_threads, 0, stream>>>(
      workspace->d_hash_table, workspace->d_chain_table, hash_size, chain_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  const u32 threads = 256;
  const u32 blocks = (input_size + threads - 1) / threads;

  find_matches_kernel<<<blocks, threads, 0, stream>>>(
      d_input, input_size, workspace->d_hash_table, workspace->d_chain_table,
      reinterpret_cast<Match *>(workspace->d_matches), config, config.hash_log);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  return Status::SUCCESS;
}

// ============================================================================
// V2: Optimized Multi-Pass Optimal Parser (10-100x faster!)
// ============================================================================

Status compute_optimal_parse_v2(const u8 *d_input, u32 input_size,
                                CompressionWorkspace *workspace,
                                const LZ77Config &config, cudaStream_t stream) {
  cudaMemset(workspace->d_costs, 0xFF, (input_size + 1) * sizeof(ParseCost));

  ParseCost initial_cost;
  initial_cost.set(0, 0);
  cudaMemcpy(workspace->d_costs, &initial_cost, sizeof(ParseCost),
             cudaMemcpyHostToDevice);

  const u32 threads = 256;
  const u32 num_blocks = (input_size + threads - 1) / threads;

  int max_passes;
  if (input_size < 100 * 1024)
    max_passes = input_size / 2; // Heuristic: need many passes
  else if (input_size < 1024 * 1024)
    max_passes = 5000;
  else
    max_passes = 10000;

  for (int pass = 0; pass < max_passes; ++pass) {
    cuda_zstd::lz77::
        optimal_parse_kernel_v2<<<num_blocks, threads, 0, stream>>>(
            input_size,
            reinterpret_cast<cuda_zstd::lz77::Match *>(workspace->d_matches),
            reinterpret_cast<cuda_zstd::lz77::ParseCost *>(workspace->d_costs));
  }

  // Check for errors occasionally or at the end
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  return Status::SUCCESS;
}

// DEPRECATED: CPU backtracking uses old ParseCost format - not used (we use
// parallel GPU)
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
// LEGACY_PARALLEL_BACKTRACK: Status backtrack_sequences_parallel(
// LEGACY_PARALLEL_BACKTRACK:     const ParseCost* d_costs,
// LEGACY_PARALLEL_BACKTRACK:     u32 input_size,
// LEGACY_PARALLEL_BACKTRACK:     CompressionWorkspace& workspace,
// LEGACY_PARALLEL_BACKTRACK:     u32* h_num_sequences,
// LEGACY_PARALLEL_BACKTRACK:     cudaStream_t stream
// LEGACY_PARALLEL_BACKTRACK: ) {
// LEGACY_PARALLEL_BACKTRACK:     // Get configuration for this input size
// LEGACY_PARALLEL_BACKTRACK:     BacktrackConfig config =
// create_backtrack_config(input_size); LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     if (!config.use_parallel) {
// LEGACY_PARALLEL_BACKTRACK:         // Shouldn't happen, but handle gracefully
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_INVALID_PARAMETER;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     u32 num_segments = config.num_segments;
// LEGACY_PARALLEL_BACKTRACK:     u32 segment_size = config.segment_size;
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Estimate max sequences per segment
// LEGACY_PARALLEL_BACKTRACK:     // Typical: 1 sequence per 100 bytes is more
// realistic LEGACY_PARALLEL_BACKTRACK:     // Add buffer for worst case (many
// small matches) LEGACY_PARALLEL_BACKTRACK:     u32 max_seq_per_segment =
// (segment_size / 50) + 1000;  // Much more conservative
// LEGACY_PARALLEL_BACKTRACK:     u32 total_max_sequences = num_segments *
// max_seq_per_segment; LEGACY_PARALLEL_BACKTRACK: LEGACY_PARALLEL_BACKTRACK: //
// Allocate host memory for segments and results LEGACY_PARALLEL_BACKTRACK:
// SegmentInfo* h_segments = new SegmentInfo[num_segments];
// LEGACY_PARALLEL_BACKTRACK:     u32* h_sequence_counts = new
// u32[num_segments]; LEGACY_PARALLEL_BACKTRACK: LEGACY_PARALLEL_BACKTRACK: //
// Initialize segment boundaries LEGACY_PARALLEL_BACKTRACK:     for (u32 i = 0;
// i < num_segments; ++i) { LEGACY_PARALLEL_BACKTRACK: h_segments[i].start_pos =
// i * segment_size; LEGACY_PARALLEL_BACKTRACK:         h_segments[i].end_pos =
// std::min((i + 1) * segment_size, input_size) - 1; LEGACY_PARALLEL_BACKTRACK:
// h_segments[i].sequence_offset = i * max_seq_per_segment;
// LEGACY_PARALLEL_BACKTRACK:         h_segments[i].num_sequences = 0;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Allocate device memory for segments and
// temporary sequence buffers LEGACY_PARALLEL_BACKTRACK:     SegmentInfo*
// d_segments = nullptr; LEGACY_PARALLEL_BACKTRACK:     u32* d_sequence_counts =
// nullptr; LEGACY_PARALLEL_BACKTRACK:     u32* d_literal_lengths_temp =
// nullptr; LEGACY_PARALLEL_BACKTRACK:     u32* d_match_lengths_temp = nullptr;
// LEGACY_PARALLEL_BACKTRACK:     u32* d_offsets_temp = nullptr;
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     cudaError_t err;
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMalloc(&d_segments, num_segments *
// sizeof(SegmentInfo)); LEGACY_PARALLEL_BACKTRACK:     if (err != cudaSuccess)
// { LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_OUT_OF_MEMORY;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMalloc(&d_sequence_counts,
// num_segments * sizeof(u32)); LEGACY_PARALLEL_BACKTRACK:     if (err !=
// cudaSuccess) { LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_OUT_OF_MEMORY;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMalloc(&d_literal_lengths_temp,
// total_max_sequences * sizeof(u32)); LEGACY_PARALLEL_BACKTRACK:     if (err !=
// cudaSuccess) { LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_OUT_OF_MEMORY;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMalloc(&d_match_lengths_temp,
// total_max_sequences * sizeof(u32)); LEGACY_PARALLEL_BACKTRACK:     if (err !=
// cudaSuccess) { LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_OUT_OF_MEMORY;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMalloc(&d_offsets_temp,
// total_max_sequences * sizeof(u32)); LEGACY_PARALLEL_BACKTRACK:     if (err !=
// cudaSuccess) { LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_OUT_OF_MEMORY;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Copy segment info to device
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMemcpyAsync(d_segments, h_segments,
// num_segments * sizeof(SegmentInfo), LEGACY_PARALLEL_BACKTRACK:
// cudaMemcpyHostToDevice, stream); LEGACY_PARALLEL_BACKTRACK:     if (err !=
// cudaSuccess) { LEGACY_PARALLEL_BACKTRACK:         printf("[ERROR] Failed to
// copy segments to device: %s\n", cudaGetErrorString(err));
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_CUDA_ERROR;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Debug: Print configuration
// LEGACY_PARALLEL_BACKTRACK:     printf("[DEBUG] Parallel backtracking
// config:\n"); LEGACY_PARALLEL_BACKTRACK:     printf("  Input size: %u bytes
// (%.2f MB)\n", input_size, input_size / (1024.0f * 1024.0f));
// LEGACY_PARALLEL_BACKTRACK:     printf("  Num segments: %u\n", num_segments);
// LEGACY_PARALLEL_BACKTRACK:     printf("  Segment size: %u bytes (%.2f KB)\n",
// segment_size, segment_size / 1024.0f); LEGACY_PARALLEL_BACKTRACK: printf("
// Max seq/segment: %u\n", max_seq_per_segment); LEGACY_PARALLEL_BACKTRACK:
// printf("  Total temp memory: %.2f MB\n", (total_max_sequences * 12) /
// (1024.0f * 1024.0f)); LEGACY_PARALLEL_BACKTRACK: LEGACY_PARALLEL_BACKTRACK:
// // Debug: Validate pointers LEGACY_PARALLEL_BACKTRACK:     printf("[DEBUG]
// Validating pointers:\n"); LEGACY_PARALLEL_BACKTRACK:     printf("  d_costs:
// %p\n", (void*)d_costs); LEGACY_PARALLEL_BACKTRACK:     printf("  d_segments:
// %p\n", (void*)d_segments); LEGACY_PARALLEL_BACKTRACK:     printf("
// d_literal_lengths_temp: %p\n", (void*)d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:     printf("  d_match_lengths_temp: %p\n",
// (void*)d_match_lengths_temp); LEGACY_PARALLEL_BACKTRACK:     printf("
// d_offsets_temp: %p\n", (void*)d_offsets_temp); LEGACY_PARALLEL_BACKTRACK:
// printf("  d_sequence_counts: %p\n", (void*)d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     if (d_costs == nullptr) {
// LEGACY_PARALLEL_BACKTRACK:         printf("[ERROR] d_costs is NULL!\n");
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_INVALID_PARAMETER;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Launch parallel backtracking kernel (one
// block per segment) LEGACY_PARALLEL_BACKTRACK:     // Use 32 threads per block
// (warp size) even though only first thread does work
// LEGACY_PARALLEL_BACKTRACK:     printf("[DEBUG] Launching kernel with %u
// blocks, 32 threads/block...\n", num_segments); LEGACY_PARALLEL_BACKTRACK:
// backtrack_segments_parallel_kernel<<<num_segments, 32>>>(
// LEGACY_PARALLEL_BACKTRACK:         d_costs,
// LEGACY_PARALLEL_BACKTRACK:         d_segments,
// LEGACY_PARALLEL_BACKTRACK:         d_literal_lengths_temp,
// LEGACY_PARALLEL_BACKTRACK:         d_match_lengths_temp,
// LEGACY_PARALLEL_BACKTRACK:         d_offsets_temp,
// LEGACY_PARALLEL_BACKTRACK:         d_sequence_counts,
// LEGACY_PARALLEL_BACKTRACK:         num_segments,
// LEGACY_PARALLEL_BACKTRACK:         max_seq_per_segment
// LEGACY_PARALLEL_BACKTRACK:     );
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaGetLastError();
// LEGACY_PARALLEL_BACKTRACK:     if (err != cudaSuccess) {
// LEGACY_PARALLEL_BACKTRACK:         printf("[ERROR] Kernel launch failed:
// %s\n", cudaGetErrorString(err)); LEGACY_PARALLEL_BACKTRACK:
// cudaFree(d_segments); LEGACY_PARALLEL_BACKTRACK: cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_CUDA_ERROR;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Wait for kernel to complete
// LEGACY_PARALLEL_BACKTRACK:     err = cudaStreamSynchronize(stream);
// LEGACY_PARALLEL_BACKTRACK:     if (err != cudaSuccess) {
// LEGACY_PARALLEL_BACKTRACK:         printf("[ERROR] Kernel execution failed:
// %s\n", cudaGetErrorString(err)); LEGACY_PARALLEL_BACKTRACK:
// cudaFree(d_segments); LEGACY_PARALLEL_BACKTRACK: cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_CUDA_ERROR;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     printf("[DEBUG] Kernel completed
// successfully\n"); LEGACY_PARALLEL_BACKTRACK: LEGACY_PARALLEL_BACKTRACK: //
// Copy sequence counts back to host for merging LEGACY_PARALLEL_BACKTRACK: err
// = cudaMemcpyAsync(h_sequence_counts, d_sequence_counts,
// LEGACY_PARALLEL_BACKTRACK:                          num_segments *
// sizeof(u32), LEGACY_PARALLEL_BACKTRACK: cudaMemcpyDeviceToHost, stream);
// LEGACY_PARALLEL_BACKTRACK:     if (err != cudaSuccess) {
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_CUDA_ERROR;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaStreamSynchronize(stream);
// LEGACY_PARALLEL_BACKTRACK:     if (err != cudaSuccess) {
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:         cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:         delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:         return Status::ERROR_CUDA_ERROR;
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Merge segments: simply concatenate (no
// boundary stitching needed if segments are independent)
// LEGACY_PARALLEL_BACKTRACK:     u32 total_sequences = 0;
// LEGACY_PARALLEL_BACKTRACK:     for (u32 i = 0; i < num_segments; ++i) {
// LEGACY_PARALLEL_BACKTRACK:         total_sequences += h_sequence_counts[i];
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Allocate host memory for merged sequences
// LEGACY_PARALLEL_BACKTRACK:     u32* h_literal_lengths = new
// u32[total_sequences]; LEGACY_PARALLEL_BACKTRACK:     u32* h_match_lengths =
// new u32[total_sequences]; LEGACY_PARALLEL_BACKTRACK:     u32* h_offsets = new
// u32[total_sequences]; LEGACY_PARALLEL_BACKTRACK: LEGACY_PARALLEL_BACKTRACK:
// // Copy all sequences from device LEGACY_PARALLEL_BACKTRACK:     u32*
// h_literal_lengths_temp = new u32[total_max_sequences];
// LEGACY_PARALLEL_BACKTRACK:     u32* h_match_lengths_temp = new
// u32[total_max_sequences]; LEGACY_PARALLEL_BACKTRACK:     u32* h_offsets_temp
// = new u32[total_max_sequences]; LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMemcpyAsync(h_literal_lengths_temp,
// d_literal_lengths_temp, LEGACY_PARALLEL_BACKTRACK: total_max_sequences *
// sizeof(u32), LEGACY_PARALLEL_BACKTRACK: cudaMemcpyDeviceToHost, stream);
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMemcpyAsync(h_match_lengths_temp,
// d_match_lengths_temp, LEGACY_PARALLEL_BACKTRACK: total_max_sequences *
// sizeof(u32), LEGACY_PARALLEL_BACKTRACK: cudaMemcpyDeviceToHost, stream);
// LEGACY_PARALLEL_BACKTRACK:     err = cudaMemcpyAsync(h_offsets_temp,
// d_offsets_temp, LEGACY_PARALLEL_BACKTRACK: total_max_sequences * sizeof(u32),
// LEGACY_PARALLEL_BACKTRACK:                          cudaMemcpyDeviceToHost,
// stream); LEGACY_PARALLEL_BACKTRACK:     cudaStreamSynchronize(stream);
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Merge: copy sequences from each segment to
// final buffer LEGACY_PARALLEL_BACKTRACK:     u32 out_idx = 0;
// LEGACY_PARALLEL_BACKTRACK:     for (u32 i = 0; i < num_segments; ++i) {
// LEGACY_PARALLEL_BACKTRACK:         u32 seg_offset =
// h_segments[i].sequence_offset; LEGACY_PARALLEL_BACKTRACK:         u32
// seg_count = h_sequence_counts[i]; LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:         for (u32 j = 0; j < seg_count; ++j) {
// LEGACY_PARALLEL_BACKTRACK:             h_literal_lengths[out_idx] =
// h_literal_lengths_temp[seg_offset + j]; LEGACY_PARALLEL_BACKTRACK:
// h_match_lengths[out_idx] = h_match_lengths_temp[seg_offset + j];
// LEGACY_PARALLEL_BACKTRACK:             h_offsets[out_idx] =
// h_offsets_temp[seg_offset + j]; LEGACY_PARALLEL_BACKTRACK: out_idx++;
// LEGACY_PARALLEL_BACKTRACK:         }
// LEGACY_PARALLEL_BACKTRACK:     }
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     *h_num_sequences = total_sequences;
// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Copy merged results to device workspace
// LEGACY_PARALLEL_BACKTRACK:     err =
// cudaMemcpyAsync(workspace.d_literal_lengths_reverse, h_literal_lengths,
// LEGACY_PARALLEL_BACKTRACK:                          total_sequences *
// sizeof(u32), cudaMemcpyHostToDevice, stream); LEGACY_PARALLEL_BACKTRACK: err
// = cudaMemcpyAsync(workspace.d_match_lengths_reverse, h_match_lengths,
// LEGACY_PARALLEL_BACKTRACK:                          total_sequences *
// sizeof(u32), cudaMemcpyHostToDevice, stream); LEGACY_PARALLEL_BACKTRACK: err
// = cudaMemcpyAsync(workspace.d_offsets_reverse, h_offsets,
// LEGACY_PARALLEL_BACKTRACK:                          total_sequences *
// sizeof(u32), cudaMemcpyHostToDevice, stream); LEGACY_PARALLEL_BACKTRACK:
// cudaStreamSynchronize(stream); LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK:     // Cleanup
// LEGACY_PARALLEL_BACKTRACK:     cudaFree(d_segments);
// LEGACY_PARALLEL_BACKTRACK:     cudaFree(d_sequence_counts);
// LEGACY_PARALLEL_BACKTRACK:     cudaFree(d_literal_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:     cudaFree(d_match_lengths_temp);
// LEGACY_PARALLEL_BACKTRACK:     cudaFree(d_offsets_temp);
// LEGACY_PARALLEL_BACKTRACK:     delete[] h_segments;
// LEGACY_PARALLEL_BACKTRACK:     delete[] h_sequence_counts;
// LEGACY_PARALLEL_BACKTRACK:     delete[] h_literal_lengths;
// LEGACY_PARALLEL_BACKTRACK:     delete[] h_match_lengths;
// LEGACY_PARALLEL_BACKTRACK:     delete[] h_offsets;
// V2 Backtracking Implementation (CPU)
Status backtrack_sequences_v2(u32 input_size, CompressionWorkspace &workspace,
                              u32 *h_num_sequences, cudaStream_t stream) {
  // 1. Allocate host memory
  ParseCost *h_costs = new ParseCost[input_size + 1];
  Match *h_matches = new Match[input_size];

  // 2. Copy from device
  cudaMemcpyAsync(h_costs, workspace.d_costs,
                  (input_size + 1) * sizeof(ParseCost), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_matches, workspace.d_matches, input_size * sizeof(Match),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // 3. Backtrack
  std::vector<u32> lit_lens;
  std::vector<u32> match_lens;
  std::vector<u32> offsets;

  // Pre-allocate to avoid reallocations
  lit_lens.reserve(input_size / 4);
  match_lens.reserve(input_size / 4);
  offsets.reserve(input_size / 4);

  u32 pos = input_size;

  // Handle trailing literals (literals at the very end of file)
  // In LZ77, the last sequence can have match_len=0

  while (pos > 0) {
    u32 parent = h_costs[pos].parent();

    if (parent >= pos) {
      // Invalid parent, likely due to insufficient passes
      // Fallback: treat as literal
      parent = pos - 1;
    }

    u32 len = pos - parent;

    if (len >= 3) { // Match
      u32 offset = h_matches[parent].offset;
      lit_lens.push_back(0);
      match_lens.push_back(len);
      offsets.push_back(offset);
    } else { // Literal
      if (lit_lens.empty()) {
        // Trailing literals
        lit_lens.push_back(1);
        match_lens.push_back(0);
        offsets.push_back(0);
      } else {
        // Add to current sequence (which is "after" this literal in reverse
        // order) So it belongs to the sequence we just pushed
        lit_lens.back()++;
      }
    }
    pos = parent;
  }

  // Reverse to get forward order
  std::reverse(lit_lens.begin(), lit_lens.end());
  std::reverse(match_lens.begin(), match_lens.end());
  std::reverse(offsets.begin(), offsets.end());

  u32 num_seqs = (u32)lit_lens.size();
  if (h_num_sequences)
    *h_num_sequences = num_seqs;


  // 4. Copy to device buffers in workspace
  if (num_seqs > workspace.max_sequences) {
    delete[] h_costs;
    delete[] h_matches;
    return Status::ERROR_OUT_OF_MEMORY;
  }

  cudaMemcpyAsync(workspace.d_literal_lengths_reverse, lit_lens.data(),
                  num_seqs * sizeof(u32), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(workspace.d_match_lengths_reverse, match_lens.data(),
                  num_seqs * sizeof(u32), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(workspace.d_offsets_reverse, offsets.data(),
                  num_seqs * sizeof(u32), cudaMemcpyHostToDevice, stream);

  delete[] h_costs;
  delete[] h_matches;
  return Status::SUCCESS;
}

// LEGACY_PARALLEL_BACKTRACK:
// LEGACY_PARALLEL_BACKTRACK: // Adaptive backtracking: chooses parallel or CPU
// based on input size Adaptive backtracking: chooses parallel or CPU based on
// input size
Status backtrack_sequences(u32 input_size, CompressionWorkspace &workspace,
                           u32 *h_num_sequences, cudaStream_t stream) {
  // V2: Always use V2 backtracking (CPU for now, but uses new ParseCost format)
  return backtrack_sequences_v2(input_size, workspace, h_num_sequences, stream);
}

} // namespace lz77
} // namespace cuda_zstd
