// ==============================================================================
// lz77_parallel.cu - Three-pass parallel LZ77 implementation
// ==============================================================================

#include "cuda_zstd_lz77.h" // For V2 kernel
#include "lz77_parallel.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

namespace cuda_zstd {
namespace lz77 {

__global__ void init_hash_table_kernel(u32 *hash_table, u32 *chain_table,
                                       u32 hash_size, u32 chain_size) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < hash_size)
    hash_table[idx] = 0xFFFFFFFF;
  if (idx < chain_size)
    chain_table[idx] = 0xFFFFFFFF;
}

__global__ void find_matches_kernel(const u8 *input, u32 input_size,
                                    u32 *hash_table, u32 *chain_table,
                                    Match *matches, const LZ77Config config,
                                    u32 hash_log) {
  u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= input_size - config.min_match)
    return;

  // DEBUG PROBES
  if (pos == 0)
    printf("[KERNEL] Input Ptr: %p\n", input);
  if (pos == 104)
    printf("[KERNEL] Input[104] = %02X\n", input[104]);
  if (pos == 174)
    printf("[KERNEL] Input[174] = %02X\n", input[174]);

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

    u32 cost = calculate_match_cost(len, offset);
    if (cost < best_cost) {
      best_cost = cost;
      best_match.offset = offset;
      best_match.length = len;
    }

    if (len >= config.good_length)
      break;

    search_pos = chain_table[search_pos & chain_mask];
  }

  matches[pos] = best_match;
}

__global__ void compute_costs_kernel_serial(const u8 *input, u32 input_size,
                                            const Match *matches,
                                            ParseCost *costs,
                                            const LZ77Config config) {
  // Serial implementation to resolve dependencies
  // One thread does everything for the block
  // (Alternatively, use scanning, but serial is easiest for correctness now)

  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Base case
  ParseCost zero_cost;
  zero_cost.set(0, 0); // parent=0 (unused for index 0)
  costs[0] = zero_cost;

  for (u32 pos = 1; pos <= input_size; ++pos) {
    u32 prev_cost = costs[pos - 1].cost();
    u32 literal_cost_val = prev_cost + calculate_literal_cost(1);

    ParseCost best_cost;
    best_cost.set(literal_cost_val, pos - 1); // parent = pos-1

    // Check matches ending at pos?
    // Matches are stored at `start` position usually.
    // If d_matches[i] stores match starting at i.
    // Optimal parse V2 usually looks FORWARD?
    // "Min cost to reach pos".
    // If we are at `i`, we can reach `i+1` (literal) or `i+len` (match).
    // This is "Forward DP".
    // The previous kernel looked like "Backward DP" or "Pull DP" (costs[pos]
    // from costs[pos-1]). But match can jump. If matches are stored at start
    // index, we should update FUTURE costs.

    // Let's implement Forward DP.
    // Initialize all costs to infinity first?
    // Or just iterate.
    // Since matches jump forward, we need to have processed `pos` before valid.
    // So we iterate `i` from 0 to input_size-1.
    // d_costs[i] is cost to reach i.
    // From i, we can go to i+1 (literal) or i+len (match).
  }
}

// Actually, let's stick to the Forward DP pattern we see in other Zstd
// implementations.
__global__ void compute_costs_kernel_forward(const u8 *input, u32 input_size,
                                             const Match *matches,
                                             ParseCost *costs,
                                             const LZ77Config config) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx != 0)
    return; // Serial

  // Initialize costs[0] = 0, others Infinity?
  // Assume kernel init happens or we do it here.
  // We'll treat costs array as initialized to MAX_U32 by init kernel?
  // Or just set costs[0].

  // Set cost[0]
  ParseCost start_c;
  start_c.set(0, 0);
  costs[0] = start_c;

  // Initialize others to infinity if not done?
  // It's safer to do inline
  for (u32 i = 1; i <= input_size; ++i) {
    costs[i].data = 0xFFFFFFFFFFFFFFFFULL; // Max cost
  }

  for (u32 pos = 0; pos < input_size; ++pos) {
    u32 current_cost = costs[pos].cost();
    if (current_cost == 0xFFFFFFFF)
      continue; // unreachable

    // 1. Literal
    u32 lit_dest = pos + 1;
    u32 lit_cost_val = current_cost + calculate_literal_cost(1);
    if (lit_cost_val < costs[lit_dest].cost()) {
      costs[lit_dest].set(lit_cost_val, pos);
    }

    // 2. Match
    Match m = matches[pos];
    if (m.length >= config.min_match) {
      u32 match_dest = pos + m.length;
      if (match_dest <= input_size) {
        u32 match_cost_val =
            current_cost + calculate_match_cost(m.length, m.offset);
        if (match_cost_val < costs[match_dest].cost()) {
          costs[match_dest].set(match_cost_val, pos);
        }
      }
    }
  }
}

// ==============================================================================
// PARALLEL GREEDY PARSING
// Each position independently decides: emit match (if found) or literal.
// This is fully parallelizable - O(1) per position with N threads.
// Trade-off: Slightly worse compression ratio (~1-5%) for massive speedup.
// ==============================================================================

// Greedy decision kernel: each thread handles one position
__global__ void
greedy_parse_kernel(const Match *matches, u32 input_size,
                    u32 *d_decisions, // Output: 0 = literal, >0 = match length
                    u32 *d_offsets_out, // Output: match offset (if match)
                    u32 min_match, u32 input_size_for_match) {

  u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= input_size)
    return;

  Match m = matches[pos];
  if (pos == 174) {
    printf("[GREEDY] Pos 174 Read Match: Len=%u Off=%u\n", m.length, m.offset);
  }

  // Greedy decision: use match if valid, else literal
  // CRITICAL: Ensure we don't pick up garbage matches from uninitialized tail
  if (m.length >= min_match && m.offset > 0 && pos < input_size_for_match) {
    d_decisions[pos] = m.length;
    d_offsets_out[pos] = m.offset;
  } else {
    d_decisions[pos] = 0; // Literal
    d_offsets_out[pos] = 0;
  }
}

// GPU Kernel: Build sequences directly on GPU (avoids 128KB D2H copy)
__global__ void build_sequences_gpu_kernel(const u32 *decisions,
                                           const u32 *offsets_in,
                                           u32 input_size, u32 *ll_out,
                                           u32 *ml_out, u32 *of_out,
                                           u32 *seq_count, u32 *has_dummy) {

  // Single thread builds sequences (sequential on GPU but no D2H transfer!)
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  u32 pos = 0;
  u32 literal_run = 0;
  u32 num_seqs = 0;

  while (pos < input_size) {
    u32 decision = decisions[pos];

    if (decision == 0) {
      literal_run++;
      pos++;
    } else {
      ll_out[num_seqs] = literal_run;
      ml_out[num_seqs] = decision;
      of_out[num_seqs] = offsets_in[pos];

      num_seqs++;
      literal_run = 0;
      pos += decision;
    }
  }

  // Handle trailing literals
  if (literal_run > 0 || num_seqs == 0) {
    // If no sequences were found, or there are leftover literals after the last
    // match, they are stored as a 'dummy' sequence for the literal copy kernel.
    ll_out[num_seqs] = literal_run;
    ml_out[num_seqs] = 0;
    of_out[num_seqs] = 0;
    // (FIX) Do NOT increment num_seqs for trailing literals.
    // They are not ZSTD sequences. They are handled by has_dummy.
    if (has_dummy)
      *has_dummy = 1; // Use 1 for true
  } else {
    if (has_dummy)
      *has_dummy = 0; // Use 0 for false
  }

  *seq_count = num_seqs;
}

// Build sequences from greedy decisions - GPU-only path
// Avoids 128KB D2H copy by doing sequence building on GPU
Status build_sequences_from_greedy(const u32 *d_decisions, const u32 *d_offsets,
                                   u32 input_size,
                                   CompressionWorkspace &workspace,
                                   u32 *h_num_sequences, u32 *out_has_dummy,
                                   cudaStream_t stream) {

  // Use workspace temp area for seq_count (already allocated)
  u32 *d_seq_count =
      (u32 *)((u8 *)workspace.d_lz77_temp + 2 * input_size * sizeof(u32));
  u32 *d_has_dummy_dev = (u32 *)(d_seq_count + 1);

  // Build sequences entirely on GPU - no large D2H transfer!
  build_sequences_gpu_kernel<<<1, 1, 0, stream>>>(
      d_decisions, d_offsets, input_size, workspace.d_literal_lengths_reverse,
      workspace.d_match_lengths_reverse, workspace.d_offsets_reverse,
      d_seq_count, d_has_dummy_dev);

  // Only copy 4 bytes (seq count) instead of 128KB!
  // (OPTIMIZATION) Use sync copy to ensure we have the count
  cudaMemcpy(h_num_sequences, d_seq_count, sizeof(u32), cudaMemcpyDeviceToHost);

  if (out_has_dummy) {
    cudaMemcpy(out_has_dummy, d_has_dummy_dev, sizeof(u32),
               cudaMemcpyDeviceToHost);
  }

  return Status::SUCCESS;
}

// ASYNC variant - writes to device memory, no sync
// Used for two-phase batched processing
Status build_sequences_from_greedy_async(
    const u32 *d_decisions, const u32 *d_offsets, u32 input_size,
    CompressionWorkspace &workspace,
    u32 *d_num_sequences, // Device pointer
    u32 *d_has_dummy,     // Device pointer (can be nullptr)
    cudaStream_t stream) {

  // Build sequences entirely on GPU - no sync!
  build_sequences_gpu_kernel<<<1, 1, 0, stream>>>(
      d_decisions, d_offsets, input_size, workspace.d_literal_lengths_reverse,
      workspace.d_match_lengths_reverse, workspace.d_offsets_reverse,
      d_num_sequences, d_has_dummy);

  // Note: has_dummy check is deferred to batch sync phase
  // The caller must handle this after batch sync

  return Status::SUCCESS;
}

// PARALLEL GREEDY PIPELINE
// Uses GPU for both match finding and greedy parsing decisions
Status lz77_parallel_greedy_pipeline(u32 input_size,
                                     CompressionWorkspace &workspace,
                                     const LZ77Config &config,
                                     u32 *h_num_sequences, u32 *out_has_dummy,
                                     cudaStream_t stream) {

  // OPTIMIZATION: Reuse workspace.d_lz77_temp instead of per-call cudaMalloc
  // Partition: [decisions: input_size*4 bytes][offsets: input_size*4 bytes]
  u32 *d_decisions = (u32 *)workspace.d_lz77_temp;
  u32 *d_offsets_out = d_decisions + input_size;

  // (OPTIMIZATION REMOVED) No sync needed here - kernel launch serializes on
  // stream

  // Launch greedy parse kernel - fully parallel!
  const u32 block_size = 256;
  const u32 num_blocks = (input_size + block_size - 1) / block_size;

  greedy_parse_kernel<<<num_blocks, block_size, 0, stream>>>(
      (const Match *)workspace.d_matches, input_size, d_decisions,
      d_offsets_out, config.min_match, input_size);

  // Build sequences from greedy decisions
  Status status = build_sequences_from_greedy(
      d_decisions, d_offsets_out, input_size, workspace, h_num_sequences,
      out_has_dummy, stream);

  // No cudaFree needed - workspace memory is managed by caller

  return status;
}

// ASYNC PARALLEL GREEDY PIPELINE (for two-phase batched processing)
// Writes to device memory, no sync - caller batches syncs
Status lz77_parallel_greedy_pipeline_async(
    u32 input_size, CompressionWorkspace &workspace, const LZ77Config &config,
    u32 *d_num_sequences, // Device pointer
    u32 *d_has_dummy,     // Device pointer
    cudaStream_t stream) {

  // OPTIMIZATION: Reuse workspace.d_lz77_temp instead of per-call cudaMalloc
  u32 *d_decisions = (u32 *)workspace.d_lz77_temp;
  u32 *d_offsets_out = d_decisions + input_size;

  // Launch greedy parse kernel - fully parallel!
  const u32 block_size = 256;
  const u32 num_blocks = (input_size + block_size - 1) / block_size;

  greedy_parse_kernel<<<num_blocks, block_size, 0, stream>>>(
      (const Match *)workspace.d_matches, input_size, d_decisions,
      d_offsets_out, config.min_match, input_size);

  // Build sequences - ASYNC version, no sync
  Status status = build_sequences_from_greedy_async(
      d_decisions, d_offsets_out, input_size, workspace, d_num_sequences,
      d_has_dummy, stream);

  return status;
}

// ==============================================================================
// UNIFIED CPU LZ77 PIPELINE
// Combines forward DP + backtracking with shared host buffers to eliminate
// redundant D2H copies. For small chunks, this is significantly faster than
// separate functions that each copy data.
// ==============================================================================
Status lz77_cpu_pipeline(u32 input_size, CompressionWorkspace &workspace,
                         const LZ77Config &config, u32 *h_num_sequences,
                         u32 *out_has_dummy, cudaStream_t stream) {

  try {
    // 1. Sync stream to ensure matches from GPU are ready
    cudaStreamSynchronize(stream);

    // 2. Copy matches D2H - ONLY ONCE for both forward DP and backtracking
    std::vector<Match> h_matches(input_size);
    cudaError_t err =
        cudaMemcpy(h_matches.data(), workspace.d_matches,
                   input_size * sizeof(Match), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      return Status::ERROR_CUDA_ERROR;
    }

    // 3. Forward DP on CPU - compute optimal parse costs
    std::vector<ParseCost> h_costs(input_size + 1);
    h_costs[0].set(0, 0);
    for (u32 i = 1; i <= input_size; ++i) {
      h_costs[i].data = 0xFFFFFFFFFFFFFFFFULL;
    }

    for (u32 pos = 0; pos < input_size; ++pos) {
      u32 current_cost = h_costs[pos].cost();
      if (current_cost == 0xFFFFFFFF)
        continue;

      // Literal transition
      u32 lit_dest = pos + 1;
      u32 lit_cost_val = current_cost + calculate_literal_cost(1);
      if (lit_cost_val < h_costs[lit_dest].cost()) {
        h_costs[lit_dest].set(lit_cost_val, pos);
      }

      // Match transition
      const Match &m = h_matches[pos];
      if (m.length >= config.min_match) {
        u32 match_dest = pos + m.length;
        if (match_dest <= input_size) {
          u32 match_cost_val =
              current_cost + calculate_match_cost(m.length, m.offset);
          if (match_cost_val < h_costs[match_dest].cost()) {
            h_costs[match_dest].set(match_cost_val, pos);
          }
        }
      }
    }

    // 4. Backtracking on CPU - using already-computed h_costs and h_matches
    // NO additional D2H copies needed!
    std::vector<u32> ll_buf, ml_buf, of_buf;
    ll_buf.reserve(input_size / 3);
    ml_buf.reserve(input_size / 3);
    of_buf.reserve(input_size / 3);

    u32 curr_pos = input_size;
    u32 literal_run = 0;
    u32 pending_ml = 0;
    u32 pending_of = 0;

    while (curr_pos > 0) {
      u32 parent = h_costs[curr_pos].parent();
      if (parent >= curr_pos)
        parent = curr_pos - 1;

      u32 len = curr_pos - parent;
      bool is_valid_match = (len >= 3);
      if (is_valid_match) {
        u32 match_offset = h_matches[parent].offset;
        if (match_offset == 0) {
          is_valid_match = false;
        }
      }

      if (is_valid_match) {
        ll_buf.push_back(literal_run);
        ml_buf.push_back(pending_ml);
        of_buf.push_back(pending_of);
        pending_ml = len;
        pending_of = h_matches[parent].offset;
        literal_run = 0;
      } else {
        literal_run += len;
      }
      curr_pos = parent;
    }

    ll_buf.push_back(literal_run);
    ml_buf.push_back(pending_ml);
    of_buf.push_back(pending_of);

    std::reverse(ll_buf.begin(), ll_buf.end());
    std::reverse(ml_buf.begin(), ml_buf.end());
    std::reverse(of_buf.begin(), of_buf.end());

    if (out_has_dummy)
      *out_has_dummy = (!ml_buf.empty() && ml_buf.back() == 0);

    u32 num_sequences = (u32)ll_buf.size();
    *h_num_sequences = num_sequences;

    // 5. Copy sequences H2D - ONLY output copy needed
    if (num_sequences > 0) {
      if (num_sequences > workspace.max_sequences) {
        return Status::ERROR_BUFFER_TOO_SMALL;
      }
      cudaMemcpyAsync(workspace.d_literal_lengths_reverse, ll_buf.data(),
                      num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(workspace.d_match_lengths_reverse, ml_buf.data(),
                      num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(workspace.d_offsets_reverse, of_buf.data(),
                      num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                      stream);
    }

    return Status::SUCCESS;
  } catch (const std::bad_alloc &) {
    return Status::ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return Status::ERROR_UNKNOWN;
  }
}

// Legacy wrapper for compatibility - now uses unified pipeline
Status compute_optimal_parse_v2(const u8 *d_input, u32 input_size,
                                CompressionWorkspace *workspace,
                                const LZ77Config &config, cudaStream_t stream) {
  // Forward DP is now done inside lz77_cpu_pipeline
  // This function is kept for API compatibility but does nothing
  // The caller should use lz77_cpu_pipeline instead
  (void)d_input;
  (void)input_size;
  (void)workspace;
  (void)config;
  (void)stream;
  return Status::SUCCESS;
}

// ==============================================================================
// Parallel Backtracking Infrastructure (Phase 2)
// ==============================================================================

BacktrackConfig create_backtrack_config(u32 input_size) {
  BacktrackConfig config;
  // Use CPU backtracking for small inputs (<= 128KB) to avoid per-call
  // cudaMalloc overhead GPU parallel is only beneficial for large inputs where
  // kernel launch overhead is amortized
  config.parallel_threshold = 128 * 1024; // 128KB threshold

  if (input_size >= config.parallel_threshold) {
    config.use_parallel = true;
    // Use larger segments for huge inputs, smaller for medium
    if (input_size > 10 * 1024 * 1024) {
      config.segment_size = 1024 * 1024; // 1MB segments for >10MB
    } else {
      config.segment_size = 256 * 1024; // 256KB segments for 1-10MB
    }
    config.num_segments =
        (input_size + config.segment_size - 1) / config.segment_size;
  } else {
    config.use_parallel = false;
    config.segment_size = input_size;
    config.num_segments = 1;
  }

  return config;
}

// GPU Kernel: Each block processes one segment
__global__ void backtrack_segments_parallel_kernel(
    const ParseCost *d_costs, const Match *d_matches, u32 input_size,
    SegmentInfo *d_segments, u32 *d_literal_lengths, u32 *d_match_lengths,
    u32 *d_offsets, u32 *d_segment_seq_counts, u32 num_segments) {

  u32 seg_idx = blockIdx.x;
  if (seg_idx >= num_segments)
    return;

  // Only thread 0 in each block does the work (sequential within segment)
  if (threadIdx.x != 0)
    return;

  SegmentInfo seg = d_segments[seg_idx];
  u32 curr_pos = seg.end_pos;
  u32 local_seq_count = 0;

  // Temporary local storage (max sequences per segment)
  // We write directly to global arrays at segment offset
  u32 write_base = seg.sequence_offset;
  u32 literal_run = 0;
  u32 pending_ml = 0, pending_of = 0;

  // Backtrack within this segment
  while (curr_pos > seg.start_pos) {
    u32 parent = d_costs[curr_pos].parent();

    // Clamp parent to valid range
    if (parent >= curr_pos)
      parent = curr_pos - 1;
    if (parent < seg.start_pos)
      parent = seg.start_pos;

    u32 len = curr_pos - parent;

    // Check for valid match
    bool is_valid_match = false;
    if (len >= 3 && parent < input_size) {
      u32 match_offset = d_matches[parent].offset;
      if (match_offset > 0) {
        is_valid_match = true;
      }
    }

    if (is_valid_match) {
      // Push pending sequence
      d_literal_lengths[write_base + local_seq_count] = literal_run;
      d_match_lengths[write_base + local_seq_count] = pending_ml;
      d_offsets[write_base + local_seq_count] = pending_of;
      local_seq_count++;

      // New pending
      pending_ml = len;
      pending_of = d_matches[parent].offset;
      literal_run = 0;
    } else {
      literal_run += len;
    }

    curr_pos = parent;
  }

  // Push final sequence for this segment
  d_literal_lengths[write_base + local_seq_count] = literal_run;
  d_match_lengths[write_base + local_seq_count] = pending_ml;
  d_offsets[write_base + local_seq_count] = pending_of;
  local_seq_count++;

  // Store count for this segment
  d_segment_seq_counts[seg_idx] = local_seq_count;
}

Status backtrack_sequences_parallel(const ParseCost *d_costs, u32 input_size,
                                    CompressionWorkspace &workspace,
                                    u32 *h_num_sequences, cudaStream_t stream) {

  BacktrackConfig config = create_backtrack_config(input_size);

  // Allocate segment info on device
  SegmentInfo *d_segments = nullptr;
  u32 *d_segment_seq_counts = nullptr;
  cudaMalloc(&d_segments, config.num_segments * sizeof(SegmentInfo));
  cudaMalloc(&d_segment_seq_counts, config.num_segments * sizeof(u32));

  // Prepare segments on host
  std::vector<SegmentInfo> h_segments(config.num_segments);
  u32 max_seqs_per_segment = config.segment_size / 3 + 1; // Rough estimate

  for (u32 i = 0; i < config.num_segments; ++i) {
    h_segments[i].start_pos = i * config.segment_size;
    h_segments[i].end_pos = std::min((i + 1) * config.segment_size, input_size);
    h_segments[i].sequence_offset = i * max_seqs_per_segment;
    h_segments[i].num_sequences = 0;
  }

  // Copy segments to device
  cudaMemcpyAsync(d_segments, h_segments.data(),
                  config.num_segments * sizeof(SegmentInfo),
                  cudaMemcpyHostToDevice, stream);

  // Launch kernel: 1 block per segment, 1 thread per block
  backtrack_segments_parallel_kernel<<<config.num_segments, 1, 0, stream>>>(
      d_costs, (const Match *)workspace.d_matches, input_size, d_segments,
      workspace.d_literal_lengths_reverse, workspace.d_match_lengths_reverse,
      workspace.d_offsets_reverse, d_segment_seq_counts, config.num_segments);

  // Get segment counts back
  std::vector<u32> h_seg_counts(config.num_segments);
  cudaMemcpyAsync(h_seg_counts.data(), d_segment_seq_counts,
                  config.num_segments * sizeof(u32), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  // Calculate total sequences and compact on host
  u32 total_seqs = 0;
  for (u32 i = 0; i < config.num_segments; ++i) {
    total_seqs += h_seg_counts[i];
  }

  // Merge segments: copy from scattered positions to contiguous
  // For simplicity, do this on CPU after copying back
  std::vector<u32> h_ll_scattered(config.num_segments * max_seqs_per_segment);
  std::vector<u32> h_ml_scattered(config.num_segments * max_seqs_per_segment);
  std::vector<u32> h_of_scattered(config.num_segments * max_seqs_per_segment);

  cudaMemcpy(h_ll_scattered.data(), workspace.d_literal_lengths_reverse,
             h_ll_scattered.size() * sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ml_scattered.data(), workspace.d_match_lengths_reverse,
             h_ml_scattered.size() * sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_of_scattered.data(), workspace.d_offsets_reverse,
             h_of_scattered.size() * sizeof(u32), cudaMemcpyDeviceToHost);

  // Compact and reverse (segments are in reverse order, sequences within are
  // reversed)
  std::vector<u32> h_ll_final, h_ml_final, h_of_final;
  h_ll_final.reserve(total_seqs);
  h_ml_final.reserve(total_seqs);
  h_of_final.reserve(total_seqs);

  // Process segments in reverse order (last segment first in file)
  for (int seg = config.num_segments - 1; seg >= 0; --seg) {
    u32 base = seg * max_seqs_per_segment;
    u32 count = h_seg_counts[seg];

    // Sequences within segment are already in reverse order, so reverse again
    for (int i = count - 1; i >= 0; --i) {
      h_ll_final.push_back(h_ll_scattered[base + i]);
      h_ml_final.push_back(h_ml_scattered[base + i]);
      h_of_final.push_back(h_of_scattered[base + i]);
    }
  }

  // Copy final contiguous results back to device
  *h_num_sequences = (u32)h_ll_final.size();

  if (*h_num_sequences > 0) {
    cudaMemcpyAsync(workspace.d_literal_lengths_reverse, h_ll_final.data(),
                    *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(workspace.d_match_lengths_reverse, h_ml_final.data(),
                    *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(workspace.d_offsets_reverse, h_of_final.data(),
                    *h_num_sequences * sizeof(u32), cudaMemcpyHostToDevice,
                    stream);
  }

  cudaFree(d_segments);
  cudaFree(d_segment_seq_counts);

  return Status::SUCCESS;
}

// Sequential CPU backtracking (for small inputs or fallback)
Status backtrack_sequences_cpu(const ParseCost *h_costs, u32 input_size,
                               u32 *h_literal_lengths, u32 *h_match_lengths,
                               u32 *h_offsets, u32 *h_num_sequences) {
  // This is essentially the logic from backtrack_sequences_v2 but operating on
  // host arrays directly
  u32 curr_pos = input_size;
  u32 seq_idx = 0;
  u32 literal_run = 0;
  u32 pending_ml = 0, pending_of = 0;

  while (curr_pos > 0) {
    u32 parent = h_costs[curr_pos].parent();
    if (parent >= curr_pos)
      parent = curr_pos - 1;

    u32 len = curr_pos - parent;

    if (len >= 3) {
      h_literal_lengths[seq_idx] = literal_run;
      h_match_lengths[seq_idx] = pending_ml;
      h_offsets[seq_idx] = pending_of;
      seq_idx++;
      pending_ml = len;
      pending_of = 0; // Would need match info
      literal_run = 0;
    } else {
      literal_run += len;
    }
    curr_pos = parent;
  }

  h_literal_lengths[seq_idx] = literal_run;
  h_match_lengths[seq_idx] = pending_ml;
  h_offsets[seq_idx] = pending_of;
  *h_num_sequences = seq_idx + 1;

  return Status::SUCCESS;
}

// V2 Backtracking Implementation (Host-side)
Status backtrack_sequences_v2(u32 input_size, CompressionWorkspace &workspace,
                              u32 *h_num_sequences, u32 *out_has_dummy,
                              cudaStream_t stream) {
  // 1. Allocate host buffers for backtracking
  // We need costs and matches on host to trace the path
  // 2. Backtrack from end
  try {
    // 1. Allocate host buffers for backtracking
    // We need costs and matches on host to trace the path
    std::vector<ParseCost> h_costs(input_size + 1);
    std::vector<Match> h_matches(input_size);

    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_costs.data(), workspace.d_costs,
                                      (input_size + 1) * sizeof(ParseCost),
                                      cudaMemcpyDeviceToHost, stream),
                      Status::ERROR_CUDA_ERROR);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_matches.data(), workspace.d_matches,
                                      input_size * sizeof(Match),
                                      cudaMemcpyDeviceToHost, stream),
                      Status::ERROR_CUDA_ERROR);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), Status::ERROR_CUDA_ERROR);

    std::vector<u32> ll_buf;
    std::vector<u32> ml_buf;
    std::vector<u32> of_buf;

    ll_buf.reserve(input_size / 3);
    ml_buf.reserve(input_size / 3);
    of_buf.reserve(input_size / 3);

    u32 curr_pos = input_size;
    u32 literal_run = 0;

    // Pending match initialization
    u32 pending_ml = 0;
    u32 pending_of = 0;

    while (curr_pos > 0) {
      u32 parent = h_costs[curr_pos].parent();
      if (parent >= curr_pos)
        parent = curr_pos - 1;

      u32 len = curr_pos - parent;

      // (FIX) Check if this is a valid match (len >= min_match AND offset > 0)
      // Matches with offset=0 indicate "no match found" during LZ77 matching
      bool is_valid_match = (len >= 3); // len >= min_match
      if (is_valid_match) {
        u32 match_offset = h_matches[parent].offset;
        if (match_offset == 0) {
          // Invalid match - offset=0 means no match was found
          // Treat this as literals instead
          is_valid_match = false;
        }
      }

      if (is_valid_match) { // Valid match
        // Push pending sequence (which is the Next Match in file order)
        // using accumulated literals as its LL
        ll_buf.push_back(literal_run);
        ml_buf.push_back(pending_ml);
        of_buf.push_back(pending_of);

        // Setup this match as new pending
        pending_ml = len;
        pending_of = h_matches[parent].offset;

        literal_run = 0;
      } else {
        // Literal (or invalid match treated as literal)
        literal_run += len;
      }
      curr_pos = parent;
    }

    // Push the final sequence (First in file)
    ll_buf.push_back(literal_run);
    ml_buf.push_back(pending_ml);
    of_buf.push_back(pending_of);

    // Reverse vectors to match Forward Order (S_0 ... S_N)
    std::reverse(ll_buf.begin(), ll_buf.end());
    std::reverse(ml_buf.begin(), ml_buf.end());
    std::reverse(of_buf.begin(), of_buf.end());

    // Check for dummy trailing sequence (ML=0)
    // We do NOT pop it, because launch_copy_literals needs it to copy the
    // trailing literals. Instead, we inform the caller.
    if (out_has_dummy)
      *out_has_dummy = 0;
    if (!ml_buf.empty()) {
      if (ml_buf.back() == 0) {
        if (out_has_dummy)
          *out_has_dummy = 1;
      }
    }

    u32 num_sequences = (u32)ll_buf.size();
    *h_num_sequences = num_sequences;

    if (num_sequences > 0) {
      if (num_sequences > workspace.max_sequences) {
        return Status::ERROR_BUFFER_TOO_SMALL;
      }

      CUDA_CHECK_RETURN(cudaMemcpyAsync(workspace.d_literal_lengths_reverse,
                                        ll_buf.data(),
                                        num_sequences * sizeof(u32),
                                        cudaMemcpyHostToDevice, stream),
                        Status::ERROR_CUDA_ERROR);
      CUDA_CHECK_RETURN(cudaMemcpyAsync(workspace.d_match_lengths_reverse,
                                        ml_buf.data(),
                                        num_sequences * sizeof(u32),
                                        cudaMemcpyHostToDevice, stream),
                        Status::ERROR_CUDA_ERROR);
      CUDA_CHECK_RETURN(cudaMemcpyAsync(workspace.d_offsets_reverse,
                                        of_buf.data(),
                                        num_sequences * sizeof(u32),
                                        cudaMemcpyHostToDevice, stream),
                        Status::ERROR_CUDA_ERROR);
    }
  } catch (const std::bad_alloc &) {
    return Status::ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return Status::ERROR_UNKNOWN;
  }

  return Status::SUCCESS;
}

Status backtrack_sequences(u32 input_size, CompressionWorkspace &workspace,
                           u32 *h_num_sequences, u32 *out_has_dummy,
                           cudaStream_t stream) {
  // Adaptive routing: use parallel GPU for large inputs, CPU for small
  BacktrackConfig config = create_backtrack_config(input_size);

  if (config.use_parallel) {
    // GPU parallel backtracking for inputs >= 1MB
    Status status = backtrack_sequences_parallel(
        (const ParseCost *)workspace.d_costs, input_size, workspace,
        h_num_sequences, stream);

    if (status != Status::SUCCESS) {
      // Fallback to CPU on GPU failure
      return backtrack_sequences_v2(input_size, workspace, h_num_sequences,
                                    out_has_dummy, stream);
    }

    // Set dummy flag based on last sequence
    if (out_has_dummy)
      *out_has_dummy = false;


    return status;
  } else {
    // CPU backtracking for small inputs (< 1MB)
    return backtrack_sequences_v2(input_size, workspace, h_num_sequences,
                                  out_has_dummy, stream);
  }
}

Status find_matches_parallel(const u8 *d_input, u32 input_size,
                             CompressionWorkspace *workspace,
                             const LZ77Config &config, cudaStream_t stream) {
  // 1. Initialize Hash Table
  u32 hash_size = 1 << config.hash_log;
  u32 chain_size = 1 << config.chain_log;

  // Use workspace sizes if available to safeguard?
  // workspace->hash_table_size is u32
  if (workspace->hash_table_size < hash_size)
    return Status::ERROR_WORKSPACE_INVALID;
  if (workspace->chain_table_size < chain_size)
    return Status::ERROR_WORKSPACE_INVALID;

  u32 block_size = 256;
  u32 max_size = (hash_size > chain_size) ? hash_size : chain_size;
  u32 grid_size = (max_size + block_size - 1) / block_size;
  // Cap grid size to avoid launching too many blocks for huge tables?
  if (grid_size > 256)
    grid_size = 256;
  // Wait, init needs to cover ALL entries. The kernel uses stride loop.
  // Step 300: `u32 stride = blockDim.x * gridDim.x; for (u32 i = idx; i <
  // hash_size; i += stride)` So any grid size is correct. 256 blocks * 256
  // threads = 65536 threads.

  init_hash_table_kernel<<<grid_size, block_size, 0, stream>>>(
      workspace->d_hash_table, workspace->d_chain_table, hash_size, chain_size);

  // 2. Find Matches
  u32 match_block_size = 128; // Tuning?
  u32 match_grid_size = (input_size + match_block_size - 1) / match_block_size;

  find_matches_kernel<<<match_grid_size, match_block_size, 0, stream>>>(
      d_input, input_size, workspace->d_hash_table, workspace->d_chain_table,
      (Match *)workspace->d_matches, // Cast void* to Match*
      config, config.hash_log);

  return Status::SUCCESS;
}

} // namespace lz77
} // namespace cuda_zstd
