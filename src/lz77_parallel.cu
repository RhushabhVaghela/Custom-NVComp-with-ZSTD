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

Status compute_optimal_parse_v2(const u8 *d_input, u32 input_size,
                                CompressionWorkspace *workspace,
                                const LZ77Config &config, cudaStream_t stream) {
  // 1. Init costs (done in kernel)
  // 2. Launch Match finder? Caller (manager) does it?
  // Usage implies compute_optimal_parse_v2 does the optimal parse PASS.
  // Matches are already in workspace->d_matches from previous step
  // (find_matches_parallel).

  // Launch Forward DP kernel (Serial)
  compute_costs_kernel_forward<<<1, 1, 0, stream>>>(
      d_input, input_size, (const Match *)workspace->d_matches,
      (ParseCost *)workspace->d_costs, config);

  return Status::SUCCESS;
}

// V2 Backtracking Implementation (Host-side)
Status backtrack_sequences_v2(u32 input_size, CompressionWorkspace &workspace,
                              u32 *h_num_sequences, bool *out_has_dummy,
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
      *out_has_dummy = false;
    if (!ml_buf.empty()) {
      if (ml_buf.back() == 0) {
        if (out_has_dummy)
          *out_has_dummy = true;
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
                           u32 *h_num_sequences, bool *out_has_dummy,
                           cudaStream_t stream) {
  // V2: Always use V2 backtracking (CPU for now, but uses new ParseCost format)
  return backtrack_sequences_v2(input_size, workspace, h_num_sequences,
                                out_has_dummy, stream);
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
