// ==============================================================================
// lz77_parallel.h - Three-pass parallel LZ77 compression
// ==============================================================================

#ifndef LZ77_PARALLEL_H
#define LZ77_PARALLEL_H

#include "common_types.h"
#include "error_context.h"
#include "lz77_types.h"
#include "workspace_manager.h"
#include <cuda_runtime.h>

namespace cuda_zstd {
namespace lz77 {

// calculate_match_cost, calculate_literal_cost removed (definitions in
// cuda_zstd_lz77.h)

__device__ inline u32 compute_hash(const u8 *data, u32 pos, u32 hash_log) {
  u32 val = (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2];
  return (val * 2654435761U) >> (32 - hash_log);
}

__device__ inline u32 match_length(const u8 *input, u32 p1, u32 p2, u32 max_len,
                                   u32 input_size) {
  u32 len = 0;
  while (len < max_len && p1 + len < input_size && p2 + len < input_size) {
    if (input[p1 + len] != input[p2 + len])
      break;
    len++;
  }
  if (p1 == 174 && p2 == 104) {
    printf("[MATCH_LENGTH] p1=174 Val=%02X p2=104 Val=%02X Len=%u\n", input[p1],
           input[p2], len);
  }
  return len;
}

Status find_matches_parallel(const u8 *d_input, u32 input_size,
                             CompressionWorkspace *workspace,
                             const LZ77Config &config, cudaStream_t stream);

Status compute_optimal_parse();

// V2: Optimized multi-pass algorithm (10-100x faster!)
Status compute_optimal_parse_v2(const u8 *d_input, u32 input_size,
                                CompressionWorkspace *workspace,
                                const LZ77Config &config, cudaStream_t stream);

// UNIFIED CPU LZ77 PIPELINE
// Combines forward DP + backtracking with shared host buffers to eliminate
// redundant D2H copies. Use this for streaming/small chunks.
Status lz77_cpu_pipeline(u32 input_size, CompressionWorkspace &workspace,
                         const LZ77Config &config, u32 *h_num_sequences,
                         bool *out_has_dummy, cudaStream_t stream);

// PARALLEL GREEDY PIPELINE (fastest - trades compression ratio for speed)
Status lz77_parallel_greedy_pipeline(u32 input_size,
                                     CompressionWorkspace &workspace,
                                     const LZ77Config &config,
                                     u32 *h_num_sequences, bool *out_has_dummy,
                                     cudaStream_t stream);

// ASYNC PARALLEL GREEDY PIPELINE (for two-phase batched processing)
// Writes seq count to DEVICE memory - no sync needed per block
// Call lz77_read_seq_counts_batch() to read all counts after batch sync
Status lz77_parallel_greedy_pipeline_async(
    u32 input_size, CompressionWorkspace &workspace, const LZ77Config &config,
    u32 *d_num_sequences, // Device pointer
    bool *d_has_dummy,    // Device pointer
    cudaStream_t stream);

// ==============================================================================
// Parallel Backtracking Infrastructure (Phase 2)
// ==============================================================================

// Segment information for parallel backtracking
struct SegmentInfo {
  u32 start_pos;       // Starting position of this segment
  u32 end_pos;         // Ending position of this segment
  u32 num_sequences;   // Number of sequences found in this segment
  u32 sequence_offset; // Offset into the global sequence buffer
};

// Configuration for backtracking mode selection
struct BacktrackConfig {
  u32 segment_size;       // Size of each segment (e.g., 1MB)
  u32 num_segments;       // Number of segments
  bool use_parallel;      // True = GPU parallel, False = CPU sequential
  u32 parallel_threshold; // Input size threshold for using parallel (default:
                          // 1MB)
};

// Create default backtracking configuration based on input size
BacktrackConfig create_backtrack_config(u32 input_size);

// Parallel GPU backtracking (for large inputs)
Status backtrack_sequences_parallel(const ParseCost *d_costs, u32 input_size,
                                    CompressionWorkspace &workspace,
                                    u32 *h_num_sequences, cudaStream_t stream);

// Sequential CPU backtracking (for small inputs or fallback)
Status backtrack_sequences_cpu(const ParseCost *h_costs, u32 input_size,
                               u32 *h_literal_lengths, u32 *h_match_lengths,
                               u32 *h_offsets, u32 *h_num_sequences);

// Adaptive backtracking (chooses parallel or sequential based on input size)
Status backtrack_sequences(u32 input_size, CompressionWorkspace &workspace,
                           u32 *h_num_sequences, bool *out_has_dummy,
                           cudaStream_t stream);

} // namespace lz77
} // namespace cuda_zstd

#endif // LZ77_PARALLEL_H
