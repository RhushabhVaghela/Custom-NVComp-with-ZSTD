// ============================================================================
// cuda_zstd_lz77.h - LZ77 Match Finding Interface
//
// (UPDATED) Includes structures and signatures for the three-pass
// parallel compression parser (Match Finding, DP Cost, Backtrack).
// (NEW) Adds streaming support via optional window buffer parameters.
// ============================================================================

#ifndef CUDA_ZSTD_LZ77_H_
#define CUDA_ZSTD_LZ77_H_

// #define CUDA_ZSTD_DEBUG_BOUNDS 1

#ifdef _MSC_VER
#include <intrin.h> // For _BitScanReverse
#endif

#include "cuda_zstd_fse.h"
#include "cuda_zstd_hash.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h"
#include <cstddef>

namespace cuda_zstd {
namespace lz77 {

// ============================================================================
// Match Structure
// ============================================================================

struct Match {
  u32 position;       // Position in input
  u32 offset;         // Match offset (distance back)
  u32 length;         // Match length
  u32 literal_length; // Literals before this match

  __host__ __device__ Match()
      : position(0), offset(0), length(0), literal_length(0) {}

  __host__ __device__ Match(u32 pos, u32 off, u32 len, u32 lit_len)
      : position(pos), offset(off), length(len), literal_length(lit_len) {}
};

// ============================================================================
// LZ77 Configuration
// ============================================================================

struct LZ77Config {
  Strategy strategy = Strategy::GREEDY;
  u32 window_log = 17;   // 128 KB window
  u32 hash_log = 20;     // Hash table size log2
  u32 chain_log = 16;    // Chain table size log2
  u32 search_depth = 8;  // Search depth for lazy matching
  u32 min_match = 3;     // Minimum match length
  u32 good_length = 32;  // Length that triggers greedy mode
  u32 nice_length = 128; // Nice match length
};

// ============================================================================
// LZ77 Statistics
// ============================================================================

struct LZ77Stats {
  u32 num_matches = 0;
  u32 num_literals = 0;
  u32 total_match_length = 0;
  u32 total_literal_length = 0;
  double avg_match_length = 0.0;
  double compression_ratio = 0.0;
};

// ============================================================================
// Optimal Parser Structures
// ============================================================================

/**
 * @brief Stores the "cheapest" cost to get to this position (for DP).
 * Uses u64 to allow atomic updates of (cost, parent_pos).
 * High 32 bits: cost
 * Low 32 bits: parent_pos
 */
struct ParseCost {
  unsigned long long data;

  __host__ __device__ ParseCost() : data(0xFFFFFFFFFFFFFFFFULL) {}

  __host__ __device__ void set(u32 cost, u32 parent_pos) {
    data = ((unsigned long long)cost << 32) | parent_pos;
  }

  __host__ __device__ u32 cost() const { return (u32)(data >> 32); }

  __host__ __device__ u32 parent() const { return (u32)(data & 0xFFFFFFFF); }
};

// V2: Simplified multi-pass kernel
__global__ void optimal_parse_kernel_v2(u32 input_size, const Match *d_matches,
                                        ParseCost *d_costs);

// ============================================================================
// LZ77 Context (Host-Side)
// ============================================================================

struct LZ77Context {
  // (MODIFIED) Hash/chain tables now point into workspace
  hash::HashTable hash_table;
  hash::ChainTable chain_table;
  LZ77Config config;
  LZ77Stats stats;

  // (MODIFIED) Device buffers now point into workspace - NOT allocated here
  Match *d_matches = nullptr;
  ParseCost *d_costs = nullptr;

  // Host buffers for dictionary training (still owned by context)
  Match *h_matches_dict_trainer = nullptr;
  u32 h_num_matches_dict_trainer = 0;
  u32 max_matches_dict_trainer = 0;

  // NEW: Window buffer with validation
  unsigned char *d_window = nullptr; // Optional, may be nullptr
  size_t window_size = 0;
  bool window_enabled = false; // Explicit flag

  // (MODIFIED) Reverse buffers now point into workspace - NOT allocated here
  u32 *d_literal_lengths_reverse = nullptr;
  u32 *d_match_lengths_reverse = nullptr;
  u32 *d_offsets_reverse = nullptr;
  u32 max_sequences_capacity = 0;

  // Safety check
  bool is_window_valid() const {
    return d_window != nullptr && window_size > 0 && window_enabled;
  }
};

// ============================================================================
// LZ77 Operations (Host Interface - UPDATED SIGNATURES)
// ============================================================================

// Initialize LZ77 context
Status init_lz77_context(LZ77Context &ctx, const LZ77Config &config,
                         size_t max_input_size);

// Free LZ77 context
Status free_lz77_context(LZ77Context &ctx);

/**
 * @brief (MODIFIED) Pass 1: Finds all best matches in parallel.
 * Accepts an optional window buffer for streaming and a workspace for temp
 * buffers.
 */
Status find_matches(
    LZ77Context &ctx, const unsigned char *d_input, size_t input_size,
    const DictionaryContent *dict,
    CompressionWorkspace *workspace,  // (NEW) Workspace for temp allocations
    const unsigned char *d_window = nullptr, // History buffer
    size_t window_size = 0,           // Size of history
    cudaStream_t stream = 0);

// Get match results (used by dictionary trainer)
Status get_matches(const LZ77Context &ctx, Match *matches, u32 *num_matches);

/**
 * @brief (MODIFIED) Pass 2 & 3: Runs DP and Backtrack.
 * Accepts workspace and optional window buffer.
 */
Status find_optimal_parse(
    LZ77Context &ctx, const unsigned char *d_input, size_t input_size,
    const DictionaryContent *dict, CompressionWorkspace *workspace,
    const unsigned char *d_window = nullptr, size_t window_size = 0,
    cudaStream_t stream = 0,
    cuda_zstd::sequence::SequenceContext *seq_ctx = nullptr,
    u32 *num_sequences_out = nullptr, size_t *literals_size_out = nullptr,
    bool is_last_block = false,
    u32 chunk_size = 131072, // Default 128KB
    u32 *total_literals_out = nullptr, bool output_raw_values = false);

void test_linkage();

// ============================================================================
// LZ77 Device Functions
// ============================================================================

// Find best match at current position (for parallel kernel)
__device__ inline Match
find_best_match_parallel(const unsigned char *input, u32 current_pos, u32 input_size,
                         const DictionaryContent *dict,
                         const hash::HashTable &hash_table,
                         const hash::ChainTable &chain_table,
                         const LZ77Config &config, u32 window_min);

// Compare bytes at two positions (handles window and dictionary lookups)
__device__ inline u32 match_length(const unsigned char *input,
                                   u32 p1, // Position in input
                                   u32 p2, // Position in input
                                   u32 max_len, const DictionaryContent *dict);

// Calculate match cost in bits
__device__ __host__ inline u32 calculate_match_cost(u32 length, u32 offset) {
  if (length == 0)
    return 1000000;
  // RFC 8878: offset bits = offset code
  u32 offset_bits =
      (offset == 0) ? 0 : (31 - cuda_zstd::utils::clz_impl(offset + 3));
  u32 length_bits = (length >= 19) ? 9 : 5; // Simplified ML cost
  return 1 + length_bits + offset_bits;
}

// Calculate literal cost in bits
__device__ __host__ inline u32 calculate_literal_cost(u32 num_literals) {
  return num_literals * 8;
}

} // namespace lz77
} // namespace cuda_zstd

#endif // CUDA_ZSTD_LZ77_H
