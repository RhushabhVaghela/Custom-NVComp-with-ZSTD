// ============================================================================
// cuda_zstd_sequence.h - Sequence Encoding/Decoding Interface
//
// (UPDATED) Header now reflects the multi-pass, parallel implementation
// found in cuda_zstd_sequence.cu.
// ============================================================================

#ifndef CUDA_ZSTD_SEQUENCE_H_
#define CUDA_ZSTD_SEQUENCE_H_

// #include "cuda_zstd_internal.h"

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>

namespace cuda_zstd {
namespace sequence {

// ============================================================================
// Sequence Structure
// ============================================================================

struct SequenceStats {
  u32 num_sequences;
  u32 num_literals;
  u32 total_match_length;
  float avg_literal_length;
  float avg_match_length;
  float avg_offset;
  u32 rep_1_count;
  u32 rep_2_count;
  u32 rep_3_count;
  SequenceStats()
      : num_sequences(0), num_literals(0), total_match_length(0),
        avg_literal_length(0), avg_match_length(0), avg_offset(0),
        rep_1_count(0), rep_2_count(0), rep_3_count(0) {}
};

struct __align__(16) Sequence {
  u32 literal_length; // Number of literals before match
  u32 match_length;   // Length of match (actual, not encoded)
  u32 match_offset;   // Offset to match (distance)
  u32 padding;        // (FIX) Align to 16 bytes to prevent CUDA stride issues

  __device__ __host__ Sequence()
      : literal_length(0), match_length(0), match_offset(0), padding(0) {}

  __device__ __host__ Sequence(u32 ll, u32 ml, u32 mo)
      : literal_length(ll), match_length(ml), match_offset(mo), padding(0) {}
};

// ============================================================================
// Context for Sequence Processing
// ============================================================================

struct SequenceContext {
  // Device output buffers populated by decompression
  Sequence *d_sequences = nullptr;
  unsigned char *d_literals_buffer = nullptr;
  u32 *d_literal_lengths = nullptr;
  u32 *d_match_lengths = nullptr;
  u32 *d_offsets = nullptr;
  u32 *d_num_sequences = nullptr; // Device pointer to sequence count

  u32 max_sequences = 0;
  u32 max_literals = 0;
  u32 num_sequences = 0;
  u32 num_literals = 0;

  // Flag to indicate if offsets are raw (Tier 4) or FSE-encoded (Tier 1)
  // true = Tier 4 raw u32 offsets (no +3 bias)
  // false = Tier 1 FSE offsets (have +3 bias, need get_actual_offset decoding)
  bool is_raw_offsets = false;

  // Tier 1: Predefined FSE Tables (Device Pointers)
  // These are populated during context initialization if Tier 1 is enabled.
  void *d_ll_table = nullptr; // Cast to fse::FSEEncodeTable*
  void *d_ml_table = nullptr; // Cast to fse::FSEEncodeTable*
  void *d_of_table = nullptr; // Cast to fse::FSEEncodeTable*

  // Tier 1: Decode Tables (Simple state→symbol lookup)
  u8 *d_ll_decode = nullptr; // State→symbol table for LL
  u8 *d_ml_decode = nullptr; // State→symbol table for ML
  u8 *d_of_decode = nullptr; // State→symbol table for OF

  // Tier 1 Optimization: Intermediate Buffers (Codes & Extra Bits)
  // These allow splitting FSE into Parallel (Prepare/Reconstruct) and Serial
  // (State Update) phases.
  u8 *d_ll_codes = nullptr;
  u8 *d_ml_codes = nullptr;
  u8 *d_of_codes = nullptr;

  u32 *d_ll_extras = nullptr;
  u32 *d_ml_extras = nullptr;
  u32 *d_of_extras = nullptr;

  u8 *d_ll_num_bits = nullptr;
  u8 *d_ml_num_bits = nullptr;
  u8 *d_of_num_bits = nullptr;
};

// ============================================================================
// Host Functions (UPDATED SIGNATURES)
// ============================================================================

/**
 * @brief Builds the sequence structs from the component arrays.
 * Required by the compression pipeline.
 */
Status build_sequences(const SequenceContext &ctx, u32 num_sequences_to_build,
                       u32 num_blocks, u32 num_threads,
                       cudaStream_t stream = 0);

/**
 * @brief Executes the sequences to reconstruct the output data.
 * This function orchestrates the multi-pass parallel decompression.
 *
 * @param is_raw_offsets If true, offsets are raw Tier 4 values (no bias).
 *                        If false, offsets are FSE-encoded with +3 bias.
 */
Status
execute_sequences(const unsigned char *d_literals, u32 literal_count,
                  const Sequence *d_sequences, u32 num_sequences,
                  unsigned char *d_output,
                  u32 *d_output_size, // Device pointer to total output size
                  bool is_raw_offsets = false, cudaStream_t stream = 0,
                  const unsigned char *output_base = nullptr, u32 output_max_size = 0,
                  u32 *d_rep_codes = nullptr // (OPIONAL) Persistent offsets
);

/**
 * @brief Compute statistics
 */
Status compute_sequence_statistics(const Sequence *d_sequences,
                                   u32 num_sequences, SequenceStats *h_stats,
                                   cudaStream_t stream = 0);

/**
 * @brief Validate sequences
 */
Status validate_sequences(const Sequence *d_sequences, u32 num_sequences,
                          u32 input_size, cudaStream_t stream = 0);

// ============================================================================
// Device Functions (REMOVED: Old copy helpers)
// ============================================================================

} // namespace sequence
} // namespace cuda_zstd

#endif // CUDA_ZSTD_SEQUENCE_H