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
  // Device output buffers populated by decompression.
  // When owns_memory == true, these 6 buffers are independently cudaMalloc'd
  // and will be cudaFree'd by the destructor. When owns_memory == false
  // (e.g. shallow copies used as per-block views), these are BORROWED pointers
  // and the destructor is a no-op.
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
  // NOTE: These are BORROWED pointers into workspace/temp memory — NOT freed
  // here.
  void *d_ll_table = nullptr; // Cast to fse::FSEEncodeTable*
  void *d_ml_table = nullptr; // Cast to fse::FSEEncodeTable*
  void *d_of_table = nullptr; // Cast to fse::FSEEncodeTable*

  // Tier 1: Decode Tables (Simple state→symbol lookup)
  u8 *d_ll_decode = nullptr; // State→symbol table for LL
  u8 *d_ml_decode = nullptr; // State→symbol table for ML
  u8 *d_of_decode = nullptr; // State→symbol table for OF

  // Tier 1 Optimization: Intermediate Buffers (Codes & Extra Bits)
  // NOTE: These are BORROWED pointers into workspace/temp memory — NOT freed
  // here.
  u8 *d_ll_codes = nullptr;
  u8 *d_ml_codes = nullptr;
  u8 *d_of_codes = nullptr;

  u32 *d_ll_extras = nullptr;
  u32 *d_ml_extras = nullptr;
  u32 *d_of_extras = nullptr;

  u8 *d_ll_num_bits = nullptr;
  u8 *d_ml_num_bits = nullptr;
  u8 *d_of_num_bits = nullptr;

  // --- Ownership flag ---
  // true  = this instance owns the 6 device buffers above; destructor frees.
  // false = shallow view / borrowed pointers; destructor is a no-op.
  // Only the canonical instance created via new+cudaMalloc (in
  // initialize_context) should set this to true. Copies (per-block views,
  // temporary code_ctx) are always non-owning.
  bool owns_memory = false;

  // Default constructor (non-owning by default — safe for stack/vector usage)
  SequenceContext() = default;

  // RAII: free the 6 owned device buffers on destruction (host-only context).
  // Only runs when owns_memory == true.
  ~SequenceContext() {
    if (owns_memory) {
      if (d_literals_buffer)
        cudaFree(d_literals_buffer);
      if (d_literal_lengths)
        cudaFree(d_literal_lengths);
      if (d_match_lengths)
        cudaFree(d_match_lengths);
      if (d_offsets)
        cudaFree(d_offsets);
      if (d_num_sequences)
        cudaFree(d_num_sequences);
      if (d_sequences)
        cudaFree(d_sequences);
    }
    d_literals_buffer = nullptr;
    d_literal_lengths = nullptr;
    d_match_lengths = nullptr;
    d_offsets = nullptr;
    d_num_sequences = nullptr;
    d_sequences = nullptr;
  }

  // Copy constructor: SHALLOW non-owning copy (safe for per-block views)
  SequenceContext(const SequenceContext &other) = default;
  SequenceContext &operator=(const SequenceContext &other) = default;

  // Move constructor: transfers ownership (source becomes non-owning)
  SequenceContext(SequenceContext &&other) noexcept
      : d_sequences(other.d_sequences),
        d_literals_buffer(other.d_literals_buffer),
        d_literal_lengths(other.d_literal_lengths),
        d_match_lengths(other.d_match_lengths), d_offsets(other.d_offsets),
        d_num_sequences(other.d_num_sequences),
        max_sequences(other.max_sequences), max_literals(other.max_literals),
        num_sequences(other.num_sequences), num_literals(other.num_literals),
        is_raw_offsets(other.is_raw_offsets), d_ll_table(other.d_ll_table),
        d_ml_table(other.d_ml_table), d_of_table(other.d_of_table),
        d_ll_decode(other.d_ll_decode), d_ml_decode(other.d_ml_decode),
        d_of_decode(other.d_of_decode), d_ll_codes(other.d_ll_codes),
        d_ml_codes(other.d_ml_codes), d_of_codes(other.d_of_codes),
        d_ll_extras(other.d_ll_extras), d_ml_extras(other.d_ml_extras),
        d_of_extras(other.d_of_extras), d_ll_num_bits(other.d_ll_num_bits),
        d_ml_num_bits(other.d_ml_num_bits),
        d_of_num_bits(other.d_of_num_bits),
        owns_memory(other.owns_memory) {
    // Source becomes non-owning to prevent double-free
    other.owns_memory = false;
    other.d_sequences = nullptr;
    other.d_literals_buffer = nullptr;
    other.d_literal_lengths = nullptr;
    other.d_match_lengths = nullptr;
    other.d_offsets = nullptr;
    other.d_num_sequences = nullptr;
  }

  SequenceContext &operator=(SequenceContext &&other) noexcept {
    if (this != &other) {
      // Free current owned memory if any
      if (owns_memory) {
        if (d_literals_buffer) cudaFree(d_literals_buffer);
        if (d_literal_lengths) cudaFree(d_literal_lengths);
        if (d_match_lengths)   cudaFree(d_match_lengths);
        if (d_offsets)         cudaFree(d_offsets);
        if (d_num_sequences)   cudaFree(d_num_sequences);
        if (d_sequences)       cudaFree(d_sequences);
      }
      // Transfer all fields
      d_sequences = other.d_sequences;
      d_literals_buffer = other.d_literals_buffer;
      d_literal_lengths = other.d_literal_lengths;
      d_match_lengths = other.d_match_lengths;
      d_offsets = other.d_offsets;
      d_num_sequences = other.d_num_sequences;
      max_sequences = other.max_sequences;
      max_literals = other.max_literals;
      num_sequences = other.num_sequences;
      num_literals = other.num_literals;
      is_raw_offsets = other.is_raw_offsets;
      d_ll_table = other.d_ll_table;
      d_ml_table = other.d_ml_table;
      d_of_table = other.d_of_table;
      d_ll_decode = other.d_ll_decode;
      d_ml_decode = other.d_ml_decode;
      d_of_decode = other.d_of_decode;
      d_ll_codes = other.d_ll_codes;
      d_ml_codes = other.d_ml_codes;
      d_of_codes = other.d_of_codes;
      d_ll_extras = other.d_ll_extras;
      d_ml_extras = other.d_ml_extras;
      d_of_extras = other.d_of_extras;
      d_ll_num_bits = other.d_ll_num_bits;
      d_ml_num_bits = other.d_ml_num_bits;
      d_of_num_bits = other.d_of_num_bits;
      owns_memory = other.owns_memory;
      // Source becomes non-owning
      other.owns_memory = false;
      other.d_sequences = nullptr;
      other.d_literals_buffer = nullptr;
      other.d_literal_lengths = nullptr;
      other.d_match_lengths = nullptr;
      other.d_offsets = nullptr;
      other.d_num_sequences = nullptr;
    }
    return *this;
  }
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