// ============================================================================
// cuda_zstd_sequence.h - Sequence Encoding/Decoding Interface
//
// (UPDATED) Header now reflects the multi-pass, parallel implementation
// found in cuda_zstd_sequence.cu.
// ============================================================================

#ifndef CUDA_ZSTD_SEQUENCE_H_
#define CUDA_ZSTD_SEQUENCE_H_

#include "cuda_zstd_types.h"
#include "cuda_zstd_internal.h"
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
    SequenceStats() :
        num_sequences(0), num_literals(0), total_match_length(0),
        avg_literal_length(0), avg_match_length(0), avg_offset(0),
        rep_1_count(0), rep_2_count(0), rep_3_count(0) {}
};

struct Sequence {
    u32 literal_length;    // Number of literals before match
    u32 match_length;      // Length of match (actual, not encoded)
    u32 match_offset;      // Offset to match (distance)
    
    __device__ __host__ Sequence() 
        : literal_length(0), match_length(0), match_offset(0) {}
    
    __device__ __host__ Sequence(u32 ll, u32 ml, u32 mo)
        : literal_length(ll), match_length(ml), match_offset(mo) {}
};

// ============================================================================
// Context for Sequence Processing
// ============================================================================

struct SequenceContext {
    // Device output buffers populated by decompression
    Sequence* d_sequences = nullptr;
    byte_t* d_literals_buffer = nullptr;
    u32* d_literal_lengths = nullptr;
    u32* d_match_lengths = nullptr;
    u32* d_offsets = nullptr;
    u32* d_num_sequences = nullptr;  // Device pointer to sequence count
    
    u32 max_sequences = 0;
    u32 max_literals = 0;
    u32 num_sequences = 0;
    u32 num_literals = 0;
};

// ============================================================================
// Host Functions (UPDATED SIGNATURES)
// ============================================================================

/**
 * @brief Builds the sequence structs from the component arrays.
 * Required by the compression pipeline.
 */
Status build_sequences(
    const SequenceContext& ctx,
    u32 num_sequences_to_build,
    u32 num_blocks,
    u32 num_threads,
    cudaStream_t stream = 0
);

/**
 * @brief Executes the sequences to reconstruct the output data.
 * This function orchestrates the multi-pass parallel decompression.
 */
Status execute_sequences(
    const byte_t* d_literals,
    u32 literal_count,
    const Sequence* d_sequences,
    u32 num_sequences,
    byte_t* d_output,
    u32* d_output_size, // Device pointer to total output size
    cudaStream_t stream = 0
);

/**
 * @brief Compute statistics
 */
Status compute_sequence_statistics(
    const Sequence* d_sequences,
    u32 num_sequences,
    SequenceStats* h_stats,
    cudaStream_t stream = 0
);

/**
 * @brief Validate sequences
 */
Status validate_sequences(
    const Sequence* d_sequences,
    u32 num_sequences,
    u32 input_size,
    cudaStream_t stream = 0
);

// ============================================================================
// Device Functions (REMOVED: Old copy helpers)
// ============================================================================

} // namespace sequence
} // namespace cuda_zstd

#endif // CUDA_ZSTD_SEQUENCE_H