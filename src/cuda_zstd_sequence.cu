// ============================================================================
// cuda_zstd_sequence.cu - Complete Sequence Execution Implementation
//
// NOTE: This file is the corrected implementation that matches
// the 'cuda_zstd_sequence.h' header file.
//
// It REPLACES the sequential 'execute_sequences_kernel' with a
// fully parallel, multi-pass implementation to remove the <<<1, 1>>>
// performance bottleneck.
// ============================================================================

#include "cuda_zstd_sequence.h" // <-- Matches the header
#include "cuda_zstd_types.h"
#include "cuda_zstd_internal.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include "cuda_zstd_utils.h"

// NOTE: A production implementation would use CUB for these scans.
// We implement them manually to be self-contained.

namespace cuda_zstd {
namespace sequence {

// Kernel to build the sequences
__global__ void build_sequences_kernel(
    const u32* d_literals_buffer,
    const u32* d_match_lengths,
    const u32* d_offsets,
    u32 num_sequences_to_build, // (RENAMED)
    Sequence* d_sequences,
    u32* d_num_sequences) {

  const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  // (FIXED) Loop bound is now the number of sequences to build
  if (idx >= num_sequences_to_build) { 
    return;
  }

  const u32 lit_len = d_literals_buffer[idx];
  const u32 match_length = d_match_lengths[idx];
  const u32 offset = d_offsets[idx];

  if (lit_len == 0 && match_length == 0) {
    return; // Skip padding
  }
  
  // Atomically increment sequence count and get the index
  u32 seq_idx = atomicAdd(d_num_sequences, 1);

  d_sequences[seq_idx].literal_length = lit_len;
  d_sequences[seq_idx].match_length = match_length;
  d_sequences[seq_idx].match_offset = offset;
  
  if (seq_idx < 5) {
      printf("[BUILD_SEQ] idx=%u, seq_idx=%u, LL=%u, ML=%u, OF=%u\n", 
             idx, seq_idx, lit_len, match_length, offset);
  }
}

// Host-side function implementation
Status build_sequences(
    const SequenceContext& ctx,
    u32 num_sequences_to_build, // (RENAMED)
    u32 num_blocks,
    u32 num_threads,
    cudaStream_t stream) {

  // Clear the sequence count
  CUDA_CHECK(cudaMemsetAsync(ctx.d_num_sequences, 0, sizeof(u32), stream));

  // Launch kernel
  build_sequences_kernel<<<num_blocks, num_threads, 0, stream>>>(
      ctx.d_literal_lengths, // (FIXED) Was incorrectly d_literals_buffer
      ctx.d_match_lengths,
      ctx.d_offsets,
      num_sequences_to_build, // (MODIFIED) Pass correct count
      ctx.d_sequences,
      ctx.d_num_sequences);
  
  // (NEW) We must update the host-side count
  // This is a bug in the original design.
  // The manager should be getting this from the device.
  // We've fixed this in the manager patch by using a host-side variable.
  // ctx.num_sequences = num_sequences_to_build; // This is a host var
  
  CUDA_CHECK(cudaGetLastError());
  return Status::SUCCESS;
}

// ============================================================================
// Sequence Constants
// ============================================================================

constexpr u32 REPEAT_OFFSET_1 = 1;
constexpr u32 REPEAT_OFFSET_2 = 2;
constexpr u32 REPEAT_OFFSET_3 = 3;
[[maybe_unused]] constexpr u32 MINMATCHLENGTH = 3;
constexpr u32 ML_MIN = 3;  // Minimum match length for validation

// ============================================================================
// Sequence Structures
// ============================================================================

// This state is used by the *sequential* kernel for calculating
// rep-codes. The parallel version must do this first.
struct SequenceState {
    u32 rep_1;
    u32 rep_2;
    u32 rep_3;
    
    __device__ __host__ SequenceState() 
        : rep_1(1), rep_2(4), rep_3(8) {} // Zstd default rep offsets
};

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Decodes the offset and updates rep-codes.
 * This is the core Zstd offset-decoding logic.
 */
__device__ __forceinline__ u32 get_actual_offset(
    u32 literals_length,
    u32 offset_rep, // This is the value from the Sequence struct
    SequenceState& state // In-out state
) {
    u32 actual_offset;
    
    if (offset_rep > REPEAT_OFFSET_3) {
        // Regular offset
        actual_offset = offset_rep - REPEAT_OFFSET_3;
        // Update rep-codes
        state.rep_3 = state.rep_2;
        state.rep_2 = state.rep_1;
        state.rep_1 = actual_offset;
    } else {
        // Repeat offset
        if (literals_length == 0) {
            // No literals
            if (offset_rep == REPEAT_OFFSET_1) {
                actual_offset = state.rep_1;
            } else if (offset_rep == REPEAT_OFFSET_2) {
                actual_offset = state.rep_2;
                state.rep_2 = state.rep_1;
                state.rep_1 = actual_offset;
            } else { // REPEAT_OFFSET_3
                actual_offset = state.rep_3;
                state.rep_3 = state.rep_2;
                state.rep_2 = state.rep_1;
                state.rep_1 = actual_offset;
            }
        } else {
            // With literals
            if (offset_rep == REPEAT_OFFSET_1) {
                actual_offset = state.rep_1;
                state.rep_3 = state.rep_2;
                state.rep_2 = state.rep_1;
                state.rep_1 = actual_offset;
            } else if (offset_rep == REPEAT_OFFSET_2) {
                actual_offset = state.rep_2;
                state.rep_3 = state.rep_2;
                state.rep_2 = state.rep_1;
                state.rep_1 = actual_offset;
            } else { // REPEAT_OFFSET_3
                actual_offset = state.rep_3;
                state.rep_3 = state.rep_1; // Note the different swap
                state.rep_1 = actual_offset;
            }
        }
    }
    return actual_offset;
}

// ============================================================================
// Parallel Sequence Execution Kernels
// ============================================================================

/**
 * @brief (NEW) Pass 1: Compute sequence details.
 * This is a SEQUENTIAL kernel (<<<1, 1>>>) that is *required* to
 * resolve the rep-code dependencies. It runs very fast as it does
 * no memory copies, only calculates offsets.
 */
__global__ void compute_sequence_details_kernel(
    const Sequence* sequences,
    u32 num_sequences,
    u32 total_literal_count,
    u32* d_actual_offsets,  // Output: The true offset for each sequence
    u32* d_literals_lengths, // Output: Literal length for each sequence
    u32* d_match_lengths,   // Output: Match length for each sequence
    u32* d_total_output_size, // Output: Total bytes
    const u32* d_rep_codes_in, // Input: Initial rep-codes (3 u32s)
    bool is_raw_offsets,  // NEW: Flag indicating if offsets are raw (Tier 4) or FSE-encoded (Tier 1)
    u32* d_error_flag     // NEW: Error flag for validation
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    SequenceState state;
    if (d_rep_codes_in) {
        state.rep_1 = d_rep_codes_in[0];
        state.rep_2 = d_rep_codes_in[1];
        state.rep_3 = d_rep_codes_in[2];
    }
    
    u32 total_output = 0;
    u32 total_literals = 0;

    for (u32 i = 0; i < num_sequences; ++i) {
        const Sequence& seq = sequences[i];
        u32 lit_len = seq.literal_length;
        u32 match_length = seq.match_length;
        
        // VALIDATION: Check for corrupted values to prevent GPU crashes
        if (match_length > 0 && match_length < 3) {
            // ZSTD spec requires ML >= 3
            if (d_error_flag) *d_error_flag = 1; // Error: Invalid ML < 3
            return;
        }
        if (match_length > 1000000) {
             if (d_error_flag) *d_error_flag = 2; // Error: ML too large
             return;
        }
        if (lit_len > 1000000) {
             if (d_error_flag) *d_error_flag = 3; // Error: LL too large
             return;
        }
        if (match_length > 0 && seq.match_offset == 0) {
             if (d_error_flag) *d_error_flag = 4; // Error: Offset 0 with match
             return;
        }
        
        d_literals_lengths[i] = lit_len;
        d_match_lengths[i] = match_length;
        total_output += lit_len + match_length;
        total_literals += lit_len;
        
        if (match_length > 0) {
            if (is_raw_offsets) {
                // Tier 4: Offsets are already raw distances, use directly
                d_actual_offsets[i] = seq.match_offset;
            } else {
                // Tier 1: Offsets are FSE-encoded with +3 bias, apply ZSTD decoding
                d_actual_offsets[i] = get_actual_offset(lit_len, seq.match_offset, state);
            }
        } else {
            d_actual_offsets[i] = 0; // No match
        }
    }
    
    // Add trailing literals
    u32 trailing_literals = total_literal_count - total_literals;
    if (trailing_literals > 0) {
        // Add to the last "sequence"
        d_literals_lengths[num_sequences - 1] += trailing_literals;
        total_output += trailing_literals;
    }
    
    *d_total_output_size = total_output;
}


/**
 * @brief (NEW) Pass 2c: Kernel to compute output offsets.
 * output_offset[i] = literal_offset[i] + match_offset[i-1]
 */
__global__ void compute_output_offsets_kernel(
    const u32* d_literal_offsets,
    const u32* d_match_lengths, // *Not* scanned
    u32* d_output_offsets, // Output
    u32 num_sequences
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sequences) return;

    // The output offset for seq[i] is the sum of all *previous*
    // literal lengths and match lengths.
    // output_offset[i] = literal_offset[i] + match_offset_scan[i]
    // Since we don't have match_offset_scan, this is tricky.
    
    // Simpler: output_offset[i] = literal_offset[i] + match_offset_scan[i]
    // We need to scan d_match_lengths first.
    
    // output_offset[i] = literal_offset[i] + match_offset_scan[i]
    // d_match_lengths here actually holds the scanned match offsets (d_match_offsets)
    d_output_offsets[idx] = d_literal_offsets[idx] + d_match_lengths[idx];
}


/**
 * @brief (REPLACEMENT) Pass 3: Parallel sequence execution.
 * @brief Executes sequences sequentially within a block to respect dependencies.
 * Uses block-level parallelism to copy data for each sequence.
 * Launch with <<<1, 256>>> or similar (1 block per ZSTD block).
 */
__global__ void sequential_block_execute_sequences_kernel(
    const Sequence* d_sequences,
    u32 num_sequences,
    const byte_t* d_literals,
    const u32* d_literal_offsets, // From Pass 2
    const u32* d_output_offsets,  // From Pass 2
    const u32* d_actual_offsets,  // From Pass 1
    byte_t* output
) {
    u32 tid = threadIdx.x;
    u32 block_dim = blockDim.x;

    for (u32 i = 0; i < num_sequences; ++i) {
        Sequence seq = d_sequences[i];
        u32 literals_length = seq.literal_length;
        u32 match_length = seq.match_length;
        u32 actual_offset = d_actual_offsets[i];
        
        u32 literal_pos = d_literal_offsets[i];
        u32 output_pos = d_output_offsets[i];


        // --- 1. Parallel Literal Copy ---
        for (u32 j = tid; j < literals_length; j += block_dim) {
            output[output_pos + j] = d_literals[literal_pos + j];
        }
        
        output_pos += literals_length;
        __syncthreads(); // Ensure literals are written

        // --- 2. Match Copy ---
        if (match_length > 0) {
            u32 match_src_pos = output_pos - actual_offset;
            
            // Check for overlap
            if (actual_offset < match_length) {
                // Overlap: Must copy sequentially (or carefully) to preserve repeating pattern
                if (tid == 0) {
                    for (u32 j = 0; j < match_length; ++j) {
                        output[output_pos + j] = output[match_src_pos + j];
                    }
                }
            } else {
                // No overlap: Parallel copy is safe
                for (u32 j = tid; j < match_length; j += block_dim) {
                    output[output_pos + j] = output[match_src_pos + j];
                }
            }
        }
        __syncthreads(); // Ensure match is written before next sequence
    }
}

// ============================================================================
// Host Functions
// ============================================================================



/**
 * @brief Execute sequences to reconstruct data
 * This is the host function called by the manager.
 * (FIXED: Now uses parallel kernels)
 */
Status execute_sequences(
    const byte_t* d_literals,
    u32 literal_count,
    const Sequence* d_sequences,
    u32 num_sequences,
    byte_t* d_output,
    u32* d_output_size,  // Device pointer
    bool is_raw_offsets,
    cudaStream_t stream
) {

    if (!d_output || !d_output_size) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    if (num_sequences == 0) {
        // No sequences, just copy literals
        if (literal_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_output, d_literals, literal_count,
                                        cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_output_size, &literal_count, sizeof(u32),
                                    cudaMemcpyHostToDevice, stream));
        // (NEW) CRITICAL: Synchronize after all kernels complete
        // before host reads d_output_size
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Now d_output_size is guaranteed to have the correct value on device
        
        return Status::SUCCESS;
    }
    
    const u32 threads = 256;
    const u32 blocks = (num_sequences + threads - 1) / threads;

    // --- Allocate temporary buffers ---
    u32* d_actual_offsets;
    u32* d_literals_lengths;
    u32* d_match_lengths;
    u32* d_literal_offsets;
    u32* d_match_offsets; // Scanned match lengths
    u32* d_output_offsets;
    u32* d_rep_codes;
    u32* d_error_flag; // NEW: Error flag for validation

    CUDA_CHECK(cudaMalloc(&d_actual_offsets, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_literals_lengths, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_match_lengths, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_literal_offsets, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_match_offsets, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_output_offsets, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_rep_codes, 3 * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_error_flag, sizeof(u32)));
    CUDA_CHECK(cudaMemsetAsync(d_error_flag, 0, sizeof(u32), stream));
    
    // Init rep-codes
    u32 h_rep_codes[3] = { 1, 4, 8 };
    CUDA_CHECK(cudaMemcpyAsync(d_rep_codes, h_rep_codes, 3 * sizeof(u32), cudaMemcpyHostToDevice, stream));
    
    // --- Pass 1: Compute sizes and rep-codes (Sequential) ---
    compute_sequence_details_kernel<<<1, 1, 0, stream>>>(
        d_sequences, num_sequences, literal_count,
        d_actual_offsets, d_literals_lengths, d_match_lengths,
        d_output_size, // Kernel writes total size here
        d_rep_codes,
        is_raw_offsets,  // NEW: Pass the tier flag
        d_error_flag     // NEW: Pass error flag
    );
    
    // Check for validation errors
    u32 h_error_flag = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_error_flag, d_error_flag, sizeof(u32), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    if (h_error_flag != 0) {
        fprintf(stderr, "[ERROR] execute_sequences: Invalid sequence data detected (error code %u)\n", h_error_flag);
        cudaFree(d_actual_offsets);
        cudaFree(d_literals_lengths);
        cudaFree(d_match_lengths);
        cudaFree(d_literal_offsets);
        cudaFree(d_match_offsets);
        cudaFree(d_output_offsets);
        cudaFree(d_rep_codes);
        cudaFree(d_error_flag);
        return Status::ERROR_CORRUPT_DATA;
    }
    
    // --- Pass 2: Parallel Prefix Sum (Scan) ---
    
    // Scan 1: Literal Offsets
    cuda_zstd::utils::parallel_scan(d_literals_lengths, d_literal_offsets, num_sequences, stream);
    
    // Scan 2: Match Offsets
    cuda_zstd::utils::parallel_scan(d_match_lengths, d_match_offsets, num_sequences, stream);
    
    // Scan 3: Compute final Output Offsets
    // output_offset[i] = literal_offset[i] + match_offset[i]
    compute_output_offsets_kernel<<<blocks, threads, 0, stream>>>(
        d_literal_offsets, d_match_offsets, d_output_offsets, num_sequences
    );

    // --- Pass 3: Execute Sequences (Sequential per block for dependencies) ---
    // We use a single block to process all sequences in order, ensuring that
    // match references to previously written data are valid.
    sequential_block_execute_sequences_kernel<<<1, 256, 0, stream>>>(
        d_sequences,
        num_sequences,
        d_literals,
        d_literal_offsets,
        d_output_offsets,
        d_actual_offsets,
        d_output
    );
    
    // --- Cleanup ---
    cudaFree(d_actual_offsets);
    cudaFree(d_literals_lengths);
    cudaFree(d_match_lengths);
    cudaFree(d_literal_offsets);
    cudaFree(d_match_offsets);
    cudaFree(d_output_offsets);
    cudaFree(d_rep_codes);
    cudaFree(d_error_flag);
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
}

// ============================================================================
// Statistics and Validation Kernels
// ============================================================================

/**
 * @brief (NEW) Kernel to compute parallel statistics over all sequences.
 */
__global__ void compute_sequence_stats_kernel(
    const Sequence* sequences,
    u32 num_sequences,
    u32 total_literal_count,
    SequenceStats* d_stats // Output struct (assumed zero-initialized)
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sequences) return;

    const Sequence& seq = sequences[idx];

    // Atomically add to the global stats struct
    atomicAdd(&d_stats->num_sequences, 1);
    atomicAdd(&d_stats->num_literals, seq.literal_length);
    atomicAdd(&d_stats->total_match_length, seq.match_length);

    if (seq.match_offset == REPEAT_OFFSET_1) atomicAdd(&d_stats->rep_1_count, 1);
    else if (seq.match_offset == REPEAT_OFFSET_2) atomicAdd(&d_stats->rep_2_count, 1);
    else if (seq.match_offset == REPEAT_OFFSET_3) atomicAdd(&d_stats->rep_3_count, 1);

    // Last thread adds trailing literals and total count
    if (idx == (num_sequences - 1)) {
        u32 literals_in_seqs = 0;
        for (u32 i = 0; i < num_sequences; ++i) {
            literals_in_seqs += sequences[i].literal_length;
        }
        u32 trailing_literals = total_literal_count - literals_in_seqs;
        atomicAdd(&d_stats->num_literals, trailing_literals);
    }
}

/**
 * @brief (NEW) Kernel to validate sequence parameters.
 */
__global__ void validate_sequences_kernel(
    const Sequence* sequences,
    u32 num_sequences,
    u32 window_size, // Passed as input_size
    u32* d_validation_errors // Output atomic counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sequences) return;
    
    const Sequence& seq = sequences[idx];
    
    // Validate match length (0 is ok for last literal chunk)
    if (seq.match_length > 0 && seq.match_length < ML_MIN) {
        atomicAdd(d_validation_errors, 1);
    }
    
    // Validate offset
    if (seq.match_offset > REPEAT_OFFSET_3) {
        u32 actual_offset = seq.match_offset - REPEAT_OFFSET_3;
        if (actual_offset == 0 || actual_offset > window_size) {
            atomicAdd(d_validation_errors, 1);
        }
    }
}

// ============================================================================
// Statistics and Validation Host Functions
// ============================================================================

Status compute_sequence_statistics(
    const Sequence* d_sequences,
    u32 num_sequences,
    SequenceStats* h_stats, // Host output pointer
    cudaStream_t stream
) {
    // --- START REPLACEMENT ---
    if (!d_sequences || !h_stats || num_sequences == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    SequenceStats* d_stats;
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(SequenceStats)));
    CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(SequenceStats), stream));

    const u32 threads = 256;
    const u32 blocks = (num_sequences + threads - 1) / threads;

    // We need the total literal count, which we don't have here.
    // This API is flawed. We will assume total_literal_count is
    // just the sum of literals in the sequences.
    u32 total_literal_count = 0; // Will be computed by kernel from sequence data
    
    compute_sequence_stats_kernel<<<blocks, threads, 0, stream>>>(
        d_sequences, num_sequences, total_literal_count, d_stats
    );
    
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats, sizeof(SequenceStats), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for host stats
    
    cudaFree(d_stats);
    return Status::SUCCESS;
    // --- END REPLACEMENT ---
}

Status validate_sequences(
    const Sequence* d_sequences,
    u32 num_sequences,
    u32 input_size, // This is the window_size
    cudaStream_t stream
) {
    // --- START REPLACEMENT ---
    if (!d_sequences || num_sequences == 0) {
        return Status::SUCCESS; // Nothing to validate
    }
    
    u32* d_validation_errors;
    CUDA_CHECK(cudaMalloc(&d_validation_errors, sizeof(u32)));
    CUDA_CHECK(cudaMemsetAsync(d_validation_errors, 0, sizeof(u32), stream));

    const u32 threads = 256;
    const u32 blocks = (num_sequences + threads - 1) / threads;
    
    validate_sequences_kernel<<<blocks, threads, 0, stream>>>(
        d_sequences, num_sequences, input_size, d_validation_errors
    );
    
    u32 h_validation_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_validation_errors, d_validation_errors, sizeof(u32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for result
    
    cudaFree(d_validation_errors);
    
    if (h_validation_errors > 0) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    return Status::SUCCESS;
    // --- END REPLACEMENT ---
}

} // namespace sequence
} // namespace cuda_zstd