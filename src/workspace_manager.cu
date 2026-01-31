// ==============================================================================
// workspace_manager.cu - Workspace Pre-Allocation Pattern (COMPLETE)
// ==============================================================================

#include "workspace_manager.h"
#include "common_types.h"
#include <cuda_runtime.h>
#include <iostream>

namespace cuda_zstd {

// ==============================================================================
// Compression Workspace Structure Definition
// ==============================================================================

// CompressionWorkspace definition moved to workspace_manager.h

// ==============================================================================
// Constructor and Destructor Implementations
// ==============================================================================

// Constructor - Initialize all pointers to nullptr
// Constructor and Destructor removed (struct is now POD in cuda_zstd_types.h)

// ==============================================================================
// Helper macro for CUDA error checking
// ==============================================================================
#define CUDA_CHECK_WORKSPACE(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return Status::ERROR_CUDA_ERROR; \
        } \
    } while(0)

// ==============================================================================
// Workspace Management Functions
// ==============================================================================

// Calculate workspace size needed for compression
size_t calculate_workspace_size(size_t max_block_size, const CompressionConfig& config) {
    size_t total = 0;

    // Hash table
    size_t hash_table_size = (1 << config.hash_log) * sizeof(u32);
    total += hash_table_size;

    // Chain table
    size_t chain_table_size = (1 << config.chain_log) * sizeof(u32);
    total += chain_table_size;

    // Matches array (assuming Match struct is ~16 bytes)
    total += max_block_size * 16;

    // Costs array (assuming ParseCost struct is ~16 bytes)
    total += max_block_size * 16;

    // Sequences (literal lengths, match lengths, offsets)
    total += max_block_size * sizeof(u32) * 3;

    // Frequencies and code lengths
    total += 256 * sizeof(u32);
    total += max_block_size * sizeof(u32) * 2;

    // Block sums for scan operations
    size_t num_blocks = (max_block_size + 1023) / 1024;
    total += num_blocks * sizeof(u32) * 2;

    // Add 10% padding for safety
    total = (total * 11) / 10;

    return total;
}

// allocate_compression_workspace and free_compression_workspace removed (implemented in cuda_zstd_types.cpp)

} // namespace cuda_zstd

