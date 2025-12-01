// ==============================================================================
// workspace_manager.cu - Workspace Pre-Allocation Pattern (COMPLETE)
// ==============================================================================

#include "workspace_manager.h"
#include "common_types.h"
#include <cuda_runtime.h>
#include <iostream>

namespace compression {

// ==============================================================================
// Compression Workspace Structure Definition
// ==============================================================================

// CompressionWorkspace definition moved to workspace_manager.h

// ==============================================================================
// Constructor and Destructor Implementations
// ==============================================================================

// Constructor - Initialize all pointers to nullptr
CompressionWorkspace::CompressionWorkspace() 
    : d_hash_table(nullptr),
      hash_table_size(0),
      d_chain_table(nullptr),
      chain_table_size(0),
      d_matches(nullptr),
      max_matches(0),
      d_costs(nullptr),
      max_costs(0),
      d_literal_lengths_reverse(nullptr),
      d_match_lengths_reverse(nullptr),
      d_offsets_reverse(nullptr),
      max_sequences(0),
      d_frequencies(nullptr),
      d_code_lengths(nullptr),
      d_bit_offsets(nullptr),
      d_block_sums(nullptr),
      d_scanned_block_sums(nullptr),
      num_blocks(0),
      d_base_ptr(nullptr),
      total_size_bytes(0),
      is_allocated(false),
      stream(nullptr),
      event_complete(nullptr)
{
    // All initialization done in initializer list
}

// Destructor - Cleanup resources if allocated
CompressionWorkspace::~CompressionWorkspace() {
    if (is_allocated) {
        // Clean up CUDA resources
        free_compression_workspace(*this);
    }

    // Clean up stream and event if created
    if (event_complete) {
        cudaEventDestroy(event_complete);
        event_complete = nullptr;
    }

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

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

// Allocate workspace with actual CUDA memory allocation
Status allocate_compression_workspace(
    CompressionWorkspace& workspace,
    size_t max_block_size,
    const CompressionConfig& config
) {
//     std::cout << "[WorkspaceManager] Allocating workspace for max block size: " 
//               << max_block_size << " bytes" << std::endl;

    // Initialize sizes
    workspace.hash_table_size = (1 << config.hash_log);
    workspace.chain_table_size = (1 << config.chain_log);
    workspace.max_matches = max_block_size;
    workspace.max_costs = max_block_size;
    workspace.max_sequences = max_block_size;
    workspace.num_blocks = (max_block_size + 1023) / 1024;

    // Allocate hash table
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_hash_table, 
                                    workspace.hash_table_size * sizeof(u32)));
//     std::cout << "  ✓ Hash table: " << (workspace.hash_table_size * sizeof(u32)) / 1024 << " KB" << std::endl;

    // Allocate chain table
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_chain_table,
                                    workspace.chain_table_size * sizeof(u32)));
//     std::cout << "  ✓ Chain table: " << (workspace.chain_table_size * sizeof(u32)) / 1024 << " KB" << std::endl;

    // Allocate matches (assume Match is 16 bytes)
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_matches,
                                    max_block_size * 16));
//     std::cout << "  ✓ Matches: " << (max_block_size * 16) / 1024 << " KB" << std::endl;

    // Allocate costs (assume ParseCost is 16 bytes)
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_costs,
                                    max_block_size * 16));
//     std::cout << "  ✓ Costs: " << (max_block_size * 16) / 1024 << " KB" << std::endl;

    // Allocate sequence arrays
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_literal_lengths_reverse,
                                    max_block_size * sizeof(u32)));
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_match_lengths_reverse,
                                    max_block_size * sizeof(u32)));
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_offsets_reverse,
                                    max_block_size * sizeof(u32)));
//     std::cout << "  ✓ Sequences: " << (max_block_size * sizeof(u32) * 3) / 1024 << " KB" << std::endl;

    // Allocate frequency and code length arrays
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_frequencies, 256 * sizeof(u32)));
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_code_lengths, max_block_size * sizeof(u32)));
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_bit_offsets, max_block_size * sizeof(u32)));

    // Allocate block sum arrays for parallel scan
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_block_sums, 
                                    workspace.num_blocks * sizeof(u32)));
    CUDA_CHECK_WORKSPACE(cudaMalloc(&workspace.d_scanned_block_sums,
                                    workspace.num_blocks * sizeof(u32)));

    workspace.is_allocated = true;

    size_t total_allocated = calculate_workspace_size(max_block_size, config);
//     std::cout << "[WorkspaceManager] Total workspace allocated: " 
//               << total_allocated / (1024 * 1024) << " MB" << std::endl;

    return Status::SUCCESS;
}

// Free workspace
Status free_compression_workspace(CompressionWorkspace& workspace) {
    if (!workspace.is_allocated) {
        return Status::SUCCESS;
    }

//     std::cout << "[WorkspaceManager] Freeing workspace..." << std::endl;

    // Free all allocated buffers
    if (workspace.d_hash_table) cudaFree(workspace.d_hash_table);
    if (workspace.d_chain_table) cudaFree(workspace.d_chain_table);
    if (workspace.d_matches) cudaFree(workspace.d_matches);
    if (workspace.d_costs) cudaFree(workspace.d_costs);
    if (workspace.d_literal_lengths_reverse) cudaFree(workspace.d_literal_lengths_reverse);
    if (workspace.d_match_lengths_reverse) cudaFree(workspace.d_match_lengths_reverse);
    if (workspace.d_offsets_reverse) cudaFree(workspace.d_offsets_reverse);
    if (workspace.d_frequencies) cudaFree(workspace.d_frequencies);
    if (workspace.d_code_lengths) cudaFree(workspace.d_code_lengths);
    if (workspace.d_bit_offsets) cudaFree(workspace.d_bit_offsets);
    if (workspace.d_block_sums) cudaFree(workspace.d_block_sums);
    if (workspace.d_scanned_block_sums) cudaFree(workspace.d_scanned_block_sums);

    // Zero out pointers
    workspace.d_hash_table = nullptr;
    workspace.d_chain_table = nullptr;
    workspace.d_matches = nullptr;
    workspace.d_costs = nullptr;
    workspace.d_literal_lengths_reverse = nullptr;
    workspace.d_match_lengths_reverse = nullptr;
    workspace.d_offsets_reverse = nullptr;
    workspace.d_frequencies = nullptr;
    workspace.d_code_lengths = nullptr;
    workspace.d_bit_offsets = nullptr;
    workspace.d_block_sums = nullptr;
    workspace.d_scanned_block_sums = nullptr;

    workspace.is_allocated = false;

//     std::cout << "[WorkspaceManager] Workspace freed successfully" << std::endl;

    return Status::SUCCESS;
}

} // namespace compression
