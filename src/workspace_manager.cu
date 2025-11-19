// ==============================================================================
// workspace_manager.cu - Workspace allocation implementation
// ==============================================================================

#include "workspace_manager.h"
#include <iostream>
#include <algorithm>

namespace compression {

constexpr size_t GPU_MEMORY_ALIGNMENT = 256;

inline size_t align_to_boundary(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// Constructor
CompressionWorkspace::CompressionWorkspace() 
    : d_hash_table(nullptr), hash_table_size(0),
      d_chain_table(nullptr), chain_table_size(0),
      d_matches(nullptr), max_matches(0),
      d_costs(nullptr), max_costs(0),
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
      stream(0),
      event_complete(nullptr)
{}

// Destructor
CompressionWorkspace::~CompressionWorkspace() {
    if (is_allocated) {
        free_compression_workspace(*this);
    }
}

size_t calculate_workspace_size(
    size_t max_block_size,
    const CompressionConfig& config
) {
    size_t total = 0;

    size_t hash_table_size = (1ULL << config.hash_log) * sizeof(u32);
    total += align_to_boundary(hash_table_size, GPU_MEMORY_ALIGNMENT);

    size_t chain_table_size = (1ULL << config.chain_log) * sizeof(u32);
    total += align_to_boundary(chain_table_size, GPU_MEMORY_ALIGNMENT);

    size_t matches_size = max_block_size * sizeof(lz77::Match);
    total += align_to_boundary(matches_size, GPU_MEMORY_ALIGNMENT);

    size_t costs_size = max_block_size * sizeof(lz77::ParseCost);
    total += align_to_boundary(costs_size, GPU_MEMORY_ALIGNMENT);

    size_t sequence_size = max_block_size * sizeof(u32);
    total += align_to_boundary(sequence_size * 3, GPU_MEMORY_ALIGNMENT);

    size_t freq_size = 256 * sizeof(u32);
    total += align_to_boundary(freq_size, GPU_MEMORY_ALIGNMENT);

    size_t code_len_size = 256 * sizeof(u32);
    total += align_to_boundary(code_len_size, GPU_MEMORY_ALIGNMENT);

    size_t bit_offset_size = max_block_size * sizeof(u32);
    total += align_to_boundary(bit_offset_size, GPU_MEMORY_ALIGNMENT);

    u32 num_blocks = (max_block_size + 1023) / 1024;
    size_t block_size = num_blocks * sizeof(u32) * 2;
    total += align_to_boundary(block_size, GPU_MEMORY_ALIGNMENT);

    total += total / 10;

    return total;
}
