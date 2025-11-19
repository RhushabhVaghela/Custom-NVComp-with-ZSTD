// ==============================================================================
// workspace_manager.h - Pre-allocated workspace pattern
// ==============================================================================

#ifndef WORKSPACE_MANAGER_H
#define WORKSPACE_MANAGER_H

#include "common_types.h"
#include "error_context.h"
#include "lz77_types.h"
#include <cuda_runtime.h>

namespace compression {

// Compression workspace structure
struct CompressionWorkspace {
    // === LZ77 Temporary Buffers ===
    u32* d_hash_table;           
    u32 hash_table_size;

    u32* d_chain_table;          
    u32 chain_table_size;

    lz77::Match* d_matches;      
    u32 max_matches;

    lz77::ParseCost* d_costs;    
    u32 max_costs;

    // === Sequence Temporary Buffers ===
    u32* d_literal_lengths_reverse;  
    u32* d_match_lengths_reverse;
    u32* d_offsets_reverse;
    u32 max_sequences;

    // === Huffman Temporary Buffers ===
    u32* d_frequencies;          
    u32* d_code_lengths;         
    u32* d_bit_offsets;          

    // === Block Processing Buffers ===
    u32* d_block_sums;           
    u32* d_scanned_block_sums;   
    u32 num_blocks;

    // === Workspace Metadata ===
    void* d_base_ptr;            
    size_t total_size_bytes;     
    bool is_allocated;           

    // === Stream Management ===
    cudaStream_t stream;         
    cudaEvent_t event_complete;  

    CompressionWorkspace();
    ~CompressionWorkspace();
};

// Configuration for workspace allocation
struct CompressionConfig {
    u32 window_log = 20;      
    u32 hash_log = 17;        
    u32 chain_log = 17;       
    u32 search_log = 8;       
    u32 min_match = 3;        
    u32 block_size = 128 * 1024;
};

// Workspace management functions
Status allocate_compression_workspace(
    CompressionWorkspace& workspace,
    size_t max_block_size,
    const CompressionConfig& config
);

Status free_compression_workspace(CompressionWorkspace& workspace);

size_t calculate_workspace_size(
    size_t max_block_size,
    const CompressionConfig& config
);

} // namespace compression

#endif // WORKSPACE_MANAGER_H
