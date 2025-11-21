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

// Forward declaration to avoid circular dependency
struct CompressionWorkspace;

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
