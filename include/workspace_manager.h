// ==============================================================================
// workspace_manager.h - Pre-allocated workspace pattern
// ==============================================================================

#ifndef WORKSPACE_MANAGER_H
#define WORKSPACE_MANAGER_H

#include "cuda_zstd_types.h"

namespace cuda_zstd {

// CompressionWorkspace and CompressionConfig are defined in cuda_zstd_types.h

// calculate_workspace_size is NOT in cuda_zstd_types.h, so declare it here.
size_t calculate_workspace_size(
    size_t max_block_size,
    const CompressionConfig& config
);




} // namespace cuda_zstd

#endif // WORKSPACE_MANAGER_H
