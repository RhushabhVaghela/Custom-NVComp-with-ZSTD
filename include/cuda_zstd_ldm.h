#ifndef CUDA_ZSTD_LDM_H_
#define CUDA_ZSTD_LDM_H_

#include "cuda_zstd_types.h"
#include "cuda_zstd_lz77.h"

namespace cuda_zstd {
namespace ldm {

struct LDMHashEntry {
    u32 position;
    u32 hash_value;
};

struct LDMContext {
    LDMHashEntry* d_hash_table;      
    u32 window_start;                
    u32 window_size;                 
    u32 matches_found;
    u32 min_match_length;
    u32 max_distance;
    u64 rolling_hash_state;
    
    LDMContext() : d_hash_table(nullptr), window_start(0), window_size(0), 
                   matches_found(0), min_match_length(8), 
                   max_distance(0), rolling_hash_state(0) {}
};

__host__ Status ldm_init_context(LDMContext& ctx, u32 window_log);
__host__ void ldm_cleanup_context(LDMContext& ctx);
__host__ Status ldm_reset(LDMContext& ctx);
__host__ Status ldm_process_block(LDMContext& ctx, const void* input, size_t input_size, lz77::Match* d_matches, u32 global_offset, cudaStream_t stream);
__host__ bool ldm_is_supported();

} // namespace ldm
} // namespace cuda_zstd

#endif
