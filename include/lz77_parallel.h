// ==============================================================================
// lz77_parallel.h - Three-pass parallel LZ77 compression
// ==============================================================================

#ifndef LZ77_PARALLEL_H
#define LZ77_PARALLEL_H

#include "common_types.h"
#include "error_context.h"
#include "lz77_types.h"
#include "workspace_manager.h"
#include <cuda_runtime.h>

namespace compression {
namespace lz77 {

__device__ inline u32 calculate_match_cost(u32 length, u32 offset) {
    if (length == 0) return 1000000;

    u32 offset_bits = (offset == 0) ? 0 : (31 - __clz(static_cast<int>(offset)));
    u32 length_bits = (length >= 19) ? 9 : 5;
    return 1 + length_bits + offset_bits;
}

__device__ inline u32 calculate_literal_cost(u32 num_literals) {
    return num_literals * 8;
}

__device__ inline u32 compute_hash(const u8* data, u32 pos, u32 hash_log) {
    u32 val = (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2];
    return (val * 2654435761U) >> (32 - hash_log);
}

__device__ inline u32 match_length(
    const u8* input,
    u32 p1,
    u32 p2,
    u32 max_len,
    u32 input_size
) {
    u32 len = 0;
    while (len < max_len && p1 + len < input_size && p2 + len < input_size) {
        if (input[p1 + len] != input[p2 + len]) break;
        len++;
    }
    return len;
}

Status find_matches_parallel(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
);

Status compute_optimal_parse(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    const LZ77Config& config,
    cudaStream_t stream
);

Status backtrack_sequences(
    u32 input_size,
    CompressionWorkspace& workspace,
    u32* h_num_sequences,
    cudaStream_t stream
);

} // namespace lz77
} // namespace compression

#endif // LZ77_PARALLEL_H
