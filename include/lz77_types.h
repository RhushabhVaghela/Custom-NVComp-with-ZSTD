// ==============================================================================
// lz77_types.h - LZ77 compression structures
// ==============================================================================

#ifndef LZ77_TYPES_H
#define LZ77_TYPES_H

#include "common_types.h"

namespace compression {
namespace lz77 {

// Match structure
struct Match {
    u32 position;         // Position in input
    u32 offset;          // Match offset (distance back)
    u32 length;          // Match length
    u32 literal_length;  // Literals before this match

    __host__ __device__ Match() 
        : position(0), offset(0), length(0), literal_length(0) {}

    __host__ __device__ Match(u32 pos, u32 off, u32 len, u32 lit_len)
        : position(pos), offset(off), length(len), literal_length(lit_len) {}
};

// Parse cost for optimal parsing (Dynamic Programming)
// V2: Packed u64 format for atomic operations
struct ParseCost {
    unsigned long long data;

    __host__ __device__ ParseCost() : data(0xFFFFFFFFFFFFFFFFULL) {}
    
    __host__ __device__ void set(u32 cost, u32 parent_pos) {
        data = ((unsigned long long)cost << 32) | parent_pos;
    }
    
    __host__ __device__ u32 cost() const {
        return (u32)(data >> 32);
    }
    
    __host__ __device__ u32 parent() const {
        return (u32)(data & 0xFFFFFFFF);
    }
};

// LZ77 configuration
struct LZ77Config {
    u32 window_log = 20;      // Window size = 2^window_log
    u32 hash_log = 17;        // Hash table size = 2^hash_log
    u32 chain_log = 17;       // Chain table size = 2^chain_log
    u32 search_depth = 8;     // Search depth for matching
    u32 min_match = 3;        // Minimum match length
    u32 good_length = 32;     // Good match length
    u32 nice_length = 128;    // Nice match length
};

} // namespace lz77
} // namespace compression

#endif // LZ77_TYPES_H
