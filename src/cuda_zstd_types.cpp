// ============================================================================
// cuda_zstd_types.cpp - Implementation of Core Types
// ============================================================================

#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_lz77.h"
#include <cstdio>
#include <mutex>

namespace cuda_zstd {

// ============================================================================
// Status String Conversion
// ============================================================================

// ============================================================================
// Enhanced Error Handling Implementation
// ============================================================================

static ErrorCallback g_error_callback = nullptr;
static ErrorContext g_last_error;
static std::mutex g_error_mutex;

const char* status_to_string(Status status) {
    switch (status) {
        case Status::SUCCESS: return "SUCCESS";
        case Status::ERROR_GENERIC: return "ERROR_GENERIC";
        case Status::ERROR_INVALID_PARAMETER: return "ERROR_INVALID_PARAMETER";
        case Status::ERROR_OUT_OF_MEMORY: return "ERROR_OUT_OF_MEMORY";
        case Status::ERROR_CUDA_ERROR: return "ERROR_CUDA_ERROR";
        case Status::ERROR_INVALID_MAGIC: return "ERROR_INVALID_MAGIC";
        case Status::ERROR_CORRUPT_DATA: return "ERROR_CORRUPT_DATA";
        case Status::ERROR_BUFFER_TOO_SMALL: return "ERROR_BUFFER_TOO_SMALL";
        case Status::ERROR_UNSUPPORTED_VERSION: return "ERROR_UNSUPPORTED_VERSION";
        case Status::ERROR_DICTIONARY_MISMATCH: return "ERROR_DICTIONARY_MISMATCH";
        case Status::ERROR_CHECKSUM_FAILED: return "ERROR_CHECKSUM_FAILED";
        case Status::ERROR_IO: return "ERROR_IO";
        case Status::ERROR_COMPRESSION: return "ERROR_COMPRESSION";
        case Status::ERROR_DECOMPRESSION: return "ERROR_DECOMPRESSION";
        case Status::ERROR_WORKSPACE_INVALID: return "ERROR_WORKSPACE_INVALID";
        case Status::ERROR_STREAM_ERROR: return "ERROR_STREAM_ERROR";
        case Status::ERROR_ALLOCATION_FAILED: return "ERROR_ALLOCATION_FAILED";
        case Status::ERROR_HASH_TABLE_FULL: return "ERROR_HASH_TABLE_FULL";
        case Status::ERROR_SEQUENCE_ERROR: return "ERROR_SEQUENCE_ERROR";
        default: return "UNKNOWN_ERROR";
    }
}

const char* get_detailed_error_message(const ErrorContext& ctx) {
    static thread_local char buffer[512];
    
    if (ctx.cuda_error != cudaSuccess) {
        snprintf(buffer, sizeof(buffer),
                 "%s at %s:%d in %s() - CUDA Error: %s (%d)%s%s",
                 status_to_string(ctx.status),
                 ctx.file ? ctx.file : "unknown",
                 ctx.line,
                 ctx.function ? ctx.function : "unknown",
                 cudaGetErrorString(ctx.cuda_error),
                 ctx.cuda_error,
                 ctx.message ? " - " : "",
                 ctx.message ? ctx.message : "");
    } else {
        snprintf(buffer, sizeof(buffer),
                 "%s at %s:%d in %s()%s%s",
                 status_to_string(ctx.status),
                 ctx.file ? ctx.file : "unknown",
                 ctx.line,
                 ctx.function ? ctx.function : "unknown",
                 ctx.message ? " - " : "",
                 ctx.message ? ctx.message : "");
    }
    
    return buffer;
}

void set_error_callback(ErrorCallback callback) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_error_callback = callback;
}

void log_error(const ErrorContext& ctx) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = ctx;
    
    if (g_error_callback) {
        g_error_callback(ctx);
    }
}

ErrorContext get_last_error() {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_error;
}

void clear_last_error() {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = ErrorContext();
}

// ============================================================================
// CompressionConfig Implementation
// ============================================================================

CompressionConfig CompressionConfig::from_level(int level) {
    CompressionConfig config;
    config.compression_mode = CompressionMode::LEVEL_BASED;
    config.level = level;
    config.use_exact_level = true;
    
    // Map level to parameters
    if (level <= 1) {
        config.strategy = Strategy::FAST;
        config.window_log = 18;
        config.hash_log = 15;
        config.chain_log = 0;
        config.search_log = 1;
    } else if (level <= 3) {
        config.strategy = Strategy::DFAST;
        config.window_log = 19;
        config.hash_log = 17;
        config.chain_log = 0;
        config.search_log = 1;
    } else if (level <= 6) {
        config.strategy = Strategy::GREEDY;
        config.window_log = 20;
        config.hash_log = 17;
        config.chain_log = 17;
        config.search_log = (level == 4) ? 2 : (level == 5) ? 4 : 8;
    } else if (level <= 12) {
        config.strategy = Strategy::LAZY;
        config.window_log = (level <= 9) ? 22 : 23;
        config.hash_log = (level <= 9) ? 18 : 19;
        config.chain_log = (level <= 9) ? 18 : 19;
        config.search_log = (level == 7) ? 8 : (level == 8) ? 16 : 
                           (level == 9) ? 32 : (level == 10) ? 64 : 
                           (level == 11) ? 128 : 256;
    } else if (level <= 15) {
        config.strategy = Strategy::LAZY2;
        config.window_log = 23;
        config.hash_log = (level <= 14) ? 19 : 20;
        config.chain_log = 19;
        config.search_log = (level <= 14) ? 256 : 512;
    } else if (level <= 18) {
        config.strategy = Strategy::BTLAZY2;
        config.window_log = 23;
        config.hash_log = 20;
        config.chain_log = 20;
        config.search_log = (level <= 17) ? 512 : 999;
    } else if (level <= 20) {
        config.strategy = Strategy::BTOPT;
        config.window_log = 23;
        config.hash_log = 20;
        config.chain_log = 20;
        config.search_log = 999;
    } else {
        config.strategy = Strategy::BTULTRA;
        config.window_log = 23;
        config.hash_log = 20;
        config.chain_log = 20;
        config.search_log = 999;
    }
    
    return config;
}

int CompressionConfig::strategy_to_default_level(Strategy s) {
    switch (s) {
        case Strategy::FAST: return 1;
        case Strategy::DFAST: return 3;
        case Strategy::GREEDY: return 5;
        case Strategy::LAZY: return 9;
        case Strategy::LAZY2: return 13;
        case Strategy::BTLAZY2: return 16;
        case Strategy::BTOPT: return 19;
        case Strategy::BTULTRA: return 22;
        default: return 3;
    }
}

Strategy CompressionConfig::level_to_strategy(int level) {
    if (level <= 1) return Strategy::FAST;
    if (level <= 3) return Strategy::DFAST;
    if (level <= 6) return Strategy::GREEDY;
    if (level <= 12) return Strategy::LAZY;
    if (level <= 15) return Strategy::LAZY2;
    if (level <= 18) return Strategy::BTLAZY2;
    if (level <= 20) return Strategy::BTOPT;
    return Strategy::BTULTRA;
}

Status CompressionConfig::validate() const {
    if (level < MIN_COMPRESSION_LEVEL || level > MAX_COMPRESSION_LEVEL) return Status::ERROR_INVALID_PARAMETER;
    if (window_log < MIN_WINDOW_LOG || window_log > MAX_WINDOW_LOG) return Status::ERROR_INVALID_PARAMETER;
    if (block_size < 1024) return Status::ERROR_INVALID_PARAMETER;
    return Status::SUCCESS;
}

CompressionConfig CompressionConfig::get_default() {
    return from_level(DEFAULT_COMPRESSION_LEVEL);
}

// ============================================================================
// Workspace Management Functions (NEW)
// ============================================================================

Status allocate_compression_workspace(
    CompressionWorkspace& workspace,
    size_t max_block_size,
    const CompressionConfig& config
) {
    // Get the global memory pool instance
    memory::MemoryPoolManager& pool = memory::get_global_pool();
    
    // Calculate required sizes based on max block size
    workspace.hash_table_size = (1u << config.hash_log);
    workspace.chain_table_size = (1u << config.chain_log);
    workspace.max_matches = max_block_size;
    workspace.max_costs = max_block_size + 1;
    workspace.max_sequences = max_block_size / 3;
    workspace.num_blocks = (max_block_size + 255) / 256;
    
    // Use memory pool for all allocations
    workspace.d_hash_table = static_cast<u32*>(
        pool.allocate(workspace.hash_table_size * sizeof(u32)));
    if (!workspace.d_hash_table) return Status::ERROR_OUT_OF_MEMORY;
    
    workspace.d_chain_table = static_cast<u32*>(
        pool.allocate(workspace.chain_table_size * sizeof(u32)));
    if (!workspace.d_chain_table) {
        pool.deallocate(workspace.d_hash_table);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_matches = pool.allocate(workspace.max_matches * sizeof(lz77::Match));
    if (!workspace.d_matches) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_costs = pool.allocate(workspace.max_costs * sizeof(lz77::ParseCost));
    if (!workspace.d_costs) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_literal_lengths_reverse = static_cast<u32*>(
        pool.allocate(workspace.max_sequences * sizeof(u32)));
    if (!workspace.d_literal_lengths_reverse) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_match_lengths_reverse = static_cast<u32*>(
        pool.allocate(workspace.max_sequences * sizeof(u32)));
    if (!workspace.d_match_lengths_reverse) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_offsets_reverse = static_cast<u32*>(
        pool.allocate(workspace.max_sequences * sizeof(u32)));
    if (!workspace.d_offsets_reverse) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_frequencies = static_cast<u32*>(
        pool.allocate(256 * sizeof(u32)));
    if (!workspace.d_frequencies) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        pool.deallocate(workspace.d_offsets_reverse);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_code_lengths = static_cast<u32*>(
        pool.allocate(max_block_size * sizeof(u32)));
    if (!workspace.d_code_lengths) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        pool.deallocate(workspace.d_offsets_reverse);
        pool.deallocate(workspace.d_frequencies);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_bit_offsets = static_cast<u32*>(
        pool.allocate(max_block_size * sizeof(u32)));
    if (!workspace.d_bit_offsets) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        pool.deallocate(workspace.d_offsets_reverse);
        pool.deallocate(workspace.d_frequencies);
        pool.deallocate(workspace.d_code_lengths);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_block_sums = static_cast<u32*>(
        pool.allocate(workspace.num_blocks * sizeof(u32)));
    if (!workspace.d_block_sums) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        pool.deallocate(workspace.d_offsets_reverse);
        pool.deallocate(workspace.d_frequencies);
        pool.deallocate(workspace.d_code_lengths);
        pool.deallocate(workspace.d_bit_offsets);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    workspace.d_scanned_block_sums = static_cast<u32*>(
        pool.allocate(workspace.num_blocks * sizeof(u32)));
    if (!workspace.d_scanned_block_sums) {
        pool.deallocate(workspace.d_hash_table);
        pool.deallocate(workspace.d_chain_table);
        pool.deallocate(workspace.d_matches);
        pool.deallocate(workspace.d_costs);
        pool.deallocate(workspace.d_literal_lengths_reverse);
        pool.deallocate(workspace.d_match_lengths_reverse);
        pool.deallocate(workspace.d_offsets_reverse);
        pool.deallocate(workspace.d_frequencies);
        pool.deallocate(workspace.d_code_lengths);
        pool.deallocate(workspace.d_bit_offsets);
        pool.deallocate(workspace.d_block_sums);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Calculate total size
    workspace.total_size_bytes =
        workspace.hash_table_size * sizeof(u32) +
        workspace.chain_table_size * sizeof(u32) +
        workspace.max_matches * sizeof(lz77::Match) +
        workspace.max_costs * sizeof(lz77::ParseCost) +
        workspace.max_sequences * sizeof(u32) * 3 +
        256 * sizeof(u32) +
        max_block_size * sizeof(u32) * 2 +
        workspace.num_blocks * sizeof(u32) * 2;
    
    workspace.is_allocated = true;
    return Status::SUCCESS;
}

Status free_compression_workspace(CompressionWorkspace& workspace) {
    if (!workspace.is_allocated) {
        return Status::SUCCESS;
    }
    
    // Use memory pool for deallocation
    memory::MemoryPoolManager& pool = memory::get_global_pool();
    
    if (workspace.d_hash_table) pool.deallocate(workspace.d_hash_table);
    if (workspace.d_chain_table) pool.deallocate(workspace.d_chain_table);
    if (workspace.d_matches) pool.deallocate(workspace.d_matches);
    if (workspace.d_costs) pool.deallocate(workspace.d_costs);
    if (workspace.d_literal_lengths_reverse) pool.deallocate(workspace.d_literal_lengths_reverse);
    if (workspace.d_match_lengths_reverse) pool.deallocate(workspace.d_match_lengths_reverse);
    if (workspace.d_offsets_reverse) pool.deallocate(workspace.d_offsets_reverse);
    if (workspace.d_frequencies) pool.deallocate(workspace.d_frequencies);
    if (workspace.d_code_lengths) pool.deallocate(workspace.d_code_lengths);
    if (workspace.d_bit_offsets) pool.deallocate(workspace.d_bit_offsets);
    if (workspace.d_block_sums) pool.deallocate(workspace.d_block_sums);
    if (workspace.d_scanned_block_sums) pool.deallocate(workspace.d_scanned_block_sums);
    
    // Reset all pointers and metadata
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
    workspace.total_size_bytes = 0;
    
    return Status::SUCCESS;
}

} // namespace cuda_zstd
