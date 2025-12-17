// ============================================================================
// cuda_zstd_types.cpp - Implementation of Core Types
// ============================================================================

#include "cuda_zstd_types.h"
#include "cuda_zstd_lz77.h"
#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_stacktrace.h"
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
// CRITICAL FIX: Use pointer to avoid static initialization heap corruption
// static std::recursive_mutex* g_error_mutex = nullptr; // Global removed

// Helper to get or create mutex (thread-safe lazy initialization)
static std::recursive_mutex &get_error_mutex() {
  static std::recursive_mutex *m = new std::recursive_mutex();
  return *m;
}

const char *status_to_string(Status status) {
  switch (status) {
  case Status::SUCCESS:
    return "SUCCESS";
  case Status::ERROR_GENERIC:
    return "ERROR_GENERIC";
  case Status::ERROR_INVALID_PARAMETER:
    return "ERROR_INVALID_PARAMETER";
  case Status::ERROR_OUT_OF_MEMORY:
    return "ERROR_OUT_OF_MEMORY";
  case Status::ERROR_CUDA_ERROR:
    return "ERROR_CUDA_ERROR";
  case Status::ERROR_INVALID_MAGIC:
    return "ERROR_INVALID_MAGIC";
  case Status::ERROR_CORRUPT_DATA:
    return "ERROR_CORRUPT_DATA";
  case Status::ERROR_BUFFER_TOO_SMALL:
    return "ERROR_BUFFER_TOO_SMALL";
  case Status::ERROR_UNSUPPORTED_VERSION:
    return "ERROR_UNSUPPORTED_VERSION";
  case Status::ERROR_DICTIONARY_MISMATCH:
    return "ERROR_DICTIONARY_MISMATCH";
  case Status::ERROR_CHECKSUM_FAILED:
    return "ERROR_CHECKSUM_FAILED";
  case Status::ERROR_IO:
    return "ERROR_IO";
  case Status::ERROR_COMPRESSION:
    return "ERROR_COMPRESSION";
  case Status::ERROR_DECOMPRESSION:
    return "ERROR_DECOMPRESSION";
  case Status::ERROR_WORKSPACE_INVALID:
    return "ERROR_WORKSPACE_INVALID";
  case Status::ERROR_STREAM_ERROR:
    return "ERROR_STREAM_ERROR";
  case Status::ERROR_ALLOCATION_FAILED:
    return "ERROR_ALLOCATION_FAILED";
  case Status::ERROR_HASH_TABLE_FULL:
    return "ERROR_HASH_TABLE_FULL";
  case Status::ERROR_SEQUENCE_ERROR:
    return "ERROR_SEQUENCE_ERROR";
  default:
    return "UNKNOWN_ERROR";
  }
}

const char *get_detailed_error_message(const ErrorContext &ctx) {
  static thread_local char buffer[512];

  if (ctx.cuda_error != cudaSuccess) {
    snprintf(buffer, sizeof(buffer),
             "%s at %s:%d in %s() - CUDA Error: %s (%d)%s%s",
             status_to_string(ctx.status), ctx.file ? ctx.file : "unknown",
             ctx.line, ctx.function ? ctx.function : "unknown",
             cudaGetErrorString(ctx.cuda_error), (int)ctx.cuda_error,
             ctx.message ? " - " : "", ctx.message ? ctx.message : "");
  } else {
    snprintf(buffer, sizeof(buffer), "%s at %s:%d in %s()%s%s",
             status_to_string(ctx.status), ctx.file ? ctx.file : "unknown",
             ctx.line, ctx.function ? ctx.function : "unknown",
             ctx.message ? " - " : "", ctx.message ? ctx.message : "");
  }

  return buffer;
}

void set_error_callback(ErrorCallback callback) {
  std::lock_guard<std::recursive_mutex> lock(get_error_mutex());
  g_error_callback = callback;
}

void log_error(const ErrorContext &ctx) {
  std::lock_guard<std::recursive_mutex> lock(get_error_mutex());
  g_last_error = ctx;

  // Print to stderr for debugging
  std::cerr << "[ERROR] " << status_to_string(ctx.status);
  if (ctx.file) {
    std::cerr << " at " << ctx.file << ":" << ctx.line;
  }
  if (ctx.function) {
    std::cerr << " in " << ctx.function << "()";
  }
  if (ctx.message) {
    std::cerr << ": " << ctx.message;
  }
  if (ctx.cuda_error != cudaSuccess) {
    std::cerr << " (CUDA: " << cudaGetErrorString(ctx.cuda_error) << ")";
  }
  std::cerr << std::endl;

  if (g_error_callback) {
    g_error_callback(ctx);
  }
}

ErrorContext get_last_error() {
  std::lock_guard<std::recursive_mutex> lock(get_error_mutex());
  return g_last_error;
}

void clear_last_error() {
  std::lock_guard<std::recursive_mutex> lock(get_error_mutex());
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
    config.chain_log = 15; // FIX: Use hash_log instead of 0 to prevent OOB
    config.search_log = 1;
  } else if (level <= 3) {
    config.strategy = Strategy::DFAST;
    config.window_log = 19;
    config.hash_log = 17;
    config.chain_log = 17; // FIX: Use hash_log instead of 0 to prevent OOB
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
    config.search_log = (level == 7)    ? 8
                        : (level == 8)  ? 16
                        : (level == 9)  ? 32
                        : (level == 10) ? 64
                        : (level == 11) ? 128
                                        : 256;
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
  case Strategy::FAST:
    return 1;
  case Strategy::DFAST:
    return 3;
  case Strategy::GREEDY:
    return 5;
  case Strategy::LAZY:
    return 9;
  case Strategy::LAZY2:
    return 13;
  case Strategy::BTLAZY2:
    return 16;
  case Strategy::BTOPT:
    return 19;
  case Strategy::BTULTRA:
    return 22;
  default:
    return 3;
  }
}

Strategy CompressionConfig::level_to_strategy(int level) {
  if (level <= 1)
    return Strategy::FAST;
  if (level <= 3)
    return Strategy::DFAST;
  if (level <= 6)
    return Strategy::GREEDY;
  if (level <= 12)
    return Strategy::LAZY;
  if (level <= 15)
    return Strategy::LAZY2;
  if (level <= 18)
    return Strategy::BTLAZY2;
  if (level <= 20)
    return Strategy::BTOPT;
  return Strategy::BTULTRA;
}

Status CompressionConfig::validate() const {
  if (level < MIN_COMPRESSION_LEVEL || level > MAX_COMPRESSION_LEVEL)
    return Status::ERROR_INVALID_PARAMETER;
  if (window_log < MIN_WINDOW_LOG || window_log > MAX_WINDOW_LOG)
    return Status::ERROR_INVALID_PARAMETER;
  if (block_size < 1024)
    return Status::ERROR_INVALID_PARAMETER;
  return Status::SUCCESS;
}

CompressionConfig CompressionConfig::get_default() {
  return from_level(DEFAULT_COMPRESSION_LEVEL);
}

// ============================================================================
// Workspace Management Functions (NEW)
// ============================================================================

Status allocate_compression_workspace(CompressionWorkspace &workspace,
                                      size_t max_block_size,
                                      const CompressionConfig &config) {
  // Get the global memory pool instance
  memory::MemoryPoolManager &pool = memory::get_global_pool();

  // Calculate required sizes based on max block size
  workspace.hash_table_size = (1u << config.hash_log);
  workspace.chain_table_size = (1u << config.chain_log);
  workspace.max_matches = max_block_size;
  workspace.max_costs = max_block_size + 1;
  workspace.max_sequences = max_block_size / 3;
  workspace.num_blocks = (max_block_size + 255) / 256;

  // Use memory pool for all allocations
  //     std::cerr << "allocate_compression_workspace: allocation d_hash_table
  //     size=" << workspace.hash_table_size * sizeof(u32) << std::endl;
  workspace.d_hash_table = static_cast<u32 *>(
      pool.allocate(workspace.hash_table_size * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_hash_table ptr=" << workspace.d_hash_table << std::endl;
  if (!workspace.d_hash_table)
    return Status::ERROR_OUT_OF_MEMORY;
  // Ensure this is a true device pointer - don't pass a host-fallback pointer
  if (!pool.is_device_pointer(workspace.d_hash_table)) {
    pool.deallocate(workspace.d_hash_table);
    return Status::ERROR_OUT_OF_MEMORY;
  }

  workspace.d_chain_table = static_cast<u32 *>(
      pool.allocate(workspace.chain_table_size * sizeof(u32)));
  if (!workspace.d_chain_table) {
    pool.deallocate(workspace.d_hash_table);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_chain_table)) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    return Status::ERROR_OUT_OF_MEMORY;
  }

  workspace.d_matches =
      pool.allocate(workspace.max_matches * sizeof(lz77::Match));
  if (!workspace.d_matches) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_matches)) {
    // Attempt to migrate a host-fallback pool entry into device memory
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_matches, workspace.max_matches * sizeof(lz77::Match));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(
          workspace.d_matches, workspace.max_matches * sizeof(lz77::Match));
    }
    if (migrated) {
      workspace.d_matches = migrated;
    } else {
      pool.deallocate(workspace.d_hash_table);
      pool.deallocate(workspace.d_chain_table);
      pool.deallocate(workspace.d_matches);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  workspace.d_costs =
      pool.allocate(workspace.max_costs * sizeof(lz77::ParseCost));
  if (!workspace.d_costs) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    pool.deallocate(workspace.d_matches);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_costs)) {
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_costs, workspace.max_costs * sizeof(lz77::ParseCost));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(
          workspace.d_costs, workspace.max_costs * sizeof(lz77::ParseCost));
    }
    if (migrated) {
      workspace.d_costs = migrated;
    } else {
      pool.deallocate(workspace.d_hash_table);
      pool.deallocate(workspace.d_chain_table);
      pool.deallocate(workspace.d_matches);
      pool.deallocate(workspace.d_costs);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  workspace.d_literal_lengths_reverse =
      static_cast<u32 *>(pool.allocate(workspace.max_sequences * sizeof(u32)));
  if (!workspace.d_literal_lengths_reverse) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    pool.deallocate(workspace.d_matches);
    pool.deallocate(workspace.d_costs);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_literal_lengths_reverse)) {
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_literal_lengths_reverse,
        workspace.max_sequences * sizeof(u32));
    if (!migrated) {
      migrated =
          pool.migrate_host_to_device(workspace.d_literal_lengths_reverse,
                                      workspace.max_sequences * sizeof(u32));
    }
    if (migrated) {
      workspace.d_literal_lengths_reverse = static_cast<u32 *>(migrated);
    } else {
      pool.deallocate(workspace.d_hash_table);
      pool.deallocate(workspace.d_chain_table);
      pool.deallocate(workspace.d_matches);
      pool.deallocate(workspace.d_costs);
      pool.deallocate(workspace.d_literal_lengths_reverse);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  workspace.d_match_lengths_reverse =
      static_cast<u32 *>(pool.allocate(workspace.max_sequences * sizeof(u32)));
  if (!workspace.d_match_lengths_reverse) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    pool.deallocate(workspace.d_matches);
    pool.deallocate(workspace.d_costs);
    pool.deallocate(workspace.d_literal_lengths_reverse);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_match_lengths_reverse)) {
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_match_lengths_reverse,
        workspace.max_sequences * sizeof(u32));
    if (!migrated) {
      migrated =
          pool.migrate_host_to_device(workspace.d_match_lengths_reverse,
                                      workspace.max_sequences * sizeof(u32));
    }
    if (migrated) {
      workspace.d_match_lengths_reverse = static_cast<u32 *>(migrated);
    } else {
      pool.deallocate(workspace.d_hash_table);
      pool.deallocate(workspace.d_chain_table);
      pool.deallocate(workspace.d_matches);
      pool.deallocate(workspace.d_costs);
      pool.deallocate(workspace.d_literal_lengths_reverse);
      pool.deallocate(workspace.d_match_lengths_reverse);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  workspace.d_offsets_reverse =
      static_cast<u32 *>(pool.allocate(workspace.max_sequences * sizeof(u32)));
  if (!workspace.d_offsets_reverse) {
    pool.deallocate(workspace.d_hash_table);
    pool.deallocate(workspace.d_chain_table);
    pool.deallocate(workspace.d_matches);
    pool.deallocate(workspace.d_costs);
    pool.deallocate(workspace.d_literal_lengths_reverse);
    pool.deallocate(workspace.d_match_lengths_reverse);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_offsets_reverse)) {
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_offsets_reverse, workspace.max_sequences * sizeof(u32));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(
          workspace.d_offsets_reverse, workspace.max_sequences * sizeof(u32));
    }
    if (migrated) {
      workspace.d_offsets_reverse = static_cast<u32 *>(migrated);
    } else {
      pool.deallocate(workspace.d_offsets_reverse);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  //     std::cerr << "allocate_compression_workspace: allocation d_frequencies
  //     size=" << 256 * sizeof(u32) << std::endl;
  workspace.d_frequencies =
      static_cast<u32 *>(pool.allocate(256 * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_frequencies ptr=" << workspace.d_frequencies << std::endl;
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
  if (!pool.is_device_pointer(workspace.d_frequencies)) {
    //         std::cerr << "allocate_compression_workspace: d_frequencies is
    //         not device memory - attempting pool migration" << std::endl;
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_frequencies, 256 * sizeof(u32));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(workspace.d_frequencies,
                                             256 * sizeof(u32));
    }
    if (migrated) {
      //             std::cerr << "allocate_compression_workspace: migrated
      //             d_frequencies host-fallback to device ptr=" << migrated <<
      //             std::endl;
      workspace.d_frequencies = static_cast<u32 *>(migrated);
    } else {
      //             std::cerr << "allocate_compression_workspace: migration of
      //             d_frequencies failed" << std::endl;
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
  }

  //     std::cerr << "allocate_compression_workspace: allocation d_code_lengths
  //     size=" << max_block_size * sizeof(u32) << std::endl;
  workspace.d_code_lengths =
      static_cast<u32 *>(pool.allocate(max_block_size * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_code_lengths ptr=" << workspace.d_code_lengths << std::endl;
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
  if (!pool.is_device_pointer(workspace.d_code_lengths)) {
    //         std::cerr << "allocate_compression_workspace: d_code_lengths is
    //         not device memory - attempting pool migration" << std::endl;
    // The code lengths size depends on the header; use 256 as a reasonable
    // default
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_code_lengths, 256 * sizeof(u32));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(workspace.d_code_lengths,
                                             256 * sizeof(u32));
    }
    if (migrated) {
      //             std::cerr << "allocate_compression_workspace: migrated
      //             d_code_lengths host-fallback to device ptr=" << migrated <<
      //             std::endl;
      workspace.d_code_lengths = static_cast<u32 *>(migrated);
    } else {
      //             std::cerr << "allocate_compression_workspace: migration of
      //             d_code_lengths failed" << std::endl;
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
  }

  //     std::cerr << "allocate_compression_workspace: allocation d_bit_offsets
  //     size=" << max_block_size * sizeof(u32) << std::endl;
  workspace.d_bit_offsets =
      static_cast<u32 *>(pool.allocate(max_block_size * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_bit_offsets ptr=" << workspace.d_bit_offsets << std::endl;
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
  if (!pool.is_device_pointer(workspace.d_bit_offsets)) {
    //         std::cerr << "allocate_compression_workspace: d_bit_offsets is
    //         not device memory - attempting pool migration" << std::endl;
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_bit_offsets, 2 * workspace.num_blocks * sizeof(u32));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(
          workspace.d_bit_offsets, 2 * workspace.num_blocks * sizeof(u32));
    }
    if (migrated) {
      //             std::cerr << "allocate_compression_workspace: migrated
      //             d_bit_offsets host-fallback to device ptr=" << migrated <<
      //             std::endl;
      workspace.d_bit_offsets = static_cast<u32 *>(migrated);
    } else {
      //             std::cerr << "allocate_compression_workspace: migration of
      //             d_bit_offsets failed" << std::endl;
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
  }

  //     std::cerr << "allocate_compression_workspace: allocation d_block_sums
  //     size=" << workspace.num_blocks * 3 * sizeof(u32) << std::endl;
  workspace.d_block_sums =
      static_cast<u32 *>(pool.allocate(workspace.num_blocks * 3 * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_block_sums ptr=" << workspace.d_block_sums << std::endl;
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
  if (!pool.is_device_pointer(workspace.d_block_sums)) {
    //         std::cerr << "allocate_compression_workspace: d_block_sums is not
    //         device memory - attempting pool migration" << std::endl;
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_block_sums, workspace.num_blocks * 3 * sizeof(u32));
    if (!migrated) {
      migrated = pool.migrate_host_to_device(
          workspace.d_block_sums, workspace.num_blocks * 3 * sizeof(u32));
    }
    if (migrated) {
      //             std::cerr << "allocate_compression_workspace: migrated
      //             d_block_sums host-fallback to device ptr=" << migrated <<
      //             std::endl;
      workspace.d_block_sums = static_cast<u32 *>(migrated);
    } else {
      //             std::cerr << "allocate_compression_workspace: migration of
      //             d_block_sums failed" << std::endl;
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
  }

  //     std::cerr << "allocate_compression_workspace: allocation
  //     d_scanned_block_sums size=" << workspace.num_blocks * 3 * sizeof(u32)
  //     << std::endl;
  workspace.d_scanned_block_sums =
      static_cast<u32 *>(pool.allocate(workspace.num_blocks * 3 * sizeof(u32)));
  //     std::cerr << "allocate_compression_workspace: after allocate
  //     d_scanned_block_sums ptr=" << workspace.d_scanned_block_sums <<
  //     std::endl;
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
  if (!pool.is_device_pointer(workspace.d_scanned_block_sums)) {
    //         std::cerr << "allocate_compression_workspace:
    //         d_scanned_block_sums is not device memory - attempting pool
    //         migration" << std::endl;
    void *migrated = pool.migrate_pool_host_entry_to_device(
        workspace.d_scanned_block_sums, workspace.num_blocks * 3 * sizeof(u32));
    if (!migrated) {
      migrated =
          pool.migrate_host_to_device(workspace.d_scanned_block_sums,
                                      workspace.num_blocks * 3 * sizeof(u32));
    }
    if (migrated) {
      //             std::cerr << "allocate_compression_workspace: migrated
      //             d_scanned_block_sums host-fallback to device ptr=" <<
      //             migrated << std::endl;
      workspace.d_scanned_block_sums = static_cast<u32 *>(migrated);
    } else {
      //             std::cerr << "allocate_compression_workspace: migration of
      //             d_scanned_block_sums failed" << std::endl;
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
      pool.deallocate(workspace.d_scanned_block_sums);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  // Allocate d_lz77_temp (Needed for RLE check)
  // Allocate modest size (e.g. 1024 bytes) as it's just for small flags
  workspace.d_lz77_temp = static_cast<u32 *>(pool.allocate(1024));
  if (!workspace.d_lz77_temp) {
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
    pool.deallocate(workspace.d_scanned_block_sums);
    return Status::ERROR_OUT_OF_MEMORY;
  }
  if (!pool.is_device_pointer(workspace.d_lz77_temp)) {
    void *migrated =
        pool.migrate_pool_host_entry_to_device(workspace.d_lz77_temp, 1024);
    if (!migrated)
      migrated = pool.migrate_host_to_device(workspace.d_lz77_temp, 1024);
    if (migrated) {
      workspace.d_lz77_temp = static_cast<u32 *>(migrated);
    } else {
      // Deallocate all and return error
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
      pool.deallocate(workspace.d_scanned_block_sums);
      pool.deallocate(workspace.d_lz77_temp);
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  // Calculate total size
  workspace.total_size_bytes = workspace.hash_table_size * sizeof(u32) +
                               workspace.chain_table_size * sizeof(u32) +
                               workspace.max_matches * sizeof(lz77::Match) +
                               workspace.max_costs * sizeof(lz77::ParseCost) +
                               workspace.max_sequences * sizeof(u32) * 3 +
                               256 * sizeof(u32) +
                               max_block_size * sizeof(u32) * 2 +
                               workspace.num_blocks * sizeof(u32) * 2 * 3;

  workspace.is_allocated = true;
  return Status::SUCCESS;
}

Status free_compression_workspace(CompressionWorkspace &workspace) {
  if (!workspace.is_allocated) {
    return Status::SUCCESS;
  }

  // Use memory pool for deallocation
  memory::MemoryPoolManager &pool = memory::get_global_pool();

  if (workspace.d_hash_table) {
    //         std::cerr << "free_compression_workspace: dealloc d_hash_table="
    //         << (void*)workspace.d_hash_table << std::endl;
    std::string st = cuda_zstd::util::capture_stacktrace(32);
    //         std::cerr << "free_compression_workspace: stacktrace: \n" << st
    //         << std::endl;
    auto _s = pool.deallocate(workspace.d_hash_table);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_hash_table -> "
    //         << status_to_string(_s) << std::endl;
  }
  if (workspace.d_chain_table) {
    //         std::cerr << "free_compression_workspace: dealloc d_chain_table="
    //         << (void*)workspace.d_chain_table << std::endl;
    auto _s = pool.deallocate(workspace.d_chain_table);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_chain_table ->
    //         " << status_to_string(_s) << std::endl;
  }
  if (workspace.d_matches) {
    //         std::cerr << "free_compression_workspace: dealloc d_matches=" <<
    //         (void*)workspace.d_matches << std::endl;
    auto _s = pool.deallocate(workspace.d_matches);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_matches -> " <<
    //         status_to_string(_s) << std::endl;
  }
  if (workspace.d_costs) {
    //         std::cerr << "free_compression_workspace: dealloc d_costs=" <<
    //         (void*)workspace.d_costs << std::endl;
    auto _s = pool.deallocate(workspace.d_costs);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_costs -> " <<
    //         status_to_string(_s) << std::endl;
  }
  if (workspace.d_literal_lengths_reverse) {
    //         std::cerr << "free_compression_workspace: dealloc
    //         d_literal_lengths_reverse=" <<
    //         (void*)workspace.d_literal_lengths_reverse << std::endl;
    auto _s = pool.deallocate(workspace.d_literal_lengths_reverse);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for
    //         d_literal_lengths_reverse -> " << status_to_string(_s) <<
    //         std::endl;
  }
  if (workspace.d_match_lengths_reverse) {
    //         std::cerr << "free_compression_workspace: dealloc
    //         d_match_lengths_reverse=" <<
    //         (void*)workspace.d_match_lengths_reverse << std::endl;
    auto _s = pool.deallocate(workspace.d_match_lengths_reverse);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for
    //         d_match_lengths_reverse -> " << status_to_string(_s) <<
    //         std::endl;
  }
  if (workspace.d_offsets_reverse) {
    //         std::cerr << "free_compression_workspace: dealloc
    //         d_offsets_reverse=" << (void*)workspace.d_offsets_reverse <<
    //         std::endl;
    auto _s = pool.deallocate(workspace.d_offsets_reverse);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_offsets_reverse
    //         -> " << status_to_string(_s) << std::endl;
  }
  if (workspace.d_frequencies) {
    //         std::cerr << "free_compression_workspace: dealloc d_frequencies="
    //         << (void*)workspace.d_frequencies << std::endl;
    auto _s = pool.deallocate(workspace.d_frequencies);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_frequencies ->
    //         " << status_to_string(_s) << std::endl;
  }
  if (workspace.d_code_lengths) {
    //         std::cerr << "free_compression_workspace: dealloc
    //         d_code_lengths=" << (void*)workspace.d_code_lengths << std::endl;
    auto _s = pool.deallocate(workspace.d_code_lengths);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_code_lengths ->
    //         " << status_to_string(_s) << std::endl;
  }
  if (workspace.d_bit_offsets) {
    //         std::cerr << "free_compression_workspace: dealloc d_bit_offsets="
    //         << (void*)workspace.d_bit_offsets << std::endl;
    auto _s = pool.deallocate(workspace.d_bit_offsets);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_bit_offsets ->
    //         " << status_to_string(_s) << std::endl;
  }
  if (workspace.d_block_sums) {
    //         std::cerr << "free_compression_workspace: dealloc d_block_sums="
    //         << (void*)workspace.d_block_sums << std::endl;
    auto _s = pool.deallocate(workspace.d_block_sums);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for d_block_sums -> "
    //         << status_to_string(_s) << std::endl;
  }
  if (workspace.d_scanned_block_sums) {
    //         std::cerr << "free_compression_workspace: dealloc
    //         d_scanned_block_sums=" << (void*)workspace.d_scanned_block_sums
    //         << std::endl;
    auto _s = pool.deallocate(workspace.d_scanned_block_sums);
    //         if (_s != Status::SUCCESS) std::cerr <<
    //         "free_compression_workspace: dealloc failed for
    //         d_scanned_block_sums -> " << status_to_string(_s) << std::endl;
  }

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

// ============================================================================
// Block Size and Level Parameter Utilities (STUBS)
// ============================================================================

size_t estimate_compressed_size(size_t uncompressed_size,
                                int compression_level) {
  // Conservative estimate: worst case is input_size + overhead
  // Zstandard overhead is roughly 0.4% for incompressible data
  (void)compression_level; // Unused
  // Increased overhead to 512 to account for CustomMetadataFrame and
  // SkippableFrameHeader
  return uncompressed_size + (uncompressed_size / 255) + 512;
}

Status validate_config(const CompressionConfig &config) {
  // Delegate to the config's own validate method
  return config.validate();
}

void apply_level_parameters(CompressionConfig &config) {
  // Stub: Level parameters are already applied in from_level()
  // This function is called by DefaultZstdManager but doesn't need to do
  // anything since config is already populated with level-appropriate values
}

// ============================================================================
// Optimal Configuration (Based on Comprehensive Benchmarks)
// ============================================================================

CompressionConfig CompressionConfig::optimal(size_t input_size) {
  // Start with default level 3
  CompressionConfig config = CompressionConfig::from_level(3);

  // 1. CPU Threshold: 1MB
  // Benchmarks show CPU is competitive/faster for small payloads due to launch
  // overhead
  config.cpu_threshold = 1024 * 1024;

  // 2. Block Size Selection (Hybrid Formula)
  // Based on benchmark results, the "Hybrid" approach yields the best
  // consistency across all input sizes by balancing block size with
  // parallelism. Logic: Target 64-256 blocks for optimal GPU occupancy, but
  // scale block size with input size to avoid excessive overhead.

  if (input_size <= 256 * 1024) {
    // Small inputs: Use input size (clamped to 256KB)
    // This matches the "Fixed_256KB" / "Fixed_Input" win for small files
    config.block_size = 256 * 1024;
  } else {
    // Large inputs: Use Hybrid logic
    // Ideal block size grows with sqrt of input
    double ideal_bs = std::sqrt((double)input_size) * 400.0;

    // Calculate target number of blocks
    size_t target_blocks = (size_t)(input_size / ideal_bs);

    // Clamp target blocks to keep parallelism in sweet spot (64-256
    // streams/blocks)
    if (target_blocks < 64)
      target_blocks = 64;
    if (target_blocks > 256)
      target_blocks = 256;

    // Calculate resulting block size
    u32 block_size = (u32)(input_size / target_blocks);

    // Round to nearest power of 2 for alignment/efficiency
    // (Simple bit manipulation to find next power of 2)
    block_size--;
    block_size |= block_size >> 1;
    block_size |= block_size >> 2;
    block_size |= block_size >> 4;
    block_size |= block_size >> 8;
    block_size |= block_size >> 16;
    block_size++;

    // Clamp to valid range (128KB - 8MB)
    if (block_size < 128 * 1024)
      block_size = 128 * 1024;
    if (block_size > 8 * 1024 * 1024)
      block_size = 8 * 1024 * 1024;

    config.block_size = block_size;
  }

  // 3. Parallelism
  // The manager handles parallelism via streams.
  // For large inputs, the 512KB block size allows good parallelism.

  return config;
}

u32 get_optimal_block_size(u32 input_size, u32 compression_level) {
  // Use the same Hybrid logic as CompressionConfig::optimal
  (void)compression_level;

  if (input_size <= 256 * 1024) {
    return 256 * 1024;
  } else {
    double ideal_bs = std::sqrt((double)input_size) * 400.0;
    size_t target_blocks = (size_t)(input_size / ideal_bs);

    if (target_blocks < 64)
      target_blocks = 64;
    if (target_blocks > 256)
      target_blocks = 256;

    u32 block_size = (u32)(input_size / target_blocks);

    // Next power of 2
    block_size--;
    block_size |= block_size >> 1;
    block_size |= block_size >> 2;
    block_size |= block_size >> 4;
    block_size |= block_size >> 8;
    block_size |= block_size >> 16;
    block_size++;

    if (block_size < 128 * 1024)
      block_size = 128 * 1024;
    if (block_size > 8 * 1024 * 1024)
      block_size = 8 * 1024 * 1024;

    return block_size;
  }
}

} // namespace cuda_zstd
