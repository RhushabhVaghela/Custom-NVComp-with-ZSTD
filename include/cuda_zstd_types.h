// ============================================================================
// cuda_zstd_types.h - Core Types and Definitions
// ============================================================================

#ifndef CUDA_ZSTD_TYPES_H_FIXED_
#define CUDA_ZSTD_TYPES_H_FIXED_

#include <stdint.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuda_zstd_primitives.h"
#include <cuda_runtime.h>

namespace cuda_zstd {

// ============================================================================
// Basic Types
// ============================================================================

// Basic Types defined in primitives.h

// ============================================================================
// Hash Structures
// ============================================================================
namespace hash {
struct HashTable {
  u32 *table;
  u32 size;
};

struct ChainTable {
  u32 *prev;
  u32 size;
};
} // namespace hash

// ============================================================================
// FSE Context (Memory Optimization)
// ============================================================================
struct FSEContext {
  /** Encoding Table Buffers (Device Pointers) */
  void *d_dev_symbol_table = nullptr;
  void *d_dev_next_state = nullptr;
  void *d_dev_nbBits_table = nullptr;
  void *d_dev_next_state_vals = nullptr;
  void *d_dev_initial_states = nullptr;
  void *d_ctable_for_encoder = nullptr;

  /** Parallel Chunk Buffers (Device Pointers) */
  void *d_chunk_start_states = nullptr;
  void *d_bitstreams = nullptr;
  void *d_chunk_bit_counts = nullptr;
  void *d_chunk_offsets = nullptr;

  /** Capacity Tracking */
  size_t bitstreams_capacity_bytes = 0;
  size_t num_chunks_capacity = 0;
  size_t symbol_table_capacity = 0; // Track max_symbol+1 capacity
};


// ============================================================================
// Bitstream Format Selection (nvcomp-style)
// ============================================================================

/**
 * @brief Selects bitstream format for compression output.
 *
 * NATIVE: GPU-optimized format with raw u32 sequence arrays.
 *         Fastest for GPU-to-GPU pipelines. NOT compatible with official zstd.
 *
 * RAW:    RFC 8878 compliant format with FSE-encoded sequences.
 *         Compatible with official zstd decoder. Slightly slower.
 */
enum class BitstreamKind : u32 {
  NATIVE = 0, // GPU-optimized, internal use only
  RAW = 1     // RFC 8878 compatible, interoperable with official zstd
};

// ============================================================================
// Enhanced Status Codes with Context
// ============================================================================

/**
 * @brief Status codes for CUDA-ZSTD operations.
 */
enum class Status : u32 {
  SUCCESS = 0,                     /**< Operation completed successfully */
  ERROR_GENERIC = 1,               /**< General error */
  ERROR_INVALID_PARAMETER = 2,     /**< Invalid argument passed to function */
  ERROR_OUT_OF_MEMORY = 3,         /**< Memory allocation failed */
  ERROR_CUDA_ERROR = 4,            /**< CUDA runtime error occurred */
  ERROR_INVALID_MAGIC = 5,         /**< Invalid ZSTD magic number */
  ERROR_CORRUPT_DATA = 6,          /**< Input data is corrupted or malformed */
  /** @deprecated Use ERROR_CORRUPT_DATA instead */
  ERROR_CORRUPTED_DATA [[deprecated("Use ERROR_CORRUPT_DATA instead")]] = 6,
  ERROR_BUFFER_TOO_SMALL = 7,      /**< Provided output buffer is too small */
  ERROR_UNSUPPORTED_VERSION = 8,   /**< Unsupported ZSTD format version */
  ERROR_DICTIONARY_MISMATCH = 9,   /**< Dictionary does not match data */
  ERROR_CHECKSUM_FAILED = 10,      /**< Frame checksum verification failed */
  ERROR_IO = 11,                   /**< I/O operation failed */
  ERROR_COMPRESSION = 12,          /**< Error occurred during compression */
  /** @deprecated Use ERROR_COMPRESSION instead */
  ERROR_COMPRESSION_FAILED [[deprecated("Use ERROR_COMPRESSION instead")]] = 12,
  ERROR_DECOMPRESSION = 13,        /**< Error occurred during decompression */
  /** @deprecated Use ERROR_DECOMPRESSION instead */
  ERROR_DECOMPRESSION_FAILED [[deprecated("Use ERROR_DECOMPRESSION instead")]] = 13,
  ERROR_WORKSPACE_INVALID = 14,    /**< Temporary workspace is invalid or too small */
  ERROR_STREAM_ERROR = 15,         /**< CUDA stream error */
  ERROR_ALLOCATION_FAILED = 16,    /**< Resource allocation failed */
  ERROR_HASH_TABLE_FULL = 17,      /**< Internal hash table limit reached */
  ERROR_SEQUENCE_ERROR = 18,       /**< Invalid match sequences encountered */
  ERROR_NOT_INITIALIZED = 19,      /**< Manager or context not initialized */
  ERROR_ALREADY_INITIALIZED = 20,  /**< Resource already initialized */
  ERROR_INVALID_STATE = 21,        /**< Operation invalid in current state */
  ERROR_TIMEOUT = 22,              /**< Operation timed out */
  ERROR_CANCELLED = 23,            /**< Operation was cancelled */
  ERROR_NOT_IMPLEMENTED = 24,      /**< Requested feature not yet implemented */
  ERROR_INTERNAL = 25,             /**< Internal consistency check failed */
  ERROR_UNKNOWN = 26,              /**< Unknown error occurred */
  ERROR_DICTIONARY_FAILED = 27,    /**< Dictionary training or parsing failed */
  ERROR_UNSUPPORTED_FORMAT = 28    /**< Unsupported frame or block format */
};


// Enhanced error context
struct ErrorContext {
  Status status = Status::SUCCESS;
  const char *file = nullptr;
  int line = 0;
  const char *function = nullptr;
  const char *message = nullptr;
  cudaError_t cuda_error = cudaSuccess;

  ErrorContext() = default;
  ErrorContext(Status s, const char *f, int l, const char *fn,
               const char *msg = nullptr)
      : status(s), file(f), line(l), function(fn), message(msg) {}
};

const char *status_to_string(Status status);
const char *get_detailed_error_message(const ErrorContext &ctx);

// Error callback type
typedef void (*ErrorCallback)(const ErrorContext &ctx);

// Global error handling
void set_error_callback(ErrorCallback callback);
void log_error(const ErrorContext &ctx);
ErrorContext get_last_error();
void clear_last_error();

// ============================================================================
// Compression Strategy
// ============================================================================

enum class Strategy : u32 {
  FAST = 0,    // Level 1
  DFAST = 1,   // Levels 2-3
  GREEDY = 2,  // Levels 4-6
  LAZY = 3,    // Levels 7-12
  LAZY2 = 4,   // Levels 13-15
  BTLAZY2 = 5, // Levels 16-18
  BTOPT = 6,   // Levels 19-20
  BTULTRA = 7  // Levels 21-22
};

// ============================================================================
// Compression Mode
// ============================================================================

enum class CompressionMode : u32 {
  LEVEL_BASED = 0,   // Use exact level (1-22)
  STRATEGY_BASED = 1 // Use strategy (library picks level)
};

// ============================================================================
// Checksum Policy
// ============================================================================

enum class ChecksumPolicy : u32 {
  NO_COMPUTE_NO_VERIFY = 0,
  COMPUTE_NO_VERIFY = 1,
  COMPUTE_AND_VERIFY = 2
};

// ============================================================================
// Compression Configuration
// ============================================================================

struct CompressionConfig {
  // Compression mode selection
  CompressionMode compression_mode = CompressionMode::LEVEL_BASED;

  // Level-based mode (NEW)
  int level = 3;               // 1-22 for exact control
  bool use_exact_level = true; // Force exact level parameters

  // Strategy-based mode (Original)
  Strategy strategy = Strategy::GREEDY; // Let strategy pick level

  // Advanced parameters (override level defaults if set)
  u32 window_log = 20;         // Window size = 1 << window_log
  u32 hash_log = 17;           // Hash table size
  u32 chain_log = 17;          // Chain table size
  u32 search_log = 8;          // Search depth
  u32 min_match = 3;           // Minimum match length (3-7)
  u32 target_length = 0;       // Target match length
  u32 block_size = 128 * 1024; // Compression block size
  // NOTE: Long Distance Matching (LDM) is supported when enabled.
  // LDM adds a separate match-finding path for large-window compression.
  bool enable_ldm = false;
  u32 ldm_hash_log = 20;
  ChecksumPolicy checksum = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;

  // Smart Router Configuration
  u32 cpu_threshold = 1024 * 1024; // 1MB default (based on benchmarks)

  // Helper functions
  static CompressionConfig from_level(int level);
  static CompressionConfig
  optimal(size_t input_size); // NEW: Get optimal config based on benchmarks
  static int strategy_to_default_level(Strategy s);
  static Strategy level_to_strategy(int level);
  Status validate() const;
  static CompressionConfig get_default();
};

// ============================================================================
// Compression Statistics
// ============================================================================

struct CompressionStats {
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t num_blocks = 0;
  uint64_t num_sequences = 0;
  uint64_t num_literals = 0;
  uint64_t matches_found = 0;
  uint64_t bytes_compressed = 0;
  uint64_t bytes_produced = 0;
  uint64_t bytes_decompressed = 0;
  uint64_t blocks_processed = 0;
  double compression_time_ms = 0.0;
  double decompression_time_ms = 0.0;

  float get_ratio() const {
    return output_bytes > 0 ? static_cast<float>(input_bytes) / output_bytes
                            : 0.0f;
  }

  double get_compression_throughput_gbps() const {
    return compression_time_ms > 0
               ? (input_bytes / 1e9) / (compression_time_ms / 1000.0)
               : 0.0;
  }
};

// ============================================================================
// Batch Processing
// ============================================================================

struct BatchItem {
  void *input_ptr = nullptr;
  void *output_ptr = nullptr;
  size_t input_size = 0;
  size_t output_size = 0;
  Status status = Status::SUCCESS;
};

// ============================================================================
// Dictionary Structures
// ============================================================================

struct DictionaryContent {
  const unsigned char *d_buffer = nullptr;
  size_t size = 0;
  u32 dict_id = 0;
};

// ============================================================================
// NVCOMP Metadata
// ============================================================================

struct NvcompMetadata {
  u32 format_version = 0;
  u32 compression_level = 0;
  u64 uncompressed_size = 0;  // u64 to support files >4GB without truncation
  u32 num_chunks = 0;
  u32 chunk_size = 0;
  u32 dictionary_id = 0;
  ChecksumPolicy checksum_policy = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
};

// ============================================================================
// Constants
// ============================================================================

constexpr u32 ZSTD_MAGIC = 0xFD2FB528;
constexpr u32 MIN_COMPRESSION_LEVEL = 1;
constexpr u32 MAX_COMPRESSION_LEVEL = 22;
constexpr u32 DEFAULT_COMPRESSION_LEVEL = 3;
constexpr u32 MIN_WINDOW_LOG = 10;
constexpr u32 MAX_WINDOW_LOG = 31;
constexpr u32 DEFAULT_BLOCK_SIZE = 128 * 1024;

// ============================================================================
// Hybrid CPU/GPU Execution System
// ============================================================================

/**
 * @brief Controls how the hybrid system routes operations between CPU and GPU.
 *
 * The hybrid system automatically selects the best execution path based on
 * data characteristics, location, and available resources. Users can override
 * the automatic selection with explicit mode preferences.
 */
enum class HybridMode : u32 {
  AUTO = 0,        ///< System decides best path based on heuristics (default)
  PREFER_CPU = 1,  ///< CPU unless data is already on GPU and output stays on GPU
  PREFER_GPU = 2,  ///< GPU unless data is on host and result stays on host
  FORCE_CPU = 3,   ///< Always use CPU path (libzstd)
  FORCE_GPU = 4,   ///< Always use GPU path (CUDA kernels)
  ADAPTIVE = 5     ///< Learn from profiling data to improve routing over time
};

/**
 * @brief Describes where data resides in the memory hierarchy.
 */
enum class DataLocation : u32 {
  HOST = 0,       ///< System RAM (host memory)
  DEVICE = 1,     ///< GPU VRAM (device memory)
  MANAGED = 2,    ///< CUDA managed/unified memory
  UNKNOWN = 3     ///< Auto-detect via cudaPointerGetAttributes
};

/**
 * @brief Which execution backend was used for an operation.
 */
enum class ExecutionBackend : u32 {
  CPU_LIBZSTD = 0,     ///< CPU path using reference libzstd
  GPU_KERNELS = 1,     ///< GPU path using CUDA kernels
  CPU_PARALLEL = 2,    ///< CPU with multi-threaded libzstd (batch)
  GPU_BATCH = 3        ///< GPU batch processing (multiple items)
};

/**
 * @brief Configuration for the hybrid CPU/GPU routing engine.
 *
 * Controls thresholds, preferences, and profiling behavior for the
 * automatic routing system. Default values are tuned for RTX 5080-class GPUs.
 */
struct HybridConfig {
  /// Routing mode
  HybridMode mode = HybridMode::AUTO;

  /// Size threshold below which CPU is always preferred (bytes).
  /// Default 1MB — GPU overhead exceeds benefit for small data.
  size_t cpu_size_threshold = 1024 * 1024;

  /// Size threshold above which data-on-device stays on GPU (bytes).
  /// Below this, it may be cheaper to transfer to host for CPU processing.
  size_t gpu_device_threshold = 64 * 1024;

  /// Enable profiling to track per-call timing for adaptive mode.
  bool enable_profiling = false;

  /// Compression level (1-22) passed through to both CPU and GPU backends.
  int compression_level = 3;

  /// Number of parallel threads for CPU batch operations.
  /// 0 = auto-detect (hardware_concurrency).
  u32 cpu_thread_count = 0;

  /// If true, use pinned (page-locked) host memory for transfers.
  bool use_pinned_memory = true;

  /// If true, overlap CPU computation with GPU transfers where possible.
  bool overlap_transfers = true;
};

/**
 * @brief Result metadata returned after a hybrid compress/decompress operation.
 *
 * Provides timing breakdown and path selection information for diagnostics.
 */
struct HybridResult {
  /// Which backend was used
  ExecutionBackend backend_used = ExecutionBackend::CPU_LIBZSTD;

  /// Where the input data was located
  DataLocation input_location = DataLocation::HOST;

  /// Where the output data was placed
  DataLocation output_location = DataLocation::HOST;

  /// Total wall-clock time for the operation (milliseconds)
  double total_time_ms = 0.0;

  /// Time spent transferring data between host and device (milliseconds)
  double transfer_time_ms = 0.0;

  /// Time spent in actual compute (compression/decompression) (milliseconds)
  double compute_time_ms = 0.0;

  /// Throughput achieved (MB/s, based on uncompressed data size)
  double throughput_mbps = 0.0;

  /// Input size processed (bytes)
  size_t input_bytes = 0;

  /// Output size produced (bytes)
  size_t output_bytes = 0;

  /// Compression ratio (input/output, >1 means compressed)
  float compression_ratio = 1.0f;

  /// Reason the routing decision was made (for diagnostics)
  const char *routing_reason = nullptr;
};

/**
 * @brief Per-item routing result for batch operations.
 */
struct BatchRoutingResult {
  size_t item_index = 0;
  ExecutionBackend backend_used = ExecutionBackend::CPU_LIBZSTD;
  Status status = Status::SUCCESS;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  double compute_time_ms = 0.0;
};

// ============================================================================
// Memory Safety Buffer Constants
// ============================================================================
// These constants define the minimum amount of free memory that must remain
// after any allocation. They prevent WSL crashes and system instability by
// ensuring the OS/GPU driver always has enough headroom.

/** VRAM safety buffer: 256 MB must remain free after every GPU allocation */
constexpr size_t VRAM_SAFETY_BUFFER_BYTES = 256ULL * 1024 * 1024;

/** RAM safety buffer: 512 MB must remain free after every host allocation */
constexpr size_t RAM_SAFETY_BUFFER_BYTES = 512ULL * 1024 * 1024;

// ============================================================================
// Compression Workspace (NEW - for workspace-based memory management)
// ============================================================================

struct CompressionWorkspace {
  // LZ77 temporary buffers
  u32 *d_hash_table; // Hash table for LZ77 matching
  u32 hash_table_size;
  u32 *d_chain_table; // Chain table for LZ77 matching
  u32 chain_table_size;
  void *d_matches; // Match array (dense)
  u32 max_matches;
  void *d_costs; // DP cost table for optimal parsing
  u32 max_costs;

  // Sequence temporary buffers
  u32 *d_literal_lengths_reverse; // Reverse sequence buffers for backtracking
  u32 *d_match_lengths_reverse;
  u32 *d_offsets_reverse;
  u32 max_sequences;

  // Huffman temporary buffers
  u32 *d_frequencies;  // Symbol frequency analysis
  u32 *d_code_lengths; // Per-symbol code lengths
  u32 *d_bit_offsets;  // Prefix sum for bit positions

  // Block processing buffers
  u32 *d_block_sums;         // Block-level reduction sums
  u32 *d_scanned_block_sums; // Scanned block sums
  u32 num_blocks;

  // Workspace metadata
  void *d_workspace;       // Base pointer to the entire workspace
  size_t total_size;       // Total workspace size (alias for total_size_bytes)
  size_t total_size_bytes; // Total workspace size
  bool is_allocated;       // Allocation status

  // (NEW) Stream management for pipelining
  cudaStream_t stream;        // Associated CUDA stream
  cudaEvent_t event_complete; // Completion event for dependencies

  // (NEW) Missing fields required by DefaultZstdManager
  u32 *d_lz77_temp;
  void *d_sequences;     // Cast to sequence::Sequence*
  void *d_fse_tables;    // Cast to fse::FSEEncodeTable*
  void *d_huffman_table; // Cast to huffman::HuffmanTable*
  void *d_bitstream;     // Pointer to output bitstream buffer
};

// ============================================================================
// Utility Functions
// ============================================================================

inline float get_compression_ratio(size_t uncompressed, size_t compressed) {
  return compressed > 0 ? static_cast<float>(uncompressed) / compressed : 0.0f;
}

inline bool is_valid_compression_level(int level) {
  return level >= MIN_COMPRESSION_LEVEL && level <= MAX_COMPRESSION_LEVEL;
}

// ============================================================================
// Workspace Management Functions (NEW)
// ============================================================================

// Forward declarations for LZ77 types
namespace lz77 {
struct Match;
struct ParseCost;
} // namespace lz77

Status allocate_compression_workspace(CompressionWorkspace &workspace,
                                      size_t max_block_size,
                                      const CompressionConfig &config);

Status free_compression_workspace(CompressionWorkspace &workspace);

// ============================================================================
// CUDA Error Checking
// ============================================================================

// Enhanced CUDA error checking with context
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t __cuda_err = (call);                                           \
    if (__cuda_err != cudaSuccess) {                                           \
      ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__,         \
                         __FUNCTION__);                                        \
      __ctx.cuda_error = __cuda_err;                                           \
      log_error(__ctx);                                                        \
      return Status::ERROR_CUDA_ERROR;                                         \
    }                                                                          \
  } while (0)
#endif // CUDA_CHECK

#ifndef CUDA_CHECK_RETURN
#define CUDA_CHECK_RETURN(call, ret)                                           \
  do {                                                                         \
    cudaError_t __cuda_err = (call);                                           \
    if (__cuda_err != cudaSuccess) {                                           \
      ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__,         \
                         __FUNCTION__);                                        \
      __ctx.cuda_error = __cuda_err;                                           \
      log_error(__ctx);                                                        \
      return ret;                                                              \
    }                                                                          \
  } while (0)
#endif // CUDA_CHECK_RETURN

// Status validation macros
// Status validation macros
#ifndef CHECK_STATUS
#define CHECK_STATUS(status)                                                   \
  do {                                                                         \
    if ((status) != Status::SUCCESS) {                                         \
      ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__);          \
      log_error(__ctx);                                                        \
      return (status);                                                         \
    }                                                                          \
  } while (0)
#endif

#ifndef CHECK_STATUS_MSG
#define CHECK_STATUS_MSG(status, msg)                                          \
  do {                                                                         \
    if ((status) != Status::SUCCESS) {                                         \
      ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__, (msg));   \
      log_error(__ctx);                                                        \
      return (status);                                                         \
    }                                                                          \
  } while (0)
#endif

// Input validation macros
#ifndef VALIDATE_NOT_NULL
#define VALIDATE_NOT_NULL(ptr, name)                                           \
  do {                                                                         \
    if (!(ptr)) {                                                              \
      ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__,  \
                         __FUNCTION__, name " is null");                       \
      log_error(__ctx);                                                        \
      return Status::ERROR_INVALID_PARAMETER;                                  \
    }                                                                          \
  } while (0)
#endif

#ifndef VALIDATE_RANGE
#define VALIDATE_RANGE(val, min, max, name)                                    \
  do {                                                                         \
    if ((val) < (min) || (val) > (max)) {                                      \
      ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__,  \
                         __FUNCTION__, name " out of range");                  \
      log_error(__ctx);                                                        \
      return Status::ERROR_INVALID_PARAMETER;                                  \
    }                                                                          \
  } while (0)
#endif
} // namespace cuda_zstd

#endif // CUDA_ZSTD_TYPES_H_FIXED_
