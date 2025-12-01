// ============================================================================
// cuda_zstd_types.h - Core Types and Definitions
// ============================================================================

#ifndef CUDA_ZSTD_TYPES_H_
#define CUDA_ZSTD_TYPES_H_

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#else
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#endif
#include <cuda_runtime.h>

#ifdef __cplusplus
namespace cuda_zstd {
#endif

// ============================================================================
// Basic Types
// ============================================================================

#ifdef __cplusplus
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using f32 = float;
using i64 = int64_t;
using f64 = double;
using byte_t = unsigned char;
#else
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef float f32;
typedef int64_t i64;
typedef double f64;
typedef unsigned char byte_t;
#endif

#ifdef __cplusplus
// ============================================================================
// Hash Structures
// ============================================================================
namespace hash {
struct HashTable
{
    u32* table;
    u32 size;
};

struct ChainTable
{
    u32* prev;
    u32 size;
};
}

// ============================================================================
// Enhanced Status Codes with Context
// ============================================================================

enum class Status : u32 {
    SUCCESS = 0,
    ERROR_GENERIC = 1,
    ERROR_INVALID_PARAMETER = 2,
    ERROR_OUT_OF_MEMORY = 3,
    ERROR_CUDA_ERROR = 4,
    ERROR_INVALID_MAGIC = 5,
    ERROR_CORRUPT_DATA = 6,
    ERROR_CORRUPTED_DATA = 6,  // Alias for ERROR_CORRUPT_DATA
    ERROR_BUFFER_TOO_SMALL = 7,
    ERROR_UNSUPPORTED_VERSION = 8,
    ERROR_DICTIONARY_MISMATCH = 9,
    ERROR_CHECKSUM_FAILED = 10,
    ERROR_IO = 11,
    ERROR_COMPRESSION = 12,
    ERROR_COMPRESSION_FAILED = 12,  // Alias for ERROR_COMPRESSION
    ERROR_DECOMPRESSION = 13,
    ERROR_DECOMPRESSION_FAILED = 13,  // Alias for ERROR_DECOMPRESSION
    ERROR_WORKSPACE_INVALID = 14,
    ERROR_STREAM_ERROR = 15,
    ERROR_ALLOCATION_FAILED = 16,
    ERROR_HASH_TABLE_FULL = 17,
    ERROR_SEQUENCE_ERROR = 18,
    ERROR_NOT_INITIALIZED = 19,
    ERROR_ALREADY_INITIALIZED = 20,
    ERROR_INVALID_STATE = 21,
    ERROR_TIMEOUT = 22,
    ERROR_CANCELLED = 23,
    ERROR_NOT_IMPLEMENTED = 24,
    ERROR_INTERNAL = 25,
    ERROR_UNKNOWN = 26,
    ERROR_DICTIONARY_FAILED = 27,
    ERROR_UNSUPPORTED_FORMAT = 28
};

// Enhanced error context
struct ErrorContext {
    Status status = Status::SUCCESS;
    const char* file = nullptr;
    int line = 0;
    const char* function = nullptr;
    const char* message = nullptr;
    cudaError_t cuda_error = cudaSuccess;
    
    ErrorContext() = default;
    ErrorContext(Status s, const char* f, int l, const char* fn, const char* msg = nullptr)
        : status(s), file(f), line(l), function(fn), message(msg) {}
};

const char* status_to_string(Status status);
const char* get_detailed_error_message(const ErrorContext& ctx);

// Error callback type
typedef void (*ErrorCallback)(const ErrorContext& ctx);

// Global error handling
void set_error_callback(ErrorCallback callback);
void log_error(const ErrorContext& ctx);
ErrorContext get_last_error();
void clear_last_error();

// ============================================================================
// Compression Strategy
// ============================================================================

enum class Strategy : u32 {
    FAST = 0,       // Level 1
    DFAST = 1,      // Levels 2-3
    GREEDY = 2,     // Levels 4-6
    LAZY = 3,       // Levels 7-12
    LAZY2 = 4,      // Levels 13-15
    BTLAZY2 = 5,    // Levels 16-18
    BTOPT = 6,      // Levels 19-20
    BTULTRA = 7     // Levels 21-22
};

// ============================================================================
// Compression Mode
// ============================================================================

enum class CompressionMode : u32 {
    LEVEL_BASED = 0,      // Use exact level (1-22)
    STRATEGY_BASED = 1    // Use strategy (library picks level)
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
    int level = 3;                          // 1-22 for exact control
    bool use_exact_level = true;            // Force exact level parameters
    
    // Strategy-based mode (Original)
    Strategy strategy = Strategy::GREEDY;   // Let strategy pick level
    
    // Advanced parameters (override level defaults if set)
    u32 window_log = 20;                    // Window size = 1 << window_log
    u32 hash_log = 17;                      // Hash table size
    u32 chain_log = 17;                     // Chain table size
    u32 search_log = 8;                     // Search depth
    u32 min_match = 3;                      // Minimum match length (3-7)
    u32 target_length = 0;                  // Target match length
    u32 block_size = 128 * 1024;            // Compression block size
    bool enable_ldm = false;                // Long distance matching
    u32 ldm_hash_log = 20;                  // LDM hash table size
    ChecksumPolicy checksum = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
    
    // Smart Router Configuration
    u32 cpu_threshold = 1024 * 1024;        // 1MB default (based on benchmarks)
    
    // Helper functions
    static CompressionConfig from_level(int level);
    static CompressionConfig optimal(size_t input_size); // NEW: Get optimal config based on benchmarks
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
        return output_bytes > 0 ? static_cast<float>(input_bytes) / output_bytes : 0.0f;
    }
    
    double get_compression_throughput_gbps() const {
        return compression_time_ms > 0 ? (input_bytes / 1e9) / (compression_time_ms / 1000.0) : 0.0;
    }
};

// ============================================================================
// Batch Processing
// ============================================================================

struct BatchItem {
    void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    Status status = Status::SUCCESS;
};

// ============================================================================
// Dictionary Structures
// ============================================================================

struct DictionaryContent {
    const byte_t* d_buffer = nullptr;
    size_t size = 0;
    u32 dict_id = 0;
};

// ============================================================================
// NVCOMP Metadata
// ============================================================================

struct NvcompMetadata {
    u32 format_version = 0;
    u32 compression_level = 0;
    u32 uncompressed_size = 0;
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
// Compression Workspace (NEW - for workspace-based memory management)
// ============================================================================

struct CompressionWorkspace {
    // LZ77 temporary buffers
    u32* d_hash_table;           // Hash table for LZ77 matching
    u32 hash_table_size;
    u32* d_chain_table;          // Chain table for LZ77 matching
    u32 chain_table_size;
    void* d_matches;             // Match array (dense)
    u32 max_matches;
    void* d_costs;               // DP cost table for optimal parsing
    u32 max_costs;
    
    // Sequence temporary buffers
    u32* d_literal_lengths_reverse;  // Reverse sequence buffers for backtracking
    u32* d_match_lengths_reverse;
    u32* d_offsets_reverse;
    u32 max_sequences;
    
    // Huffman temporary buffers
    u32* d_frequencies;          // Symbol frequency analysis
    u32* d_code_lengths;         // Per-symbol code lengths
    u32* d_bit_offsets;          // Prefix sum for bit positions
    
    // Block processing buffers
    u32* d_block_sums;           // Block-level reduction sums
    u32* d_scanned_block_sums;   // Scanned block sums
    u32 num_blocks;
    
    // Workspace metadata
    void* d_workspace;           // Base pointer to the entire workspace
    size_t total_size;           // Total workspace size (alias for total_size_bytes)
    size_t total_size_bytes;     // Total workspace size
    bool is_allocated;           // Allocation status
    
    // (NEW) Stream management for pipelining
    cudaStream_t stream;         // Associated CUDA stream
    cudaEvent_t event_complete;  // Completion event for dependencies
    
    // (NEW) Missing fields required by DefaultZstdManager
    u32* d_lz77_temp;
    void* d_sequences;           // Cast to sequence::Sequence*
    void* d_fse_tables;          // Cast to fse::FSEEncodeTable*
    void* d_huffman_table;       // Cast to huffman::HuffmanTable*
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
}

Status allocate_compression_workspace(
    CompressionWorkspace& workspace,
    size_t max_block_size,
    const CompressionConfig& config
);

Status free_compression_workspace(CompressionWorkspace& workspace);

// ============================================================================
// CUDA Error Checking
// ============================================================================

// Enhanced CUDA error checking with context
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t __cuda_err = (call); \
    if (__cuda_err != cudaSuccess) { \
        ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__, __FUNCTION__); \
        __ctx.cuda_error = __cuda_err; \
        log_error(__ctx); \
        return Status::ERROR_CUDA_ERROR; \
    } \
} while(0)
#endif // CUDA_CHECK

#ifndef CUDA_CHECK_RETURN
#define CUDA_CHECK_RETURN(call, ret) do { \
    cudaError_t __cuda_err = (call); \
    if (__cuda_err != cudaSuccess) { \
        ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__, __FUNCTION__); \
        __ctx.cuda_error = __cuda_err; \
        log_error(__ctx); \
        return ret; \
    } \
} while(0)
 #endif // CUDA_CHECK_RETURN

// Status validation macros
// Status validation macros
#ifndef CHECK_STATUS
#define CHECK_STATUS(status) do { \
    if ((status) != Status::SUCCESS) { \
        ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__); \
        log_error(__ctx); \
        return (status); \
    } \
} while(0)
#endif

#ifndef CHECK_STATUS_MSG
#define CHECK_STATUS_MSG(status, msg) do { \
    if ((status) != Status::SUCCESS) { \
        ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__, (msg)); \
        log_error(__ctx); \
        return (status); \
    } \
} while(0)
#endif

// Input validation macros
#ifndef VALIDATE_NOT_NULL
#define VALIDATE_NOT_NULL(ptr, name) do { \
    if (!(ptr)) { \
        ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__, __FUNCTION__, \
                          name " is null"); \
        log_error(__ctx); \
        return Status::ERROR_INVALID_PARAMETER; \
    } \
} while(0)
#endif

#ifndef VALIDATE_RANGE
#define VALIDATE_RANGE(val, min, max, name) do { \
    if ((val) < (min) || (val) > (max)) { \
        ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__, __FUNCTION__, \
                          name " out of range"); \
        log_error(__ctx); \
        return Status::ERROR_INVALID_PARAMETER; \
    } \
} while(0)
#endif
#endif // __cplusplus

#ifdef __cplusplus
} // namespace cuda_zstd
#endif

#endif // CUDA_ZSTD_TYPES_H
