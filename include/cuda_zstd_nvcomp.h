// ============================================================================
// cuda_zstd_nvcomp.h - NVCOMP v5.0 Compatibility Layer
// ============================================================================

#ifndef CUDA_ZSTD_NVCOMP_H
#define CUDA_ZSTD_NVCOMP_H

#include "cuda_zstd_types.h"
#include "cuda_zstd_manager.h"

// Forward declare NVCOMP types (include nvcomp headers in implementation)
struct nvcompBatchedZstdOpts_t;

namespace cuda_zstd {
namespace nvcomp_v5 {

// ============================================================================
// NVCOMP v5.0 Format Compatibility
// ============================================================================

// Check if compressed data is NVCOMP v5.0 compatible
bool is_nvcomp_v5_zstd_format(
    const void* compressed_data,
    size_t compressed_size
);

// Get NVCOMP v5.0 format version
constexpr u32 get_nvcomp_v5_format_version() {
    return 0x00050000;  // Version 5.0.0
}

// Check format version compatibility
bool is_compatible_with_nvcomp_v5(u32 format_version);

// ============================================================================
// NVCOMP v5.0 Options Conversion
// ============================================================================

// NVCOMP v5.0 options structure (mirrors nvcompBatchedZstdOpts_t)
struct NvcompV5Options {
    int level;                    // Compression level (1-22)
    int algorithm;                // Reserved for future use
    u32 chunk_size;               // Chunk size for batch processing
    bool enable_checksum;         // Enable checksums
    
    NvcompV5Options()
        : level(3), algorithm(0),
          chunk_size(64 * 1024),
          enable_checksum(false) {}
};

// Convert our config to NVCOMP v5.0 options
NvcompV5Options to_nvcomp_v5_opts(const CompressionConfig& config);

// Convert NVCOMP v5.0 options to our config
CompressionConfig from_nvcomp_v5_opts(const NvcompV5Options& opts);

#ifdef NVCOMP_V5_AVAILABLE
// If NVCOMP v5.0 headers are available, provide direct conversion
NvcompV5Options from_nvcomp_opts(const nvcompBatchedZstdOpts_t& opts);
nvcompBatchedZstdOpts_t to_nvcomp_opts(const NvcompV5Options& opts);
#endif

// ============================================================================
// NVCOMP v5.0 Manager Factory
// ============================================================================

// Create manager compatible with NVCOMP v5.0 options
std::unique_ptr<ZstdManager> create_nvcomp_v5_manager(
    const NvcompV5Options& opts
);

#ifdef NVCOMP_V5_AVAILABLE
std::unique_ptr<ZstdManager> create_nvcomp_v5_manager(
    const nvcompBatchedZstdOpts_t& opts
);
#endif

// ============================================================================
// NVCOMP v5.0 Batch Manager
// ============================================================================

// Batch manager compatible with NVCOMP v5.0 patterns
class NvcompV5BatchManager {
public:
    explicit NvcompV5BatchManager(const NvcompV5Options& opts);
    ~NvcompV5BatchManager();
    
    // Get workspace size for batch compression (NVCOMP v5.0 pattern)
    size_t get_compress_temp_size(
        const size_t* chunk_sizes,
        size_t num_chunks,
        cudaStream_t stream = 0
    ) const;
    
    // Get workspace size for batch decompression
    size_t get_decompress_temp_size(
        const size_t* compressed_sizes,
        size_t num_chunks,
        cudaStream_t stream = 0
    ) const;
    
    // Get maximum compressed chunk size
    size_t get_max_compressed_chunk_size(size_t uncompressed_chunk_size) const;
    
    // Batch compress (NVCOMP v5.0 async pattern)
    Status compress_async(
        const void* const* d_uncompressed_ptrs,
        const size_t* uncompressed_sizes,
        size_t num_chunks,
        void* const* d_compressed_ptrs,
        size_t* compressed_sizes,
        void* d_temp_storage,
        size_t temp_storage_bytes,
        cudaStream_t stream = 0
    );
    
    // Batch decompress (NVCOMP v5.0 async pattern)
    Status decompress_async(
        const void* const* d_compressed_ptrs,
        const size_t* compressed_sizes,
        size_t num_chunks,
        void* const* d_uncompressed_ptrs,
        size_t* uncompressed_sizes,
        void* d_temp_storage,
        size_t temp_storage_bytes,
        cudaStream_t stream = 0
    );
    
    // Get compression statistics
    const CompressionStats& get_stats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// NVCOMP v5.0 Metadata Functions
// ============================================================================

// Extended metadata for NVCOMP v5.0
struct NvcompV5Metadata {
    u32 format_version;           // NVCOMP format version
    u32 library_version;          // Library version
    int compression_level;        // Compression level used
    u32 uncompressed_size;        // Original size
    u32 compressed_size;          // Compressed size
    u32 num_chunks;               // Number of chunks
    u32 chunk_size;               // Size per chunk
    u32 dictionary_id;            // Dictionary ID (0 if none)
    ChecksumPolicy checksum_policy; // Checksum policy
    bool has_dictionary;          // Whether dictionary was used
    u64 checksum;                 // Data checksum (if enabled)
    
    NvcompV5Metadata()
        : format_version(get_nvcomp_v5_format_version()),
          library_version(0x00010000),
          compression_level(3),
          uncompressed_size(0),
          compressed_size(0),
          num_chunks(0),
          chunk_size(0),
          dictionary_id(0),
          checksum_policy(ChecksumPolicy::NO_COMPUTE_NO_VERIFY),
          has_dictionary(false),
          checksum(0) {}
};

// Extract metadata from compressed data (NVCOMP v5.0 pattern)
Status get_metadata_async(
    const void* d_compressed_data,
    size_t compressed_size,
    NvcompV5Metadata* h_metadata,
    cudaStream_t stream = 0
);

// Get metadata synchronously
Status get_metadata(
    const void* d_compressed_data,
    size_t compressed_size,
    NvcompV5Metadata& metadata
);

// Validate metadata
bool validate_metadata(const NvcompV5Metadata& metadata);

// ============================================================================
// NVCOMP v5.0 Utility Functions
// ============================================================================

// Get decompressed size from compressed data (NVCOMP v5.0 async pattern)
Status get_decompressed_size_async(
    const void* d_compressed_data,
    size_t compressed_size,
    size_t* h_decompressed_size,
    cudaStream_t stream = 0
);

// Get number of chunks in compressed data
Status get_num_chunks(
    const void* d_compressed_data,
    size_t compressed_size,
    size_t* num_chunks
);

// Get individual chunk sizes
Status get_chunk_sizes(
    const void* d_compressed_data,
    size_t compressed_size,
    size_t* chunk_sizes,
    size_t max_chunks
);

// ============================================================================
// NVCOMP v5.0 Error Handling
// ============================================================================

// Convert our Status to NVCOMP-style error code
int status_to_nvcomp_error(Status status);

// Convert NVCOMP error code to our Status
Status nvcomp_error_to_status(int nvcomp_error);

// Get error string for NVCOMP v5.0
const char* get_nvcomp_v5_error_string(int error_code);

// ============================================================================
// NVCOMP v5.0 C API (for compatibility)
// ============================================================================

extern "C" {

// C API types
typedef void* nvcompZstdManagerHandle;

// Manager creation
nvcompZstdManagerHandle nvcomp_zstd_create_manager_v5(
    int compression_level
);

void nvcomp_zstd_destroy_manager_v5(
    nvcompZstdManagerHandle handle
);

// Compression
int nvcomp_zstd_compress_async_v5(
    nvcompZstdManagerHandle handle,
    const void* d_uncompressed,
    size_t uncompressed_size,
    void* d_compressed,
    size_t* compressed_size,
    void* d_temp,
    size_t temp_size,
    cudaStream_t stream
);

// Decompression
int nvcomp_zstd_decompress_async_v5(
    nvcompZstdManagerHandle handle,
    const void* d_compressed,
    size_t compressed_size,
    void* d_uncompressed,
    size_t* uncompressed_size,
    void* d_temp,
    size_t temp_size,
    cudaStream_t stream
);

// Workspace queries
size_t nvcomp_zstd_get_compress_temp_size_v5(
    nvcompZstdManagerHandle handle,
    size_t uncompressed_size
);

size_t nvcomp_zstd_get_decompress_temp_size_v5(
    nvcompZstdManagerHandle handle,
    size_t compressed_size
);

// Metadata
int nvcomp_zstd_get_metadata_v5(
    const void* d_compressed_data,
    size_t compressed_size,
    NvcompV5Metadata* h_metadata,
    cudaStream_t stream
);

} // extern "C"

// ============================================================================
// NVCOMP v5.0 Benchmark Helpers
// ============================================================================

// Benchmark compression across all levels (NVCOMP v5.0 pattern)
struct NvcompV5BenchmarkResult {
    int level;
    double compress_time_ms;
    double decompress_time_ms;
    double compress_throughput_mbps;
    double decompress_throughput_mbps;
    float compression_ratio;
    size_t compressed_size;
};

// Run benchmark for specific level
NvcompV5BenchmarkResult benchmark_level(
    const void* d_input,
    size_t input_size,
    int level,
    int iterations = 100,
    cudaStream_t stream = 0
);

// Run benchmark for all levels
std::vector<NvcompV5BenchmarkResult> benchmark_all_levels(
    const void* d_input,
    size_t input_size,
    int iterations = 100,
    cudaStream_t stream = 0
);

// Print benchmark results
void print_benchmark_results(
    const std::vector<NvcompV5BenchmarkResult>& results
);

} // namespace nvcomp_v5
} // namespace cuda_zstd

#endif // CUDA_ZSTD_NVCOMP_H