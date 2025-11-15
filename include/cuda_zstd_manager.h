// ==============================================================================
// cuda_zstd_manager.h - Complete Manager Interface (PRODUCTION READY)
// ==============================================================================

#ifndef CUDA_ZSTD_MANAGER_H_
#define CUDA_ZSTD_MANAGER_H_

#include "cuda_zstd_types.h"
#include "cuda_zstd_dictionary.h"
#include <cuda_runtime.h>
#ifdef __cplusplus
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#endif

#ifdef __cplusplus
namespace cuda_zstd {

// ==============================================================================
// FORWARD DECLARATIONS
// ==============================================================================

namespace lz77 { struct LZ77Context; }
namespace sequence { struct SequenceContext; }
namespace fse { struct FSEEncodeTable; struct FSEDecodeTable; }
namespace huffman { struct HuffmanContext; }
namespace xxhash { u64 compute_xxh64(const byte_t*, u32, u64, cudaStream_t); }

// ==============================================================================
// MANAGER BASE CLASS
// ==============================================================================

class ZstdManager {
public:
    virtual ~ZstdManager() = default;
    
    // Configuration
    virtual Status configure(const CompressionConfig& config) = 0;
    virtual CompressionConfig get_config() const = 0;
    
    // Workspace queries
    virtual size_t get_compress_temp_size(size_t uncompressed_size) const = 0;
    virtual size_t get_decompress_temp_size(size_t compressed_size) const = 0;
    virtual size_t get_max_compressed_size(size_t uncompressed_size) const = 0;
    
    // Core operations
    virtual Status compress(const void* uncompressed_data, size_t uncompressed_size,
                    void* compressed_data,
                    size_t* compressed_size,
                    void* temp_workspace,
                    size_t temp_size,
                    const void* dict_buffer,
                    size_t dict_size,
                    cudaStream_t stream) = 0;
    
    virtual Status decompress(
        const void* compressed_data,
        size_t compressed_size,
        void* uncompressed_data,
        size_t* uncompressed_size,
        void* temp_workspace,
        size_t temp_size,
        cudaStream_t stream = 0
    ) = 0;
    
    // Dictionary support
    virtual Status set_dictionary(const dictionary::Dictionary& dict) = 0;
    virtual Status get_dictionary(dictionary::Dictionary& dict) const = 0;
    virtual Status clear_dictionary() = 0;
    
    // Statistics & configuration
    virtual const CompressionStats& get_stats() const = 0;
    virtual Status set_compression_level(int level) = 0;
    virtual int get_compression_level() const = 0;
    virtual void reset_stats() = 0;
};

// ==============================================================================
// BATCH MANAGER
// ==============================================================================

class ZstdBatchManager : public ZstdManager {
public:
    ZstdBatchManager();
    explicit ZstdBatchManager(const CompressionConfig& config);
    ~ZstdBatchManager() override;
    
    // Base interface
    Status configure(const CompressionConfig& config) override;
    CompressionConfig get_config() const override;
    
    size_t get_compress_temp_size(size_t uncompressed_size) const override;
    size_t get_decompress_temp_size(size_t compressed_size) const override;
    size_t get_max_compressed_size(size_t uncompressed_size) const override;
    
    Status compress(
        const void* uncompressed_data,
        size_t uncompressed_size,
        void* compressed_data,
        size_t* compressed_size,
        void* temp_workspace,
        size_t temp_size,
        const void* dict_buffer,
        size_t dict_size,
        cudaStream_t stream = 0
    ) override;
    
    Status decompress(
        const void* compressed_data,
        size_t compressed_size,
        void* uncompressed_data,
        size_t* uncompressed_size,
        void* temp_workspace,
        size_t temp_size,
        cudaStream_t stream = 0
    ) override;
    
    Status set_dictionary(const dictionary::Dictionary& dict) override;
    Status get_dictionary(dictionary::Dictionary& dict) const override;
    Status clear_dictionary() override;
    
    const CompressionStats& get_stats() const override;
    Status set_compression_level(int level) override;
    int get_compression_level() const override;
    void reset_stats() override;
    
    // Batch-specific operations
    Status compress_batch(
        const std::vector<BatchItem>& items,
        void* temp_workspace,
        size_t temp_size,
        cudaStream_t stream = 0
    );
    
    Status decompress_batch(
        const std::vector<BatchItem>& items,
        void* temp_workspace,
        size_t temp_size,
        cudaStream_t stream = 0
    );
    
    size_t get_batch_compress_temp_size(
        const std::vector<size_t>& uncompressed_sizes
    ) const;
    
    size_t get_batch_decompress_temp_size(
        const std::vector<size_t>& compressed_sizes
    ) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ==============================================================================
// STREAMING MANAGER
// ==============================================================================

class ZstdStreamingManager {
public:
    ZstdStreamingManager();
    explicit ZstdStreamingManager(const CompressionConfig& config);
    ~ZstdStreamingManager();
    
    // Initialization
    Status init_compression(cudaStream_t stream = 0);
    Status init_decompression(cudaStream_t stream = 0);
    
    // Streaming compression
    Status compress_chunk(
        const void* input,
        size_t input_size,
        void* output,
        size_t* output_size,
        bool is_last_chunk,
        cudaStream_t stream = 0
    );
    
    // Streaming decompression
    Status decompress_chunk(
        const void* input,
        size_t input_size,
        void* output,
        size_t* output_size,
        bool* is_last_chunk,
        cudaStream_t stream = 0
    );
    
    // Control
    Status reset();
    Status flush(cudaStream_t stream = 0);
    
    // Configuration
    Status set_config(const CompressionConfig& config);
    Status set_dictionary(const dictionary::Dictionary& dict);
    CompressionConfig get_config() const;
    
    // Queries
    size_t get_temp_size() const;
    bool is_compression_initialized() const;
    bool is_decompression_initialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ==============================================================================
// FACTORY FUNCTIONS
// ==============================================================================

std::unique_ptr<ZstdManager> create_manager(int compression_level = 3);
std::unique_ptr<ZstdManager> create_manager(const CompressionConfig& config);
std::unique_ptr<ZstdBatchManager> create_batch_manager(int compression_level = 3);
std::unique_ptr<ZstdStreamingManager> create_streaming_manager(int compression_level = 3);

// ==============================================================================
// CONVENIENCE FUNCTIONS (Single-Shot)
// ==============================================================================

Status compress_simple(
    const void* uncompressed_data,
    size_t uncompressed_size,
    void* compressed_data,
    size_t* compressed_size,
    int compression_level = 3,
    cudaStream_t stream = 0
);

Status decompress_simple(
    const void* compressed_data,
    size_t compressed_size,
    void* uncompressed_data,
    size_t* uncompressed_size,
    cudaStream_t stream = 0
);

Status compress_with_dict(
    const void* uncompressed_data,
    size_t uncompressed_size,
    void* compressed_data,
    size_t* compressed_size,
    const dictionary::Dictionary& dict,
    int compression_level = 3,
    cudaStream_t stream = 0
);

Status decompress_with_dict(
    const void* compressed_data,
    size_t compressed_size,
    void* uncompressed_data,
    size_t* uncompressed_size,
    const dictionary::Dictionary& dict,
    cudaStream_t stream = 0
);

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

Status get_decompressed_size(
    const void* compressed_data,
    size_t compressed_size,
    size_t* decompressed_size
);

Status validate_compressed_data(
    const void* compressed_data,
    size_t compressed_size,
    bool check_checksum = true
);

size_t estimate_compressed_size(
    size_t uncompressed_size,
    int compression_level
);

Status validate_config(const CompressionConfig& config);
void apply_level_parameters(CompressionConfig& config);
u32 get_optimal_block_size(u32 input_size, u32 compression_level);

// ==============================================================================
// NVCOMP INTEGRATION
// ==============================================================================

constexpr const char* get_format_name() {
    return "cuda_zstd";
}

constexpr u32 get_format_version() {
    return 0x00010000; // 1.0.0
}

bool is_nvcomp_zstd_format(
    const void* compressed_data,
    size_t compressed_size
);

Status extract_metadata(
    const void* compressed_data,
    size_t compressed_size,
    NvcompMetadata& metadata
);

// ==============================================================================
// ENHANCED PERFORMANCE PROFILING
// ==============================================================================

struct DetailedPerformanceMetrics {
    // Timing breakdown (milliseconds)
    double lz77_time_ms = 0.0;
    double hash_build_time_ms = 0.0;
    double match_finding_time_ms = 0.0;
    double optimal_parse_time_ms = 0.0;
    double sequence_generation_time_ms = 0.0;
    
    double entropy_encode_time_ms = 0.0;
    double fse_encode_time_ms = 0.0;
    double huffman_encode_time_ms = 0.0;
    
    double entropy_decode_time_ms = 0.0;
    double total_time_ms = 0.0;
    
    // Throughput metrics
    double compression_throughput_mbps = 0.0;
    double decompression_throughput_mbps = 0.0;
    
    // Memory metrics
    size_t peak_memory_bytes = 0;
    size_t current_memory_bytes = 0;
    size_t workspace_size_bytes = 0;
    
    // Compression statistics
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    float compression_ratio = 0.0f;
    
    // Kernel metrics
    uint32_t kernel_launches = 0;
    double avg_kernel_time_ms = 0.0;
    
    // Memory bandwidth
    double read_bandwidth_gbps = 0.0;
    double write_bandwidth_gbps = 0.0;
    double total_bandwidth_gbps = 0.0;
    
    // GPU utilization
    float gpu_utilization_percent = 0.0f;
    float memory_utilization_percent = 0.0f;
    
    DetailedPerformanceMetrics() = default;
    
    void print() const;
    void export_csv(const char* filename) const;
};

// Legacy struct for backward compatibility
using PerformanceMetrics = DetailedPerformanceMetrics;

class PerformanceProfiler {
public:
    static void enable_profiling(bool enable);
    static bool is_profiling_enabled();
    static const DetailedPerformanceMetrics& get_metrics();
    static void reset_metrics();
    static void print_metrics();
    
    // Enhanced profiling
    static void start_timer(const char* name);
    static void stop_timer(const char* name);
    static double get_timer_ms(const char* name);
    
    // Enhanced profiling with std::string support
    static void start_timer(const std::string& name);
    static void stop_timer(const std::string& name);
    static double get_timer_ms(const std::string& name);
    
    // Component-specific profiling
    static void record_lz77_time(double ms);
    static void record_fse_time(double ms);
    static void record_huffman_time(double ms);
    static void record_memory_usage(size_t bytes);
    static void record_kernel_launch();
    
    // Export functionality
    static void export_metrics_csv(const char* filename);
    static void export_metrics_json(const char* filename);
    static void export_metrics_csv(const std::string& filename);
    static void export_metrics_json(const std::string& filename);
    
private:
    static bool profiling_enabled_;
    static DetailedPerformanceMetrics metrics_;
    static std::unordered_map<std::string, double> timers_;
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timer_start_;
    static std::unordered_map<std::string, cudaEvent_t> cuda_timers_;
    static std::mutex profiler_mutex_;
};

} // namespace cuda_zstd
#endif

// ==============================================================================
// C API FOR COMPATIBILITY
// ==============================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_zstd_manager_t cuda_zstd_manager_t;
typedef struct cuda_zstd_dict_t cuda_zstd_dict_t;

// Manager lifecycle
cuda_zstd_manager_t* cuda_zstd_create_manager(int compression_level);
void cuda_zstd_destroy_manager(cuda_zstd_manager_t* manager);

// Compression/Decompression
int cuda_zstd_compress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
);

int cuda_zstd_decompress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
);

// Workspace queries
size_t cuda_zstd_get_compress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t src_size
);

size_t cuda_zstd_get_decompress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t compressed_size
);

// Dictionary training
cuda_zstd_dict_t* cuda_zstd_train_dictionary(
    const void** samples,
    const size_t* sample_sizes,
    size_t num_samples,
    size_t dict_size
);

void cuda_zstd_destroy_dictionary(cuda_zstd_dict_t* dict);

int cuda_zstd_set_dictionary(
    cuda_zstd_manager_t* manager,
    cuda_zstd_dict_t* dict
);

// Error handling
const char* cuda_zstd_get_error_string(int error_code);
int cuda_zstd_is_error(int code);

#ifdef __cplusplus
}
#endif // __cplusplus



#endif // CUDA_ZSTD_MANAGER_H
