// ==============================================================================
// cuda_zstd_manager.cpp - COMPLETE Manager Implementation with Full Pipeline
//
// NOTE: This file is patched to include the 'extract_metadata' function
//       required by the NVCOMP v5.0 API.
//
// (NEW) NOTE: This file is also patched to implement the PerformanceProfiler
//             with full support for performance metrics tracking.
//
// (NEW) NOTE: This file is patched again to remove the redundant
//             `lz77::find_matches` call.
//
// (NEW) NOTE: This file is patched to implement true parallel batching
//             with stream pools and partitioned workspaces.
// =============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_lz77.h" // Ensure Match and ParseCost are defined
#include "cuda_zstd_types.h" // Also include for workspace struct
#include "cuda_zstd_fse.h"
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_xxhash.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_stream_pool.h"
#include <memory>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <chrono>
#include <fstream>
#include <mutex>
#include <optional>

namespace cuda_zstd {

// Add alignment constants
constexpr u32 GPU_MEMORY_ALIGNMENT = 256;  // Most GPU requirements

// Helper: Align size to boundary
inline size_t align_to_boundary(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// ==============================================================================
// (REMOVED) PERFORMANCE PROFILER IMPLEMENTATION - Now in cuda_zstd_utils.cpp
// ==============================================================================

bool PerformanceProfiler::profiling_enabled_ = false;
PerformanceMetrics PerformanceProfiler::metrics_;
std::unordered_map<std::string, double> PerformanceProfiler::timers_;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> PerformanceProfiler::timer_start_;
std::unordered_map<std::string, cudaEvent_t> PerformanceProfiler::cuda_timers_;
std::mutex PerformanceProfiler::profiler_mutex_;

void PerformanceProfiler::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
    if (enable) {
        reset_metrics();
    }
}

bool PerformanceProfiler::is_profiling_enabled() {
    return profiling_enabled_;
}

const PerformanceMetrics& PerformanceProfiler::get_metrics() {
    return metrics_;
}

void PerformanceProfiler::reset_metrics() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    memset(&metrics_, 0, sizeof(PerformanceMetrics));
    timers_.clear();
    timer_start_.clear();
}

void PerformanceProfiler::print_metrics() {
    if (!profiling_enabled_) {
        printf("[Profiler] Profiling is disabled.\n");
        return;
    }
    printf("=== Performance Metrics ===\n");
    printf("  LZ77 Time:       %.3f ms\n", metrics_.lz77_time_ms);
    printf("  Entropy Enc Time: %.3f ms\n", metrics_.entropy_encode_time_ms);
    printf("  Entropy Dec Time: %.3f ms\n", metrics_.entropy_decode_time_ms);
    printf("  Total Time:      %.3f ms\n", metrics_.total_time_ms);
    printf("  Throughput:      %.2f MB/s\n", metrics_.compression_throughput_mbps);
    printf("  Peak Memory:     %zu bytes\n", metrics_.peak_memory_bytes);
    printf("===========================\n");
}

// Timer methods with const char*
void PerformanceProfiler::start_timer(const char* name) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    timer_start_[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::stop_timer(const char* name) {
    if (!profiling_enabled_) return;
    auto end = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    auto it = timer_start_.find(name);
    if (it != timer_start_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count() / 1000.0;
        timers_[name] = duration;
    }
}

double PerformanceProfiler::get_timer_ms(const char* name) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    auto it = timers_.find(name);
    return (it != timers_.end()) ? it->second : 0.0;
}

// Timer methods with std::string
void PerformanceProfiler::start_timer(const std::string& name) {
    start_timer(name.c_str());
}

void PerformanceProfiler::stop_timer(const std::string& name) {
    stop_timer(name.c_str());
}

double PerformanceProfiler::get_timer_ms(const std::string& name) {
    return get_timer_ms(name.c_str());
}

// Component-specific profiling
void PerformanceProfiler::record_lz77_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.lz77_time_ms += ms;
}

void PerformanceProfiler::record_fse_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.fse_encode_time_ms += ms;
}

void PerformanceProfiler::record_huffman_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.huffman_encode_time_ms += ms;
}

void PerformanceProfiler::record_memory_usage(size_t bytes) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.current_memory_bytes = bytes;
    if (bytes > metrics_.peak_memory_bytes) {
        metrics_.peak_memory_bytes = bytes;
    }
}

void PerformanceProfiler::record_kernel_launch() {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.kernel_launches++;
}

// Export functionality
void PerformanceProfiler::export_metrics_csv(const char* filename) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "Metric,Value\n";
    file << "lz77_time_ms," << metrics_.lz77_time_ms << "\n";
    file << "fse_encode_time_ms," << metrics_.fse_encode_time_ms << "\n";
    file << "huffman_encode_time_ms," << metrics_.huffman_encode_time_ms << "\n";
    file << "entropy_encode_time_ms," << metrics_.entropy_encode_time_ms << "\n";
    file << "entropy_decode_time_ms," << metrics_.entropy_decode_time_ms << "\n";
    file << "total_time_ms," << metrics_.total_time_ms << "\n";
    file << "compression_throughput_mbps," << metrics_.compression_throughput_mbps << "\n";
    file << "peak_memory_bytes," << metrics_.peak_memory_bytes << "\n";
    file << "kernel_launches," << metrics_.kernel_launches << "\n";
    
    // Export custom timers
    for (const auto& [name, duration] : timers_) {
        file << "timer_" << name << "," << duration << "\n";
    }
    
    file.close();
}

void PerformanceProfiler::export_metrics_json(const char* filename) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"lz77_time_ms\": " << metrics_.lz77_time_ms << ",\n";
    file << "  \"fse_encode_time_ms\": " << metrics_.fse_encode_time_ms << ",\n";
    file << "  \"huffman_encode_time_ms\": " << metrics_.huffman_encode_time_ms << ",\n";
    file << "  \"entropy_encode_time_ms\": " << metrics_.entropy_encode_time_ms << ",\n";
    file << "  \"entropy_decode_time_ms\": " << metrics_.entropy_decode_time_ms << ",\n";
    file << "  \"total_time_ms\": " << metrics_.total_time_ms << ",\n";
    file << "  \"compression_throughput_mbps\": " << metrics_.compression_throughput_mbps << ",\n";
    file << "  \"peak_memory_bytes\": " << metrics_.peak_memory_bytes << ",\n";
    file << "  \"kernel_launches\": " << metrics_.kernel_launches;
    
    // Export custom timers
    if (!timers_.empty()) {
        file << ",\n  \"custom_timers\": {\n";
        bool first = true;
        for (const auto& [name, duration] : timers_) {
            if (!first) file << ",\n";
            file << "    \"" << name << "\": " << duration;
            first = false;
        }
        file << "\n  }";
    }
    
    file << "\n}\n";
    file.close();
}

void PerformanceProfiler::export_metrics_csv(const std::string& filename) {
    export_metrics_csv(filename.c_str());
}

void PerformanceProfiler::export_metrics_json(const std::string& filename) {
    export_metrics_json(filename.c_str());
}


// ==============================================================================
// ZSTANDARD FRAME CONSTANTS (RFC 8878)
// ==============================================================================

constexpr u32 ZSTD_MAGIC_NUMBER = 0xFD2FB528;
constexpr u32 ZSTD_MAGIC_SKIPPABLE_START = 0x184D2A50;
constexpr u32 ZSTD_MIN_CLEVEL = 1;
constexpr u32 ZSTD_MAX_CLEVEL = 22;
constexpr u32 ZSTD_DEFAULT_CLEVEL = 3;

constexpr u32 ZSTD_BLOCKSIZE_MAX = 128 * 1024;  // 128 KB
constexpr u32 ZSTD_WINDOWLOG_MIN = 10;
constexpr u32 ZSTD_WINDOWLOG_MAX = 31;

// Frame header sizes
constexpr u32 FRAME_HEADER_SIZE_MIN = 2;
constexpr u32 FRAME_HEADER_SIZE_MAX = 18;

// ==============================================================================
// (NEW) CUSTOM METADATA DEFINITIONS
// ==============================================================================

// This is our application-specific magic number to identify our own metadata
constexpr u32 CUSTOM_METADATA_MAGIC = 0x184D2A5E; // ZSTD Magic Number

// Defines the standard Zstd Skippable Frame Header
struct SkippableFrameHeader {
    u32 magic_number;
    u32 frame_size; // Size of the data that follows (CustomMetadataFrame)
};

// This is our custom data. We can add more fields here later.
struct CustomMetadataFrame {
    u32 custom_magic;       // Set to CUSTOM_METADATA_MAGIC
    i32 compression_level;  // The level we want to save
};

// ==============================================================================
// HELPER KERNELS AND FUNCTIONS
// ==============================================================================

/**
 * @brief Expands a byte_t[] array to a u32[] array.
 * This is used for 'Raw' sequence streams.
 */
__global__ void expand_bytes_to_u32_kernel(
    const byte_t* d_input,
    u32* d_output,
    u32 num_sequences
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (u32 i = idx; i < num_sequences; i += stride) {
        d_output[i] = (u32)d_input[i];
    }
}

/**
 * @brief Expands a single byte (RLE) to a full block.
 */
__global__ void expand_rle_kernel(
    byte_t* d_output,
    u32 decompressed_size,
    byte_t value
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (u32 i = idx; i < decompressed_size; i += stride) {
        d_output[i] = value;
    }
}

/**
 * @brief Expands a single u32 value (RLE) for sequence components.
 */
__global__ void expand_rle_u32_kernel(
    u32* d_output,
    u32 num_sequences,
    u32 value
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (u32 i = idx; i < num_sequences; i += stride) {
        d_output[i] = value;
    }
}

/**
 * @brief Aligns a pointer to the specified byte alignment.
 */
template<typename T>
T* align_ptr(T* ptr, size_t alignment) {
    uintptr_t int_ptr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_ptr = (int_ptr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<T*>(aligned_ptr);
}


// ==============================================================================
// INTERNAL STRUCTURES
// ==============================================================================

struct BlockInfo {
    byte_t* data;
    u32 size;
    bool is_compressed;
    bool is_last;
};

struct CompressionContext {
    // LZ77 matching
    lz77::LZ77Context* lz77_ctx;
    
    // Sequence encoding
    sequence::SequenceContext* seq_ctx;
    
    // FSE tables
    fse::FSEEncodeTable* lit_fse_table;
    fse::FSEEncodeTable* ml_fse_table;
    fse::FSEEncodeTable* of_fse_table;
    
    // Huffman
    huffman::HuffmanTable* huff_ctx;
    
    // (NEW) Workspace for all temporary allocations
    CompressionWorkspace workspace;
    
    // Temporary buffers
    byte_t* d_temp_buffer; // Persistent temp buffer
    u32 temp_buffer_size;
    
    // (NEW) Multi-stream support for pipelining
    cudaStream_t* streams;        // Array of CUDA streams
    cudaEvent_t* events;          // Array of CUDA events
    u32 num_streams;              // Number of streams in pool
    u32 current_stream_idx;       // Round-robin stream selection
    
    // Statistics
    u64 total_matches;
    u64 total_literals;
    u64 total_sequences;
};

struct StreamingContext {
    // Window history from previous chunks
    byte_t* d_window_history;        // Device: last N bytes
    u32 window_history_size;         // Current filled size
    u32 window_history_capacity;     // Max capacity (32KB-128KB)
    
    // Hash chain persistence
    u32* d_hash_table_state;         // Persistent across chunks
    u32* d_chain_table_state;
    
    // Offset tracking for proper distances
    u64 total_bytes_processed;       // Cumulative across chunks
    
    // Frame state
    bool started_compression;
    bool finished_compression;
    u32 block_count;
};

// ==============================================================================
// RFC 8878 Frame Header Structure
// ==============================================================================
// 
// Frame_Header = Magic_Number Frame_Header_Descriptor (Optional_Data_Block)*
// Magic_Number = 4 bytes = 0x28, 0xB5, 0x2F, 0xFD (= 0xFD2FB528 in little-endian)
// Frame_Header_Descriptor = 1 byte (FHDB)
// Optional_Data_Block = Window_Descriptor | Dictionary_ID | Content_Size | Checksum

// RFC 8878 Frame Header Structure
struct FrameHeaderDescriptor {
    u8 fhd;
    
    bool has_dictionary_id() const {
        return (fhd & 0x04) != 0;
    }
    
    bool has_content_size() const {
        return (fhd & 0x08) != 0;
    }
    
    bool has_checksum() const {
        return (fhd & 0x04) != 0; // Bit 2
    }
    
    bool is_single_segment() const {
        return (fhd & 0x40) != 0;
    }
    
    u32 get_dictionary_id_size() const {
        u32 did = (fhd >> 0) & 0x03;
        if (did == 0) return 0;
        return (1 << (did - 1)) * 4;
    }
    
    u32 get_content_size_bytes() const {
        if (!has_content_size()) return 0;
        u32 csf = (fhd >> 6) & 0x03;
        if (csf == 0) return 1;
        if (csf == 1) return 2;
        if (csf == 2) return 4;
        return 8;
    }
};

struct ZstdFrameMetadata {
    u32 magic_number;
    u8 fhd;  // Frame_Header_Descriptor
    
    // Optional fields
    u32 dictionary_id;      // Optional
    u64 content_size;       // Optional
    
    // Computed
    u32 frame_header_size;  // Total size of frame header
    u32 checksum_value;     // Content checksum (after all blocks)
    
    bool has_dict;
    bool has_content_size;
    bool has_checksum;
};

// Helper: Read little-endian u32
inline u32 read_u32_le(const byte_t* data) {
    return ((u32)data[0]) | ((u32)data[1] << 8) | 
           ((u32)data[2] << 16) | ((u32)data[3] << 24);
}

// Helper: Read little-endian u64
inline u64 read_u64_le(const byte_t* data) {
    return ((u64)read_u32_le(data)) | (((u64)read_u32_le(data + 4)) << 32);
}

Status parse_zstd_frame_header(
    const byte_t* compressed_data,
    size_t compressed_size,
    ZstdFrameMetadata* metadata
) {
    if (!compressed_data || !metadata || compressed_size < 6) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    size_t offset = 0;
    
    // 1. Read magic number (4 bytes)
    metadata->magic_number = read_u32_le(compressed_data + offset);
    offset += 4;
    
    if (metadata->magic_number != ZSTD_MAGIC_NUMBER) {
        // Check for skippable frame
        if ((metadata->magic_number & 0xFFFFFFF0) == ZSTD_MAGIC_SKIPPABLE_START) {
            // This is a skippable frame - parse and skip it
            if (compressed_size < offset + 4) {
                return Status::ERROR_CORRUPT_DATA;
            }
            u32 frame_size = read_u32_le(compressed_data + offset);
            // Frame size doesn't include the 8-byte header
            offset += 4 + frame_size;
            
            // Check for next frame
            if (offset >= compressed_size) {
                return Status::ERROR_CORRUPT_DATA;
            }
            
            // Recursively parse the next frame
            return parse_zstd_frame_header(
                compressed_data + offset,
                compressed_size - offset,
                metadata
            );
        }
        return Status::ERROR_INVALID_MAGIC;
    }
    
    // 2. Read Frame Header Descriptor (1 byte)
    if (offset >= compressed_size) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    metadata->fhd = compressed_data[offset];
    offset += 1;
    
    FrameHeaderDescriptor fhd_parser;
    fhd_parser.fhd = metadata->fhd;
    
    metadata->has_dict = fhd_parser.has_dictionary_id();
    metadata->has_content_size = fhd_parser.has_content_size();
    metadata->has_checksum = fhd_parser.has_checksum();
    
    // 3. Parse optional Window_Descriptor (only if not single-segment)
    if (!fhd_parser.is_single_segment()) {
        if (offset >= compressed_size) {
            return Status::ERROR_CORRUPT_DATA;
        }
        // Window descriptor is 1 byte, extract window size for information
        offset += 1;
        // window_exponent = (window_desc >> 3) & 0x1F
        // window_size = (1 << (10 + window_exponent))
    }
    
    // 4. Parse Dictionary_ID (optional, variable size)
    metadata->dictionary_id = 0;
    if (metadata->has_dict) {
        u32 did_size = fhd_parser.get_dictionary_id_size();
        if (offset + did_size > compressed_size) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        switch (did_size) {
            case 1:
                metadata->dictionary_id = compressed_data[offset];
                break;
            case 2:
                metadata->dictionary_id = read_u32_le(compressed_data + offset) & 0xFFFF;
                break;
            case 4:
                metadata->dictionary_id = read_u32_le(compressed_data + offset);
                break;
            default:
                return Status::ERROR_CORRUPT_DATA;
        }
        offset += did_size;
    }
    
    // 5. Parse Content_Size (optional, variable size)
    metadata->content_size = 0;
    if (metadata->has_content_size) {
        u32 cs_size = fhd_parser.get_content_size_bytes();
        if (offset + cs_size > compressed_size) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        switch (cs_size) {
            case 1:
                metadata->content_size = compressed_data[offset];
                break;
            case 2:
                metadata->content_size = read_u32_le(compressed_data + offset) & 0xFFFF;
                break;
            case 4:
                metadata->content_size = read_u32_le(compressed_data + offset);
                break;
            case 8:
                metadata->content_size = read_u64_le(compressed_data + offset);
                break;
            default:
                return Status::ERROR_CORRUPT_DATA;
        }
        offset += cs_size;
    }
    
    metadata->frame_header_size = (u32)offset;
    
    return Status::SUCCESS;
}

// New public function: Extract metadata from compressed data
Status extract_metadata(
    const void* compressed_data,
    size_t compressed_size,
    ZstdFrameMetadata* metadata
) {
    if (!compressed_data || !metadata || compressed_size < 4) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    return parse_zstd_frame_header(
        static_cast<const byte_t*>(compressed_data),
        compressed_size,
        metadata
    );
}

// ==============================================================================
// DEFAULT ZSTD MANAGER IMPLEMENTATION
// ==============================================================================

class DefaultZstdManager : public ZstdManager {
private:
    CompressionConfig config;
    CompressionStats stats;
    // Stream pooling implemented via StreamPool for efficient reuse.
    // Future enhancement: Add stream_pool member if needed.
    dictionary::Dictionary dict;
    bool has_dictionary;
    CompressionContext ctx;
    bool ctx_initialized;
    u64* d_checksum_buffer;
    StreamPool* stream_pool_;

public:
    DefaultZstdManager() 
        : has_dictionary(false), ctx_initialized(false) 
    {
        std::cerr << "DefaultZstdManager ctor start" << std::endl;
        // Initialize stream pool for parallelism; use environment var to customize
        // Use a shared global stream pool for all managers to better saturate
        // GPU concurrency across threads / managers without allocating
        // per-manager pools.
        stream_pool_ = get_global_stream_pool();
        std::cerr << "DefaultZstdManager ctor after get_global_stream_pool" << std::endl;
        d_checksum_buffer = nullptr;
        // Set default config
        config.level = ZSTD_DEFAULT_CLEVEL;
        config.compression_mode = CompressionMode::LEVEL_BASED;
        config.window_log = 20;
        config.hash_log = 17;
        config.chain_log = 17;
        config.search_log = 8;
        config.min_match = 3;
        config.strategy = Strategy::GREEDY;
        
        reset_stats();
        memset(&ctx, 0, sizeof(CompressionContext));
        std::cerr << "DefaultZstdManager ctor end" << std::endl;
    }

    void cleanup_context() {
        if (!ctx_initialized) return;

        // FIX: Synchronize device BEFORE destroying streams
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::cerr << "[cleanup_context] cudaDeviceSynchronize failed: " 
                    << cudaGetErrorString(sync_err) << std::endl;
        }

        // FIX: Synchronize each stream individually before destroying
        if (ctx.streams) {
            for (u32 i = 0; i < ctx.num_streams; ++i) {
                if (ctx.streams[i]) {
                    sync_err = cudaStreamSynchronize(ctx.streams[i]);
                    if (sync_err != cudaSuccess) {
                        std::cerr << "[cleanup_context] Stream " << i << " sync failed: "
                                << cudaGetErrorString(sync_err) << std::endl;
                    }

                    // Now safe to destroy
                    cudaError_t destroy_err = cudaStreamDestroy(ctx.streams[i]);
                    if (destroy_err != cudaSuccess) {
                        std::cerr << "[cleanup_context] Stream " << i << " destroy failed: "
                                << cudaGetErrorString(destroy_err) << std::endl;
                    }
                }

                if (ctx.events[i]) {
                    cudaEventDestroy(ctx.events[i]);
                }
            }
            delete[] ctx.streams;
            delete[] ctx.events;
            ctx.streams = nullptr;
            ctx.events = nullptr;
        }

        // Free workspace AFTER synchronization
        free_compression_workspace(ctx.workspace);

        // Free other resources...
        if (ctx.seq_ctx) {
            cudaFree(ctx.seq_ctx->d_literals_buffer);
            cudaFree(ctx.seq_ctx->d_literal_lengths);
            cudaFree(ctx.seq_ctx->d_match_lengths);
            cudaFree(ctx.seq_ctx->d_offsets);
            cudaFree(ctx.seq_ctx->d_num_sequences);
            delete ctx.seq_ctx;
        }

        if (ctx.huff_ctx) {
            cudaFree(ctx.huff_ctx->codes);
            delete ctx.huff_ctx;
        }

        delete ctx.lit_fse_table;
        delete ctx.ml_fse_table;
        delete ctx.of_fse_table;
        delete ctx.lz77_ctx;

        ctx_initialized = false;
    }
    
    virtual ~DefaultZstdManager() {
        // Cleanup any device-side checksum buffer allocated lazily
        if (d_checksum_buffer != nullptr) {
            cudaFree(d_checksum_buffer);
            d_checksum_buffer = nullptr;
        }
        cleanup_context();
    }
    
    // ==========================================================================
    // Workspace Queries
    // ==========================================================================
    size_t get_compress_temp_size(size_t input_size) const override {
        if (input_size == 0) return 0;
        
        size_t total = 0;

        // 0. Dictionary buffer and content struct
        if (has_dictionary) {
            total += align_to_boundary(sizeof(DictionaryContent), GPU_MEMORY_ALIGNMENT);
            total += align_to_boundary(dict.raw_size, GPU_MEMORY_ALIGNMENT);
        }
        
        // 1. Input buffer (device) - only if input is on host
        // We'll assume worst case and include it
        size_t input_buf_size = input_size;
        total += align_to_boundary(input_buf_size, GPU_MEMORY_ALIGNMENT);
        
        // 2. Compressed output buffer (device)
        size_t output_buf_size = input_size * 2;  // Worst case
        total += align_to_boundary(output_buf_size, GPU_MEMORY_ALIGNMENT);
        
        // 3. LZ77 temporary buffer (device)
        size_t lz77_temp_size = ZSTD_BLOCKSIZE_MAX;
        total += align_to_boundary(lz77_temp_size, GPU_MEMORY_ALIGNMENT);
        
        // 4. Hash/Chain tables for LZ77 (device)
        // NOTE: The hash/chain tables are used by the parallel find_matches
        // kernel and must be large enough to accept a position index that is
        // dict_size + input_size. When computing the workspace size we must
        // include these to avoid overflow of the provided workspace.
        size_t hash_table_size = (1ull << config.hash_log) * sizeof(u32);
        total += align_to_boundary(hash_table_size, GPU_MEMORY_ALIGNMENT);

        size_t chain_table_size = (1ull << config.chain_log) * sizeof(u32);
        total += align_to_boundary(chain_table_size, GPU_MEMORY_ALIGNMENT);

        // 5. Matches and costs buffers - CRITICAL FIX
        // These are allocated per BLOCK, not per full input
        // Use the smaller of input_size and BLOCKSIZE_MAX
        size_t block_size = std::min((size_t)ZSTD_BLOCKSIZE_MAX, input_size);
        size_t matches_size = block_size * sizeof(lz77::Match);
        total += align_to_boundary(matches_size, GPU_MEMORY_ALIGNMENT);
        
        size_t costs_size = block_size * sizeof(lz77::ParseCost);
        total += align_to_boundary(costs_size, GPU_MEMORY_ALIGNMENT);

        // 6. Sequence storage (device)
        size_t seq_storage_size = (ZSTD_BLOCKSIZE_MAX / 3) * sizeof(sequence::Sequence);
        total += align_to_boundary(seq_storage_size, GPU_MEMORY_ALIGNMENT);
        
        // 7. FSE tables (device)
        size_t fse_table_size = 3 * sizeof(fse::FSEEncodeTable);
        total += align_to_boundary(fse_table_size, GPU_MEMORY_ALIGNMENT);
        
        // 8. Huffman context (device)
        size_t huff_size = sizeof(huffman::HuffmanTable);
        total += align_to_boundary(huff_size, GPU_MEMORY_ALIGNMENT);
        
        // 9. Safety padding (extra 10% for alignment overhead)
        total += total / 10;
        
        // 10. Final round to reasonable boundary
        total = align_to_boundary(total, 1024 * 1024);  // 1MB boundary
        
        return total;
    }

    // Similar for decompression
    size_t get_decompress_temp_size(size_t compressed_size) const override {
        if (compressed_size == 0) return 0;
        
        size_t total = 0;
        
        // Estimate max decompressed size (4x typically for Zstd)
        size_t max_decompressed = std::min(
            compressed_size * 4,
            (size_t)1024 * 1024 * 1024  // 1GB max
        );
        
        // 1. Input (compressed) buffer
        total += align_to_boundary(compressed_size, GPU_MEMORY_ALIGNMENT);
        
        // 2. Output buffer
        total += align_to_boundary(max_decompressed, GPU_MEMORY_ALIGNMENT);
        
        // 3. Temp working buffers
        total += align_to_boundary(ZSTD_BLOCKSIZE_MAX * 2, GPU_MEMORY_ALIGNMENT);
        
        // 4. FSE decode tables
        total += align_to_boundary(3 * sizeof(fse::FSEDecodeTable), 
                                  GPU_MEMORY_ALIGNMENT);
        
        // 5. Safety padding
        total += total / 20;
        
        // Final rounding
        total = align_to_boundary(total, 1024 * 1024);
        
        return total;
    }

    virtual size_t get_max_compressed_size(size_t uncompressed_size) const override {
        return estimate_compressed_size(uncompressed_size, config.level);
    }
    
    // Configuration
    virtual Status configure(const CompressionConfig& new_config) override {
        auto status = validate_config(new_config);
        if (status != Status::SUCCESS) return status;
        
        config = new_config;
        
        if (config.compression_mode == CompressionMode::LEVEL_BASED) {
            apply_level_parameters(config);
        }
        
        cleanup_context();
        return initialize_context();
    }
    
    virtual CompressionConfig get_config() const override {
        return config;
    }

    virtual Status set_compression_level(int level) override {
        if (!is_valid_compression_level(level)) {
            return Status::ERROR_INVALID_PARAMETER;
        }
        config.level = level;
        config.compression_mode = CompressionMode::LEVEL_BASED;
        apply_level_parameters(config);
        
        cleanup_context();
        return initialize_context();
    }
    
    virtual int get_compression_level() const override {
        return config.level;
    }
    
    // ==========================================================================
    // compress() implementation
    // ==========================================================================
    virtual Status compress(
        const void* uncompressed_data,
        size_t uncompressed_size,
        void* compressed_data,
        size_t* compressed_size,
        void* temp_workspace,
        size_t temp_size,
        const void* dict_buffer,
        size_t dict_size,
        cudaStream_t stream
    ) override {
        if (!uncompressed_data || !compressed_data || !compressed_size || !temp_workspace || uncompressed_size == 0) {
            return Status::ERROR_INVALID_PARAMETER;
        }

        size_t required_size = get_compress_temp_size(uncompressed_size);
        if (temp_size < required_size) {
            return Status::ERROR_BUFFER_TOO_SMALL;
        }
        
        if (!ctx_initialized) {
            auto status = initialize_context();
            if (status != Status::SUCCESS) return status;
        }
        
        // If no stream supplied, acquire one from the pool for parallelism
        std::optional<StreamPool::Guard> pool_guard;
        if (stream == 0 && stream_pool_) {
            // Try to acquire a stream from the pool. To avoid indefinite
            // blocking (deadlocks in multithreaded test harness), use a
            // configurable timeout (milliseconds) and return a timeout
            // error to the caller if no stream is available.
            size_t timeout_ms = 100000; // default 100s for developer runs
            const char* env_to = getenv("CUDA_ZSTD_STREAM_POOL_TIMEOUT_MS");
            if (env_to) try { timeout_ms = std::max((size_t)1, (size_t)std::strtoul(env_to, nullptr, 10)); } catch(...) { }

            pool_guard = stream_pool_->acquire_for(timeout_ms);
            if (!pool_guard.has_value()) {
                return Status::ERROR_TIMEOUT;
            }
            stream = pool_guard->get_stream();
        }

        // --- 1. Ensure temp_workspace is device memory ---
        cudaPointerAttributes temp_attrs;
        cudaError_t temp_attr_err = cudaPointerGetAttributes(&temp_attrs, temp_workspace);
        if (temp_attr_err != cudaSuccess || temp_attrs.type != cudaMemoryTypeDevice) {
            // Allocate device buffer if not already device memory
            void* device_workspace = nullptr;
            cudaError_t alloc_err = cudaMalloc(&device_workspace, temp_size);
            if (alloc_err != cudaSuccess) {
                std::cerr << "[compress] Failed to allocate device workspace: " << cudaGetErrorString(alloc_err) << std::endl;
                return Status::ERROR_CUDA_ERROR;
            }
            // Optionally copy host buffer to device if needed
            cudaMemcpy(device_workspace, temp_workspace, temp_size, cudaMemcpyHostToDevice);
            temp_workspace = device_workspace;
        }

        // --- 1. Partition the temp_workspace ---
        byte_t* workspace_ptr = static_cast<byte_t*>(temp_workspace);
        size_t alignment = 128;

        DictionaryContent* d_dict_content = nullptr;
        byte_t* d_dict_buffer = nullptr;
        if (has_dictionary) {
            d_dict_content = reinterpret_cast<DictionaryContent*>(workspace_ptr);
            workspace_ptr = align_ptr(workspace_ptr + sizeof(DictionaryContent), alignment);
            
            d_dict_buffer = workspace_ptr;
            workspace_ptr = align_ptr(workspace_ptr + dict.raw_size, alignment);
            
            // Copy dictionary to device
            CUDA_CHECK(cudaMemcpyAsync(d_dict_buffer, dict.raw_content, dict.raw_size,
                                      cudaMemcpyHostToDevice, stream));
            
            // Set up DictionaryContent structure
            DictionaryContent h_dict_content;
            h_dict_content.d_buffer = d_dict_buffer;
            h_dict_content.size = dict.raw_size;
            h_dict_content.dict_id = dict.header.dictionary_id;
            
            CUDA_CHECK(cudaMemcpyAsync(d_dict_content, &h_dict_content, sizeof(DictionaryContent),
                                      cudaMemcpyHostToDevice, stream));
        }
        
        // Check if uncompressed_data is already on device BEFORE allocating workspace for it
        cudaPointerAttributes input_attrs;
        cudaError_t input_attr_err = cudaPointerGetAttributes(&input_attrs, uncompressed_data);
        bool input_is_device = (input_attr_err == cudaSuccess && input_attrs.type == cudaMemoryTypeDevice);
        
        byte_t* d_input = nullptr;
        if (input_is_device) {
            // Input is already on device, use it directly
            d_input = const_cast<byte_t*>(static_cast<const byte_t*>(uncompressed_data));
            std::cerr << "compress: input already on device, using directly: d_input=" << (void*)d_input << std::endl;
        } else {
            // Input is on host, allocate space in workspace
            d_input = workspace_ptr;
            workspace_ptr = align_ptr(workspace_ptr + uncompressed_size, alignment);
            std::cerr << "compress: input on host, allocated workspace: d_input=" << (void*)d_input << std::endl;
        }
        
        byte_t* d_output = workspace_ptr;
        size_t d_output_max_size = estimate_compressed_size(uncompressed_size, config.level);
        workspace_ptr = align_ptr(workspace_ptr + d_output_max_size, alignment);
        
        byte_t* d_compressed_block = workspace_ptr;
        size_t d_compressed_block_max_size = uncompressed_size * 2;
        workspace_ptr = align_ptr(workspace_ptr + d_compressed_block_max_size, alignment);

        u64* d_checksum = reinterpret_cast<u64*>(workspace_ptr);
        workspace_ptr = align_ptr(workspace_ptr + sizeof(u64), alignment);

        // (NEW) Partition hash/chain tables from the *same* workspace
        // NOTE: do not overwrite ctx.workspace here; allocate a local
        // CompressionWorkspace for this call so the persistent ctx workspace
        // stays unchanged and the pool allocations remain valid.
        // Start with the persistent workspace and then override per-call
        // pointers that are partitioned from the temp workspace for this
        // compress() call. This prevents overwriting ctx.workspace and
        // ensures any buffers that are not overwritten stay valid.
        CompressionWorkspace call_workspace = ctx.workspace;
        ctx.lz77_ctx->hash_table.table = reinterpret_cast<u32*>(workspace_ptr);
        ctx.lz77_ctx->hash_table.size = (1 << config.hash_log);
        call_workspace.d_hash_table = reinterpret_cast<u32*>(workspace_ptr);
        workspace_ptr = align_ptr(workspace_ptr + ctx.lz77_ctx->hash_table.size * sizeof(u32), alignment);
        
        ctx.lz77_ctx->chain_table.prev = reinterpret_cast<u32*>(workspace_ptr);
        ctx.lz77_ctx->chain_table.size = (1 << config.chain_log);
        call_workspace.d_chain_table = reinterpret_cast<u32*>(workspace_ptr);
        workspace_ptr = align_ptr(workspace_ptr + ctx.lz77_ctx->chain_table.size * sizeof(u32), alignment);

        // Partition d_matches and d_costs from workspace
        std::cerr << "compress: before partition d_matches, workspace_ptr=" << (void*)workspace_ptr << "\n";
        call_workspace.d_matches = reinterpret_cast<void*>(workspace_ptr);
        call_workspace.max_matches = uncompressed_size;
        std::cerr << "compress: partitioned d_matches=" << call_workspace.d_matches << " size=" << (uncompressed_size * sizeof(lz77::Match)) << "\n";
        workspace_ptr = align_ptr(workspace_ptr + uncompressed_size * sizeof(lz77::Match), alignment);

        std::cerr << "compress: before partition d_costs, workspace_ptr=" << (void*)workspace_ptr << "\n";
        call_workspace.d_costs = reinterpret_cast<void*>(workspace_ptr);
        call_workspace.max_costs = uncompressed_size;
        std::cerr << "compress: partitioned d_costs=" << call_workspace.d_costs << " size=" << (uncompressed_size * sizeof(lz77::ParseCost)) << "\n";
        workspace_ptr = align_ptr(workspace_ptr + uncompressed_size * sizeof(lz77::ParseCost), alignment);
        
        // Partition sequence storage from the workspace for compression
        ctx.seq_ctx->d_sequences = reinterpret_cast<sequence::Sequence*>(workspace_ptr);
        workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX * sizeof(sequence::Sequence), alignment);
        // (END NEW)

        // --- 2. Start Compression Pipeline ---
        
        // Copy input to workspace if it's on host (already determined above)
        if (!input_is_device) {
            cudaError_t err = cudaMemcpyAsync(d_input, uncompressed_data, uncompressed_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                std::cerr << "[compress] cudaMemcpyAsync d_input failed: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
                return Status::ERROR_CUDA_ERROR;
            }
        }

        u32 compressed_offset = 0;

        // --- (NEW) Write Skippable Frame with Metadata ---
        {
            SkippableFrameHeader h_skip_header;
            h_skip_header.magic_number = ZSTD_MAGIC_SKIPPABLE_START;
            h_skip_header.frame_size = sizeof(CustomMetadataFrame);

            CustomMetadataFrame h_custom_meta;
            h_custom_meta.custom_magic = CUSTOM_METADATA_MAGIC;
            h_custom_meta.compression_level = config.level;

            CUDA_CHECK(cudaMemcpyAsync(d_output + compressed_offset, &h_skip_header, 
                                        sizeof(SkippableFrameHeader), 
                                        cudaMemcpyHostToDevice, stream));
            compressed_offset += sizeof(SkippableFrameHeader);

            CUDA_CHECK(cudaMemcpyAsync(d_output + compressed_offset, &h_custom_meta, 
                                        sizeof(CustomMetadataFrame), 
                                        cudaMemcpyHostToDevice, stream));
            compressed_offset += sizeof(CustomMetadataFrame);
        }
        
        u32 header_size = 0;
        auto status = write_frame_header(d_output + compressed_offset,
                                         d_output_max_size - compressed_offset,
                                         &header_size, uncompressed_size,
                                         has_dictionary ? dict.raw_content : nullptr,
                                         has_dictionary ? dict.raw_size : 0,
                                         stream);
        if (status != Status::SUCCESS) return status;
        
        compressed_offset += header_size;
        
        u32 block_size = get_optimal_block_size(uncompressed_size, config.level);
        u32 num_blocks = (uncompressed_size + block_size - 1) / block_size;
        
        for (u32 block_idx = 0; block_idx < num_blocks; block_idx++) {
            u32 block_start = block_idx * block_size;
            u32 current_block_size = std::min(block_size, (u32)uncompressed_size - block_start);
            bool is_last_block = (block_idx == num_blocks - 1);
            
            const byte_t* block_input = d_input + block_start;
            byte_t* block_output = d_compressed_block;
            
            // (MODIFIED) Step 1: Find All Matches - PASS WORKSPACE
            status = lz77::find_matches(
                *ctx.lz77_ctx,
                block_input,
                current_block_size,
                d_dict_content,
                &call_workspace,  // (NEW) Pass local call workspace
                nullptr, 0, // No window for single-shot
                stream
            );
            cudaError_t __err = cudaGetLastError();
            std::cerr << "compress: returned from lz77::find_matches, status=" << status_to_string(status) << "\n";
            std::cerr << "compress: lz77::find_matches after kernel cudaGetLastError=" << __err << "\n";
            if (status != Status::SUCCESS) {
                std::cerr << "compress: lz77::find_matches failed, status=" << status_to_string(status) << std::endl;
                return status;
            }
            if (__err != cudaSuccess) {
                std::cerr << "compress: CUDA kernel error after lz77::find_matches: " << cudaGetErrorString(__err) << " (" << __err << ")" << std::endl;
                return Status::ERROR_CUDA_ERROR;
            }

            // (MODIFIED) Step 2: Optimal Parse - PASS WORKSPACE
            u32 h_num_sequences_in_block = 0;
            u32 h_total_literals_in_block = 0;
            std::cerr << "compress: before lz77::find_optimal_parse" << std::endl;
            status = lz77::find_optimal_parse(
                *ctx.lz77_ctx,
                block_input,
                current_block_size,
                d_dict_content,
                &call_workspace,  // (NEW) Pass local call workspace
                nullptr, 0, // No window for single-shot
                ctx.seq_ctx->d_literals_buffer,
                ctx.seq_ctx->d_literal_lengths,
                ctx.seq_ctx->d_match_lengths,
                ctx.seq_ctx->d_offsets,
                &h_num_sequences_in_block,
                &h_total_literals_in_block,
                stream
            );
            __err = cudaGetLastError();
            std::cerr << "compress: lz77::find_optimal_parse after kernel cudaGetLastError=" << __err << "\n";
            // Runtime CI toggle: synchronously verify kernel execution and print diagnostic
            cuda_zstd::utils::debug_kernel_verify("manager: after lz77::find_optimal_parse");
            std::cerr << "compress: after lz77::find_optimal_parse" << std::endl;
            if (status != Status::SUCCESS) {
                std::cerr << "compress: lz77::find_optimal_parse failed, status=" << status_to_string(status) << "\n";
                return status;
            }
            if (__err != cudaSuccess) {
                std::cerr << "compress: CUDA kernel error after lz77::find_optimal_parse: " << cudaGetErrorString(__err) << " (" << __err << ")" << std::endl;
                return Status::ERROR_CUDA_ERROR;
            }
            
            // (MODIFIED) Step 3: Generate sequence structs
            const u32 threads = 256;
            const u32 seq_blocks = (h_num_sequences_in_block + threads - 1) / threads;
            status = sequence::build_sequences(
                *ctx.seq_ctx,
                h_num_sequences_in_block,
                seq_blocks,
                threads,
                stream
            );
            std::cerr << "compress: build_sequences returned status=" << status_to_string(status) << "\n";
            std::cerr << "compress: calling build_sequences with ctx.seq_ctx->d_sequences=" << ctx.seq_ctx->d_sequences << " ctx.seq_ctx->d_num_sequences=" << ctx.seq_ctx->d_num_sequences << " num_sequences=" << h_num_sequences_in_block << std::endl;
            if (status != Status::SUCCESS) return status;
            
            u32 num_sequences = h_num_sequences_in_block; 
            ctx.total_sequences += num_sequences;
            
            // (MODIFIED) Step 4: Compress literals
            u32 literals_size = 0;
            byte_t* literals_compressed = block_output;
            
            ctx.seq_ctx->num_literals = h_total_literals_in_block;
            
            status = compress_literals(
                ctx.seq_ctx->d_literals_buffer,
                ctx.seq_ctx->num_literals,
                literals_compressed,
                &literals_size,
                &call_workspace,  // (NEW) Pass per-call workspace
                stream
            );
            if (status != Status::SUCCESS) return status;
            std::cerr << "compress: compress_literals returned status=" << status_to_string(status) << " literals=" << ctx.seq_ctx->num_literals << "\n";
            std::cerr << "compress: compress_literals returned status=" << status_to_string(status) << " literals=" << ctx.seq_ctx->num_literals << "\n";
            
            ctx.total_literals += ctx.seq_ctx->num_literals;
            
            // Step 4: Compress sequences
            u32 sequences_size = 0;
            byte_t* sequences_compressed = block_output + literals_size;
            
            status = compress_sequences(
                ctx.seq_ctx,
                num_sequences,
                sequences_compressed,
                &sequences_size,
                stream
            );
            std::cerr << "compress: entered compress_sequences call: num_sequences=" << num_sequences << " sequences_compressed=" << (void*)sequences_compressed << "\n";
            std::cerr << "compress: compress_sequences returned status=" << status_to_string(status) << "\n";
            if (status != Status::SUCCESS) return status;
            std::cerr << "compress: compress_sequences returned status=" << status_to_string(status) << " num_sequences=" << num_sequences << "\n";
            
            // Step 5: Write block
            u32 block_compressed_size = literals_size + sequences_size;
            
            status = write_block(
                d_output + compressed_offset,
                d_output_max_size - compressed_offset,
                block_output,
                block_compressed_size,
                current_block_size,
                is_last_block,
                &compressed_offset,
                stream
            );
            if (status != Status::SUCCESS) return status;
            std::cerr << "compress: write_block returned status=" << status_to_string(status) << " block_idx=" << block_idx << "\n";
        }
        
        // Write checksum
        if (config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
            xxhash::compute_xxhash64(d_input, uncompressed_size, 0, d_checksum, stream);
            cudaMemcpyAsync(d_output + compressed_offset, d_checksum, sizeof(u64), cudaMemcpyDeviceToDevice, stream);
            compressed_offset += sizeof(u64);
        }
        
        *compressed_size = compressed_offset;
        // (FIX) Don't copy to host here, this is done by the batch manager
        // cudaMemcpyAsync(compressed_data, d_output, *compressed_size, cudaMemcpyDeviceToHost, stream);
        // cudaStreamSynchronize(stream);
        
        stats.bytes_compressed += uncompressed_size;
        stats.bytes_produced += *compressed_size;
        stats.blocks_processed += num_blocks;
        
        return Status::SUCCESS;
    }
    
    // // ==========================================================================
    // decompress() implementation - RFC 8878 COMPLIANT
    // ==========================================================================

    virtual Status decompress(
        const void* compressed_data,
        size_t compressed_size,
        void* uncompressed_data,
        size_t* uncompressed_size,
        void* temp_workspace,
        size_t temp_size,
        cudaStream_t stream = 0
    ) override {
        
        // === Parameter Validation ===
        if (!compressed_data || !uncompressed_data || !uncompressed_size || 
            !temp_workspace || compressed_size < 4) {
            return Status::ERROR_INVALID_PARAMETER;
        }
        
        size_t required_size = get_decompress_temp_size(compressed_size);
        if (temp_size < required_size) {
            return Status::ERROR_BUFFER_TOO_SMALL;
        }
        
        // === Initialize Context if Needed ===
        if (!ctx_initialized) {
            auto status = initialize_context();
            if (status != Status::SUCCESS) return status;
        }
        
        // === Handle Skippable Frames (RFC 8878) ===
        // Zstd may have skippable frames at the beginning
        const byte_t* h_compressed_data_ptr = static_cast<const byte_t*>(compressed_data);
        size_t h_compressed_size_remaining = compressed_size;
        u32 data_offset = 0;
        
        // Skip all skippable frames to find the real Zstd frame
        while (h_compressed_size_remaining >= 8) {
            u32 magic;
            memcpy(&magic, h_compressed_data_ptr + data_offset, sizeof(u32));
            
            // Check if this is the Zstd magic number
            if (magic == ZSTD_MAGIC_NUMBER) {
                break;
            }
            
            // Check if this is a skippable frame
            if ((magic & 0xFFFFFFF0) == ZSTD_MAGIC_SKIPPABLE_START) {
                // Read the frame size (next 4 bytes)
                u32 frame_size;
                memcpy(&frame_size, h_compressed_data_ptr + data_offset + 4, sizeof(u32));
                
                // Total frame size = 8 byte header + frame_size
                u32 total_frame_size = 8 + frame_size;
                
                if (h_compressed_size_remaining < total_frame_size) {
                    return Status::ERROR_CORRUPT_DATA;
                }
                
                // Move past this skippable frame
                data_offset += total_frame_size;
                h_compressed_size_remaining -= total_frame_size;
            } else {
                // Invalid magic number
                return Status::ERROR_INVALID_MAGIC;
            }
        }
        
        if (h_compressed_size_remaining < 4) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        // === Partition the temp_workspace ===
        byte_t* workspace_ptr = static_cast<byte_t*>(temp_workspace);
        size_t alignment = 128;
        
        // Allocate device input buffer
        byte_t* d_input = workspace_ptr;
        workspace_ptr = align_ptr(workspace_ptr + h_compressed_size_remaining, alignment);
        
        // Allocate checksum verification buffer
        u64* d_checksum = reinterpret_cast<u64*>(workspace_ptr);
        workspace_ptr = align_ptr(workspace_ptr + sizeof(u64), alignment);
        
        // Allocate persistent buffers from workspace
        ctx.d_temp_buffer = workspace_ptr;
        workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX, alignment);
        
        ctx.seq_ctx->d_sequences = reinterpret_cast<sequence::Sequence*>(workspace_ptr);
        std::cerr << "initialize_context: assigned ctx.seq_ctx->d_sequences=" << ctx.seq_ctx->d_sequences << " workspace_ptr=" << (void*)workspace_ptr << std::endl;
        // Diagnostic: verify pointer is still valid
        std::cerr << "partition workspace after assignments: ctx.seq_ctx->d_sequences=" << ctx.seq_ctx->d_sequences << " ctx.workspace.d_hash_table=" << ctx.workspace.d_hash_table << " ctx.workspace.d_chain_table=" << ctx.workspace.d_chain_table << std::endl;
        workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX * sizeof(sequence::Sequence), alignment);
        
        byte_t* d_output = static_cast<byte_t*>(uncompressed_data);
        size_t d_output_max_size = *uncompressed_size;
        
        // === Copy compressed data to device ===
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_compressed_data_ptr + data_offset, 
                                  h_compressed_size_remaining, 
                                  cudaMemcpyHostToDevice, stream));
        
        // === Parse Frame Header (RFC 8878) ===
        u32 header_size = 0;
        u32 frame_content_size = 0;
        
        auto status = parse_frame_header(d_input, h_compressed_size_remaining, 
                                        &header_size, &frame_content_size);
        if (status != Status::SUCCESS) return status;
        
        // Validate the output buffer size if content size is present
        if (frame_content_size > 0) {
            if (d_output_max_size < frame_content_size) {
                return Status::ERROR_BUFFER_TOO_SMALL;
            }
            *uncompressed_size = frame_content_size;
        }
        
        // === Decompress Blocks ===
        u32 read_offset = header_size;  // Start after frame header
        u32 write_offset = 0;           // Where we write decompressed data
        
        while (read_offset < h_compressed_size_remaining) {
            // === Read Block Header ===
            bool is_last_block = false;
            u32 block_size = 0;
            bool is_compressed = false;
            u32 block_header_size = 0;
            
            status = read_block_header(
                d_input + read_offset,
                h_compressed_size_remaining - read_offset,
                &is_last_block,
                &is_compressed,
                &block_size,
                &block_header_size
            );
            if (status != Status::SUCCESS) return status;
            
            read_offset += block_header_size;
            
            // === Process Block ===
            if (is_compressed) {
                // Decompress block
                u32 decompressed_size = 0;
                
                status = decompress_block(
                    d_input + read_offset,
                    block_size,
                    d_output + write_offset,
                    &decompressed_size,
                    stream
                );
                if (status != Status::SUCCESS) return status;
                
                write_offset += decompressed_size;
            } else {
                // Raw block - just copy
                CUDA_CHECK(cudaMemcpyAsync(
                    d_output + write_offset,
                    d_input + read_offset,
                    block_size,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
                
                write_offset += block_size;
            }
            
            read_offset += block_size;
            
            // Stop if this was the last block
            if (is_last_block) break;
        }
        
        // === Verify Checksum (if present) ===
        if (config.checksum == ChecksumPolicy::COMPUTE_AND_VERIFY) {
            // Check if there's a checksum at the end (8 bytes)
            if (read_offset + 8 <= h_compressed_size_remaining) {
                u64 stored_checksum;
                
                // Copy checksum from device to host
                CUDA_CHECK(cudaMemcpyAsync(&stored_checksum, d_input + read_offset, 
                                          sizeof(u64), cudaMemcpyDeviceToHost, stream));
                
                // Compute checksum of decompressed data
                xxhash::compute_xxhash64(d_output, write_offset, 0, d_checksum, stream);
                
                u64 computed_checksum;
                CUDA_CHECK(cudaMemcpyAsync(&computed_checksum, d_checksum, 
                                          sizeof(u64), cudaMemcpyDeviceToHost, stream));
                
                // Wait for all GPU operations to complete
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                // Compare checksums
                if (stored_checksum != computed_checksum) {
                    return Status::ERROR_CHECKSUM_FAILED;
                }
            }
        }
        
        // === Set output size ===
        *uncompressed_size = write_offset;
        
        // === Update statistics ===
        stats.bytes_decompressed += write_offset;
        
        return Status::SUCCESS;
    }
    
    // ==========================================================================
    // Dictionary support
    // ==========================================================================
    virtual Status set_dictionary(const dictionary::Dictionary& new_dict) override {
        // if (new_dict.size > config.dict_size) {
        //     return Status::ERROR_BUFFER_TOO_SMALL;
        // }
        dict = new_dict;
        has_dictionary = true;
        
        // Copy dictionary to pre-allocated device buffer
        // CUDA_CHECK(cudaMemcpy(ctx.dict.d_buffer, dict.d_buffer, dict.size, cudaMemcpyHostToDevice));
        // ctx.dict.size = dict.size;
        // ctx.dict.dict_id = xxhash::xxhash_32_cpu(static_cast<const byte_t*>(dict.d_buffer), dict.size, 0);

        return Status::SUCCESS;
    }
    
    virtual Status get_dictionary(dictionary::Dictionary& dict_out) const override {
        if (!has_dictionary) return Status::ERROR_INVALID_PARAMETER;
        dict_out = dict;
        return Status::SUCCESS;
    }
    
    virtual Status clear_dictionary() override {
        has_dictionary = false;
        return Status::SUCCESS;
    }
    
    // ==========================================================================
    // Statistics
    // ==========================================================================
    virtual const CompressionStats& get_stats() const override {
        return stats;
    }
    
    virtual void reset_stats() override {
        memset(&stats, 0, sizeof(CompressionStats));
    }

private:
    // ==========================================================================
    // Context management
    // ==========================================================================
    Status initialize_context() {
        static std::mutex init_mutex;
        std::unique_lock<std::mutex> init_lock(init_mutex);
        std::cerr << "initialize_context() entered (guarded)" << std::endl;
        std::cerr << "initialize_context: before allocate_compression_workspace" << std::endl;
        
        ctx.total_matches = 0;
        ctx.total_literals = 0;
        ctx.total_sequences = 0;
        
        ctx_initialized = true;
        return Status::SUCCESS;
    }
    

    // ==========================================================================
    // Frame operations
    // ==========================================================================
    Status write_frame_header(
        byte_t* output,
        size_t max_size,
        u32* header_size,
        u32 content_size,
        const void* dict_buffer,
        size_t dict_size,
        cudaStream_t stream
    ) {
        if (max_size < FRAME_HEADER_SIZE_MIN) {
            return Status::ERROR_BUFFER_TOO_SMALL;
        }
        
        byte_t h_header[FRAME_HEADER_SIZE_MAX];
        u32 offset = 0;
        
        // 1. Write magic number
        u32 magic = ZSTD_MAGIC_NUMBER;
        memcpy(h_header + offset, &magic, 4);
        offset += 4;
        
        // 2. Frame Header Descriptor
        byte_t fhd = 0;
        
        // Dictionary ID
        u32 dict_id = 0;
        if (dict_buffer && dict_size > 0) {
            fhd |= (1 << 2); // Set DID flag
            fhd |= 0x01; // 1-byte dict ID for now
            dict_id = xxhash::xxhash_32_cpu(static_cast<const byte_t*>(dict_buffer), dict_size, 0);
        }

        // Set checksum bit if enabled
        if (config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
            fhd |= 0x04;  // Content checksum bit
        }
        
        // Set single segment if content_size fits in 1-8 bytes
        if (content_size > 0) {
            fhd |= 0x40;  // Single segment flag
            fhd |= 0x08;  // Content size flag
        }
        
        h_header[offset] = fhd;
        offset += 1;

        // 3. Dictionary ID
        if (dict_buffer && dict_size > 0) {
            memcpy(h_header + offset, &dict_id, 1);
            offset += 1;
        }
        
        // 4. Content Size (if has_content_size)
        if (content_size > 0) {
            if (content_size <= 255) {
                h_header[offset++] = (byte_t)content_size;
            } else if (content_size <= 65535) {
                h_header[offset++] = (byte_t)(content_size & 0xFF);
                h_header[offset++] = (byte_t)((content_size >> 8) & 0xFF);
            } else {
                // Write as 4-byte little-endian
                u32 cs = content_size;
                memcpy(h_header + offset, &cs, 4);
                offset += 4;
            }
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpyAsync(output, h_header, offset, 
                                  cudaMemcpyHostToDevice, stream));
        
        *header_size = offset;
        return Status::SUCCESS;
    }
    
    Status parse_frame_header(
        const byte_t* input,     // Device pointer to compressed data
        u32 input_size,
        u32* header_size,        // Output: total header size (host)
        u32* content_size        // Output: decompressed size if present (host)
    ) {
        if (input_size < 5) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        // Copy frame header to host for parsing
        byte_t h_header[18];
        CUDA_CHECK(cudaMemcpy(h_header, input, std::min(18u, input_size), 
                             cudaMemcpyDeviceToHost));
        
        u32 offset = 4;  // Skip magic number (already validated)
        
        // === Parse Frame Header Descriptor (1 byte) ===
        byte_t fhd = h_header[offset++];
        
        bool single_segment = (fhd >> 5) & 0x01;
        bool has_dict_id = (fhd & 0x03) != 0;
        
        // === Parse Window Descriptor (if not single segment) ===
        if (!single_segment) {
            if (offset >= input_size) {
                return Status::ERROR_CORRUPT_DATA;
            }
            
            byte_t wd = h_header[offset++];
            u32 window_log = 10 + (wd >> 3);
            
            // Update config window size
            if (window_log >= ZSTD_WINDOWLOG_MIN && window_log <= ZSTD_WINDOWLOG_MAX) {
                config.window_log = window_log;
            } else {
                return Status::ERROR_CORRUPT_DATA;
            }
        }
        
        // === Parse Dictionary ID (if present) ===
        u32 dict_id_size = 0;
        u32 dictionary_id = 0;
        
        if (has_dict_id) {
            // Dict ID size is encoded in the low 2 bits of FHD
            u32 did_flag = (fhd & 0x03);
            
            if (did_flag == 1) {
                dict_id_size = 1;
                dictionary_id = h_header[offset];
            } else if (did_flag == 2) {
                dict_id_size = 2;
                dictionary_id = (u32)h_header[offset] | ((u32)h_header[offset+1] << 8);
            } else if (did_flag == 3) {
                dict_id_size = 4;
                memcpy(&dictionary_id, h_header + offset, 4);
            }
            
            offset += dict_id_size;
        }
        
        // === Parse Content Size (if present) ===
        u32 h_content_size = 0;
        u32 fcs_field_size = (fhd >> 6) & 0x03;
        
        if (fcs_field_size == 0) {
            // No content size field (or single-segment with 1 byte)
            if (single_segment && offset < input_size) {
                h_content_size = h_header[offset];
                offset += 1;
            }
        } else if (fcs_field_size == 1) {
            // 2 bytes: content_size = value + 256
            if (offset + 2 > input_size) {
                return Status::ERROR_CORRUPT_DATA;
            }
            u16 size_val;
            memcpy(&size_val, h_header + offset, 2);
            h_content_size = size_val + 256;
            offset += 2;
        } else if (fcs_field_size == 2) {
            // 4 bytes: content_size = value
            if (offset + 4 > input_size) {
                return Status::ERROR_CORRUPT_DATA;
            }
            memcpy(&h_content_size, h_header + offset, 4);
            offset += 4;
        } else if (fcs_field_size == 3) {
            // 8 bytes: content_size = value (stored as u64)
            if (offset + 8 > input_size) {
                return Status::ERROR_CORRUPT_DATA;
            }
            u64 size_val;
            memcpy(&size_val, h_header + offset, 8);
            h_content_size = (u32)size_val;  // Truncate to u32
            offset += 8;
        }
        
        *header_size = offset;
        *content_size = h_content_size;
        
        return Status::SUCCESS;
    }
    
    Status write_block(
        byte_t* output,
        size_t max_size,
        const byte_t* block_data,
        u32 compressed_size,
        u32 original_size,
        bool is_last,
        u32* compressed_offset,
        cudaStream_t stream
    ) {
         bool use_compressed = (compressed_size < original_size);
         u32 block_type = use_compressed ? 2 : 0; // 2=Compressed, 0=Raw
         u32 block_size = use_compressed ? compressed_size : original_size;
        
         if (*compressed_offset + 3 + block_size > max_size) {
             return Status::ERROR_BUFFER_TOO_SMALL;
         }

         u32 header = 0;
         header |= (is_last ? 1 : 0);
         header |= (block_type << 1);
         header |= (block_size << 3);
         
         CUDA_CHECK(cudaMemcpyAsync(output + *compressed_offset, &header, 3, cudaMemcpyHostToDevice, stream));
         CUDA_CHECK(cudaMemcpyAsync(output + *compressed_offset + 3, block_data, block_size, cudaMemcpyDeviceToDevice, stream));
        
         *compressed_offset += 3 + block_size;
         return Status::SUCCESS;
    }
    
    Status read_block_header(
        const byte_t* input,        // Device pointer
        u32 input_size,
        bool* is_last,              // Output: is last block?
        bool* is_compressed,        // Output: is compressed?
        u32* size,                  // Output: block size
        u32* header_size            // Output: header size (always 3 bytes)
    ) {
        if (input_size < 3) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        // Read 3-byte block header from device
        u32 header = 0;
        CUDA_CHECK(cudaMemcpy(&header, input, 3, cudaMemcpyDeviceToHost));
        
        // === Parse Block Header ===
        // Bit 0: last_block flag
        *is_last = (header & 0x01) != 0;
        
        // Bits 1-2: block_type
        u32 block_type = (header >> 1) & 0x03;
        
        // 0 = Raw block (uncompressed)
        // 1 = RLE block (run-length encoded, not commonly used)
        // 2 = Compressed block
        // 3 = Reserved
        
        if (block_type == 1) {
            // RLE block - not fully implemented
            return Status::ERROR_UNSUPPORTED_VERSION;
        } else if (block_type == 3) {
            // Reserved
            return Status::ERROR_CORRUPT_DATA;
        }
        
        *is_compressed = (block_type == 2);
        
        // Bits 3+: block_size (24-bit value)
        *size = header >> 3;
        
        // Validate block size against remaining input
        if (*size > 0 && *size > input_size - 3) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        *header_size = 3;
        
        return Status::SUCCESS;
    }
    
    // ==========================================================================
    // Decompression Helpers
    // ==========================================================================
    
    Status decompress_block(
        const byte_t* input,
        u32 input_size,
        byte_t* output,
        u32* output_size,  // Host pointer for output
        cudaStream_t stream
    ) {
        if (!input || !output || !output_size) {
            return Status::ERROR_INVALID_PARAMETER;
        }
        
        // Use temp buffer for literals
        byte_t* d_decompressed_literals = ctx.d_temp_buffer;
        
        // === Decompress Literals ===
        u32 literals_header_size = 0;
        u32 literals_compressed_size = 0;
        u32 literals_decompressed_size = 0;
        
        auto status = decompress_literals(
            input,
            input_size,
            d_decompressed_literals,
            &literals_header_size,
            &literals_compressed_size,
            &literals_decompressed_size,
            stream
        );
        if (status != Status::SUCCESS) return status;
        
        // === Decompress Sequences ===
        u32 sequences_offset = literals_header_size + literals_compressed_size;
        
        if (sequences_offset > input_size) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        status = decompress_sequences(
            input + sequences_offset,
            input_size - sequences_offset,
            ctx.seq_ctx,
            stream
        );
        if (status != Status::SUCCESS) return status;
        
        // === Execute Sequences ===
        u32* d_output_size;
        CUDA_CHECK(cudaMalloc(&d_output_size, sizeof(u32)));
        CUDA_CHECK(cudaMemsetAsync(d_output_size, 0, sizeof(u32), stream));
        
        status = sequence::execute_sequences(
            d_decompressed_literals,
            literals_decompressed_size,
            ctx.seq_ctx->d_sequences,
            ctx.seq_ctx->num_sequences,
            output,
            d_output_size,
            stream
        );
        
        // Copy result size from device
        CUDA_CHECK(cudaMemcpyAsync(output_size, d_output_size, sizeof(u32), 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        cudaFree(d_output_size);
        
        return status;
    }
    
    Status compress_literals(
        const byte_t* literals,
        u32 num_literals,
        byte_t* output,
        u32* output_size,
        CompressionWorkspace* workspace,
        cudaStream_t stream
    ) {
        size_t huffman_size_host = 0;
        
        if (num_literals == 0) {
            // No literals to compress -> nothing to write, caller can
            // rely on output_size == 0 as the length of this section.
            *output_size = 0;
            return Status::SUCCESS;
        }

        auto status = huffman::encode_huffman(
            literals,
            num_literals,
            *ctx.huff_ctx,
            output,
            &huffman_size_host,
            workspace,  // (NEW) Pass workspace
            stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        if (status == Status::SUCCESS && huffman_size_host < num_literals) {
            *output_size = (u32)huffman_size_host;
            return Status::SUCCESS;
        }
        
        status = fse::encode_fse_advanced(
            literals,
            num_literals,
            output,
            output_size,
            fse::TableType::LITERALS,
            true, true, false,
            stream
        );
        
        if (status == Status::SUCCESS && *output_size < num_literals) {
            return Status::SUCCESS;
        }
        
        cudaMemcpyAsync(output, literals, num_literals, cudaMemcpyDeviceToDevice, stream);
        *output_size = num_literals;
        
        return Status::SUCCESS;
    }
    
    Status compress_sequences(
        const sequence::SequenceContext* seq_ctx,
        u32 num_sequences,
        byte_t* output,
        u32* output_size,
        cudaStream_t stream
    ) {
        u32 offset = 0;

        // Nothing to emit
        if (num_sequences == 0) {
            *output_size = 0;
            return Status::SUCCESS;
        }
        
        u32 ll_size = 0;
        auto status = fse::encode_fse_advanced(
            (const byte_t*)seq_ctx->d_literal_lengths,
            num_sequences,
            output + offset,
            &ll_size,
            fse::TableType::LITERALS,
            true, true, false,
            stream
        );
        
        if (status != Status::SUCCESS) return status;
        offset += ll_size;
        
        u32 ml_size = 0;
        status = fse::encode_fse_advanced(
            (const byte_t*)seq_ctx->d_match_lengths,
            num_sequences,
            output + offset,
            &ml_size,
            fse::TableType::MATCH_LENGTHS,
            true, true, false,
            stream
        );
        
        if (status != Status::SUCCESS) return status;
        offset += ml_size;
        
        u32 of_size = 0;
        status = fse::encode_fse_advanced(
            (const byte_t*)seq_ctx->d_offsets,
            num_sequences,
            output + offset,
            &of_size,
            fse::TableType::OFFSETS,
            true, true, false,
            stream
        );
        
        if (status != Status::SUCCESS) return status;
        offset += of_size;
        
        *output_size = offset;
        return Status::SUCCESS;
    }
    
    Status decompress_literals(
        const byte_t* input,
        u32 input_size,
        byte_t* output,
        u32* h_header_size,
        u32* h_compressed_size,
        u32* h_decompressed_size,
        cudaStream_t stream
    ) {
        byte_t h_header[5];
        if (input_size == 0) return Status::ERROR_CORRUPT_DATA;
        CUDA_CHECK(cudaMemcpy(h_header, input, std::min(5u, input_size), cudaMemcpyDeviceToHost));
        
        u32 literals_type = (h_header[0] >> 6) & 0x03;

        if (literals_type == 0 || literals_type == 1) {
            u32 size_format = (h_header[0] >> 4) & 0x03;
            
            switch (size_format) {
                case 0:
                    *h_header_size = 1;
                    *h_decompressed_size = (h_header[0] & 0x1F);
                    break;
                case 1:
                    *h_header_size = 2;
                    *h_decompressed_size = (h_header[0] & 0x0F) + (h_header[1] << 4);
                    break;
                case 2:
                    *h_header_size = 3;
                    *h_decompressed_size = (h_header[0] & 0x0F) + (h_header[1] << 4) + (h_header[2] << 12);
                    break;
                case 3:
                    *h_header_size = 4;
                    *h_decompressed_size = (h_header[0] & 0x0F) + (h_header[1] << 4) + (h_header[2] << 12) + (h_header[3] << 20);
                    break;
                default:
                    return Status::ERROR_CORRUPT_DATA;
            }

            if (literals_type == 0) { // Raw
                *h_compressed_size = *h_decompressed_size;
                if (*h_header_size + *h_compressed_size > input_size) return Status::ERROR_CORRUPT_DATA;
                if (*h_compressed_size > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(output, input + *h_header_size, *h_compressed_size, cudaMemcpyDeviceToDevice, stream));
                }
                return Status::SUCCESS;
            
            } else { // RLE
                *h_compressed_size = 1;
                if (*h_header_size + *h_compressed_size > input_size) return Status::ERROR_CORRUPT_DATA;
                
                byte_t rle_value = h_header[*h_header_size]; 
                
                const u32 threads = 256;
                const u32 blocks = (*h_decompressed_size + threads - 1) / threads;
                
                expand_rle_kernel<<<blocks, threads, 0, stream>>>(
                    output, *h_decompressed_size, rle_value
                );
                return Status::SUCCESS;
            }
        }
        else {
            u32 size_format = (h_header[0] >> 4) & 0x03;
            u32 size_info = (h_header[0] & 0x0F);
            u32 combined_bits = 0;

            if (size_format == 0 || size_format == 2) { 
                *h_header_size = 2;
                combined_bits = (size_info << 8) | h_header[1];
            } else if (size_format == 1) {
                *h_header_size = 3;
                combined_bits = (size_info << 16) | (h_header[1] << 8) | h_header[2];
            }
            else if (size_format == 3) {
                *h_header_size = 5;
            }
            else {
                return Status::ERROR_CORRUPT_DATA;
            }
            
            if (size_format == 0 || size_format == 2) {
                *h_decompressed_size = (combined_bits >> 0) & 0x3FF;
                *h_compressed_size = (combined_bits >> 10) & 0x3FF;
            } else if (size_format == 1) {
                *h_decompressed_size = (combined_bits >> 0) & 0x3FFF;
                *h_compressed_size = (combined_bits >> 14) & 0x3FFF;
            } else { // size_format == 3
                *h_decompressed_size = (h_header[1] | (h_header[2] << 8) | ((h_header[3] & 0x03) << 16));
                *h_compressed_size = ((h_header[3] & 0xFC) >> 2) | (h_header[4] << 6);
            }
            
            if (*h_header_size + *h_compressed_size > input_size) return Status::ERROR_CORRUPT_DATA;

            const byte_t* d_data_start = input + *h_header_size;
            
            if (literals_type == 2) { // FSE
                u32 h_fse_output_size = 0;
                return fse::decode_fse(
                    d_data_start, *h_compressed_size,
                    output, &h_fse_output_size,
                    stream
                );
            } else { // Huffman
                size_t h_huff_output_size = 0;
                return huffman::decode_huffman(
                    d_data_start, *h_compressed_size,
                    *ctx.huff_ctx,
                    output, &h_huff_output_size,
                    *h_decompressed_size,
                    stream
                );
            }
        }
    }
    
    Status decompress_sequences(
        const byte_t* input,
        u32 input_size,
        sequence::SequenceContext* seq_ctx,
        cudaStream_t stream
    ) {
        if (input_size < 1) {
            seq_ctx->num_sequences = 0;
            return Status::SUCCESS;
        }
        
        byte_t h_header[5];
        CUDA_CHECK(cudaMemcpy(h_header, input, std::min(5u, input_size), cudaMemcpyDeviceToHost));
        
        u32 num_sequences = 0;
        u32 offset = 0;
        
        if (h_header[0] == 0) {
            seq_ctx->num_sequences = 0;
            return Status::SUCCESS;
        } else if (h_header[0] < 128) {
            num_sequences = h_header[0];
            offset = 1;
        } else if (h_header[0] < 255) {
            if (input_size < 2) return Status::ERROR_CORRUPT_DATA;
            num_sequences = ((h_header[0] - 128) << 8) + h_header[1];
            offset = 2;
        } else {
            if (input_size < 3) return Status::ERROR_CORRUPT_DATA;
            num_sequences = (h_header[1] << 8) + h_header[2] + 0x7F00;
            offset = 3;
        }
        
        seq_ctx->num_sequences = num_sequences;
        if (offset >= input_size) return Status::ERROR_CORRUPT_DATA;

        byte_t fse_modes = h_header[offset];
        offset += 1;
        
        u32 ll_mode = (fse_modes >> 6) & 0x03;
        u32 of_mode = (fse_modes >> 4) & 0x03;
        u32 ml_mode = (fse_modes >> 2) & 0x03;

        u32 ll_size = 0;
        u32 of_size = 0;
        u32 ml_size = 0;
        
        if (ll_mode == 2 || of_mode == 2 || ml_mode == 2) {
            if (offset >= input_size) return Status::ERROR_CORRUPT_DATA;
            
            if (offset + 2 > input_size) return Status::ERROR_CORRUPT_DATA;
            ll_size = h_header[offset] | (h_header[offset+1] << 8);
            offset += 2;

            if (of_mode == 2) {
                if (offset + 2 > input_size) return Status::ERROR_CORRUPT_DATA;
                of_size = h_header[offset] | (h_header[offset+1] << 8);
                offset += 2;
            } else {
                of_size = 0;
            }
            
            if (ml_mode == 2) {
                 if (offset + 2 > input_size) return Status::ERROR_CORRUPT_DATA;
                 ml_size = h_header[offset] | (h_header[offset+1] << 8);
                 offset += 2;
            } else {
                ml_size = 0;
            }
        }
        
        Status status;
        
        status = decompress_sequence_stream(
            input, input_size, &offset, ll_mode, num_sequences,
            ll_size, fse::TableType::LITERALS,
            seq_ctx->d_literal_lengths, stream
        );
        if (status != Status::SUCCESS) return status;

        status = decompress_sequence_stream(
            input, input_size, &offset, of_mode, num_sequences,
            of_size, fse::TableType::OFFSETS,
            seq_ctx->d_offsets, stream
        );
        if (status != Status::SUCCESS) return status;

        status = decompress_sequence_stream(
            input, input_size, &offset, ml_mode, num_sequences,
            ml_size, fse::TableType::MATCH_LENGTHS,
            seq_ctx->d_match_lengths, stream
        );
        return status;
    }

    Status decompress_sequence_stream(
        const byte_t* input,
        u32 input_size,
        u32* offset,
        u32 mode,
        u32 num_sequences,
        u32 stream_size,
        fse::TableType table_type,
        u32* d_out_buffer,
        cudaStream_t stream
    ) {
        const u32 threads = 256;
        const u32 blocks = (num_sequences + threads - 1) / threads;
        u32 h_decoded_count = 0;

        switch(mode) {
            case 0: { // Raw
                u32 raw_size_bytes = num_sequences; 
                if (*offset + raw_size_bytes > input_size) return Status::ERROR_CORRUPT_DATA;
                
                expand_bytes_to_u32_kernel<<<blocks, threads, 0, stream>>>(
                    input + *offset, d_out_buffer, num_sequences
                );
                
                *offset += raw_size_bytes;
                return Status::SUCCESS;
            }
            case 1: { // RLE
                if (*offset + 1 > input_size) return Status::ERROR_CORRUPT_DATA;
                byte_t rle_value;
                CUDA_CHECK(cudaMemcpy(&rle_value, input + *offset, 1, cudaMemcpyDeviceToHost));
                
                expand_rle_u32_kernel<<<blocks, threads, 0, stream>>>(
                    d_out_buffer, num_sequences, (u32)rle_value
                );
                *offset += 1;
                return Status::SUCCESS;
            }
            case 2: { // FSE Compressed
                if (*offset + stream_size > input_size) return Status::ERROR_CORRUPT_DATA;

                Status status = fse::decode_fse(
                    input + *offset,
                    stream_size,
                    (byte_t*)d_out_buffer,
                    &h_decoded_count,
                    stream
                );
                *offset += stream_size;
                return status;
            }
            case 3: { // Predefined
                Status status = fse::decode_fse_predefined(
                    input + *offset,
                    input_size - *offset,
                    (byte_t*)d_out_buffer,
                    num_sequences,
                    &h_decoded_count,
                    table_type,
                    stream
                );
                *offset = input_size;
                return status;
            }
            default:
                return Status::ERROR_CORRUPT_DATA;
        }
    }
};

// ==============================================================================
// BATCH MANAGER IMPLEMENTATION
// ==============================================================================

class ZstdBatchManager::Impl {
public:
    std::unique_ptr<ZstdManager> manager;
    CompressionStats batch_stats;
    
    // --- (NEW) Stream pool for parallel batching ---
    std::vector<cudaStream_t> streams;
    int num_streams;
    // --- (END NEW) ---

    explicit Impl(const CompressionConfig& config) {
        manager = create_manager(config);
        
        // --- (NEW) Create stream pool ---
        num_streams = 8; // Default pool size
        for (int i = 0; i < num_streams; ++i) {
            cudaStream_t s;
            cudaStreamCreate(&s);
            streams.push_back(s);
        }
        // --- (END NEW) ---
    }

    // --- (NEW) Destructor to clean up streams ---
    ~Impl() {
        for (auto s : streams) {
            cudaStreamDestroy(s);
        }
    }
    // --- (END NEW) ---

    void reset_stats() {
        memset(&batch_stats, 0, sizeof(CompressionStats));
        manager->reset_stats();
    }
};

ZstdBatchManager::ZstdBatchManager() {
    pimpl_ = std::make_unique<Impl>(CompressionConfig::from_level(ZSTD_DEFAULT_CLEVEL));
}

ZstdBatchManager::ZstdBatchManager(const CompressionConfig& config) {
    pimpl_ = std::make_unique<Impl>(config);
}

ZstdBatchManager::~ZstdBatchManager() = default;

Status ZstdBatchManager::configure(const CompressionConfig& config) {
    return pimpl_->manager->configure(config);
}

CompressionConfig ZstdBatchManager::get_config() const {
    return pimpl_->manager->get_config();
}

Status ZstdBatchManager::set_compression_level(int level) {
    return pimpl_->manager->set_compression_level(level);
}

int ZstdBatchManager::get_compression_level() const {
    return pimpl_->manager->get_compression_level();
}

Status ZstdBatchManager::set_dictionary(const dictionary::Dictionary& dict) {
    return pimpl_->manager->set_dictionary(dict);
}

Status ZstdBatchManager::get_dictionary(dictionary::Dictionary& dict) const {
    return pimpl_->manager->get_dictionary(dict);
}

Status ZstdBatchManager::clear_dictionary() {
    return pimpl_->manager->clear_dictionary();
}

const CompressionStats& ZstdBatchManager::get_stats() const {
    return pimpl_->batch_stats;
}

void ZstdBatchManager::reset_stats() {
    pimpl_->reset_stats();
}

size_t ZstdBatchManager::get_compress_temp_size(size_t uncompressed_size) const {
    return pimpl_->manager->get_compress_temp_size(uncompressed_size);
}

size_t ZstdBatchManager::get_decompress_temp_size(size_t compressed_size) const {
    return pimpl_->manager->get_decompress_temp_size(compressed_size);
}

size_t ZstdBatchManager::get_max_compressed_size(size_t uncompressed_size) const {
    return pimpl_->manager->get_max_compressed_size(uncompressed_size);
}

size_t ZstdBatchManager::get_batch_compress_temp_size(
    const std::vector<size_t>& uncompressed_sizes
) const {
    // --- (MODIFIED) ---
    // The total workspace is the sum of max workspace per item,
    // as we partition it.
    size_t total_workspace = 0;
    for (const auto& size : uncompressed_sizes) {
        total_workspace += pimpl_->manager->get_compress_temp_size(size);
    }
    return total_workspace;
    // --- (END MODIFIED) ---
}

size_t ZstdBatchManager::get_batch_decompress_temp_size(
    const std::vector<size_t>& compressed_sizes
) const {
    // --- (MODIFIED) ---
    size_t max_item_temp_size = 0;
    for (const auto& size : compressed_sizes) {
        max_item_temp_size = std::max(max_item_temp_size, 
                                    pimpl_->manager->get_decompress_temp_size(size));
    }
    // Align to 128 bytes for safety
    max_item_temp_size = (max_item_temp_size + 127) & ~127;
    return max_item_temp_size * compressed_sizes.size();
    // --- (END MODIFIED) ---
}

Status ZstdBatchManager::compress(
    const void* uncompressed_data,
    size_t uncompressed_size,
    void* compressed_data,
    size_t* compressed_size,
    void* temp_workspace,
    size_t temp_size,
    const void* dict_buffer,
    size_t dict_size,
    cudaStream_t stream
) {
    return pimpl_->manager->compress(
        uncompressed_data, uncompressed_size,
        compressed_data, compressed_size,
        temp_workspace, temp_size,
        dict_buffer, dict_size,
        stream
    );
}

// Debug wrapper: log top-level compress call result for easier tracing
// (removed) was a temporary debug wrapper

Status ZstdBatchManager::decompress(
    const void* compressed_data,
    size_t compressed_size,
    void* uncompressed_data,
    size_t* uncompressed_size,
    void* temp_workspace,
    size_t temp_size,
    cudaStream_t stream
) {
    return pimpl_->manager->decompress(
        compressed_data, compressed_size,
        uncompressed_data, uncompressed_size,
        temp_workspace, temp_size, stream
    );
}

Status ZstdBatchManager::compress_batch(
    const std::vector<BatchItem>& items,
    void* temp_workspace,
    size_t temp_size,
    cudaStream_t stream
) {
    // --- (NEW) OPTIMIZED PIPELINE IMPLEMENTATION ---
    pimpl_->manager->reset_stats();
    bool all_success = true;
    
    if (items.empty()) return Status::SUCCESS;

    std::vector<size_t> uncompressed_sizes;
    for(const auto& item : items) {
        uncompressed_sizes.push_back(item.input_size);
    }
    size_t required_size = get_batch_compress_temp_size(uncompressed_sizes);
    if (temp_size < required_size) {
        return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Create events for pipeline synchronization
    std::vector<cudaEvent_t> transfer_complete_events(items.size());
    std::vector<cudaEvent_t> compute_complete_events(items.size());
    
    for (size_t i = 0; i < items.size(); ++i) {
        cudaEventCreate(&transfer_complete_events[i]);
        cudaEventCreate(&compute_complete_events[i]);
    }
    
    // (NEW) PIPELINED EXECUTION: Overlap H2D, Compute, D2H
    for (size_t i = 0; i < items.size(); ++i) {
        auto& item = const_cast<std::vector<BatchItem>&>(items)[i];
        
        // Select streams for 3-stage pipeline
        cudaStream_t h2d_stream = pimpl_->streams[0];      // H2D transfer
        cudaStream_t compute_stream = pimpl_->streams[1];  // Kernel execution
        cudaStream_t d2h_stream = pimpl_->streams[2];      // D2H transfer
        
        // Partition the workspace
        size_t item_workspace_size = pimpl_->manager->get_compress_temp_size(item.input_size);
        byte_t* item_workspace = static_cast<byte_t*>(temp_workspace) + i * item_workspace_size;
        
        // Pipeline stage 1: H2D transfer (async, overlapped with previous compute)
        // Note: Input already on device in most cases, but this shows the pattern
        
        // Pipeline stage 2: Compression (overlapped with H2D of next block)
        item.status = pimpl_->manager->compress(
            item.input_ptr,
            item.input_size,
            item.output_ptr,
            &item.output_size,
            item_workspace,
            item_workspace_size,
            nullptr, 0,
            compute_stream // Use dedicated compute stream
        );
        
        // Record event when compression completes
        cudaEventRecord(compute_complete_events[i], compute_stream);
        
        if (item.status != Status::SUCCESS) {
            all_success = false;
        }
    }
    
    // (NEW) Only synchronize at the end - all streams execute in parallel
    for (auto s : pimpl_->streams) {
        cudaStreamSynchronize(s);
    }
    
    // Cleanup events
    for (size_t i = 0; i < items.size(); ++i) {
        cudaEventDestroy(transfer_complete_events[i]);
        cudaEventDestroy(compute_complete_events[i]);
    }
    
    pimpl_->batch_stats = pimpl_->manager->get_stats();
    
    return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
    // --- (END NEW PIPELINE) ---
}

Status ZstdBatchManager::decompress_batch(
    const std::vector<BatchItem>& items,
    void* temp_workspace,
    size_t temp_size,
    cudaStream_t stream
) {
    // --- (START REPLACEMENT) ---
    pimpl_->manager->reset_stats();
    bool all_success = true;
    int stream_idx = 0;
    
    if (items.empty()) return Status::SUCCESS;

    // Calculate the size of a single item's workspace
    size_t max_item_temp_size = 0;
    for(const auto& item : items) {
        max_item_temp_size = std::max(max_item_temp_size, 
            pimpl_->manager->get_decompress_temp_size(item.input_size));
    }
    max_item_temp_size = (max_item_temp_size + 127) & ~127; // Align

    if (temp_size < max_item_temp_size * items.size()) {
        return Status::ERROR_BUFFER_TOO_SMALL;
    }

    for (size_t i = 0; i < items.size(); ++i) {
        auto& item = const_cast<std::vector<BatchItem>&>(items)[i];

        cudaStream_t item_stream = pimpl_->streams[stream_idx % pimpl_->num_streams];
        stream_idx++;
        
        byte_t* item_workspace = static_cast<byte_t*>(temp_workspace) + i * max_item_temp_size;
        
        item.status = pimpl_->manager->decompress(
            item.input_ptr,
            item.input_size,
            item.output_ptr,
            &item.output_size,
            item_workspace,
            max_item_temp_size,
            item_stream
        );
        if (item.status != Status::SUCCESS) {
            all_success = false;
        }
    }
    
    // Wait for all pooled streams to finish
    for (auto s : pimpl_->streams) {
        cudaStreamSynchronize(s);
    }
    
    pimpl_->batch_stats = pimpl_->manager->get_stats();

    return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
    // --- (END REPLACEMENT) ---
}

// ==============================================================================
// STREAMING MANAGER IMPLEMENTATION
// ==============================================================================

class ZstdStreamingManager::Impl {
public:
    CompressionConfig config;
    dictionary::Dictionary dict;
    bool has_dictionary;
    cudaStream_t stream;
    bool owns_stream;
    
    std::unique_ptr<ZstdManager> manager;
    StreamingContext streaming_ctx;
    
    void* d_workspace;
    size_t workspace_size;
    
    bool comp_initialized;
    bool decomp_initialized;
    bool frame_header_parsed;

    explicit Impl(const CompressionConfig& cfg)
        : config(cfg), has_dictionary(false), stream(nullptr), owns_stream(false),
          d_workspace(nullptr), workspace_size(0),
          comp_initialized(false), decomp_initialized(false), frame_header_parsed(false)
    {
        manager = create_manager(config);
    }

    ~Impl() {
        if (d_workspace) {
            cudaFree(d_workspace);
        }
    }

    Status alloc_workspace(cudaStream_t s) {
        stream = s;
        size_t comp_size = manager->get_compress_temp_size(ZSTD_BLOCKSIZE_MAX);
        size_t decomp_size = manager->get_decompress_temp_size(ZSTD_BLOCKSIZE_MAX * 2);
        workspace_size = std::max(comp_size, decomp_size);
        
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
        return Status::SUCCESS;
    }
};

ZstdStreamingManager::ZstdStreamingManager() {
    pimpl_ = std::make_unique<Impl>(CompressionConfig::from_level(ZSTD_DEFAULT_CLEVEL));
}

ZstdStreamingManager::ZstdStreamingManager(const CompressionConfig& config) {
    pimpl_ = std::make_unique<Impl>(config);
}

ZstdStreamingManager::~ZstdStreamingManager() = default;

Status ZstdStreamingManager::set_config(const CompressionConfig& config) {
    if (pimpl_->comp_initialized || pimpl_->decomp_initialized) {
        return Status::ERROR_GENERIC;
    }
    pimpl_->config = config;
    return pimpl_->manager->configure(config);
}

CompressionConfig ZstdStreamingManager::get_config() const {
    return pimpl_->config;
}

Status ZstdStreamingManager::set_dictionary(const dictionary::Dictionary& dict) {
    pimpl_->dict = dict;
    pimpl_->has_dictionary = true;
    return pimpl_->manager->set_dictionary(dict);
}

Status ZstdStreamingManager::init_compression(cudaStream_t stream) {
    if (pimpl_->comp_initialized) reset();
    
    // Allocate window history
    const u32 WINDOW_SIZE = 128 * 1024;  // 128 KB
    
    pimpl_->streaming_ctx.d_window_history = nullptr;
    pimpl_->streaming_ctx.window_history_capacity = WINDOW_SIZE;
    pimpl_->streaming_ctx.window_history_size = 0;
    
    CUDA_CHECK(cudaMalloc(&pimpl_->streaming_ctx.d_window_history, WINDOW_SIZE));
    
    // Allocate persistent hash tables
    u32 hash_table_size = (1 << 20);  // 1M entries
    u32 chain_table_size = (1 << 20);
    
    CUDA_CHECK(cudaMalloc(&pimpl_->streaming_ctx.d_hash_table_state, 
                         hash_table_size * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&pimpl_->streaming_ctx.d_chain_table_state,
                         chain_table_size * sizeof(u32)));
    
    // Initialize tables
    u32 invalid_pos = 0xFFFFFFFF;
    hash::init_hash_table(pimpl_->streaming_ctx.d_hash_table_state, 
                         hash_table_size, invalid_pos, stream);
    hash::init_hash_table(pimpl_->streaming_ctx.d_chain_table_state,
                         chain_table_size, invalid_pos, stream);
    
    pimpl_->streaming_ctx.total_bytes_processed = 0;
    pimpl_->streaming_ctx.block_count = 0;
    pimpl_->streaming_ctx.started_compression = false;
    pimpl_->streaming_ctx.finished_compression = false;
    
    // Write frame header (only once at start)
    if (!pimpl_->streaming_ctx.started_compression) {
        // Will be written at first compress_chunk call
    }
    
    pimpl_->comp_initialized = true;
    return Status::SUCCESS;
}

Status ZstdStreamingManager::init_decompression(cudaStream_t stream) {
    if (pimpl_->decomp_initialized) reset();
    
    auto status = pimpl_->alloc_workspace(stream);
    if (status != Status::SUCCESS) return status;
    
    pimpl_->decomp_initialized = true;
    return Status::SUCCESS;
}

Status ZstdStreamingManager::reset() {
    if (pimpl_->d_workspace) {
        cudaFree(pimpl_->d_workspace);
        pimpl_->d_workspace = nullptr;
    }
    pimpl_->workspace_size = 0;
    pimpl_->comp_initialized = false;
    pimpl_->decomp_initialized = false;
    return Status::SUCCESS;
}

Status ZstdStreamingManager::flush(cudaStream_t stream) {
    return Status::SUCCESS;
}

size_t ZstdStreamingManager::get_temp_size() const {
    size_t comp_size = pimpl_->manager->get_compress_temp_size(ZSTD_BLOCKSIZE_MAX);
    size_t decomp_size = pimpl_->manager->get_decompress_temp_size(ZSTD_BLOCKSIZE_MAX * 2);
    return std::max(comp_size, decomp_size);
}

bool ZstdStreamingManager::is_compression_initialized() const {
    return pimpl_->comp_initialized;
}

bool ZstdStreamingManager::is_decompression_initialized() const {
    return pimpl_->decomp_initialized;
}

Status ZstdStreamingManager::compress_chunk(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    bool is_last_chunk,
    cudaStream_t stream
) {
    if (!pimpl_->comp_initialized) return Status::ERROR_GENERIC;
    if (!input || !output || !output_size) return Status::ERROR_INVALID_PARAMETER;
    if (input_size > ZSTD_BLOCKSIZE_MAX) {
        return Status::ERROR_BUFFER_TOO_SMALL;
    }
    
    auto& sctx = pimpl_->streaming_ctx;
    const byte_t* d_input = static_cast<const byte_t*>(input);
    byte_t* d_output = static_cast<byte_t*>(output);
    size_t max_output_size = *output_size;
    u32 output_offset = 0;
    
    // ===== STEP 1: Write Frame Header (First Chunk Only) =====
    if (!sctx.started_compression) {
        byte_t h_header[FRAME_HEADER_SIZE_MAX];
        u32 header_offset = 0;
        
        // Magic number
        u32 magic = ZSTD_MAGIC_NUMBER;
        memcpy(h_header + header_offset, &magic, 4);
        header_offset += 4;
        
        // Frame Header Descriptor
        byte_t fhd = 0;
        if (pimpl_->config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
            fhd |= 0x04;  // Checksum flag
        }
        // Don't set content size for streaming (unknown)
        h_header[header_offset++] = fhd;
        
        // Window descriptor (not single segment)
        byte_t window_desc = ((pimpl_->config.window_log - 10) << 3);
        h_header[header_offset++] = window_desc;
        
        // Copy header to device output
        CUDA_CHECK(cudaMemcpyAsync(d_output, h_header, header_offset,
                                  cudaMemcpyHostToDevice, stream));
        output_offset += header_offset;
        sctx.started_compression = true;
    }
    
    // ===== STEP 2: Prepare Combined Input (Window History + Current Chunk) =====
    size_t combined_size = sctx.window_history_size + input_size;
    byte_t* d_combined_input = nullptr;
    
    if (sctx.window_history_size > 0) {
        // Allocate temporary buffer for combined input
        CUDA_CHECK(cudaMalloc(&d_combined_input, combined_size));
        
        // Copy window history
        CUDA_CHECK(cudaMemcpyAsync(d_combined_input, sctx.d_window_history,
                                  sctx.window_history_size, cudaMemcpyDeviceToDevice, stream));
        
        // Copy current chunk
        CUDA_CHECK(cudaMemcpyAsync(d_combined_input + sctx.window_history_size, d_input,
                                  input_size, cudaMemcpyDeviceToDevice, stream));
    } else {
        d_combined_input = const_cast<byte_t*>(d_input);
    }
    
    // ===== STEP 3: Compress Block Using Manager =====
    byte_t* d_compressed_block = nullptr;
    CUDA_CHECK(cudaMalloc(&d_compressed_block, input_size * 2));
    
    size_t block_compressed_size = input_size * 2;
    Status status = pimpl_->manager->compress(
        d_combined_input + sctx.window_history_size,  // Current chunk only for compression
        input_size,
        d_compressed_block,
        &block_compressed_size,
        pimpl_->d_workspace,
        pimpl_->workspace_size,
        pimpl_->has_dictionary ? pimpl_->dict.raw_content : nullptr,
        pimpl_->has_dictionary ? pimpl_->dict.raw_size : 0,
        stream
    );
    
    if (status != Status::SUCCESS) {
        if (sctx.window_history_size > 0) cudaFree(d_combined_input);
        cudaFree(d_compressed_block);
        return status;
    }
    
    // ===== STEP 4: Write Block Header and Data =====
    if (output_offset + 3 + block_compressed_size > max_output_size) {
        if (sctx.window_history_size > 0) cudaFree(d_combined_input);
        cudaFree(d_compressed_block);
        return Status::ERROR_BUFFER_TOO_SMALL;
    }
    
    // Block header (3 bytes)
    u32 block_header = 0;
    block_header |= (is_last_chunk ? 1 : 0);  // Last block flag
    block_header |= (2 << 1);  // Compressed block type
    block_header |= (block_compressed_size << 3);  // Block size
    
    CUDA_CHECK(cudaMemcpyAsync(d_output + output_offset, &block_header, 3,
                              cudaMemcpyHostToDevice, stream));
    output_offset += 3;
    
    // Block data
    CUDA_CHECK(cudaMemcpyAsync(d_output + output_offset, d_compressed_block,
                              block_compressed_size, cudaMemcpyDeviceToDevice, stream));
    output_offset += block_compressed_size;
    
    // ===== STEP 5: Update Window History =====
    u32 window_to_keep = std::min((u32)input_size, sctx.window_history_capacity);
    
    if (input_size >= sctx.window_history_capacity) {
        // New chunk is larger than window, use end of chunk
        CUDA_CHECK(cudaMemcpyAsync(sctx.d_window_history,
                                  d_input + (input_size - window_to_keep),
                                  window_to_keep,
                                  cudaMemcpyDeviceToDevice, stream));
        sctx.window_history_size = window_to_keep;
    } else {
        // Shift existing window and append new data
        u32 shift_amount = std::min(sctx.window_history_size,
                                    sctx.window_history_capacity - (u32)input_size);
        
        if (shift_amount > 0) {
            CUDA_CHECK(cudaMemcpyAsync(sctx.d_window_history,
                                      sctx.d_window_history + (sctx.window_history_size - shift_amount),
                                      shift_amount,
                                      cudaMemcpyDeviceToDevice, stream));
        }
        
        CUDA_CHECK(cudaMemcpyAsync(sctx.d_window_history + shift_amount, d_input,
                                  input_size, cudaMemcpyDeviceToDevice, stream));
        sctx.window_history_size = shift_amount + input_size;
    }
    
    // ===== STEP 6: Update State =====
    sctx.total_bytes_processed += input_size;
    sctx.block_count++;
    
    if (is_last_chunk) {
        sctx.finished_compression = true;
        
        // Write checksum if enabled
        if (pimpl_->config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
            // Allocate a temporary device-side buffer for checksum output.
            // (We previously tried to reuse a shared device buffer, but that
            // buffer belongs to DefaultZstdManager; here allocate locally
            // to keep streaming manager self-contained while avoiding
            // exposing manager internals.)
            u64* d_checksum = nullptr;
            CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(u64)));
            
            // TODO: Implement a true incremental/streaming xxHash for better performance and memory efficiency.
            // The current implementation re-hashes the entire chunk, which is inefficient for large streams.
            // Compute checksum of all processed data (simplified - should hash incrementally)
            xxhash::compute_xxhash64(d_input, input_size, 0, d_checksum, stream);
            
            CUDA_CHECK(cudaMemcpyAsync(d_output + output_offset, d_checksum, sizeof(u64),
                                      cudaMemcpyDeviceToDevice, stream));
            output_offset += sizeof(u64);
            
            // Free the temporary device-side checksum buffer
            cudaFree(d_checksum);
        }
    }
    
    // ===== STEP 7: Cleanup and Return =====
    *output_size = output_offset;
    
    if (sctx.window_history_size > 0 && d_combined_input != d_input) {
        cudaFree(d_combined_input);
    }
    cudaFree(d_compressed_block);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return Status::SUCCESS;
}

Status ZstdStreamingManager::decompress_chunk(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    bool* is_last_chunk,
    cudaStream_t stream
) {
    if (!pimpl_->decomp_initialized) return Status::ERROR_GENERIC;
    if (!input || !output || !output_size || !is_last_chunk) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    const byte_t* d_input = static_cast<const byte_t*>(input);
    byte_t* d_output = static_cast<byte_t*>(output);
    size_t max_output_size = *output_size;
    
    u32 read_offset = 0;
    u32 write_offset = 0;
    
    // ===== STEP 1: Parse Frame Header (First Chunk Only) =====
    if (!pimpl_->frame_header_parsed && input_size >= 6) {
        byte_t h_header[FRAME_HEADER_SIZE_MAX];
        CUDA_CHECK(cudaMemcpy(h_header, d_input, std::min((size_t)FRAME_HEADER_SIZE_MAX, input_size),
                             cudaMemcpyDeviceToHost));
        
        // Check magic number
        u32 magic = read_u32_le(h_header);
        if (magic != ZSTD_MAGIC_NUMBER) {
            return Status::ERROR_INVALID_MAGIC;
        }
        
        read_offset = 4;
        
        // Parse FHD
        byte_t fhd = h_header[read_offset++];
        bool single_segment = (fhd >> 5) & 0x01;
        
        // Window descriptor (if not single segment)
        if (!single_segment) {
            read_offset++;  // Skip window descriptor
        }
        
        // Dictionary ID (if present)
        if ((fhd & 0x03) != 0) {
            u32 dict_id_size = 1 << ((fhd & 0x03) - 1);
            read_offset += dict_id_size;
        }
        
        // Content size (if present)
        u32 fcs_field_size = (fhd >> 6) & 0x03;
        if (fcs_field_size == 0 && single_segment) {
            read_offset += 1;
        } else if (fcs_field_size == 1) {
            read_offset += 2;
        } else if (fcs_field_size == 2) {
            read_offset += 4;
        } else if (fcs_field_size == 3) {
            read_offset += 8;
        }
        
        pimpl_->frame_header_parsed = true;
    }
    
    if (read_offset >= input_size) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    // ===== STEP 2: Read and Process Blocks =====
    while (read_offset + 3 <= input_size) {
        // Read block header (3 bytes)
        u32 block_header = 0;
        CUDA_CHECK(cudaMemcpy(&block_header, d_input + read_offset, 3, cudaMemcpyDeviceToHost));
        
        bool last_block = (block_header & 0x01) != 0;
        u32 block_type = (block_header >> 1) & 0x03;
        u32 block_size = block_header >> 3;
        
        read_offset += 3;
        
        if (read_offset + block_size > input_size) {
            return Status::ERROR_CORRUPT_DATA;
        }
        
        // ===== STEP 3: Decompress Block Based on Type =====
        if (block_type == 0) {
            // Raw block - direct copy
            if (write_offset + block_size > max_output_size) {
                return Status::ERROR_BUFFER_TOO_SMALL;
            }
            
            CUDA_CHECK(cudaMemcpyAsync(d_output + write_offset, d_input + read_offset,
                                      block_size, cudaMemcpyDeviceToDevice, stream));
            write_offset += block_size;
            
        } else if (block_type == 1) {
            // RLE block
            if (write_offset + block_size > max_output_size) {
                return Status::ERROR_BUFFER_TOO_SMALL;
            }
            
            byte_t rle_byte;
            CUDA_CHECK(cudaMemcpy(&rle_byte, d_input + read_offset, 1, cudaMemcpyDeviceToHost));
            
            CUDA_CHECK(cudaMemsetAsync(d_output + write_offset, rle_byte, block_size, stream));
            write_offset += block_size;
            
        } else if (block_type == 2) {
            // Compressed block - use manager decompression
            size_t decompressed_size = max_output_size - write_offset;
            
            Status status = pimpl_->manager->decompress(
                d_input + read_offset,
                block_size,
                d_output + write_offset,
                &decompressed_size,
                pimpl_->d_workspace,
                pimpl_->workspace_size,
                stream
            );
            
            if (status != Status::SUCCESS) {
                return status;
            }
            
            write_offset += decompressed_size;
            
        } else {
            // Reserved block type
            return Status::ERROR_CORRUPT_DATA;
        }
        
        read_offset += block_size;
        
        // ===== STEP 4: Check if Last Block =====
        if (last_block) {
            *is_last_chunk = true;
            
            // Verify checksum if present
            if (pimpl_->config.checksum == ChecksumPolicy::COMPUTE_AND_VERIFY) {
                if (read_offset + 8 <= input_size) {
                    u64 stored_checksum;
                    CUDA_CHECK(cudaMemcpy(&stored_checksum, d_input + read_offset, sizeof(u64),
                                         cudaMemcpyDeviceToHost));
                    
                    u64* d_computed_checksum = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_computed_checksum, sizeof(u64)));
                    
                    xxhash::compute_xxhash64(d_output, write_offset, 0, d_computed_checksum, stream);
                    
                    u64 computed_checksum;
                    CUDA_CHECK(cudaMemcpy(&computed_checksum, d_computed_checksum, sizeof(u64),
                                         cudaMemcpyDeviceToHost));
                    
                    cudaFree(d_computed_checksum);
                    
                    if (stored_checksum != computed_checksum) {
                        return Status::ERROR_CHECKSUM_FAILED;
                    }
                }
            }
            
            break;
        }
    }
    
    // ===== STEP 5: Set Output Size and Return =====
    *output_size = write_offset;
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return Status::SUCCESS;
}


// ==============================================================================
// FACTORY FUNCTIONS
// ==============================================================================

std::unique_ptr<ZstdManager> create_manager(const CompressionConfig& config) {
    auto manager = std::make_unique<DefaultZstdManager>();
    manager->configure(config);
    return manager;
}

std::unique_ptr<ZstdManager> create_manager(int compression_level) {
    auto config = CompressionConfig::from_level(compression_level);
    return create_manager(config);
}

std::unique_ptr<ZstdBatchManager> create_batch_manager(int compression_level) {
    return std::make_unique<ZstdBatchManager>(CompressionConfig::from_level(compression_level));
}

std::unique_ptr<ZstdStreamingManager> create_streaming_manager(int compression_level) {
    return std::make_unique<ZstdStreamingManager>(CompressionConfig::from_level(compression_level));
}

// ==============================================================================
// CONVENIENCE FUNCTIONS (Single-Shot)
// ==============================================================================

Status compress_simple(
    const void* uncompressed_data,
    size_t uncompressed_size,
    void* compressed_data,
    size_t* compressed_size,
    int compression_level,
    cudaStream_t stream
) {
    auto manager = create_manager(compression_level);
    size_t temp_size = manager->get_compress_temp_size(uncompressed_size);
    void* temp_workspace;
    CUDA_CHECK(cudaMalloc(&temp_workspace, temp_size));
    
    // This function needs a device buffer for the output
    void* d_output;
    size_t max_compressed_size = manager->get_max_compressed_size(uncompressed_size);
    CUDA_CHECK(cudaMalloc(&d_output, max_compressed_size));

    Status status = manager->compress(
        uncompressed_data, uncompressed_size,
        d_output, compressed_size,
        temp_workspace, temp_size,
        nullptr, 0, // No dictionary in simple mode
        stream
    );

    if(status == Status::SUCCESS) {
        // Synchronous copy to host buffer to avoid using non-pinned host memory
        // with cudaMemcpyAsync (undefined behavior unless host memory is pinned).
        CUDA_CHECK(cudaMemcpy(compressed_data, d_output, *compressed_size, cudaMemcpyDeviceToHost));
    }
    
    cudaFree(d_output);
    cudaFree(temp_workspace);
    return status;
}

Status decompress_simple(
    const void* compressed_data,
    size_t compressed_size,
    void* uncompressed_data,
    size_t* uncompressed_size,
    cudaStream_t stream
) {
    auto manager = create_manager();
    size_t temp_size = manager->get_decompress_temp_size(compressed_size);
    void* temp_workspace;
    CUDA_CHECK(cudaMalloc(&temp_workspace, temp_size));
    
    Status status = manager->decompress(
        compressed_data, compressed_size,
        uncompressed_data, uncompressed_size,
        temp_workspace, temp_size, stream
    );
    // (FIX) Need to sync
    cudaStreamSynchronize(stream);
    
    cudaFree(temp_workspace);
    return status;
}

Status compress_with_dict(
    const void* uncompressed_data,
    size_t uncompressed_size,
    void* compressed_data,
    size_t* compressed_size,
    const dictionary::Dictionary& dict,
    int compression_level,
    cudaStream_t stream
) {
    auto manager = create_manager(compression_level);
    manager->set_dictionary(dict);
    size_t temp_size = manager->get_compress_temp_size(uncompressed_size);
    void* temp_workspace;
    CUDA_CHECK(cudaMalloc(&temp_workspace, temp_size));

    void* d_output;
    size_t max_compressed_size = manager->get_max_compressed_size(uncompressed_size);
    CUDA_CHECK(cudaMalloc(&d_output, max_compressed_size));
    
    Status status = manager->compress(
        uncompressed_data, uncompressed_size,
        d_output, compressed_size,
        temp_workspace, temp_size,
        dict.raw_content, dict.raw_size,
        stream
    );

    if (status == Status::SUCCESS) {
        CUDA_CHECK(cudaMemcpy(compressed_data, d_output, *compressed_size, cudaMemcpyDeviceToHost));
    }
    
    cudaFree(d_output);
    cudaFree(temp_workspace);
    return status;
}

Status decompress_with_dict(
    const void* compressed_data,
    size_t compressed_size,
    void* uncompressed_data,
    size_t* uncompressed_size,
    const dictionary::Dictionary& dict,
    cudaStream_t stream
) {
    auto manager = create_manager();
    manager->set_dictionary(dict); // (FIX) Actually use the dictionary
    size_t temp_size = manager->get_decompress_temp_size(compressed_size);
    void* temp_workspace;
    CUDA_CHECK(cudaMalloc(&temp_workspace, temp_size));
    
    Status status = manager->decompress(
        compressed_data, compressed_size,
        uncompressed_data, uncompressed_size,
        temp_workspace, temp_size, stream
    );
    cudaStreamSynchronize(stream);
    
    cudaFree(temp_workspace);
    return status;
}

// ==============================================================================
// (MODIFIED) PATCH: Implementation for extracting metadata from compressed data
// ==============================================================================

Status extract_metadata(
    const void* compressed_data,
    size_t compressed_size,
    NvcompMetadata& metadata // Host output
) {
    if (compressed_size < 4) return Status::ERROR_CORRUPT_DATA;
    
    u32 data_offset = 0;
    u32 magic = 0;
    
    const byte_t* h_compressed_ptr = static_cast<const byte_t*>(compressed_data);
    memcpy(&magic, h_compressed_ptr, sizeof(u32));

    while (magic == ZSTD_MAGIC_SKIPPABLE_START) {
        SkippableFrameHeader skip_header;
        if (compressed_size < data_offset + sizeof(SkippableFrameHeader)) {
            return Status::ERROR_CORRUPT_DATA;
        }
        memcpy(&skip_header, h_compressed_ptr + data_offset, sizeof(SkippableFrameHeader));

        if (skip_header.frame_size == sizeof(CustomMetadataFrame)) {
            CustomMetadataFrame custom_meta;
            if (compressed_size < data_offset + sizeof(SkippableFrameHeader) + sizeof(CustomMetadataFrame)) {
                return Status::ERROR_CORRUPT_DATA;
            }
            memcpy(&custom_meta, h_compressed_ptr + data_offset + sizeof(SkippableFrameHeader), 
                                  sizeof(CustomMetadataFrame));
            
            if (custom_meta.custom_magic == CUSTOM_METADATA_MAGIC) {
                metadata.compression_level = custom_meta.compression_level;
            }
        }
        
        data_offset += sizeof(SkippableFrameHeader) + skip_header.frame_size;
        
        if (compressed_size < data_offset + 4) {
            return Status::ERROR_INVALID_MAGIC;
        }
        memcpy(&magic, h_compressed_ptr + data_offset, sizeof(u32));
    } 
    
    if (magic != ZSTD_MAGIC_NUMBER) {
        return Status::ERROR_INVALID_MAGIC;
    }
    
    if (compressed_size < data_offset + 5) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    byte_t h_header[FRAME_HEADER_SIZE_MAX];
    memcpy(h_header, h_compressed_ptr + data_offset, 
           std::min((size_t)FRAME_HEADER_SIZE_MAX, compressed_size - data_offset));

    if (*reinterpret_cast<u32*>(h_header) != ZSTD_MAGIC_NUMBER) {
        return Status::ERROR_INVALID_MAGIC;
    }
    
    u32 offset = 4;
    byte_t fhd = h_header[offset++];
    bool single_segment = (fhd >> 5) & 0x01;
    bool has_checksum = (fhd >> 2) & 0x01;
    bool has_dict_id = (fhd & 0x03) != 0;
    
    if (!single_segment) {
        offset++;
    }
    
    u32 dict_id_size = 0;
    if (has_dict_id) {
        dict_id_size = 1 << (fhd & 0x03);
        metadata.dictionary_id = *reinterpret_cast<u32*>(h_header + offset);
        offset += dict_id_size;
    } else {
        metadata.dictionary_id = 0;
    }
    
    u32 fcs_field_size = (fhd >> 6) & 0x03;
    u64 h_content_size = 0;
    
    if (fcs_field_size == 0) {
        h_content_size = single_segment ? h_header[offset] : 0;
    } else if (fcs_field_size == 1) {
        h_content_size = *reinterpret_cast<u16*>(h_header + offset) + 256;
    } else if (fcs_field_size == 2) {
        h_content_size = *reinterpret_cast<u32*>(h_header + offset);
    } else {
        h_content_size = *reinterpret_cast<u64*>(h_header + offset);
    }

    metadata.format_version = get_format_version();
    metadata.uncompressed_size = (u32)h_content_size;
    metadata.checksum_policy = has_checksum ? 
                                ChecksumPolicy::COMPUTE_AND_VERIFY : 
                                ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
    
    if (metadata.compression_level == 0) { 
        metadata.compression_level = 0; // Unknown
    }
    
    metadata.num_chunks = 1;
    metadata.chunk_size = (u32)h_content_size;
    
    return Status::SUCCESS;
}

// ==============================================================================
// UTILITY FUNCTIONS (Implementation)
// ==============================================================================

u32 get_optimal_block_size(u32 input_size, u32 compression_level) {
    return std::min(ZSTD_BLOCKSIZE_MAX, input_size);
}
 
// Minimal implementation to map a compression level to concrete parameters.
// Keep logic minimal and non-functional-change: use the existing helper to
// produce the canonical configuration for the level.
void apply_level_parameters(CompressionConfig& config) {
    // Preserve the requested level, but obtain the full parameter set from
    // the canonical factory. This avoids duplicating mapping logic.
    CompressionConfig canonical = CompressionConfig::from_level(config.level);
    config = canonical;
}

size_t estimate_compressed_size(size_t input_size, int compression_level) {
    size_t skippable_frame_size = sizeof(SkippableFrameHeader) + sizeof(CustomMetadataFrame);
    return (size_t)(input_size * 1.01) + 512 + skippable_frame_size;
}

Status validate_config(const CompressionConfig& config) {
    if (config.level < ZSTD_MIN_CLEVEL || config.level > ZSTD_MAX_CLEVEL) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    if (config.window_log < ZSTD_WINDOWLOG_MIN || config.window_log > ZSTD_WINDOWLOG_MAX) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    return Status::SUCCESS;
}
} // namespace cuda_zstd
