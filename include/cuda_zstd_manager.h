// ==============================================================================
// cuda_zstd_manager.h - Complete Manager Interface (PRODUCTION READY)
// ==============================================================================

#ifndef CUDA_ZSTD_MANAGER_H_
#define CUDA_ZSTD_MANAGER_H_

#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#endif

#include "performance_profiler.h"

#ifdef __cplusplus
namespace cuda_zstd {

// ==============================================================================
// FORWARD DECLARATIONS
// ==============================================================================

namespace lz77 {
struct LZ77Context;
}
namespace sequence {
struct SequenceContext;
}
namespace fse {
struct FSEEncodeTable;
struct FSEDecodeTable;
} // namespace fse
namespace huffman {
struct HuffmanContext;
}
namespace xxhash {
u64 compute_xxh64(const unsigned char *, u32, u64, cudaStream_t);
}

// ==============================================================================
// MANAGER BASE CLASS
// ==============================================================================

class ZstdManager {
public:
  virtual ~ZstdManager() = default;

  // Configuration
  virtual Status configure(const CompressionConfig &config) = 0;
  virtual CompressionConfig get_config() const = 0;

  // Workspace queries
  virtual size_t get_compress_temp_size(size_t uncompressed_size) const = 0;
  virtual size_t get_decompress_temp_size(size_t compressed_size) const = 0;
  virtual size_t get_max_compressed_size(size_t uncompressed_size) const = 0;

  // Core operations
  virtual Status compress(const void *uncompressed_data,
                          size_t uncompressed_size, void *compressed_data,
                          size_t *compressed_size, void *temp_workspace,
                          size_t temp_size, const void *dict_buffer,
                          size_t dict_size, cudaStream_t stream) = 0;

  virtual Status decompress(const void *compressed_data, size_t compressed_size,
                            void *uncompressed_data, size_t *uncompressed_size,
                            void *temp_workspace, size_t temp_size,
                            cudaStream_t stream = 0) = 0;

  // Dictionary support
  virtual Status set_dictionary(const dictionary::Dictionary &dict) = 0;
  virtual Status get_dictionary(dictionary::Dictionary &dict) const = 0;
  virtual Status clear_dictionary() = 0;

  // Statistics & configuration
  virtual const CompressionStats &get_stats() const = 0;
  virtual Status set_compression_level(int level) = 0;
  virtual int get_compression_level() const = 0;
  virtual void reset_stats() = 0;

  // Smart Path Selection
  enum class ExecutionPath {
    CPU,       // Small payloads (< 64KB)
    GPU_BATCH, // Standard Block Parallel (Default)
    GPU_CHUNK  // Intra-Block Parallel (> 1MB)
  };

  static ExecutionPath select_execution_path(size_t size,
                                             int cpu_threshold = 65536);

  // Optimization: Pre-allocate/Pre-compute tables for persistent reusable
  // managers
  virtual Status preallocate_tables(cudaStream_t stream = 0) {
    return Status::SUCCESS;
  }
  virtual Status free_tables(cudaStream_t stream = 0) {
    return Status::SUCCESS;
  }
};

// ==============================================================================
// BATCH MANAGER
// ==============================================================================

/**
 * @brief Batch processing manager for GPU-accelerated Zstandard operations.
 * 
 * The ZstdBatchManager provides high-level APIs for compressing and decompressing
 * batches of data chunks. It automatically handles resource management, 
 * workspace partitioning, and stream synchronization to maximize GPU utilization.
 */
class ZstdBatchManager : public ZstdManager {

public:
  ZstdBatchManager();
  explicit ZstdBatchManager(const CompressionConfig &config);
  ~ZstdBatchManager() override;

  // Base interface
  Status configure(const CompressionConfig &config) override;
  CompressionConfig get_config() const override;

  size_t get_compress_temp_size(size_t uncompressed_size) const override;
  size_t get_decompress_temp_size(size_t compressed_size) const override;
  size_t get_max_compressed_size(size_t uncompressed_size) const override;

  Status compress(const void *uncompressed_data, size_t uncompressed_size,
                  void *compressed_data, size_t *compressed_size,
                  void *temp_workspace, size_t temp_size,
                  const void *dict_buffer, size_t dict_size,
                  cudaStream_t stream = 0) override;

  Status decompress(const void *compressed_data, size_t compressed_size,
                    void *uncompressed_data, size_t *uncompressed_size,
                    void *temp_workspace, size_t temp_size,
                    cudaStream_t stream = 0) override;

  Status set_dictionary(const dictionary::Dictionary &dict) override;
  Status get_dictionary(dictionary::Dictionary &dict) const override;
  Status clear_dictionary() override;

  const CompressionStats &get_stats() const override;
  Status set_compression_level(int level) override;
  int get_compression_level() const override;
  void reset_stats() override;

  // Batch-specific operations
  Status compress_batch(const std::vector<BatchItem> &items,
                        void *temp_workspace, size_t temp_size,
                        cudaStream_t stream = 0);

  Status decompress_batch(const std::vector<BatchItem> &items,
                          void *temp_workspace, size_t temp_size,
                          cudaStream_t stream = 0);

  size_t get_batch_compress_temp_size(
      const std::vector<size_t> &uncompressed_sizes) const;

  size_t get_batch_decompress_temp_size(
      const std::vector<size_t> &compressed_sizes) const;

  // ============================================================================
  // INFERENCE-READY API (For Holographic/JIT Inference)
  // ============================================================================
  // These methods support zero-malloc decompression using pre-allocated
  // "Zipper Buffers" that rotate during inference. Ideal for:
  // - LLM inference with layer-wise streaming
  // - Double-buffered decompression (decompress N+1 while computing N)
  // - Minimizing VRAM allocation overhead during inference
  // ============================================================================

  /**
   * @brief Decompress directly into a pre-allocated output buffer.
   *
   * Unlike decompress_batch(), this method does NOT allocate output memory.
   * The caller provides pre-allocated buffers (e.g., a rotating "Zipper
   * Buffer").
   *
   * Use case: Inference where you decompress into a fixed VRAM buffer,
   * compute, and then overwrite with the next layer's decompressed data.
   *
   * @param compressed_data Pointer to compressed data (device memory)
   * @param compressed_size Size of compressed data in bytes
   * @param preallocated_output Pre-allocated output buffer (device memory)
   * @param output_capacity Capacity of the output buffer
   * @param actual_output_size [out] Actual decompressed size written
   * @param temp_workspace Temporary workspace (device memory)
   * @param temp_size Size of workspace
   * @param stream CUDA stream for async operation
   * @return Status::OK on success
   */
  Status decompress_to_preallocated(const void *compressed_data,
                                    size_t compressed_size,
                                    void *preallocated_output,
                                    size_t output_capacity,
                                    size_t *actual_output_size,
                                    void *temp_workspace, size_t temp_size,
                                    cudaStream_t stream = 0);

  /**
   * @brief Decompress batch into pre-allocated output buffers (inference mode).
   *
   * For batch decompression where ALL output buffers are provided by caller.
   * Enables true zero-malloc inference with rotating buffer pools.
   *
   * @param items Vector of BatchItem with pre-set output pointers and
   * capacities (output_data must point to valid pre-allocated memory)
   * @param temp_workspace Shared workspace
   * @param temp_size Workspace size
   * @param stream CUDA stream
   * @return Status::OK on success
   */
  Status decompress_batch_preallocated(
      std::vector<BatchItem> &items, // Note: non-const, output_size is set
      void *temp_workspace, size_t temp_size, cudaStream_t stream = 0);

  /**
   * @brief Async decompress that doesn't synchronize (caller manages sync).
   *
   * For pipelined inference: decompress layer N+1 while computing layer N.
   * The caller MUST synchronize the stream before reading output.
   *
   * @param compressed_data Compressed input (device or pinned host memory)
   * @param compressed_size Size of compressed data
   * @param preallocated_output Pre-allocated output (device memory)
   * @param output_capacity Output buffer capacity
   * @param d_actual_size Device pointer to write actual size (async-safe)
   * @param temp_workspace Workspace
   * @param temp_size Workspace size
   * @param stream Stream for async operation (caller syncs)
   * @return Status::OK if launch succeeded (not completion)
   */
  Status decompress_async_no_sync(
      const void *compressed_data, size_t compressed_size,
      void *preallocated_output, size_t output_capacity,
      size_t *d_actual_size, // Device pointer for async size write
      void *temp_workspace, size_t temp_size, cudaStream_t stream);

  // ============================================================================
  // INFERENCE UTILITY METHODS
  // ============================================================================

  /**
   * @brief Query workspace size for inference with pre-allocated output.
   *
   * @param max_compressed_size Maximum compressed chunk size expected
   * @param max_output_size Maximum decompressed output size expected
   * @return Required workspace size in bytes
   */
  size_t get_inference_workspace_size(size_t max_compressed_size,
                                      size_t max_output_size) const;

  /**
   * @brief Allocate a reusable inference workspace once at init time.
   *
   * Returns a workspace that can be reused for all inference calls.
   *
   * @param max_compressed_size Max compressed size expected
   * @param max_output_size Max output size expected
   * @param[out] workspace_ptr Set to allocated workspace (device memory)
   * @param[out] workspace_size Set to allocated size
   * @return Status::OK on success
   */
  Status allocate_inference_workspace(size_t max_compressed_size,
                                      size_t max_output_size,
                                      void **workspace_ptr,
                                      size_t *workspace_size);

  /**
   * @brief Free inference workspace allocated by allocate_inference_workspace.
   */
  Status free_inference_workspace(void *workspace_ptr);

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

// ==============================================================================
// STREAMING MANAGER
// ==============================================================================

/**
 * @brief Streaming compression manager for chunked data processing.
 * 
 * NOTE: This is a basic implementation that compresses each chunk independently.
 * Full streaming compression with sliding window history (for optimal compression
 * ratios across chunks) is not yet implemented. Each chunk is compressed as a
 * complete independent ZSTD frame.
 * 
 * Current limitations:
 * - No sliding window history between chunks (5-10% compression loss vs full streaming)
 * - No hash chain persistence across chunk boundaries
 * - Each chunk produces a complete ZSTD frame with full headers
 * 
 * For applications requiring maximum compression ratios on large streaming data,
 * consider using the batch manager with larger block sizes instead.
 */
class ZstdStreamingManager {
public:
  ZstdStreamingManager();
  explicit ZstdStreamingManager(const CompressionConfig &config);
  ~ZstdStreamingManager();

  // Initialization
  Status init_compression(cudaStream_t stream = 0, size_t max_chunk_size = 0);
  Status init_decompression(cudaStream_t stream = 0);

  // Streaming compression
  // NOTE: Each chunk is compressed independently. No window history is maintained
  // between chunks, which may result in lower compression ratios compared to
  // full streaming implementations.
  Status compress_chunk(const void *input, size_t input_size, void *output,
                        size_t *output_size, bool is_last_chunk,
                        cudaStream_t stream = 0);

  // Streaming decompression
  Status decompress_chunk(const void *input, size_t input_size, void *output,
                          size_t *output_size, bool *is_last_chunk,
                          cudaStream_t stream = 0);

  // Control
  Status reset();
  Status flush(cudaStream_t stream = 0);

  // Configuration
  Status set_config(const CompressionConfig &config);
  Status set_dictionary(const dictionary::Dictionary &dict);
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
std::unique_ptr<ZstdManager> create_manager(const CompressionConfig &config);
std::unique_ptr<ZstdBatchManager>
create_batch_manager(int compression_level = 3);
std::unique_ptr<ZstdStreamingManager>
create_streaming_manager(int compression_level = 3);

// ==============================================================================
// CONVENIENCE FUNCTIONS (Single-Shot)
// ==============================================================================

Status compress_simple(const void *uncompressed_data, size_t uncompressed_size,
                       void *compressed_data, size_t *compressed_size,
                       int compression_level = 3, cudaStream_t stream = 0);

Status decompress_simple(const void *compressed_data, size_t compressed_size,
                         void *uncompressed_data, size_t *uncompressed_size,
                         cudaStream_t stream = 0);

Status compress_with_dict(const void *uncompressed_data,
                          size_t uncompressed_size, void *compressed_data,
                          size_t *compressed_size,
                          const dictionary::Dictionary &dict,
                          int compression_level = 3, cudaStream_t stream = 0);

Status decompress_with_dict(const void *compressed_data, size_t compressed_size,
                            void *uncompressed_data, size_t *uncompressed_size,
                            const dictionary::Dictionary &dict,
                            cudaStream_t stream = 0);

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

Status get_decompressed_size(const void *compressed_data,
                             size_t compressed_size, size_t *decompressed_size);

Status validate_compressed_data(const void *compressed_data,
                                size_t compressed_size,
                                bool check_checksum = true);

size_t estimate_compressed_size(size_t uncompressed_size,
                                int compression_level);

Status validate_config(const CompressionConfig &config);
void apply_level_parameters(CompressionConfig &config);
u32 get_optimal_block_size(u32 input_size, u32 compression_level);

// ==============================================================================
// NVCOMP INTEGRATION
// ==============================================================================

constexpr const char *get_format_name() { return "cuda_zstd"; }

constexpr u32 get_format_version() {
  return 0x00010000; // 1.0.0
}

bool is_nvcomp_zstd_format(const void *compressed_data, size_t compressed_size);

Status extract_metadata(const void *compressed_data, size_t compressed_size,
                        NvcompMetadata &metadata);

// ==============================================================================
// ENHANCED PERFORMANCE PROFILING
// ==============================================================================

// Defined in performance_profiler.h
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
cuda_zstd_manager_t *cuda_zstd_create_manager(int compression_level);
void cuda_zstd_destroy_manager(cuda_zstd_manager_t *manager);

// Compression/Decompression
int cuda_zstd_compress(cuda_zstd_manager_t *manager, const void *src,
                       size_t src_size, void *dst, size_t *dst_size,
                       void *workspace, size_t workspace_size,
                       cudaStream_t stream);

int cuda_zstd_decompress(cuda_zstd_manager_t *manager, const void *src,
                         size_t src_size, void *dst, size_t *dst_size,
                         void *workspace, size_t workspace_size,
                         cudaStream_t stream);

// Workspace queries
size_t cuda_zstd_get_compress_workspace_size(cuda_zstd_manager_t *manager,
                                             size_t src_size);

size_t cuda_zstd_get_decompress_workspace_size(cuda_zstd_manager_t *manager,
                                               size_t compressed_size);

// Dictionary training
cuda_zstd_dict_t *cuda_zstd_train_dictionary(const void **samples,
                                             const size_t *sample_sizes,
                                             size_t num_samples,
                                             size_t dict_size);

void cuda_zstd_destroy_dictionary(cuda_zstd_dict_t *dict);

int cuda_zstd_set_dictionary(cuda_zstd_manager_t *manager,
                             cuda_zstd_dict_t *dict);

// Error handling
const char *cuda_zstd_get_error_string(int error_code);
int cuda_zstd_is_error(int code);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CUDA_ZSTD_MANAGER_H
