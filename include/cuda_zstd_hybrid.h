// ============================================================================
// cuda_zstd_hybrid.h - Hybrid CPU/GPU Compression/Decompression Engine
// ============================================================================
//
// Automatically routes compression and decompression operations to the best
// execution backend (CPU or GPU) based on data characteristics, location,
// and available resources.
//
// Routing Decision Matrix:
//
// COMPRESSION:
// +-----------------------+----------------+-----------------------------------+
// | Scenario              | Chosen Path    | Reason                            |
// +-----------------------+----------------+-----------------------------------+
// | Host data, any size   | CPU (libzstd)  | CPU is 100-790x faster than GPU   |
// | Device data, <64KB    | GPU (in-place) | Avoid transfer overhead            |
// | Device data, >=64KB   | D2H->CPU->H2D  | CPU still faster even with xfer   |
// | Device data, output   | GPU (in-place) | No transfer needed                 |
// |   stays on device     |                |                                    |
// +-----------------------+----------------+-----------------------------------+
//
// DECOMPRESSION:
// +-----------------------+----------------+-----------------------------------+
// | Scenario              | Chosen Path    | Reason                            |
// +-----------------------+----------------+-----------------------------------+
// | Host data, any size   | CPU (libzstd)  | CPU is 74x faster than GPU         |
// | Device data, output   | GPU (in-place) | No transfer needed                 |
// |   stays on device     |                |                                    |
// | Device data, >1MB     | D2H->CPU->H2D  | CPU still faster with transfer     |
// | Device data, <1MB     | GPU (in-place) | Transfer overhead dominates         |
// +-----------------------+----------------+-----------------------------------+
//
// ============================================================================

#ifndef CUDA_ZSTD_HYBRID_H_
#define CUDA_ZSTD_HYBRID_H_

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
#include <memory>
#include <vector>

namespace cuda_zstd {

// ============================================================================
// Hybrid Engine Class
// ============================================================================

/**
 * @brief Hybrid CPU/GPU compression engine that automatically routes
 *        operations to the best backend.
 *
 * Usage:
 * @code
 *   HybridConfig config;
 *   config.mode = HybridMode::AUTO;
 *   config.compression_level = 3;
 *
 *   HybridEngine engine(config);
 *
 *   // Compress host data (auto-routes to CPU)
 *   HybridResult result;
 *   engine.compress(host_input, size, host_output, &out_size,
 *                   DataLocation::HOST, DataLocation::HOST, &result);
 *
 *   // Decompress device data to device (auto-routes to GPU)
 *   engine.decompress(d_compressed, comp_size, d_output, &decomp_size,
 *                     DataLocation::DEVICE, DataLocation::DEVICE, &result);
 * @endcode
 */
class HybridEngine {
public:
  HybridEngine();
  explicit HybridEngine(const HybridConfig &config);
  ~HybridEngine();

  // Non-copyable, movable
  HybridEngine(const HybridEngine &) = delete;
  HybridEngine &operator=(const HybridEngine &) = delete;
  HybridEngine(HybridEngine &&) noexcept;
  HybridEngine &operator=(HybridEngine &&) noexcept;

  // ============================================================================
  // Configuration
  // ============================================================================

  /**
   * @brief Update the hybrid engine configuration.
   */
  Status configure(const HybridConfig &config);

  /**
   * @brief Get the current configuration.
   */
  HybridConfig get_config() const;

  /**
   * @brief Set compression level (1-22).
   */
  Status set_compression_level(int level);

  // ============================================================================
  // Single-Item Operations
  // ============================================================================

  /**
   * @brief Compress data using the best available backend.
   *
   * @param input         Input data pointer (host or device, as specified)
   * @param input_size    Size of input data in bytes
   * @param output        Output buffer pointer (host or device, as specified)
   * @param output_size   [in/out] Capacity on input, actual compressed size on output
   * @param input_loc     Where the input data resides
   * @param output_loc    Where the output should be placed
   * @param result        [out] Optional result with timing and path info
   * @param stream        CUDA stream for GPU operations
   * @return Status::SUCCESS on success
   */
  Status compress(const void *input, size_t input_size,
                  void *output, size_t *output_size,
                  DataLocation input_loc = DataLocation::HOST,
                  DataLocation output_loc = DataLocation::HOST,
                  HybridResult *result = nullptr,
                  cudaStream_t stream = 0);

  /**
   * @brief Decompress data using the best available backend.
   *
   * @param input         Compressed data pointer (host or device)
   * @param input_size    Size of compressed data in bytes
   * @param output        Output buffer pointer (host or device)
   * @param output_size   [in/out] Capacity on input, actual decompressed size on output
   * @param input_loc     Where the compressed data resides
   * @param output_loc    Where the decompressed output should be placed
   * @param result        [out] Optional result with timing and path info
   * @param stream        CUDA stream for GPU operations
   * @return Status::SUCCESS on success
   */
  Status decompress(const void *input, size_t input_size,
                    void *output, size_t *output_size,
                    DataLocation input_loc = DataLocation::HOST,
                    DataLocation output_loc = DataLocation::HOST,
                    HybridResult *result = nullptr,
                    cudaStream_t stream = 0);

  // ============================================================================
  // Batch Operations
  // ============================================================================

  /**
   * @brief Compress multiple items, routing each to the best backend.
   *
   * Items may be routed to different backends. CPU items are processed
   * in parallel threads; GPU items use the CUDA stream.
   *
   * @param inputs        Array of input data pointers
   * @param input_sizes   Array of input data sizes
   * @param outputs       Array of output buffer pointers
   * @param output_sizes  [in/out] Array of capacities / actual sizes
   * @param count         Number of items
   * @param input_loc     Where the input data resides
   * @param output_loc    Where the outputs should be placed
   * @param results       [out] Optional per-item results (array of count)
   * @param stream        CUDA stream for GPU operations
   * @return Status::SUCCESS if all items succeeded
   */
  Status compress_batch(const void *const *inputs, const size_t *input_sizes,
                        void **outputs, size_t *output_sizes,
                        size_t count,
                        DataLocation input_loc = DataLocation::HOST,
                        DataLocation output_loc = DataLocation::HOST,
                        BatchRoutingResult *results = nullptr,
                        cudaStream_t stream = 0);

  /**
   * @brief Decompress multiple items, routing each to the best backend.
   */
  Status decompress_batch(const void *const *inputs, const size_t *input_sizes,
                          void **outputs, size_t *output_sizes,
                          size_t count,
                          DataLocation input_loc = DataLocation::HOST,
                          DataLocation output_loc = DataLocation::HOST,
                          BatchRoutingResult *results = nullptr,
                          cudaStream_t stream = 0);

  // ============================================================================
  // Query Functions
  // ============================================================================

  /**
   * @brief Get the maximum compressed size for a given input size.
   */
  size_t get_max_compressed_size(size_t input_size) const;

  /**
   * @brief Query which backend would be selected for given parameters.
   *
   * Does not execute any operation — purely advisory.
   */
  ExecutionBackend query_routing(size_t data_size,
                                 DataLocation input_loc,
                                 DataLocation output_loc,
                                 bool is_compression) const;

  /**
   * @brief Get cumulative statistics from all operations.
   */
  CompressionStats get_stats() const;

  /**
   * @brief Reset cumulative statistics.
   */
  void reset_stats();

  /**
   * @brief Detect the memory location of a pointer automatically.
   */
  static DataLocation detect_location(const void *ptr);

  // ============================================================================
  // Profiling (ADAPTIVE mode)
  // ============================================================================

  /**
   * @brief Get the average throughput observed for a given backend.
   */
  double get_observed_throughput(ExecutionBackend backend,
                                 bool is_compression) const;

  /**
   * @brief Reset profiling history.
   */
  void reset_profiling();

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// Convenience Functions (Single-Shot)
// ============================================================================

/**
 * @brief Compress data using the hybrid engine with default settings.
 *
 * Creates a temporary HybridEngine, compresses, and returns.
 * For repeated operations, create a HybridEngine instance and reuse it.
 */
Status hybrid_compress(const void *input, size_t input_size,
                       void *output, size_t *output_size,
                       DataLocation input_loc = DataLocation::HOST,
                       DataLocation output_loc = DataLocation::HOST,
                       int compression_level = 3,
                       HybridResult *result = nullptr,
                       cudaStream_t stream = 0);

/**
 * @brief Decompress data using the hybrid engine with default settings.
 */
Status hybrid_decompress(const void *input, size_t input_size,
                         void *output, size_t *output_size,
                         DataLocation input_loc = DataLocation::HOST,
                         DataLocation output_loc = DataLocation::HOST,
                         HybridResult *result = nullptr,
                         cudaStream_t stream = 0);

// ============================================================================
// Factory
// ============================================================================

/**
 * @brief Create a hybrid engine with the given configuration.
 */
std::unique_ptr<HybridEngine> create_hybrid_engine(
    const HybridConfig &config = HybridConfig{});

/**
 * @brief Create a hybrid engine with just a compression level.
 */
std::unique_ptr<HybridEngine> create_hybrid_engine(int compression_level);

} // namespace cuda_zstd
#endif // __cplusplus

// ============================================================================
// C API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_zstd_hybrid_engine_t cuda_zstd_hybrid_engine_t;

/**
 * @brief C API hybrid configuration.
 */
typedef struct {
  unsigned int mode;               // HybridMode enum value
  size_t cpu_size_threshold;       // Bytes below which CPU is preferred
  size_t gpu_device_threshold;     // Min device-resident size for GPU path
  int compression_level;           // 1-22
  int enable_profiling;            // 0 or 1
  unsigned int cpu_thread_count;   // 0 = auto
} cuda_zstd_hybrid_config_t;

/**
 * @brief C API hybrid result.
 */
typedef struct {
  unsigned int backend_used;       // ExecutionBackend enum value
  unsigned int input_location;     // DataLocation enum value
  unsigned int output_location;    // DataLocation enum value
  double total_time_ms;
  double transfer_time_ms;
  double compute_time_ms;
  double throughput_mbps;
  size_t input_bytes;
  size_t output_bytes;
  float compression_ratio;
} cuda_zstd_hybrid_result_t;

// Engine lifecycle
cuda_zstd_hybrid_engine_t *cuda_zstd_hybrid_create(
    const cuda_zstd_hybrid_config_t *config);

cuda_zstd_hybrid_engine_t *cuda_zstd_hybrid_create_default(void);

void cuda_zstd_hybrid_destroy(cuda_zstd_hybrid_engine_t *engine);

// Single-item operations
// input_loc and output_loc: 0=HOST, 1=DEVICE, 2=MANAGED, 3=UNKNOWN(auto-detect)
int cuda_zstd_hybrid_compress(cuda_zstd_hybrid_engine_t *engine,
                               const void *input, size_t input_size,
                               void *output, size_t *output_size,
                               unsigned int input_loc,
                               unsigned int output_loc,
                               cuda_zstd_hybrid_result_t *result,
                               cudaStream_t stream);

int cuda_zstd_hybrid_decompress(cuda_zstd_hybrid_engine_t *engine,
                                 const void *input, size_t input_size,
                                 void *output, size_t *output_size,
                                 unsigned int input_loc,
                                 unsigned int output_loc,
                                 cuda_zstd_hybrid_result_t *result,
                                 cudaStream_t stream);

// Query
size_t cuda_zstd_hybrid_max_compressed_size(
    cuda_zstd_hybrid_engine_t *engine, size_t input_size);

unsigned int cuda_zstd_hybrid_query_routing(
    cuda_zstd_hybrid_engine_t *engine,
    size_t data_size, unsigned int input_loc,
    unsigned int output_loc, int is_compression);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ZSTD_HYBRID_H_
