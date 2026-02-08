// ============================================================================
// cuda_zstd_hybrid.cu - Hybrid CPU/GPU Compression/Decompression Engine
// ============================================================================
//
// Implementation of the HybridEngine that automatically routes compression
// and decompression operations to the best execution backend (CPU libzstd
// or GPU CUDA kernels) based on data location, size, and configuration.
//
// ============================================================================

#include "cuda_zstd_hybrid.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include "cuda_zstd_types.h"

#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace cuda_zstd {

// ============================================================================
// Timing helper
// ============================================================================

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Profiling history (for ADAPTIVE mode)
// ============================================================================

struct ThroughputSample {
  double throughput_mbps;
};

struct ProfilingHistory {
  // Rolling window of recent throughput observations
  static constexpr size_t MAX_SAMPLES = 64;

  ThroughputSample cpu_compress[MAX_SAMPLES] = {};
  size_t cpu_compress_count = 0;

  ThroughputSample gpu_compress[MAX_SAMPLES] = {};
  size_t gpu_compress_count = 0;

  ThroughputSample cpu_decompress[MAX_SAMPLES] = {};
  size_t cpu_decompress_count = 0;

  ThroughputSample gpu_decompress[MAX_SAMPLES] = {};
  size_t gpu_decompress_count = 0;

  void record(ExecutionBackend backend, bool is_compression,
              double throughput_mbps) {
    ThroughputSample *arr = nullptr;
    size_t *count = nullptr;

    if (is_compression) {
      if (backend == ExecutionBackend::CPU_LIBZSTD ||
          backend == ExecutionBackend::CPU_PARALLEL) {
        arr = cpu_compress;
        count = &cpu_compress_count;
      } else {
        arr = gpu_compress;
        count = &gpu_compress_count;
      }
    } else {
      if (backend == ExecutionBackend::CPU_LIBZSTD ||
          backend == ExecutionBackend::CPU_PARALLEL) {
        arr = cpu_decompress;
        count = &cpu_decompress_count;
      } else {
        arr = gpu_decompress;
        count = &gpu_decompress_count;
      }
    }

    if (arr && count) {
      size_t idx = (*count) % MAX_SAMPLES;
      arr[idx].throughput_mbps = throughput_mbps;
      (*count)++;
    }
  }

  double get_average(ExecutionBackend backend, bool is_compression) const {
    const ThroughputSample *arr = nullptr;
    size_t count = 0;

    if (is_compression) {
      if (backend == ExecutionBackend::CPU_LIBZSTD ||
          backend == ExecutionBackend::CPU_PARALLEL) {
        arr = cpu_compress;
        count = cpu_compress_count;
      } else {
        arr = gpu_compress;
        count = gpu_compress_count;
      }
    } else {
      if (backend == ExecutionBackend::CPU_LIBZSTD ||
          backend == ExecutionBackend::CPU_PARALLEL) {
        arr = cpu_decompress;
        count = cpu_decompress_count;
      } else {
        arr = gpu_decompress;
        count = gpu_decompress_count;
      }
    }

    if (!arr || count == 0)
      return 0.0;

    size_t n = std::min(count, MAX_SAMPLES);
    double sum = 0.0;
    size_t start = (count > MAX_SAMPLES) ? (count - MAX_SAMPLES) : 0;
    for (size_t i = 0; i < n; ++i) {
      sum += arr[(start + i) % MAX_SAMPLES].throughput_mbps;
    }
    return sum / static_cast<double>(n);
  }

  void reset() {
    cpu_compress_count = 0;
    gpu_compress_count = 0;
    cpu_decompress_count = 0;
    gpu_decompress_count = 0;
  }
};

// ============================================================================
// HybridEngine::Impl
// ============================================================================

class HybridEngine::Impl {
public:
  HybridConfig config_;
  std::unique_ptr<ZstdManager> gpu_manager_;
  CompressionStats cumulative_stats_;
  ProfilingHistory profiling_;
  mutable std::mutex mutex_;

  // Static routing reason strings (lifetime tied to process)
  static constexpr const char *REASON_FORCE_CPU = "FORCE_CPU mode";
  static constexpr const char *REASON_FORCE_GPU = "FORCE_GPU mode";
  static constexpr const char *REASON_HOST_TO_HOST =
      "Host-to-host: CPU is 100-790x faster";
  static constexpr const char *REASON_HOST_DATA_CPU_PREFERRED =
      "Host data with PREFER_CPU: avoid transfer overhead";
  static constexpr const char *REASON_DEVICE_SMALL =
      "Device data below gpu_device_threshold: GPU avoids transfer";
  static constexpr const char *REASON_DEVICE_TO_DEVICE_GPU =
      "Device-to-device: GPU avoids all transfers";
  static constexpr const char *REASON_DEVICE_LARGE_CPU =
      "Device data above threshold: D2H->CPU->H2D still faster";
  static constexpr const char *REASON_DEVICE_DECOMPRESS_SMALL =
      "Device decompress below 1MB: GPU avoids transfer overhead";
  static constexpr const char *REASON_MANAGED_CPU =
      "Managed memory: CPU can access directly, faster";
  static constexpr const char *REASON_ADAPTIVE_CPU =
      "Adaptive: CPU historically faster";
  static constexpr const char *REASON_ADAPTIVE_GPU =
      "Adaptive: GPU historically faster";
  static constexpr const char *REASON_GPU_FALLBACK =
      "GPU decompression failed, fell back to CPU";
  static constexpr const char *REASON_DEFAULT_CPU = "Default: CPU path";

  // ------------------------------------------------------------------
  Impl() : Impl(HybridConfig{}) {}

  explicit Impl(const HybridConfig &config) : config_(config) {
    reset_stats_internal();
  }

  // ------------------------------------------------------------------
  // Lazy GPU manager initialization (avoids GPU alloc if CPU-only)
  // ------------------------------------------------------------------
  ZstdManager &get_gpu_manager() {
    if (!gpu_manager_) {
      gpu_manager_ = create_manager(config_.compression_level);
    }
    return *gpu_manager_;
  }

  // ------------------------------------------------------------------
  // Routing Decision Engine
  // ------------------------------------------------------------------

  ExecutionBackend decide_route(size_t data_size, DataLocation input_loc,
                                DataLocation output_loc, bool is_compression,
                                const char **out_reason) const {
    const char *reason = REASON_DEFAULT_CPU;

    // Force modes bypass all heuristics
    if (config_.mode == HybridMode::FORCE_CPU) {
      reason = REASON_FORCE_CPU;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::CPU_LIBZSTD;
    }
    if (config_.mode == HybridMode::FORCE_GPU) {
      reason = REASON_FORCE_GPU;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::GPU_KERNELS;
    }

    // ADAPTIVE mode: use profiling history if available
    if (config_.mode == HybridMode::ADAPTIVE) {
      double cpu_tp = profiling_.get_average(ExecutionBackend::CPU_LIBZSTD,
                                             is_compression);
      double gpu_tp = profiling_.get_average(ExecutionBackend::GPU_KERNELS,
                                             is_compression);
      if (cpu_tp > 0.0 && gpu_tp > 0.0) {
        if (gpu_tp > cpu_tp * 1.2) {
          // GPU is >20% faster — use it
          reason = REASON_ADAPTIVE_GPU;
          if (out_reason)
            *out_reason = reason;
          return ExecutionBackend::GPU_KERNELS;
        } else {
          reason = REASON_ADAPTIVE_CPU;
          if (out_reason)
            *out_reason = reason;
          return ExecutionBackend::CPU_LIBZSTD;
        }
      }
      // Fall through to AUTO heuristics if not enough profiling data
    }

    // ----------------------------------------------------------------
    // AUTO / PREFER_CPU / PREFER_GPU heuristic routing
    // ----------------------------------------------------------------

    bool input_on_device =
        (input_loc == DataLocation::DEVICE);
    bool output_on_device =
        (output_loc == DataLocation::DEVICE);
    bool both_on_device = input_on_device && output_on_device;
    bool both_on_host =
        (!input_on_device && !output_on_device);

    // PREFER_CPU: always CPU unless both sides are on device
    if (config_.mode == HybridMode::PREFER_CPU) {
      if (both_on_device) {
        reason = REASON_DEVICE_TO_DEVICE_GPU;
        if (out_reason)
          *out_reason = reason;
        return ExecutionBackend::GPU_KERNELS;
      }
      reason = REASON_HOST_DATA_CPU_PREFERRED;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::CPU_LIBZSTD;
    }

    // PREFER_GPU: always GPU unless both sides are on host
    if (config_.mode == HybridMode::PREFER_GPU) {
      if (both_on_host) {
        reason = REASON_HOST_TO_HOST;
        if (out_reason)
          *out_reason = reason;
        return ExecutionBackend::CPU_LIBZSTD;
      }
      reason = REASON_DEVICE_TO_DEVICE_GPU;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::GPU_KERNELS;
    }

    // AUTO mode (default)

    // Case 1: Both host — CPU always wins
    if (both_on_host) {
      reason = REASON_HOST_TO_HOST;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::CPU_LIBZSTD;
    }

    // Case 2: Managed memory — CPU can access directly without transfer
    if (input_loc == DataLocation::MANAGED ||
        output_loc == DataLocation::MANAGED) {
      reason = REASON_MANAGED_CPU;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::CPU_LIBZSTD;
    }

    // Case 3: Both on device — GPU avoids all transfers
    if (both_on_device) {
      reason = REASON_DEVICE_TO_DEVICE_GPU;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::GPU_KERNELS;
    }

    // Case 4: Data on device but output goes to host (or vice versa)
    // For small data on device, GPU avoids transfer overhead
    if (input_on_device && data_size < config_.gpu_device_threshold) {
      reason = REASON_DEVICE_SMALL;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::GPU_KERNELS;
    }

    // For decompression of small device data (<1MB), GPU is better
    if (!is_compression && input_on_device &&
        data_size < config_.cpu_size_threshold) {
      reason = REASON_DEVICE_DECOMPRESS_SMALL;
      if (out_reason)
        *out_reason = reason;
      return ExecutionBackend::GPU_KERNELS;
    }

    // Default: CPU is faster for everything else (even with transfer)
    reason = REASON_DEVICE_LARGE_CPU;
    if (out_reason)
      *out_reason = reason;
    return ExecutionBackend::CPU_LIBZSTD;
  }

  // ------------------------------------------------------------------
  // Auto-detect data location using cudaPointerGetAttributes
  // ------------------------------------------------------------------

  static DataLocation detect_location_impl(const void *ptr) {
    if (!ptr)
      return DataLocation::HOST;

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
      cudaGetLastError(); // Clear error
      return DataLocation::HOST; // Assume host if query fails
    }

    switch (attrs.type) {
    case cudaMemoryTypeDevice:
      return DataLocation::DEVICE;
    case cudaMemoryTypeManaged:
      return DataLocation::MANAGED;
    case cudaMemoryTypeHost:
      return DataLocation::HOST;
    default:
      return DataLocation::HOST;
    }
  }

  // Resolve UNKNOWN locations to actual values
  DataLocation resolve_location(DataLocation loc, const void *ptr) const {
    if (loc == DataLocation::UNKNOWN) {
      return detect_location_impl(ptr);
    }
    return loc;
  }

  // ------------------------------------------------------------------
  // Fill HybridResult timing and metadata
  // ------------------------------------------------------------------

  void fill_result(HybridResult *result, ExecutionBackend backend,
                   DataLocation in_loc, DataLocation out_loc,
                   double total_ms, double transfer_ms, double compute_ms,
                   size_t in_bytes, size_t out_bytes,
                   const char *reason) {
    if (!result)
      return;
    result->backend_used = backend;
    result->input_location = in_loc;
    result->output_location = out_loc;
    result->total_time_ms = total_ms;
    result->transfer_time_ms = transfer_ms;
    result->compute_time_ms = compute_ms;
    result->input_bytes = in_bytes;
    result->output_bytes = out_bytes;
    result->routing_reason = reason;

    if (out_bytes > 0 && in_bytes > 0) {
      result->compression_ratio =
          static_cast<float>(in_bytes) / static_cast<float>(out_bytes);
    }

    if (total_ms > 0.0 && in_bytes > 0) {
      result->throughput_mbps =
          (static_cast<double>(in_bytes) / (1024.0 * 1024.0)) /
          (total_ms / 1000.0);
    }
  }

  // ------------------------------------------------------------------
  // CPU compress path
  // ------------------------------------------------------------------

  Status cpu_compress(const void *input, size_t input_size, void *output,
                      size_t *output_size, DataLocation in_loc,
                      DataLocation out_loc, double *transfer_ms,
                      double *compute_ms, cudaStream_t stream) {
    *transfer_ms = 0.0;
    *compute_ms = 0.0;

    // Get host-accessible input
    const void *h_input = input;
    std::vector<unsigned char> h_input_buf;
    bool input_on_device = (in_loc == DataLocation::DEVICE);

    if (input_on_device) {
      h_input_buf.resize(input_size);
      auto t0 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(h_input_buf.data(), input, input_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t0, Clock::now());
      h_input = h_input_buf.data();
    }

    // Compress on CPU
    size_t out_cap = *output_size;
    if (out_cap == 0) {
      out_cap = ZSTD_compressBound(input_size);
    }

    bool output_on_device = (out_loc == DataLocation::DEVICE);
    std::vector<unsigned char> h_output_buf;
    void *h_output = output;
    if (output_on_device) {
      h_output_buf.resize(out_cap);
      h_output = h_output_buf.data();
    }

    auto t1 = Clock::now();
    size_t cSize = ZSTD_compress(h_output, out_cap, h_input, input_size,
                                 config_.compression_level);
    *compute_ms = elapsed_ms(t1, Clock::now());

    if (ZSTD_isError(cSize)) {
      return Status::ERROR_COMPRESSION;
    }

    // Copy output to device if needed
    if (output_on_device) {
      auto t2 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(output, h_output_buf.data(), cSize,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t2, Clock::now());
    }

    *output_size = cSize;
    return Status::SUCCESS;
  }

  // ------------------------------------------------------------------
  // CPU decompress path
  // ------------------------------------------------------------------

  Status cpu_decompress(const void *input, size_t input_size, void *output,
                        size_t *output_size, DataLocation in_loc,
                        DataLocation out_loc, double *transfer_ms,
                        double *compute_ms, cudaStream_t stream) {
    *transfer_ms = 0.0;
    *compute_ms = 0.0;

    // Get host-accessible input
    const void *h_input = input;
    std::vector<unsigned char> h_input_buf;
    bool input_on_device = (in_loc == DataLocation::DEVICE);

    if (input_on_device) {
      h_input_buf.resize(input_size);
      auto t0 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(h_input_buf.data(), input, input_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t0, Clock::now());
      h_input = h_input_buf.data();
    }

    // Determine output capacity
    size_t out_cap = *output_size;
    if (out_cap == 0) {
      // Try to read from frame header
      unsigned long long frame_size =
          ZSTD_getFrameContentSize(h_input, input_size);
      if (frame_size == ZSTD_CONTENTSIZE_UNKNOWN ||
          frame_size == ZSTD_CONTENTSIZE_ERROR) {
        return Status::ERROR_INVALID_PARAMETER;
      }
      out_cap = static_cast<size_t>(frame_size);
    }

    bool output_on_device = (out_loc == DataLocation::DEVICE);
    std::vector<unsigned char> h_output_buf;
    void *h_output = output;
    if (output_on_device) {
      h_output_buf.resize(out_cap);
      h_output = h_output_buf.data();
    }

    auto t1 = Clock::now();
    size_t dSize = ZSTD_decompress(h_output, out_cap, h_input, input_size);
    *compute_ms = elapsed_ms(t1, Clock::now());

    if (ZSTD_isError(dSize)) {
      return Status::ERROR_DECOMPRESSION;
    }

    // Copy output to device if needed
    if (output_on_device) {
      auto t2 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(output, h_output_buf.data(), dSize,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t2, Clock::now());
    }

    *output_size = dSize;
    return Status::SUCCESS;
  }

  // ------------------------------------------------------------------
  // GPU compress path (delegates to DefaultZstdManager)
  // ------------------------------------------------------------------

  Status gpu_compress(const void *input, size_t input_size, void *output,
                      size_t *output_size, DataLocation in_loc,
                      DataLocation out_loc, double *transfer_ms,
                      double *compute_ms, cudaStream_t stream) {
    *transfer_ms = 0.0;
    *compute_ms = 0.0;

    auto &mgr = get_gpu_manager();

    // The DefaultZstdManager expects device pointers and handles
    // D2H internally for its CPU path. We need to ensure input/output
    // are on device, or let the manager's internal pointer detection handle it.

    // If input is on host, we need to upload it
    const void *d_input = input;
    void *d_input_alloc = nullptr;
    bool input_on_host = (in_loc == DataLocation::HOST);

    if (input_on_host) {
      auto t0 = Clock::now();
      cudaError_t err = safe_cuda_malloc(&d_input_alloc, input_size);
      if (err != cudaSuccess) {
        return Status::ERROR_OUT_OF_MEMORY;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_input_alloc, input, input_size,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t0, Clock::now());
      d_input = d_input_alloc;
    }

    // Allocate workspace
    size_t workspace_size = mgr.get_compress_temp_size(input_size);
    void *workspace = nullptr;
    if (workspace_size > 0) {
      cudaError_t err = safe_cuda_malloc(&workspace, workspace_size);
      if (err != cudaSuccess) {
        if (d_input_alloc)
          cudaFree(d_input_alloc);
        return Status::ERROR_OUT_OF_MEMORY;
      }
    }

    // If output goes to host, we need a device buffer
    void *d_output = output;
    void *d_output_alloc = nullptr;
    bool output_on_host = (out_loc == DataLocation::HOST);
    size_t out_cap = *output_size;

    if (output_on_host) {
      if (out_cap == 0) {
        out_cap = mgr.get_max_compressed_size(input_size);
      }
      cudaError_t err = safe_cuda_malloc(&d_output_alloc, out_cap);
      if (err != cudaSuccess) {
        if (d_input_alloc)
          cudaFree(d_input_alloc);
        if (workspace)
          cudaFree(workspace);
        return Status::ERROR_OUT_OF_MEMORY;
      }
      d_output = d_output_alloc;
      *output_size = out_cap;
    }

    // Compress
    auto t1 = Clock::now();
    Status status = mgr.compress(d_input, input_size, d_output, output_size,
                                 workspace, workspace_size, nullptr, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *compute_ms = elapsed_ms(t1, Clock::now());

    // Copy output back to host if needed
    if (status == Status::SUCCESS && output_on_host) {
      auto t2 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(output, d_output_alloc, *output_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t2, Clock::now());
    }

    // Cleanup
    if (d_input_alloc)
      cudaFree(d_input_alloc);
    if (d_output_alloc)
      cudaFree(d_output_alloc);
    if (workspace)
      cudaFree(workspace);

    return status;
  }

  // ------------------------------------------------------------------
  // GPU decompress path (delegates to DefaultZstdManager)
  // ------------------------------------------------------------------

  Status gpu_decompress(const void *input, size_t input_size, void *output,
                        size_t *output_size, DataLocation in_loc,
                        DataLocation out_loc, double *transfer_ms,
                        double *compute_ms, cudaStream_t stream) {
    *transfer_ms = 0.0;
    *compute_ms = 0.0;

    auto &mgr = get_gpu_manager();

    // Upload input to device if on host
    const void *d_input = input;
    void *d_input_alloc = nullptr;
    bool input_on_host = (in_loc == DataLocation::HOST);

    if (input_on_host) {
      auto t0 = Clock::now();
      cudaError_t err = safe_cuda_malloc(&d_input_alloc, input_size);
      if (err != cudaSuccess) {
        return Status::ERROR_OUT_OF_MEMORY;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_input_alloc, input, input_size,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t0, Clock::now());
      d_input = d_input_alloc;
    }

    // Workspace
    size_t workspace_size = mgr.get_decompress_temp_size(input_size);
    void *workspace = nullptr;
    if (workspace_size > 0) {
      cudaError_t err = safe_cuda_malloc(&workspace, workspace_size);
      if (err != cudaSuccess) {
        if (d_input_alloc)
          cudaFree(d_input_alloc);
        return Status::ERROR_OUT_OF_MEMORY;
      }
    }

    // Device output buffer
    void *d_output = output;
    void *d_output_alloc = nullptr;
    bool output_on_host = (out_loc == DataLocation::HOST);
    size_t out_cap = *output_size;

    if (output_on_host) {
      if (out_cap == 0) {
        return Status::ERROR_INVALID_PARAMETER;
      }
      cudaError_t err = safe_cuda_malloc(&d_output_alloc, out_cap);
      if (err != cudaSuccess) {
        if (d_input_alloc)
          cudaFree(d_input_alloc);
        if (workspace)
          cudaFree(workspace);
        return Status::ERROR_OUT_OF_MEMORY;
      }
      d_output = d_output_alloc;
    }

    // Decompress
    auto t1 = Clock::now();
    Status status = mgr.decompress(d_input, input_size, d_output, output_size,
                                   workspace, workspace_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *compute_ms = elapsed_ms(t1, Clock::now());

    // Copy output back to host
    if (status == Status::SUCCESS && output_on_host) {
      auto t2 = Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(output, d_output_alloc, *output_size,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      *transfer_ms += elapsed_ms(t2, Clock::now());
    }

    // Cleanup
    if (d_input_alloc)
      cudaFree(d_input_alloc);
    if (d_output_alloc)
      cudaFree(d_output_alloc);
    if (workspace)
      cudaFree(workspace);

    return status;
  }

  // ------------------------------------------------------------------
  // Stats
  // ------------------------------------------------------------------

  void update_stats(size_t in_bytes, size_t out_bytes) {
    cumulative_stats_.input_bytes += in_bytes;
    cumulative_stats_.output_bytes += out_bytes;
    cumulative_stats_.blocks_processed += 1;
  }

  void reset_stats_internal() {
    cumulative_stats_ = {};
  }
};

// ============================================================================
// HybridEngine public methods
// ============================================================================

HybridEngine::HybridEngine() : pimpl_(std::make_unique<Impl>()) {}

HybridEngine::HybridEngine(const HybridConfig &config)
    : pimpl_(std::make_unique<Impl>(config)) {}

HybridEngine::~HybridEngine() = default;

HybridEngine::HybridEngine(HybridEngine &&) noexcept = default;
HybridEngine &HybridEngine::operator=(HybridEngine &&) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

Status HybridEngine::configure(const HybridConfig &config) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  pimpl_->config_ = config;
  // If GPU manager already exists, update its compression level
  if (pimpl_->gpu_manager_) {
    pimpl_->gpu_manager_->set_compression_level(config.compression_level);
  }
  return Status::SUCCESS;
}

HybridConfig HybridEngine::get_config() const {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  return pimpl_->config_;
}

Status HybridEngine::set_compression_level(int level) {
  if (level < 1 || level > 22) {
    return Status::ERROR_INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  pimpl_->config_.compression_level = level;
  if (pimpl_->gpu_manager_) {
    pimpl_->gpu_manager_->set_compression_level(level);
  }
  return Status::SUCCESS;
}

// ============================================================================
// Compress
// ============================================================================

Status HybridEngine::compress(const void *input, size_t input_size,
                              void *output, size_t *output_size,
                              DataLocation input_loc, DataLocation output_loc,
                              HybridResult *result, cudaStream_t stream) {
  if (!input || !output || !output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(pimpl_->mutex_);

  auto total_start = Clock::now();

  // Resolve UNKNOWN locations
  DataLocation in_loc = pimpl_->resolve_location(input_loc, input);
  DataLocation out_loc = pimpl_->resolve_location(output_loc, output);

  // Route
  const char *reason = nullptr;
  ExecutionBackend backend =
      pimpl_->decide_route(input_size, in_loc, out_loc, true, &reason);

  double transfer_ms = 0.0, compute_ms = 0.0;
  Status status;

  if (backend == ExecutionBackend::CPU_LIBZSTD) {
    status = pimpl_->cpu_compress(input, input_size, output, output_size,
                                  in_loc, out_loc, &transfer_ms, &compute_ms,
                                  stream);
  } else {
    status = pimpl_->gpu_compress(input, input_size, output, output_size,
                                  in_loc, out_loc, &transfer_ms, &compute_ms,
                                  stream);
  }

  double total_ms = elapsed_ms(total_start, Clock::now());

  if (status == Status::SUCCESS) {
    pimpl_->update_stats(input_size, *output_size);

    if (pimpl_->config_.enable_profiling) {
      double tp = (total_ms > 0.0)
                      ? (static_cast<double>(input_size) / (1024.0 * 1024.0)) /
                            (total_ms / 1000.0)
                      : 0.0;
      pimpl_->profiling_.record(backend, true, tp);
    }
  }

  pimpl_->fill_result(result, backend, in_loc, out_loc, total_ms, transfer_ms,
                      compute_ms, input_size,
                      (status == Status::SUCCESS) ? *output_size : 0, reason);

  return status;
}

// ============================================================================
// Decompress
// ============================================================================

Status HybridEngine::decompress(const void *input, size_t input_size,
                                void *output, size_t *output_size,
                                DataLocation input_loc, DataLocation output_loc,
                                HybridResult *result, cudaStream_t stream) {
  if (!input || !output || !output_size || input_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(pimpl_->mutex_);

  auto total_start = Clock::now();

  DataLocation in_loc = pimpl_->resolve_location(input_loc, input);
  DataLocation out_loc = pimpl_->resolve_location(output_loc, output);

  const char *reason = nullptr;
  ExecutionBackend backend =
      pimpl_->decide_route(input_size, in_loc, out_loc, false, &reason);

  double transfer_ms = 0.0, compute_ms = 0.0;
  Status status;

  if (backend == ExecutionBackend::CPU_LIBZSTD) {
    status = pimpl_->cpu_decompress(input, input_size, output, output_size,
                                    in_loc, out_loc, &transfer_ms, &compute_ms,
                                    stream);
  } else {
    // GPU path — attempt GPU decompression, fall back to CPU on failure
    size_t saved_output_size = *output_size;
    status = pimpl_->gpu_decompress(input, input_size, output, output_size,
                                    in_loc, out_loc, &transfer_ms, &compute_ms,
                                    stream);

    if (status != Status::SUCCESS &&
        pimpl_->config_.mode != HybridMode::FORCE_GPU) {
      // GPU failed (likely Huffman-encoded literals from CPU-compressed data).
      // Fall back to CPU decompression.
      *output_size = saved_output_size;
      transfer_ms = 0.0;
      compute_ms = 0.0;
      status = pimpl_->cpu_decompress(input, input_size, output, output_size,
                                      in_loc, out_loc, &transfer_ms,
                                      &compute_ms, stream);
      backend = ExecutionBackend::CPU_LIBZSTD;
      reason = Impl::REASON_GPU_FALLBACK;
    }
  }

  double total_ms = elapsed_ms(total_start, Clock::now());

  if (status == Status::SUCCESS) {
    pimpl_->update_stats(input_size, *output_size);

    if (pimpl_->config_.enable_profiling) {
      double tp = (total_ms > 0.0)
                      ? (static_cast<double>(*output_size) /
                         (1024.0 * 1024.0)) /
                            (total_ms / 1000.0)
                      : 0.0;
      pimpl_->profiling_.record(backend, false, tp);
    }
  }

  pimpl_->fill_result(result, backend, in_loc, out_loc, total_ms, transfer_ms,
                      compute_ms, input_size,
                      (status == Status::SUCCESS) ? *output_size : 0, reason);

  return status;
}

// ============================================================================
// Batch Compress
// ============================================================================

Status HybridEngine::compress_batch(const void *const *inputs,
                                    const size_t *input_sizes, void **outputs,
                                    size_t *output_sizes, size_t count,
                                    DataLocation input_loc,
                                    DataLocation output_loc,
                                    BatchRoutingResult *results,
                                    cudaStream_t stream) {
  if (!inputs || !input_sizes || !outputs || !output_sizes || count == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Process each item individually with routing.
  // CPU items could be parallelized with threads, but for simplicity and
  // correctness we process sequentially. The main benefit of the hybrid
  // layer is routing, not batch parallelism (the GPU manager already
  // handles batch parallelism internally).

  bool any_failed = false;

  for (size_t i = 0; i < count; ++i) {
    HybridResult item_result;
    Status status = compress(inputs[i], input_sizes[i], outputs[i],
                             &output_sizes[i], input_loc, output_loc,
                             &item_result, stream);

    if (results) {
      results[i].item_index = i;
      results[i].backend_used = item_result.backend_used;
      results[i].status = status;
      results[i].input_bytes = input_sizes[i];
      results[i].output_bytes = output_sizes[i];
      results[i].compute_time_ms = item_result.compute_time_ms;
    }

    if (status != Status::SUCCESS) {
      any_failed = true;
    }
  }

  return any_failed ? Status::ERROR_COMPRESSION : Status::SUCCESS;
}

// ============================================================================
// Batch Decompress
// ============================================================================

Status HybridEngine::decompress_batch(const void *const *inputs,
                                      const size_t *input_sizes, void **outputs,
                                      size_t *output_sizes, size_t count,
                                      DataLocation input_loc,
                                      DataLocation output_loc,
                                      BatchRoutingResult *results,
                                      cudaStream_t stream) {
  if (!inputs || !input_sizes || !outputs || !output_sizes || count == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  bool any_failed = false;

  for (size_t i = 0; i < count; ++i) {
    HybridResult item_result;
    Status status = decompress(inputs[i], input_sizes[i], outputs[i],
                                &output_sizes[i], input_loc, output_loc,
                                &item_result, stream);

    if (results) {
      results[i].item_index = i;
      results[i].backend_used = item_result.backend_used;
      results[i].status = status;
      results[i].input_bytes = input_sizes[i];
      results[i].output_bytes = output_sizes[i];
      results[i].compute_time_ms = item_result.compute_time_ms;
    }

    if (status != Status::SUCCESS) {
      any_failed = true;
    }
  }

  return any_failed ? Status::ERROR_DECOMPRESSION : Status::SUCCESS;
}

// ============================================================================
// Query Functions
// ============================================================================

size_t HybridEngine::get_max_compressed_size(size_t input_size) const {
  return ZSTD_compressBound(input_size);
}

ExecutionBackend HybridEngine::query_routing(size_t data_size,
                                             DataLocation input_loc,
                                             DataLocation output_loc,
                                             bool is_compression) const {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  return pimpl_->decide_route(data_size, input_loc, output_loc, is_compression,
                              nullptr);
}

CompressionStats HybridEngine::get_stats() const {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  return pimpl_->cumulative_stats_;
}

void HybridEngine::reset_stats() {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  pimpl_->reset_stats_internal();
}

DataLocation HybridEngine::detect_location(const void *ptr) {
  return Impl::detect_location_impl(ptr);
}

// ============================================================================
// Profiling (ADAPTIVE mode)
// ============================================================================

double HybridEngine::get_observed_throughput(ExecutionBackend backend,
                                             bool is_compression) const {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  return pimpl_->profiling_.get_average(backend, is_compression);
}

void HybridEngine::reset_profiling() {
  std::lock_guard<std::mutex> lock(pimpl_->mutex_);
  pimpl_->profiling_.reset();
}

// ============================================================================
// Convenience Functions
// ============================================================================

Status hybrid_compress(const void *input, size_t input_size, void *output,
                       size_t *output_size, DataLocation input_loc,
                       DataLocation output_loc, int compression_level,
                       HybridResult *result, cudaStream_t stream) {
  HybridConfig config;
  config.compression_level = compression_level;
  HybridEngine engine(config);
  return engine.compress(input, input_size, output, output_size, input_loc,
                         output_loc, result, stream);
}

Status hybrid_decompress(const void *input, size_t input_size, void *output,
                         size_t *output_size, DataLocation input_loc,
                         DataLocation output_loc, HybridResult *result,
                         cudaStream_t stream) {
  HybridConfig config;
  HybridEngine engine(config);
  return engine.decompress(input, input_size, output, output_size, input_loc,
                           output_loc, result, stream);
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<HybridEngine> create_hybrid_engine(const HybridConfig &config) {
  return std::make_unique<HybridEngine>(config);
}

std::unique_ptr<HybridEngine> create_hybrid_engine(int compression_level) {
  HybridConfig config;
  config.compression_level = compression_level;
  return std::make_unique<HybridEngine>(config);
}

} // namespace cuda_zstd

// ============================================================================
// C API Implementation
// ============================================================================

struct cuda_zstd_hybrid_engine_t {
  cuda_zstd::HybridEngine engine;
};

extern "C" {

cuda_zstd_hybrid_engine_t *
cuda_zstd_hybrid_create(const cuda_zstd_hybrid_config_t *config) {
  auto *wrapper = new (std::nothrow) cuda_zstd_hybrid_engine_t();
  if (!wrapper)
    return nullptr;

  if (config) {
    cuda_zstd::HybridConfig cfg;
    cfg.mode = static_cast<cuda_zstd::HybridMode>(config->mode);
    cfg.cpu_size_threshold = config->cpu_size_threshold;
    cfg.gpu_device_threshold = config->gpu_device_threshold;
    cfg.compression_level = config->compression_level;
    cfg.enable_profiling = (config->enable_profiling != 0);
    cfg.cpu_thread_count = config->cpu_thread_count;
    wrapper->engine.configure(cfg);
  }

  return wrapper;
}

cuda_zstd_hybrid_engine_t *cuda_zstd_hybrid_create_default(void) {
  return cuda_zstd_hybrid_create(nullptr);
}

void cuda_zstd_hybrid_destroy(cuda_zstd_hybrid_engine_t *engine) {
  delete engine;
}

int cuda_zstd_hybrid_compress(cuda_zstd_hybrid_engine_t *engine,
                              const void *input, size_t input_size,
                              void *output, size_t *output_size,
                              unsigned int input_loc, unsigned int output_loc,
                              cuda_zstd_hybrid_result_t *result,
                              cudaStream_t stream) {
  if (!engine)
    return static_cast<int>(cuda_zstd::Status::ERROR_INVALID_PARAMETER);

  cuda_zstd::HybridResult cpp_result;
  cuda_zstd::Status status = engine->engine.compress(
      input, input_size, output, output_size,
      static_cast<cuda_zstd::DataLocation>(input_loc),
      static_cast<cuda_zstd::DataLocation>(output_loc), &cpp_result, stream);

  if (result) {
    result->backend_used = static_cast<unsigned int>(cpp_result.backend_used);
    result->input_location =
        static_cast<unsigned int>(cpp_result.input_location);
    result->output_location =
        static_cast<unsigned int>(cpp_result.output_location);
    result->total_time_ms = cpp_result.total_time_ms;
    result->transfer_time_ms = cpp_result.transfer_time_ms;
    result->compute_time_ms = cpp_result.compute_time_ms;
    result->throughput_mbps = cpp_result.throughput_mbps;
    result->input_bytes = cpp_result.input_bytes;
    result->output_bytes = cpp_result.output_bytes;
    result->compression_ratio = cpp_result.compression_ratio;
  }

  return static_cast<int>(status);
}

int cuda_zstd_hybrid_decompress(cuda_zstd_hybrid_engine_t *engine,
                                const void *input, size_t input_size,
                                void *output, size_t *output_size,
                                unsigned int input_loc, unsigned int output_loc,
                                cuda_zstd_hybrid_result_t *result,
                                cudaStream_t stream) {
  if (!engine)
    return static_cast<int>(cuda_zstd::Status::ERROR_INVALID_PARAMETER);

  cuda_zstd::HybridResult cpp_result;
  cuda_zstd::Status status = engine->engine.decompress(
      input, input_size, output, output_size,
      static_cast<cuda_zstd::DataLocation>(input_loc),
      static_cast<cuda_zstd::DataLocation>(output_loc), &cpp_result, stream);

  if (result) {
    result->backend_used = static_cast<unsigned int>(cpp_result.backend_used);
    result->input_location =
        static_cast<unsigned int>(cpp_result.input_location);
    result->output_location =
        static_cast<unsigned int>(cpp_result.output_location);
    result->total_time_ms = cpp_result.total_time_ms;
    result->transfer_time_ms = cpp_result.transfer_time_ms;
    result->compute_time_ms = cpp_result.compute_time_ms;
    result->throughput_mbps = cpp_result.throughput_mbps;
    result->input_bytes = cpp_result.input_bytes;
    result->output_bytes = cpp_result.output_bytes;
    result->compression_ratio = cpp_result.compression_ratio;
  }

  return static_cast<int>(status);
}

size_t cuda_zstd_hybrid_max_compressed_size(cuda_zstd_hybrid_engine_t *engine,
                                            size_t input_size) {
  if (!engine)
    return 0;
  return engine->engine.get_max_compressed_size(input_size);
}

unsigned int cuda_zstd_hybrid_query_routing(cuda_zstd_hybrid_engine_t *engine,
                                            size_t data_size,
                                            unsigned int input_loc,
                                            unsigned int output_loc,
                                            int is_compression) {
  if (!engine)
    return static_cast<unsigned int>(cuda_zstd::ExecutionBackend::CPU_LIBZSTD);
  return static_cast<unsigned int>(engine->engine.query_routing(
      data_size, static_cast<cuda_zstd::DataLocation>(input_loc),
      static_cast<cuda_zstd::DataLocation>(output_loc),
      is_compression != 0));
}

} // extern "C"
