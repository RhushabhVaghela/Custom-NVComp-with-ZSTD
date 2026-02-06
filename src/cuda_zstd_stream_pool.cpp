#include "cuda_zstd_stream_pool.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace cuda_zstd {

StreamPool::StreamPool(size_t pool_size) {
  if (pool_size == 0)
    pool_size = 1;

  // Pre-allocate vector capacity but don't resize yet
  resources_.reserve(pool_size);

  const char *dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
  bool debug = (dbg && std::atoi(dbg) != 0);

  // Verify CUDA context is initialized before creating streams
  int device_count = 0;
  cudaError_t ctx_err = cudaGetDeviceCount(&device_count);
  if (ctx_err != cudaSuccess || device_count == 0) {
    // fprintf(stderr,
    //         "[CRITICAL] StreamPool: CUDA not available or no devices
    //         found\n");
    // Create a minimal pool with null streams - will fail gracefully later
    resources_.clear();
    return;
  }

  // Ensure CUDA context is active on current thread
  int current_device = 0;
  cudaError_t dev_err = cudaGetDevice(&current_device);
  if (dev_err != cudaSuccess) {
    // fprintf(
    //     stderr,
    //     "[WARN] StreamPool: Cannot get current CUDA device, trying device
    //     0\n");
    cudaError_t set_err = cudaSetDevice(0);
    if (set_err != cudaSuccess) {
      // fprintf(stderr, "[CRITICAL] StreamPool: Cannot set CUDA device 0\n");
      resources_.clear();
      return;
    }
  }

  size_t created_count = 0;
  for (size_t i = 0; i < pool_size; ++i) {
    PerStreamResources r;
    // Initialize stream to 0 before creation attempt
    r.stream = nullptr;

    // Create stream with non-blocking flag for better performance
    cudaError_t err =
        cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking);

    if (err != cudaSuccess) {
      // fprintf(
      //     stderr,
      //     "[CRITICAL] StreamPool: cudaStreamCreate failed at index %zu:
      //     %s\n", i, cudaGetErrorString(err));

      // If we haven't created any streams yet, this is fatal
      if (created_count == 0) {
        // Try one more time with default stream flags
        err = cudaStreamCreate(&r.stream);
        if (err != cudaSuccess) {
          // fprintf(stderr,
          //         "[CRITICAL] StreamPool: Fallback stream creation also "
          //         "failed: %s\n",
          //         cudaGetErrorString(err));
          resources_.clear();
          return;
        }
        // Success with fallback - add this stream
        resources_.push_back(r);
        free_idx_.push(static_cast<int>(resources_.size() - 1));
        created_count++;
      }
      // If we have some streams, just stop creating more
      break;
    }

    // Successfully created stream - add to pool
    resources_.push_back(r);
    free_idx_.push(static_cast<int>(resources_.size() - 1));
    created_count++;
  }

  if (debug) {
  }
}

StreamPool::~StreamPool() {
  // Lock to ensure no concurrent access during destruction
  std::unique_lock<std::mutex> lock(mtx_);

  // Clear the free queue
  while (!free_idx_.empty()) {
    free_idx_.pop();
  }

  // Check if CUDA is still initialized before cleanup
  int device_count = 0;
  cudaError_t init_check = cudaGetDeviceCount(&device_count);
  bool cuda_initialized = (init_check == cudaSuccess);

  if (!cuda_initialized) {
    // CUDA context is already torn down, skip cleanup
    return;
  }

  // Unlock during CUDA operations to avoid deadlocks
  lock.unlock();

  // Cleanup resources
  for (auto &r : resources_) {
    if (r.stream != nullptr) {
      // Ensure stream has completed any work before destroying
      cudaError_t sync_err = cudaStreamSynchronize(r.stream);
      if (sync_err != cudaSuccess && sync_err != cudaErrorCudartUnloading) {
        // Log error but continue cleanup
        // fprintf(stderr, "[WARN] StreamPool: cudaStreamSynchronize failed "
        //                 "during destruction\n");
      }

      cudaError_t destroy_err = cudaStreamDestroy(r.stream);
      if (destroy_err != cudaSuccess &&
          destroy_err != cudaErrorCudartUnloading) {
        // fprintf(
        //     stderr,
        //     "[WARN] StreamPool: cudaStreamDestroy failed during
        //     destruction\n");
      }
      r.stream = nullptr;
    }
  }
}

int StreamPool::acquire_index() {
  std::unique_lock<std::mutex> lock(mtx_);

  // Check if pool has any resources
  if (resources_.empty()) {
    fprintf(stderr, "[ERROR] StreamPool: No streams available in pool\n");
    return -1;
  }

  while (free_idx_.empty()) {
    cv_.wait(lock);
  }

  int idx = free_idx_.front();
  free_idx_.pop();

  // Validate index is within bounds
  if (idx < 0 || static_cast<size_t>(idx) >= resources_.size()) {
    fprintf(stderr, "[CRITICAL] StreamPool: Invalid index %d (pool size %zu)\n",
            idx, resources_.size());
    return -1;
  }

  return idx;
}

int StreamPool::acquire_index_for(size_t timeout_ms) {
  std::unique_lock<std::mutex> lock(mtx_);

  // Check if pool has any resources
  if (resources_.empty()) {
    fprintf(stderr, "[ERROR] StreamPool: No streams available in pool\n");
    return -1;
  }

  if (timeout_ms == 0) {
    // A zero timeout means block indefinitely
    while (free_idx_.empty()) {
      cv_.wait(lock);
    }
  } else {
    bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [this]() { return !free_idx_.empty(); });
    if (!ok) {
      return -1;
    }
  }

  int idx = free_idx_.front();
  free_idx_.pop();

  // Validate index is within bounds
  if (idx < 0 || static_cast<size_t>(idx) >= resources_.size()) {
    fprintf(stderr, "[CRITICAL] StreamPool: Invalid index %d (pool size %zu)\n",
            idx, resources_.size());
    return -1;
  }

  return idx;
}

void StreamPool::release_index(int idx) {
  {
    std::unique_lock<std::mutex> lock(mtx_);

    // Validate index before releasing
    if (idx < 0 || static_cast<size_t>(idx) >= resources_.size()) {
      fprintf(stderr,
              "[CRITICAL] StreamPool: Attempted to release invalid index %d\n",
              idx);
      return;
    }

    free_idx_.push(idx);
  }
  cv_.notify_one();
}

StreamPool::Guard StreamPool::acquire() {
  int idx = acquire_index();
  if (idx < 0) {
    // Return an invalid guard if acquisition failed
    return Guard(nullptr, -1);
  }
  return Guard(this, idx);
}

std::optional<StreamPool::Guard> StreamPool::acquire_for(size_t timeout_ms) {
  int idx = acquire_index_for(timeout_ms);
  if (idx < 0)
    return std::nullopt;
  return Guard(this, idx);
}

StreamPool::Guard::Guard(StreamPool *pool, int idx)
    : pool_(pool), resources_(nullptr), idx_(-1) {

  // Validate inputs
  if (pool == nullptr || idx < 0) {
    pool_ = nullptr;
    idx_ = -1;
    resources_ = nullptr;
    return;
  }

  // Lock to safely access pool resources
  std::unique_lock<std::mutex> lock(pool->mtx_);

  // Validate index is within bounds
  if (static_cast<size_t>(idx) >= pool->resources_.size()) {
    fprintf(stderr,
            "[CRITICAL] StreamPool::Guard: Index %d out of bounds (size %zu)\n",
            idx, pool->resources_.size());
    pool_ = nullptr;
    idx_ = -1;
    resources_ = nullptr;
    return;
  }

  // Validate stream is not null
  if (pool->resources_[idx].stream == nullptr) {
    fprintf(stderr,
            "[CRITICAL] StreamPool::Guard: Stream at index %d is null\n", idx);
    pool_ = nullptr;
    idx_ = -1;
    resources_ = nullptr;
    return;
  }

  // All validations passed - initialize guard
  idx_ = idx;
  resources_ = &pool->resources_[idx];
}

StreamPool::Guard::~Guard() {
  if (pool_ != nullptr && idx_ >= 0) {
    pool_->release_index(idx_);
    idx_ = -1;
    pool_ = nullptr;
    resources_ = nullptr;
  }
}

size_t StreamPool::available_count() const {
  std::unique_lock<std::mutex> lock(mtx_);
  return free_idx_.size();
}

} // namespace cuda_zstd
