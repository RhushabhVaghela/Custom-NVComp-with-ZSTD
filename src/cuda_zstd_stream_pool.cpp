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
  //     // std::cerr << "StreamPool ctor() pool_size=" << pool_size <<
  //     std::endl;
  if (pool_size == 0)
    pool_size = 1;

  resources_.resize(pool_size);
  const char *dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
  bool debug = (dbg && std::atoi(dbg) != 0);

  for (size_t i = 0; i < pool_size; ++i) {
    PerStreamResources &r = resources_[i];

    // Initialize stream to 0 before creation attempt
    r.stream = 0;
    
    // Ensure CUDA context is ready before creating stream
    // This helps avoid illegal memory access on some GPUs
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
      // Context may not be ready, try to get device properties
      int device = 0;
      cudaError_t dev_err = cudaGetDevice(&device);
      if (dev_err != cudaSuccess) {
        // Cannot get device, skip stream creation for this index
        fprintf(stderr, "[WARN] StreamPool: Cannot get CUDA device, skipping stream %zu\n", i);
        break;
      }
    }
    
    cudaError_t err = cudaStreamCreate(&r.stream);

    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "[CRITICAL] StreamPool: cudaStreamCreate failed at index %zu: %s\n",
          i, cudaGetErrorString(err));
      // Cleanup any resources created so far
      for (size_t j = 0; j < i; ++j) {
        cudaStreamDestroy(resources_[j].stream);
      }
      resources_.clear();
      // Create at least one working stream with default flags
      cudaError_t fallback_err = cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking);
      if (fallback_err == cudaSuccess) {
        resources_.push_back(r);
        free_idx_.push((int)0);
        fprintf(stderr, "[INFO] StreamPool: Created fallback stream successfully\n");
      } else {
        fprintf(stderr, "[CRITICAL] StreamPool: Fallback stream creation also failed\n");
      }
      break;
    }

    free_idx_.push((int)i);
  }
}

StreamPool::~StreamPool() {
  //     // std::cerr << "StreamPool dtor" << std::endl;
  std::unique_lock<std::mutex> lock(mtx_);
  while (!free_idx_.empty())
    free_idx_.pop();

  // Check if CUDA is still initialized before cleanup
  // cudaGetDeviceCount will return cudaErrorInitializationError if CUDA is
  // deinitialized
  int device_count = 0;
  cudaError_t init_check = cudaGetDeviceCount(&device_count);
  bool cuda_initialized =
      (init_check == cudaSuccess || init_check == cudaErrorNoDevice);

  if (!cuda_initialized) {
    // CUDA context is already torn down, skip cleanup to avoid errors
    //         // std::cerr << "StreamPool dtor: CUDA already deinitialized,
    //         skipping cleanup" << std::endl;
    return;
  }

  // Cleanup resources
  for (auto &r : resources_) {
    if (r.stream) {
      // Ensure stream has completed any work before destroying.
      cudaError_t sync_err = cudaStreamSynchronize(r.stream);
      if (sync_err != cudaSuccess && sync_err != cudaErrorCudartUnloading) {
        //                 // std::cerr << "StreamPool::~StreamPool:
        //                 cudaStreamSynchronize failed -> " << sync_err <<
        //                 std::endl;
      }
      cudaError_t destroy_err = cudaStreamDestroy(r.stream);
      if (destroy_err != cudaSuccess &&
          destroy_err != cudaErrorCudartUnloading) {
        //                 // std::cerr << "StreamPool::~StreamPool:
        //                 cudaStreamDestroy failed -> " << destroy_err <<
        //                 std::endl;
      }
      r.stream = 0;
    }
  }
}

int StreamPool::acquire_index() {
  std::unique_lock<std::mutex> lock(mtx_);
  while (free_idx_.empty()) {
    const char *dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
    bool debug = (dbg && std::atoi(dbg) != 0);
    //            if (debug) std::cerr << "StreamPool: waiting for free
    //            index..." << std::endl;
    cv_.wait(lock);
  }
  int idx = free_idx_.front();
  free_idx_.pop();
  return idx;
}

int StreamPool::acquire_index_for(size_t timeout_ms) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (timeout_ms == 0) {
    // A zero timeout means block indefinitely (preserve existing behaviour)
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
  return idx;
}

void StreamPool::release_index(int idx) {
  {
    std::unique_lock<std::mutex> lock(mtx_);
    free_idx_.push(idx);
    const char *dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
    bool debug = (dbg && std::atoi(dbg) != 0);
    //             if (debug) std::cerr << "StreamPool: released index " << idx
    //             << std::endl;
  }
  cv_.notify_one();
}

StreamPool::Guard StreamPool::acquire() {
  int idx = acquire_index();
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
  idx_ = idx;
  if (idx_ >= 0)
    resources_ = &pool_->resources_[idx_];
}

StreamPool::Guard::~Guard() {
  if (pool_ && idx_ >= 0) {
    pool_->release_index(idx_);
    idx_ = -1;
  }
}

} // namespace cuda_zstd
