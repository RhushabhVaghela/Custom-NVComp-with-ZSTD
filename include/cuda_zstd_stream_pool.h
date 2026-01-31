#pragma once

#include "cuda_zstd_types.h"
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

namespace cuda_zstd {

class StreamPool {
public:
  struct PerStreamResources {
    cudaStream_t stream;
  };

  class Guard {
  public:
    Guard(StreamPool *pool, int idx);
    ~Guard();
    Guard(Guard &&) noexcept = default;
    Guard &operator=(Guard &&) noexcept = default;
    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;

    // Get the CUDA stream - returns nullptr if guard is invalid
    cudaStream_t get_stream() const {
      return resources_ ? resources_->stream : nullptr;
    }

    // Check if this guard is valid and can be used
    bool is_valid() const {
      return pool_ != nullptr && resources_ != nullptr && idx_ >= 0;
    }

    // Get the pool index (for debugging)
    int get_index() const { return idx_; }

  private:
    StreamPool *pool_;
    PerStreamResources *resources_;
    int idx_;
  };

  explicit StreamPool(size_t pool_size = 8);
  ~StreamPool();

  // Return the configured size of the pool (number of streams)
  size_t size() const { return resources_.size(); }

  // Check if the pool is valid and has working streams
  bool is_valid() const { return !resources_.empty(); }

  // Return the number of available (free) streams
  size_t available_count() const;

  // Acquire a stream and its associated resources, blocking if none available
  // Returns an invalid Guard if pool has no streams - always check is_valid()
  Guard acquire();

  // Acquire a stream with a timeout (milliseconds). Returns an empty optional
  // if no stream could be acquired within the timeout period.
  std::optional<Guard> acquire_for(size_t timeout_ms);

  // Non-copyable
  StreamPool(const StreamPool &) = delete;
  StreamPool &operator=(const StreamPool &) = delete;

private:
  std::vector<PerStreamResources> resources_;
  std::queue<int> free_idx_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;

  // Internal helpers
  int acquire_index();
  int acquire_index_for(size_t timeout_ms);
  void release_index(int idx);
};

} // namespace cuda_zstd
