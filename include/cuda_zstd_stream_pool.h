#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cuda_runtime.h>
#include "cuda_zstd_types.h"

namespace cuda_zstd {

class StreamPool {
public:
    struct PerStreamResources {
        cudaStream_t stream;
        // A small device-side buffer reserved for checksums per stream
        u64* checksum_buf;
    };

    class Guard {
    public:
        Guard(StreamPool* pool, int idx);
        ~Guard();
        Guard(Guard&&) noexcept = default;
        Guard& operator=(Guard&&) noexcept = default;
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        cudaStream_t get_stream() const { return resources_->stream; }
        u64* get_checksum_buf() const { return resources_->checksum_buf; }
    private:
        StreamPool* pool_;
        PerStreamResources* resources_;
        int idx_;
    };

    explicit StreamPool(size_t pool_size = 8);
    ~StreamPool();

    // Acquire a stream and its associated resources, blocking if none available
    Guard acquire();

    // Non-copyable
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

private:
    std::vector<PerStreamResources> resources_;
    std::queue<int> free_idx_;
    std::mutex mtx_;
    std::condition_variable cv_;

    // Internal helpers
    int acquire_index();
    void release_index(int idx);
};

// Global stream pool accessor: a shared pool usable across managers. The pool
// is created on first use with the provided size or defaults to environment
// variable `CUDA_ZSTD_STREAM_POOL_SIZE` (if set) or 8 if not.
StreamPool* get_global_stream_pool(size_t default_size = 8);

} // namespace cuda_zstd
