// ==============================================================================
// stream_pool.h - Stream pooling for parallel batching
// ==============================================================================

#ifndef STREAM_POOL_H
#define STREAM_POOL_H

#include "common_types.h"
#include "error_context.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <optional>

namespace cuda_zstd {

class StreamPool {
public:
    explicit StreamPool(size_t num_streams = 8);
    ~StreamPool();

    class Guard {
    public:
        Guard(StreamPool* pool, cudaStream_t stream);
        ~Guard();

        cudaStream_t get_stream() const { return stream_; }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        Guard(Guard&& other) noexcept;
        Guard& operator=(Guard&& other) noexcept;

    private:
        StreamPool* pool_;
        cudaStream_t stream_;
    };

    cudaStream_t acquire();
    cudaStream_t acquire_with_timeout(size_t timeout_ms);
    void release(cudaStream_t stream);

    Guard acquire_guard();
    std::optional<Guard> acquire_guard_with_timeout(size_t timeout_ms);

    void synchronize_all();
    size_t size() const { return num_streams_; }
    size_t available() const;

private:
    std::vector<cudaStream_t> streams_;
    std::vector<bool> available_;
    size_t num_streams_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

struct BatchItem {
    void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    Status status = Status::SUCCESS;
};

} // namespace cuda_zstd

#endif // STREAM_POOL_H
