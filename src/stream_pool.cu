// ==============================================================================
// stream_pool.cu - Stream pool implementation
// ==============================================================================

#include "stream_pool.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace cuda_zstd {

StreamPool::StreamPool(size_t num_streams) 
    : num_streams_(num_streams),
      streams_(num_streams),
      available_(num_streams, true)
{
//     std::cout << "[StreamPool] Creating pool with " << num_streams << " streams" << std::endl;

    for (size_t i = 0; i < num_streams_; ++i) {
        cudaError_t err = cudaStreamCreate(&streams_[i]);
        if (err != cudaSuccess) {
//             std::cerr << "[StreamPool] Failed to create stream " << i 
//                       << ": " << cudaGetErrorString(err) << std::endl;
            for (size_t j = 0; j < i; ++j) {
                cudaStreamDestroy(streams_[j]);
            }
            throw std::runtime_error("Failed to create CUDA streams");
        }
    }

//     std::cout << "[StreamPool] Successfully created " << num_streams << " streams" << std::endl;
}

StreamPool::~StreamPool() {
//     std::cout << "[StreamPool] Destroying pool" << std::endl;
    synchronize_all();

    for (size_t i = 0; i < num_streams_; ++i) {
        cudaStreamDestroy(streams_[i]);
    }
}

cudaStream_t StreamPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);

    cv_.wait(lock, [this] {
        return std::any_of(available_.begin(), available_.end(), [](bool avail) { return avail; });
    });

    for (size_t i = 0; i < num_streams_; ++i) {
        if (available_[i]) {
            available_[i] = false;
            return streams_[i];
        }
    }

    return nullptr;
}

cudaStream_t StreamPool::acquire_with_timeout(size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    auto timeout = std::chrono::milliseconds(timeout_ms);

    bool success = cv_.wait_for(lock, timeout, [this] {
        return std::any_of(available_.begin(), available_.end(), [](bool avail) { return avail; });
    });

    if (!success) {
//         std::cerr << "[StreamPool] Timeout acquiring stream after " << timeout_ms << "ms" << std::endl;
        return nullptr;
    }

    for (size_t i = 0; i < num_streams_; ++i) {
        if (available_[i]) {
            available_[i] = false;
            return streams_[i];
        }
    }

    return nullptr;
}

void StreamPool::release(cudaStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex_);

    for (size_t i = 0; i < num_streams_; ++i) {
        if (streams_[i] == stream) {
            available_[i] = true;
            cv_.notify_one();
            return;
        }
    }

//     std::cerr << "[StreamPool] Warning: Attempted to release unknown stream" << std::endl;
}

StreamPool::Guard StreamPool::acquire_guard() {
    cudaStream_t stream = acquire();
    return Guard(this, stream);
}

std::optional<StreamPool::Guard> StreamPool::acquire_guard_with_timeout(size_t timeout_ms) {
    cudaStream_t stream = acquire_with_timeout(timeout_ms);
    if (stream == nullptr) {
        return std::nullopt;
    }
    return Guard(this, stream);
}

void StreamPool::synchronize_all() {
    for (size_t i = 0; i < num_streams_; ++i) {
        cudaStreamSynchronize(streams_[i]);
    }
}

size_t StreamPool::available() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return std::count(available_.begin(), available_.end(), true);
}

StreamPool::Guard::Guard(StreamPool* pool, cudaStream_t stream)
    : pool_(pool), stream_(stream)
{}

StreamPool::Guard::~Guard() {
    if (pool_ != nullptr && stream_ != nullptr) {
        pool_->release(stream_);
    }
}

StreamPool::Guard::Guard(Guard&& other) noexcept
    : pool_(other.pool_), stream_(other.stream_)
{
    other.pool_ = nullptr;
    other.stream_ = nullptr;
}

StreamPool::Guard& StreamPool::Guard::operator=(Guard&& other) noexcept {
    if (this != &other) {
        if (pool_ != nullptr && stream_ != nullptr) {
            pool_->release(stream_);
        }
        pool_ = other.pool_;
        stream_ = other.stream_;
        other.pool_ = nullptr;
        other.stream_ = nullptr;
    }
    return *this;
}

} // namespace cuda_zstd
