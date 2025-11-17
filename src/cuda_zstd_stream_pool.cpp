#include "cuda_zstd_stream_pool.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <iostream>
#include <ostream>
#include <optional>

namespace cuda_zstd {

StreamPool::StreamPool(size_t pool_size) {
    fprintf(stderr, "StreamPool ctor() pool_size=%zu\n", pool_size);
    if (pool_size == 0) pool_size = 1;

    resources_.resize(pool_size);
        const char* dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
        bool debug = (dbg && std::atoi(dbg) != 0);
    for (size_t i = 0; i < pool_size; ++i) {
        PerStreamResources& r = resources_[i];
        r.checksum_buf = nullptr;
            if (debug) std::cerr << "StreamPool: creating stream " << i << " of " << pool_size << std::endl;
            cudaError_t err = cudaStreamCreate(&r.stream);
        fprintf(stderr, "StreamPool: created stream index=%zu, err=%d\n", i, (int)err);
        if (err != cudaSuccess) {
            // Cleanup any resources created so far
                if (debug) std::cerr << "StreamPool: cudaStreamCreate failed at " << i << " -> " << err << std::endl;
            for (size_t j = 0; j < i; ++j) {
                cudaStreamDestroy(resources_[j].stream);
            }
            throw std::runtime_error("Failed to create CUDA stream pool");
        }
        // Pre-allocate checksum buffer for each stream (u64)
            if (debug) std::cerr << "StreamPool: creating checksum_buf " << i << std::endl;
        err = cudaMalloc(&r.checksum_buf, sizeof(u64));
        fprintf(stderr, "StreamPool: allocated checksum_buf for index=%zu, err=%d\n", i, (int)err);
        if (err != cudaSuccess) {
            for (size_t j = 0; j <= i; ++j) {
                if (resources_[j].checksum_buf) cudaFree(resources_[j].checksum_buf);
                cudaStreamDestroy(resources_[j].stream);
            }
            throw std::runtime_error("Failed to allocate checksum buffer for stream pool");
        }

        free_idx_.push((int)i);
            if (debug) std::cerr << "StreamPool: pushed free index " << i << "\n";
    }
}

StreamPool::~StreamPool() {
    fprintf(stderr, "StreamPool dtor\n");
    std::unique_lock<std::mutex> lock(mtx_);
    while (!free_idx_.empty()) free_idx_.pop();
    // Cleanup resources
    for (auto &r : resources_) {
        if (r.checksum_buf) {
            cudaFree(r.checksum_buf);
            r.checksum_buf = nullptr;
        }
        if (r.stream) {
            cudaStreamDestroy(r.stream);
            r.stream = 0;
        }
    }
}

int StreamPool::acquire_index() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (free_idx_.empty()) {
           const char* dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
           bool debug = (dbg && std::atoi(dbg) != 0);
           if (debug) std::cerr << "StreamPool: waiting for free index..." << std::endl;
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
        bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){ return !free_idx_.empty(); });
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
            const char* dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
            bool debug = (dbg && std::atoi(dbg) != 0);
            if (debug) std::cerr << "StreamPool: released index " << idx << std::endl;
    }
    cv_.notify_one();
}


StreamPool::Guard StreamPool::acquire() {
    int idx = acquire_index();
    return Guard(this, idx);
}

std::optional<StreamPool::Guard> StreamPool::acquire_for(size_t timeout_ms) {
    int idx = acquire_index_for(timeout_ms);
    if (idx < 0) return std::nullopt;
    return Guard(this, idx);
}

StreamPool::Guard::Guard(StreamPool* pool, int idx)
    : pool_(pool), resources_(nullptr), idx_(-1) {
    idx_ = idx;
    if (idx_ >= 0) resources_ = &pool_->resources_[idx_];
}

StreamPool::Guard::~Guard() {
    if (pool_ && idx_ >= 0) {
        pool_->release_index(idx_);
        idx_ = -1;
    }
}

} // namespace cuda_zstd

// Global singleton
namespace {
    std::unique_ptr<cuda_zstd::StreamPool> g_stream_pool;
    std::once_flag g_stream_pool_once_flag;
}

cuda_zstd::StreamPool* cuda_zstd::get_global_stream_pool(size_t default_size) {
    fprintf(stderr, "get_global_stream_pool called. default_size=%zu\n", default_size);
    std::call_once(g_stream_pool_once_flag, [&](){
        fprintf(stderr, "get_global_stream_pool: initializing pool...\n");
        if (!g_stream_pool) {
            const char* env_pool = getenv("CUDA_ZSTD_STREAM_POOL_SIZE");
            size_t size = default_size;
            if (env_pool) {
                try { size = std::max((size_t)1, (size_t)std::stoi(env_pool)); } catch (...) { }
            }
            try {
                g_stream_pool = std::make_unique<cuda_zstd::StreamPool>(size);
            } catch (...) {
                // If allocation fails, keep g_stream_pool null.
                g_stream_pool.reset();
            }
            fprintf(stderr, "get_global_stream_pool: init complete\n");
        }
    });
    return g_stream_pool.get();
}

size_t cuda_zstd::get_global_stream_pool_size() {
    if (!g_stream_pool) return 0;
    return g_stream_pool->size();
}
