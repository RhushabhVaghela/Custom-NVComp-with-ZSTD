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
    std::cerr << "StreamPool ctor() pool_size=" << pool_size << std::endl;
    if (pool_size == 0) pool_size = 1;

    resources_.resize(pool_size);
    const char* dbg = getenv("CUDA_ZSTD_DEBUG_POOL");
    bool debug = (dbg && std::atoi(dbg) != 0);
    
    for (size_t i = 0; i < pool_size; ++i) {
        PerStreamResources& r = resources_[i];
        
        if (debug) std::cerr << "StreamPool: creating stream " << i << " of " << pool_size << std::endl;
        cudaError_t err = cudaStreamCreate(&r.stream);
        std::cerr << "StreamPool: created stream index=" << i << ", err=" << (int)err << std::endl;
        
        if (err != cudaSuccess) {
            // Cleanup any resources created so far
            std::cerr << "StreamPool: cudaStreamCreate failed at " << i << " -> " << err << std::endl;
            for (size_t j = 0; j < i; ++j) {
                cudaStreamDestroy(resources_[j].stream);
            }
            throw std::runtime_error("Failed to create CUDA stream pool");
        }

        free_idx_.push((int)i);
        if (debug) std::cerr << "StreamPool: pushed free index " << i << std::endl;
    }
}

StreamPool::~StreamPool() {
    std::cerr << "StreamPool dtor" << std::endl;
    std::unique_lock<std::mutex> lock(mtx_);
    while (!free_idx_.empty()) free_idx_.pop();
    
    // Check if CUDA is still initialized before cleanup
    // cudaGetDeviceCount will return cudaErrorInitializationError if CUDA is deinitialized
    int device_count = 0;
    cudaError_t init_check = cudaGetDeviceCount(&device_count);
    bool cuda_initialized = (init_check == cudaSuccess || init_check == cudaErrorNoDevice);
    
    if (!cuda_initialized) {
        // CUDA context is already torn down, skip cleanup to avoid errors
        std::cerr << "StreamPool dtor: CUDA already deinitialized, skipping cleanup" << std::endl;
        return;
    }
    
    // Cleanup resources
    for (auto &r : resources_) {
        if (r.stream) {
            // Ensure stream has completed any work before destroying.
            cudaError_t sync_err = cudaStreamSynchronize(r.stream);
            if (sync_err != cudaSuccess && sync_err != cudaErrorCudartUnloading) {
                std::cerr << "StreamPool::~StreamPool: cudaStreamSynchronize failed -> " << sync_err << std::endl;
            }
            cudaError_t destroy_err = cudaStreamDestroy(r.stream);
            if (destroy_err != cudaSuccess && destroy_err != cudaErrorCudartUnloading) {
                std::cerr << "StreamPool::~StreamPool: cudaStreamDestroy failed -> " << destroy_err << std::endl;
            }
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
    std::cerr << "get_global_stream_pool called. default_size=" << default_size << std::endl;
    std::call_once(g_stream_pool_once_flag, [&](){
        std::cerr << "get_global_stream_pool: initializing pool..." << std::endl;
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
            std::cerr << "get_global_stream_pool: init complete" << std::endl;
        }
    });
    return g_stream_pool.get();
}

size_t cuda_zstd::get_global_stream_pool_size() {
    if (!g_stream_pool) return 0;
    return g_stream_pool->size();
}
