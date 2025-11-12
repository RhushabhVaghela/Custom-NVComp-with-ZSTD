// ============================================================================
// cuda_zstd_memory_pool.h - GPU Memory Pool Manager
// ============================================================================

#ifndef CUDA_ZSTD_MEMORY_POOL_H
#define CUDA_ZSTD_MEMORY_POOL_H

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace cuda_zstd {
namespace memory {

// ============================================================================
// Pool Statistics
// ============================================================================

struct PoolStats {
    uint64_t total_allocations = 0;
    uint64_t total_deallocations = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t pool_grows = 0;
    size_t peak_memory_usage = 0;
    size_t current_memory_usage = 0;
    size_t total_pool_capacity = 0;
    
    double get_hit_rate() const {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
    }
};

// ============================================================================
// Pool Entry
// ============================================================================

struct PoolEntry {
    void* ptr = nullptr;
    size_t size = 0;
    bool in_use = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    
    PoolEntry() = default;
    PoolEntry(void* p, size_t s) : ptr(p), size(s), in_use(false), stream(nullptr), ready_event(nullptr) {}
};

// ============================================================================
// Memory Pool Manager
// ============================================================================

class MemoryPoolManager {
public:
    // Pool size configuration
    static constexpr size_t SIZE_4KB = 4 * 1024;
    static constexpr size_t SIZE_16KB = 16 * 1024;
    static constexpr size_t SIZE_64KB = 64 * 1024;
    static constexpr size_t SIZE_256KB = 256 * 1024;
    static constexpr size_t SIZE_1MB = 1024 * 1024;
    static constexpr size_t SIZE_4MB = 4 * 1024 * 1024;
    static constexpr int NUM_POOL_SIZES = 6;
    
    // Constructor with configurable pool sizes
    explicit MemoryPoolManager(bool enable_defrag = true);
    ~MemoryPoolManager();
    
    // Disable copy and move
    MemoryPoolManager(const MemoryPoolManager&) = delete;
    MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;
    
    // Allocation interface
    void* allocate(size_t size, cudaStream_t stream = 0);
    Status deallocate(void* ptr);
    
    // Async allocation with stream synchronization
    void* allocate_async(size_t size, cudaStream_t stream);
    
    // Pool management
    Status prewarm(size_t total_memory);
    Status prewarm_by_sizes(const std::vector<size_t>& allocation_sizes);
    Status defragment();
    void clear();
    
    // Statistics
    PoolStats get_statistics() const;
    void reset_statistics();
    void print_statistics() const;
    
    // Configuration
    void set_growth_factor(float factor);
    void enable_defragmentation(bool enable);
    void set_max_pool_size(size_t max_size);
    
private:
    // Pool storage - one pool per size class
    std::vector<PoolEntry> pools_[NUM_POOL_SIZES];
    mutable std::mutex pool_mutexes_[NUM_POOL_SIZES];
    
    // Pool size thresholds
    static constexpr size_t POOL_SIZES[NUM_POOL_SIZES] = {
        SIZE_4KB, SIZE_16KB, SIZE_64KB, SIZE_256KB, SIZE_1MB, SIZE_4MB
    };
    
    // Configuration
    float growth_factor_ = 1.5f;
    bool defrag_enabled_ = true;
    size_t max_pool_size_ = 1024ULL * 1024 * 1024 * 2;  // 2GB default max
    
    // Statistics (atomic for thread safety)
    mutable std::atomic<uint64_t> total_allocations_{0};
    mutable std::atomic<uint64_t> total_deallocations_{0};
    mutable std::atomic<uint64_t> cache_hits_{0};
    mutable std::atomic<uint64_t> cache_misses_{0};
    mutable std::atomic<uint64_t> pool_grows_{0};
    mutable std::atomic<size_t> peak_memory_usage_{0};
    mutable std::atomic<size_t> current_memory_usage_{0};
    
    // Helper functions
    int get_pool_index(size_t size) const;
    size_t round_up_to_pool_size(size_t size) const;
    Status grow_pool(int pool_idx, size_t min_entries = 1);
    PoolEntry* find_free_entry(int pool_idx, cudaStream_t stream);
    void* allocate_from_cuda(size_t size);
    void update_peak_usage(size_t current_usage);
};

// ============================================================================
// Global Pool Instance (Optional Singleton)
// ============================================================================

MemoryPoolManager& get_global_pool();
void destroy_global_pool();

} // namespace memory
} // namespace cuda_zstd

#endif // CUDA_ZSTD_MEMORY_POOL_H