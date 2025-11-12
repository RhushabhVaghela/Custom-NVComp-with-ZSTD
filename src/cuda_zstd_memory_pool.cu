// ============================================================================
// cuda_zstd_memory_pool.cu - GPU Memory Pool Manager Implementation
// ============================================================================

#include "cuda_zstd_memory_pool.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace cuda_zstd {
namespace memory {

// ============================================================================
// Static Pool Sizes Definition
// ============================================================================

constexpr size_t MemoryPoolManager::POOL_SIZES[NUM_POOL_SIZES];

// ============================================================================
// Constructor and Destructor
// ============================================================================

MemoryPoolManager::MemoryPoolManager(bool enable_defrag)
    : defrag_enabled_(enable_defrag) {
    // Pre-allocate vector capacity for each pool to avoid reallocation
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        pools_[i].reserve(16);  // Start with capacity for 16 entries per pool
    }
}

MemoryPoolManager::~MemoryPoolManager() {
    clear();
}

// ============================================================================
// Helper Functions
// ============================================================================

int MemoryPoolManager::get_pool_index(size_t size) const {
    // Find the appropriate pool index for the requested size
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        if (size <= POOL_SIZES[i]) {
            return i;
        }
    }
    return -1;  // Size exceeds largest pool
}

size_t MemoryPoolManager::round_up_to_pool_size(size_t size) const {
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        if (size <= POOL_SIZES[i]) {
            return POOL_SIZES[i];
        }
    }
    // Round up to nearest MB for very large allocations
    return ((size + SIZE_1MB - 1) / SIZE_1MB) * SIZE_1MB;
}

void MemoryPoolManager::update_peak_usage(size_t current_usage) {
    size_t current_peak = peak_memory_usage_.load(std::memory_order_relaxed);
    while (current_usage > current_peak) {
        if (peak_memory_usage_.compare_exchange_weak(current_peak, current_usage,
                                                      std::memory_order_relaxed)) {
            break;
        }
    }
}

void* MemoryPoolManager::allocate_from_cuda(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    
    size_t new_usage = current_memory_usage_.fetch_add(size, std::memory_order_relaxed) + size;
    update_peak_usage(new_usage);
    
    return ptr;
}

Status MemoryPoolManager::grow_pool(int pool_idx, size_t min_entries) {
    if (pool_idx < 0 || pool_idx >= NUM_POOL_SIZES) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    size_t entry_size = POOL_SIZES[pool_idx];
    size_t current_size = pools_[pool_idx].size();
    size_t new_entries = std::max(min_entries, 
                                   static_cast<size_t>(current_size * growth_factor_));
    new_entries = std::max(new_entries, size_t(4));  // At least 4 new entries
    
    // Check if we would exceed max pool size
    size_t new_memory = new_entries * entry_size;
    size_t current_total = current_memory_usage_.load(std::memory_order_relaxed);
    if (current_total + new_memory > max_pool_size_) {
        // Try to allocate just what we need
        new_entries = min_entries;
        new_memory = new_entries * entry_size;
        if (current_total + new_memory > max_pool_size_) {
            return Status::ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Allocate new entries
    for (size_t i = 0; i < new_entries; ++i) {
        void* ptr = allocate_from_cuda(entry_size);
        if (!ptr) {
            return Status::ERROR_OUT_OF_MEMORY;
        }
        pools_[pool_idx].emplace_back(ptr, entry_size);
    }
    
    pool_grows_.fetch_add(1, std::memory_order_relaxed);
    return Status::SUCCESS;
}

PoolEntry* MemoryPoolManager::find_free_entry(int pool_idx, cudaStream_t stream) {
    auto& pool = pools_[pool_idx];
    
    for (auto& entry : pool) {
        if (!entry.in_use) {
            // Check if the entry is ready (stream synchronization)
            if (entry.ready_event != nullptr) {
                cudaError_t err = cudaEventQuery(entry.ready_event);
                if (err == cudaErrorNotReady) {
                    continue;  // Entry not ready yet
                } else if (err == cudaSuccess) {
                    cudaEventDestroy(entry.ready_event);
                    entry.ready_event = nullptr;
                } else {
                    // Error querying event, destroy it anyway
                    cudaEventDestroy(entry.ready_event);
                    entry.ready_event = nullptr;
                }
            }
            
            entry.in_use = true;
            entry.stream = stream;
            return &entry;
        }
    }
    
    return nullptr;
}

// ============================================================================
// Allocation Interface
// ============================================================================

void* MemoryPoolManager::allocate(size_t size, cudaStream_t stream) {
    if (size == 0) {
        return nullptr;
    }
    
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    
    int pool_idx = get_pool_index(size);
    
    // For very large allocations, bypass the pool
    if (pool_idx < 0) {
        cache_misses_.fetch_add(1, std::memory_order_relaxed);
        return allocate_from_cuda(size);
    }
    
    std::lock_guard<std::mutex> lock(pool_mutexes_[pool_idx]);
    
    // Try to find a free entry in the pool
    PoolEntry* entry = find_free_entry(pool_idx, stream);
    
    if (entry) {
        cache_hits_.fetch_add(1, std::memory_order_relaxed);
        return entry->ptr;
    }
    
    // No free entry found, grow the pool
    cache_misses_.fetch_add(1, std::memory_order_relaxed);
    Status status = grow_pool(pool_idx, 1);
    if (status != Status::SUCCESS) {
        return nullptr;
    }
    
    // Try again after growing
    entry = find_free_entry(pool_idx, stream);
    if (entry) {
        return entry->ptr;
    }
    
    return nullptr;
}

void* MemoryPoolManager::allocate_async(size_t size, cudaStream_t stream) {
    void* ptr = allocate(size, stream);
    
    // For async allocations, we don't wait for the stream
    // The caller is responsible for stream synchronization
    return ptr;
}

Status MemoryPoolManager::deallocate(void* ptr) {
    if (!ptr) {
        return Status::SUCCESS;
    }
    
    total_deallocations_.fetch_add(1, std::memory_order_relaxed);
    
    // Search all pools to find this pointer
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        
        for (auto& entry : pools_[i]) {
            if (entry.ptr == ptr) {
                if (!entry.in_use) {
                    return Status::ERROR_INVALID_PARAMETER;  // Double free
                }
                
                // Create an event to track when the stream is done with this memory
                if (entry.stream != nullptr && entry.stream != 0) {
                    cudaEvent_t event;
                    cudaError_t err = cudaEventCreate(&event);
                    if (err == cudaSuccess) {
                        cudaEventRecord(event, entry.stream);
                        entry.ready_event = event;
                    }
                }
                
                entry.in_use = false;
                entry.stream = nullptr;
                return Status::SUCCESS;
            }
        }
    }
    
    // Not in pool, must be a direct CUDA allocation
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        return Status::ERROR_CUDA_ERROR;
    }
    
    return Status::SUCCESS;
}

// ============================================================================
// Pool Management
// ============================================================================

Status MemoryPoolManager::prewarm(size_t total_memory) {
    // Distribute memory across pool sizes proportionally
    // Strategy: Smaller sizes get more entries, larger sizes get fewer
    
    size_t remaining = total_memory;
    
    // Weights for each pool size (smaller = more weight)
    const float weights[NUM_POOL_SIZES] = {4.0f, 3.0f, 2.5f, 2.0f, 1.5f, 1.0f};
    float total_weight = 0.0f;
    for (float w : weights) total_weight += w;
    
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        size_t pool_memory = static_cast<size_t>(total_memory * weights[i] / total_weight);
        size_t num_entries = pool_memory / POOL_SIZES[i];
        
        if (num_entries > 0) {
            std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
            Status status = grow_pool(i, num_entries);
            if (status != Status::SUCCESS) {
                return status;
            }
        }
    }
    
    return Status::SUCCESS;
}

Status MemoryPoolManager::prewarm_by_sizes(const std::vector<size_t>& allocation_sizes) {
    // Count how many allocations of each size we need
    std::vector<size_t> counts(NUM_POOL_SIZES, 0);
    
    for (size_t size : allocation_sizes) {
        int idx = get_pool_index(size);
        if (idx >= 0) {
            counts[idx]++;
        }
    }
    
    // Pre-allocate entries for each pool
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        if (counts[i] > 0) {
            std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
            Status status = grow_pool(i, counts[i]);
            if (status != Status::SUCCESS) {
                return status;
            }
        }
    }
    
    return Status::SUCCESS;
}

Status MemoryPoolManager::defragment() {
    if (!defrag_enabled_) {
        return Status::SUCCESS;
    }
    
    // Defragmentation strategy:
    // 1. Identify pools with many free entries
    // 2. Free entries at the end of the pool vector
    // 3. Compact the pool by removing freed entries
    
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        auto& pool = pools_[i];
        
        // Count free entries
        size_t free_count = 0;
        for (const auto& entry : pool) {
            if (!entry.in_use && entry.ready_event == nullptr) {
                free_count++;
            }
        }
        
        // If more than 50% are free, compact
        if (free_count > pool.size() / 2 && pool.size() > 8) {
            std::vector<PoolEntry> compacted;
            compacted.reserve(pool.size() - free_count / 2);
            
            size_t freed_memory = 0;
            
            for (auto& entry : pool) {
                if (entry.in_use || entry.ready_event != nullptr) {
                    compacted.push_back(std::move(entry));
                } else if (compacted.size() < pool.size() - free_count / 2) {
                    compacted.push_back(std::move(entry));
                } else {
                    // Free this entry
                    if (entry.ptr) {
                        cudaFree(entry.ptr);
                        freed_memory += entry.size;
                    }
                }
            }
            
            pool = std::move(compacted);
            current_memory_usage_.fetch_sub(freed_memory, std::memory_order_relaxed);
        }
    }
    
    return Status::SUCCESS;
}

void MemoryPoolManager::clear() {
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        
        for (auto& entry : pools_[i]) {
            if (entry.ready_event != nullptr) {
                cudaEventSynchronize(entry.ready_event);
                cudaEventDestroy(entry.ready_event);
            }
            if (entry.ptr) {
                cudaFree(entry.ptr);
            }
        }
        
        pools_[i].clear();
    }
    
    current_memory_usage_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Statistics
// ============================================================================

PoolStats MemoryPoolManager::get_statistics() const {
    PoolStats stats;
    stats.total_allocations = total_allocations_.load(std::memory_order_relaxed);
    stats.total_deallocations = total_deallocations_.load(std::memory_order_relaxed);
    stats.cache_hits = cache_hits_.load(std::memory_order_relaxed);
    stats.cache_misses = cache_misses_.load(std::memory_order_relaxed);
    stats.pool_grows = pool_grows_.load(std::memory_order_relaxed);
    stats.peak_memory_usage = peak_memory_usage_.load(std::memory_order_relaxed);
    stats.current_memory_usage = current_memory_usage_.load(std::memory_order_relaxed);
    
    // Calculate total pool capacity
    size_t total_capacity = 0;
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        total_capacity += pools_[i].size() * POOL_SIZES[i];
    }
    stats.total_pool_capacity = total_capacity;
    
    return stats;
}

void MemoryPoolManager::reset_statistics() {
    total_allocations_.store(0, std::memory_order_relaxed);
    total_deallocations_.store(0, std::memory_order_relaxed);
    cache_hits_.store(0, std::memory_order_relaxed);
    cache_misses_.store(0, std::memory_order_relaxed);
    pool_grows_.store(0, std::memory_order_relaxed);
    // Note: We don't reset peak_memory_usage as it's cumulative
}

void MemoryPoolManager::print_statistics() const {
    PoolStats stats = get_statistics();
    
    std::cout << "\n========================================\n";
    std::cout << "Memory Pool Statistics\n";
    std::cout << "========================================\n";
    std::cout << "Total Allocations:    " << stats.total_allocations << "\n";
    std::cout << "Total Deallocations:  " << stats.total_deallocations << "\n";
    std::cout << "Cache Hits:           " << stats.cache_hits << "\n";
    std::cout << "Cache Misses:         " << stats.cache_misses << "\n";
    std::cout << "Hit Rate:             " << std::fixed << std::setprecision(2) 
              << (stats.get_hit_rate() * 100.0) << "%\n";
    std::cout << "Pool Grows:           " << stats.pool_grows << "\n";
    std::cout << "Current Usage:        " << (stats.current_memory_usage / 1024.0 / 1024.0) 
              << " MB\n";
    std::cout << "Peak Usage:           " << (stats.peak_memory_usage / 1024.0 / 1024.0) 
              << " MB\n";
    std::cout << "Total Pool Capacity:  " << (stats.total_pool_capacity / 1024.0 / 1024.0) 
              << " MB\n";
    
    std::cout << "\nPool Details:\n";
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        size_t total = pools_[i].size();
        size_t in_use = 0;
        for (const auto& entry : pools_[i]) {
            if (entry.in_use) in_use++;
        }
        
        std::cout << "  " << std::setw(8) << (POOL_SIZES[i] / 1024) << " KB: "
                  << std::setw(4) << in_use << " / " << std::setw(4) << total 
                  << " in use\n";
    }
    std::cout << "========================================\n\n";
}

// ============================================================================
// Configuration
// ============================================================================

void MemoryPoolManager::set_growth_factor(float factor) {
    if (factor >= 1.0f) {
        growth_factor_ = factor;
    }
}

void MemoryPoolManager::enable_defragmentation(bool enable) {
    defrag_enabled_ = enable;
}

void MemoryPoolManager::set_max_pool_size(size_t max_size) {
    max_pool_size_ = max_size;
}

// ============================================================================
// Global Pool Instance
// ============================================================================

static MemoryPoolManager* g_pool_instance = nullptr;
static std::mutex g_pool_mutex;

MemoryPoolManager& get_global_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (!g_pool_instance) {
        g_pool_instance = new MemoryPoolManager(true);
    }
    return *g_pool_instance;
}

void destroy_global_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_pool_instance) {
        delete g_pool_instance;
        g_pool_instance = nullptr;
    }
}

} // namespace memory
} // namespace cuda_zstd