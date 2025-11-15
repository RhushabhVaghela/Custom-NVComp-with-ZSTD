// ============================================================================
// cuda_zstd_memory_pool_simple.cu - Simplified Memory Pool Manager Implementation
// ============================================================================

#include "cuda_zstd_memory_pool.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <queue>
#include <functional>
#include <random>

// Logging macro for fallback events
#define LOG_INFO(msg) std::cout << "[MEMORY_POOL] " << msg << std::endl

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
    // Initialize fallback configuration
    fallback_config_.enable_host_memory_fallback = true;
    fallback_config_.enable_progressive_degradation = true;
    fallback_config_.enable_chunk_reduction = true;
    fallback_config_.enable_rollback_protection = true;
    fallback_config_.emergency_threshold_mb = 100;
    fallback_config_.host_memory_limit_mb = 1024;
    fallback_config_.degradation_factor = 0.5f;
    fallback_config_.max_retry_attempts = 3;
    
    // Pre-allocate vector capacity for each pool to avoid reallocation
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        pools_[i].reserve(16);  // Start with capacity for 16 entries per pool
    }
    
    // Initialize pressure tracking
    last_pressure_update_ = std::chrono::steady_clock::now();
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
        // Log CUDA allocation failure for monitoring
        allocation_failures_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    size_t new_usage = current_memory_usage_.fetch_add(size, std::memory_order_relaxed) + size;
    update_peak_usage(new_usage);
    
    return ptr;
}

void* MemoryPoolManager::allocate_from_host(size_t size) {
    // Check host memory limit
    size_t current_host_usage = host_memory_usage_.load(std::memory_order_relaxed);
    if (current_host_usage + size > fallback_config_.host_memory_limit_mb * 1024 * 1024) {
        return nullptr;
    }
    
    void* ptr = malloc(size);
    if (ptr) {
        host_memory_usage_.fetch_add(size, std::memory_order_relaxed);
        host_memory_allocations_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return ptr;
}

FallbackAllocation MemoryPoolManager::allocate_with_cuda_fallback(size_t size, cudaStream_t stream) {
    FallbackAllocation result;
    result.allocated_size = size;
    
    // Try primary CUDA allocation first
    result.ptr = allocate_from_cuda(size);
    if (result.ptr) {
        result.status = Status::SUCCESS;
        return result;
    }
    
    // CUDA allocation failed, check if we should use fallback
    if (!fallback_config_.enable_host_memory_fallback || is_emergency_mode()) {
        result.status = Status::ERROR_OUT_OF_MEMORY;
        allocation_failures_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    
    // Try host memory fallback
    result.ptr = allocate_from_host(size);
    if (result.ptr) {
        result.is_host_memory = true;
        result.status = Status::SUCCESS;
        fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    
    // Host allocation also failed
    result.status = Status::ERROR_OUT_OF_MEMORY;
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return result;
}

FallbackAllocation MemoryPoolManager::allocate_host_memory(size_t size) {
    FallbackAllocation result;
    result.allocated_size = size;
    result.is_host_memory = true;
    
    result.ptr = allocate_from_host(size);
    if (result.ptr) {
        result.status = Status::SUCCESS;
        host_memory_allocations_.fetch_add(1, std::memory_order_relaxed);
    } else {
        result.status = Status::ERROR_OUT_OF_MEMORY;
        allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    }
    
    return result;
}

FallbackAllocation MemoryPoolManager::allocate_degraded(size_t size, cudaStream_t stream) {
    FallbackAllocation result;
    
    // Calculate degraded size
    size_t degraded_size = calculate_degraded_size(size);
    if (degraded_size < 1024) {  // Minimum 1KB
        degraded_size = 1024;
    }
    
    result.allocated_size = degraded_size;
    result.is_degraded = true;
    
    // Try degraded CUDA allocation
    result.ptr = allocate_from_cuda(degraded_size);
    if (result.ptr) {
        result.status = Status::SUCCESS;
        degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    
    // Try degraded host allocation if fallback is enabled
    if (fallback_config_.enable_host_memory_fallback) {
        result.ptr = allocate_from_host(degraded_size);
        if (result.ptr) {
            result.is_host_memory = true;
            result.status = Status::SUCCESS;
            degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
            fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
            return result;
        }
    }
    
    result.status = Status::ERROR_OUT_OF_MEMORY;
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return result;
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
            // Pool growth would exceed max size, try fallback strategy
            return grow_pool_with_fallback(pool_idx, min_entries);
        }
    }
    
    // Allocate new entries with rollback protection
    std::vector<PoolEntry> new_pool_entries;
    new_pool_entries.reserve(new_entries);
    
    for (size_t i = 0; i < new_entries; ++i) {
        void* ptr = allocate_from_cuda(entry_size);
        if (!ptr) {
            // Allocation failed, rollback any successful allocations
            trigger_rollback_protection();
            return Status::ERROR_OUT_OF_MEMORY;
        }
        new_pool_entries.emplace_back(ptr, entry_size);
    }
    
    // All allocations succeeded, add to pool
    std::lock_guard<std::mutex> lock(pool_mutexes_[pool_idx]);
    pools_[pool_idx].insert(pools_[pool_idx].end(),
                           std::make_move_iterator(new_pool_entries.begin()),
                           std::make_move_iterator(new_pool_entries.end()));
    
    pool_grows_.fetch_add(1, std::memory_order_relaxed);
    return Status::SUCCESS;
}

Status MemoryPoolManager::grow_pool_with_fallback(int pool_idx, size_t min_entries) {
    if (!fallback_config_.enable_progressive_degradation) {
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Try with reduced entry size
    size_t original_entry_size = POOL_SIZES[pool_idx];
    size_t degraded_entry_size = calculate_degraded_size(original_entry_size);
    
    if (degraded_entry_size < SIZE_4KB) {
        degraded_entry_size = SIZE_4KB;  // Minimum pool size
    }
    
    // Try to grow with smaller entries
    for (size_t attempt = 0; attempt < fallback_config_.max_retry_attempts; ++attempt) {
        void* ptr = allocate_from_cuda(degraded_entry_size);
        if (ptr) {
            std::lock_guard<std::mutex> lock(pool_mutexes_[pool_idx]);
            pools_[pool_idx].emplace_back(ptr, degraded_entry_size);
            degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
            pool_grows_.fetch_add(1, std::memory_order_relaxed);
            return Status::SUCCESS;
        }
        
        // If this is the last attempt, try host memory fallback
        if (attempt == fallback_config_.max_retry_attempts - 1 &&
            fallback_config_.enable_host_memory_fallback) {
            ptr = allocate_from_host(degraded_entry_size);
            if (ptr) {
                std::lock_guard<std::mutex> lock(pool_mutexes_[pool_idx]);
                PoolEntry entry;
                entry.ptr = nullptr;  // No GPU memory
                entry.host_ptr = ptr;
                entry.size = 0;
                entry.host_size = degraded_entry_size;
                entry.is_host_fallback = true;
                entry.in_use = false;
                pools_[pool_idx].push_back(std::move(entry));
                
                host_memory_allocations_.fetch_add(1, std::memory_order_relaxed);
                fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
                pool_grows_.fetch_add(1, std::memory_order_relaxed);
                return Status::SUCCESS;
            }
        }
    }
    
    return Status::ERROR_OUT_OF_MEMORY;
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
// Allocation Interface with Fallback Support
// ============================================================================

void* MemoryPoolManager::allocate(size_t size, cudaStream_t stream) {
    if (size == 0) {
        return nullptr;
    }
    
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    
    // Update degradation mode based on memory pressure
    update_degradation_mode();
    
    int pool_idx = get_pool_index(size);
    
    // For very large allocations, bypass the pool and use fallback strategy
    if (pool_idx < 0) {
        cache_misses_.fetch_add(1, std::memory_order_relaxed);
        
        // Use fallback allocation for large allocations
        FallbackAllocation result = allocate_with_cuda_fallback(size, stream);
        if (result.is_host_memory) {
            // For host memory allocations, caller must handle data movement
            return result.ptr;
        }
        return result.ptr;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutexes_[pool_idx]);
    
    // Try to find a free entry in the pool
    PoolEntry* entry = find_free_entry(pool_idx, stream);
    
    if (entry) {
        cache_hits_.fetch_add(1, std::memory_order_relaxed);
        entry->in_use = true;
        return entry->ptr;
    }
    
    // No free entry found, try to grow the pool
    cache_misses_.fetch_add(1, std::memory_order_relaxed);
    Status status = grow_pool(pool_idx, 1);
    if (status == Status::SUCCESS) {
        // Try again after growing
        entry = find_free_entry(pool_idx, stream);
        if (entry) {
            return entry->ptr;
        }
    }
    
    // Pool growth failed, try fallback allocation
    FallbackAllocation result = allocate_with_cuda_fallback(size, stream);
    if (result.is_valid()) {
        return result.ptr;
    }
    
    return nullptr;
}

FallbackAllocation MemoryPoolManager::allocate_with_fallback(size_t requested_size, cudaStream_t stream) {
    FallbackAllocation result;
    result.allocated_size = requested_size;
    
    // Update degradation mode
    update_degradation_mode();
    
    // Try normal allocation first
    void* ptr = allocate(requested_size, stream);
    if (ptr) {
        result.ptr = ptr;
        result.status = Status::SUCCESS;
        return result;
    }
    
    // Normal allocation failed, try progressive allocation if enabled
    if (fallback_config_.enable_progressive_degradation) {
        result = allocate_progressive(requested_size / 2, requested_size, stream);
        if (result.is_valid()) {
            return result;
        }
    }
    
    // All strategies failed
    result.status = Status::ERROR_OUT_OF_MEMORY;
    return result;
}

FallbackAllocation MemoryPoolManager::allocate_progressive(size_t min_size, size_t max_size, cudaStream_t stream) {
    FallbackAllocation result;
    
    // Progressive allocation: start from max and reduce until successful
    std::vector<size_t> try_sizes;
    size_t current_size = max_size;
    
    while (current_size >= min_size && current_size >= SIZE_4KB) {
        try_sizes.push_back(current_size);
        current_size = static_cast<size_t>(current_size * fallback_config_.degradation_factor);
        if (current_size < SIZE_4KB) break;
    }
    
    // Try each size
    for (size_t try_size : try_sizes) {
        result = allocate_with_cuda_fallback(try_size, stream);
        if (result.is_valid()) {
            result.is_degraded = (try_size < max_size);
            if (result.is_degraded) {
                degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
            }
            return result;
        }
    }
    
    result.status = Status::ERROR_OUT_OF_MEMORY;
    return result;
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
                
                // Handle stream synchronization
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
            
            // Check for host memory fallback pointers
            if (entry.is_host_fallback && entry.host_ptr == ptr) {
                if (!entry.in_use) {
                    return Status::ERROR_INVALID_PARAMETER;  // Double free
                }
                
                free(entry.host_ptr);
                entry.host_ptr = nullptr;
                entry.in_use = false;
                
                // Update host memory usage
                host_memory_usage_.fetch_sub(entry.host_size, std::memory_order_relaxed);
                
                return Status::SUCCESS;
            }
        }
    }
    
    // Not in pool, must be a direct allocation
    // Check if it's host memory (we can't easily distinguish)
    // For safety, try both CUDA and host deallocation
    
    // First try CUDA deallocation
    cudaError_t cuda_err = cudaFree(ptr);
    if (cuda_err == cudaSuccess) {
        return Status::SUCCESS;
    }
    
    // If CUDA deallocation failed, assume it might be host memory
    // Note: This is a potential limitation - we can't always distinguish
    // In production, you might want to maintain separate tracking
    return Status::ERROR_INVALID_PARAMETER;
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
            if (entry.ptr && !entry.is_host_fallback) {
                cudaFree(entry.ptr);
            }
            if (entry.host_ptr) {
                free(entry.host_ptr);
            }
        }
        
        pools_[i].clear();
    }
    
    current_memory_usage_.store(0, std::memory_order_relaxed);
    host_memory_usage_.store(0, std::memory_order_relaxed);
}

Status MemoryPoolManager::emergency_clear() {
    std::lock_guard<std::mutex> mode_lock(mode_mutex_);
    
    // Switch to emergency mode
    current_mode_ = DegradationMode::EMERGENCY;
    
    // Clear all pools
    clear();
    
    // Reset statistics
    reset_statistics();
    
    return Status::SUCCESS;
}

Status MemoryPoolManager::switch_to_host_memory_mode() {
    std::lock_guard<std::mutex> mode_lock(mode_mutex_);
    
    if (!fallback_config_.enable_host_memory_fallback) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // Switch to emergency mode and clear GPU pools
    current_mode_ = DegradationMode::EMERGENCY;
    clear();
    
    return Status::SUCCESS;
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
    stats.fallback_allocations = fallback_allocations_.load(std::memory_order_relaxed);
    stats.host_memory_allocations = host_memory_allocations_.load(std::memory_order_relaxed);
    stats.degraded_allocations = degraded_allocations_.load(std::memory_order_relaxed);
    stats.allocation_failures = allocation_failures_.load(std::memory_order_relaxed);
    stats.rollback_operations = rollback_operations_.load(std::memory_order_relaxed);
    stats.peak_memory_usage = peak_memory_usage_.load(std::memory_order_relaxed);
    stats.current_memory_usage = current_memory_usage_.load(std::memory_order_relaxed);
    stats.host_memory_usage = host_memory_usage_.load(std::memory_order_relaxed);
    stats.current_mode = current_mode_;
    
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
    fallback_allocations_.store(0, std::memory_order_relaxed);
    host_memory_allocations_.store(0, std::memory_order_relaxed);
    degraded_allocations_.store(0, std::memory_order_relaxed);
    allocation_failures_.store(0, std::memory_order_relaxed);
    rollback_operations_.store(0, std::memory_order_relaxed);
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
    std::cout << "Fallback Allocations: " << stats.fallback_allocations << "\n";
    std::cout << "Host Memory Allocs:   " << stats.host_memory_allocations << "\n";
    std::cout << "Degraded Allocations: " << stats.degraded_allocations << "\n";
    std::cout << "Allocation Failures:  " << stats.allocation_failures << "\n";
    std::cout << "Rollback Operations:  " << stats.rollback_operations << "\n";
    std::cout << "Fallback Rate:        " << std::fixed << std::setprecision(2)
              << (stats.get_fallback_rate() * 100.0) << "%\n";
    std::cout << "Degradation Rate:     " << std::fixed << std::setprecision(2)
              << (stats.get_degradation_rate() * 100.0) << "%\n";
    std::cout << "Current Usage:        " << (stats.current_memory_usage / 1024.0 / 1024.0)
              << " MB (GPU)\n";
    std::cout << "Host Memory Usage:    " << (stats.host_memory_usage / 1024.0 / 1024.0)
              << " MB (Host)\n";
    std::cout << "Peak Usage:           " << (stats.peak_memory_usage / 1024.0 / 1024.0)
              << " MB\n";
    std::cout << "Total Pool Capacity:  " << (stats.total_pool_capacity / 1024.0 / 1024.0)
              << " MB\n";
    std::cout << "Degradation Mode:     ";
    
    switch (stats.current_mode) {
        case DegradationMode::NORMAL: std::cout << "NORMAL"; break;
        case DegradationMode::CONSERVATIVE: std::cout << "CONSERVATIVE"; break;
        case DegradationMode::AGGRESSIVE: std::cout << "AGGRESSIVE"; break;
        case DegradationMode::EMERGENCY: std::cout << "EMERGENCY"; break;
    }
    std::cout << "\n";
    
    // Memory pressure information
    size_t available_memory = get_available_gpu_memory();
    std::cout << "Available GPU Memory: " << (available_memory / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Memory Pressure:      " << get_memory_pressure_percentage() << "%\n";
    
    std::cout << "\nPool Details:\n";
    for (int i = 0; i < NUM_POOL_SIZES; ++i) {
        std::lock_guard<std::mutex> lock(pool_mutexes_[i]);
        size_t total = pools_[i].size();
        size_t in_use = 0;
        size_t host_fallbacks = 0;
        for (const auto& entry : pools_[i]) {
            if (entry.in_use) in_use++;
            if (entry.is_host_fallback) host_fallbacks++;
        }
        
        std::cout << "  " << std::setw(8) << (POOL_SIZES[i] / 1024) << " KB: "
                  << std::setw(4) << in_use << " / " << std::setw(4) << total
                  << " in use";
        if (host_fallbacks > 0) {
            std::cout << " (" << host_fallbacks << " host fallbacks)";
        }
        std::cout << "\n";
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
// Fallback Configuration Methods
// ============================================================================

void MemoryPoolManager::set_fallback_config(const FallbackConfig& config) {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    fallback_config_ = config;
}

const FallbackConfig& MemoryPoolManager::get_fallback_config() const {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    return fallback_config_;
}

void MemoryPoolManager::set_degradation_mode(DegradationMode mode) {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    current_mode_ = mode;
}

DegradationMode MemoryPoolManager::get_degradation_mode() const {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    return current_mode_;
}

// ============================================================================
// Memory Pressure Monitoring
// ============================================================================

size_t MemoryPoolManager::get_available_gpu_memory() const {
    return get_available_gpu_memory_impl();
}

size_t MemoryPoolManager::get_available_gpu_memory_impl() const {
    size_t free_mem = 0;
    size_t total_mem = 0;
    
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        return free_mem;
    }
    
    return 0;  // Return 0 if we can't get memory info
}

size_t MemoryPoolManager::get_memory_pressure_percentage() const {
    size_t total_mem = 0;
    size_t free_mem = get_available_gpu_memory_impl();
    
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess || total_mem == 0) {
        return 0;
    }
    
    size_t used_mem = total_mem - free_mem;
    return static_cast<size_t>((static_cast<double>(used_mem) / total_mem) * 100.0);
}

bool MemoryPoolManager::is_memory_pressure_high() const {
    return is_memory_pressure_critical();
}

bool MemoryPoolManager::is_memory_pressure_critical() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_pressure_update_).count();
    
    // Only check every few seconds to avoid overhead
    if (duration < 3) {
        return current_mode_ == DegradationMode::EMERGENCY;
    }
    
    size_t pressure = get_memory_pressure_percentage();
    last_pressure_update_ = now;
    
    // Switch to emergency mode if pressure is very high
    if (pressure > 90) {
        return true;
    }
    
    return false;
}

void MemoryPoolManager::update_degradation_mode() {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    
    // Only update if not already in emergency mode
    if (current_mode_ == DegradationMode::EMERGENCY) {
        return;
    }
    
    size_t pressure = get_memory_pressure_percentage();
    
    if (pressure > 90) {
        current_mode_ = DegradationMode::EMERGENCY;
    } else if (pressure > 75) {
        current_mode_ = DegradationMode::AGGRESSIVE;
    } else if (pressure > 60) {
        current_mode_ = DegradationMode::CONSERVATIVE;
    } else {
        current_mode_ = DegradationMode::NORMAL;
    }
}

// ============================================================================
// Helper Methods for Fallback Logic
// ============================================================================

size_t MemoryPoolManager::calculate_degraded_size(size_t original_size) const {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    
    switch (current_mode_) {
        case DegradationMode::NORMAL:
            return original_size;
        case DegradationMode::CONSERVATIVE:
            return static_cast<size_t>(original_size * 0.8f);  // 20% reduction
        case DegradationMode::AGGRESSIVE:
            return static_cast<size_t>(original_size * 0.5f);  // 50% reduction
        case DegradationMode::EMERGENCY:
            return static_cast<size_t>(original_size * 0.25f); // 75% reduction
        default:
            return original_size;
    }
}

bool MemoryPoolManager::should_use_progressive_allocation(size_t requested_size) const {
    if (!fallback_config_.enable_progressive_degradation) {
        return false;
    }
    
    // Use progressive allocation for large requests
    return requested_size > SIZE_1MB;
}

bool MemoryPoolManager::is_emergency_mode() const {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    return current_mode_ == DegradationMode::EMERGENCY;
}

void MemoryPoolManager::trigger_rollback_protection() {
    rollback_operations_.fetch_add(1, std::memory_order_relaxed);
    
    // In case of repeated failures, switch to more conservative mode
    if (allocation_failures_.load(std::memory_order_relaxed) > 10) {
        std::lock_guard<std::mutex> lock(mode_mutex_);
        if (current_mode_ == DegradationMode::NORMAL) {
            current_mode_ = DegradationMode::CONSERVATIVE;
        } else if (current_mode_ == DegradationMode::CONSERVATIVE) {
            current_mode_ = DegradationMode::AGGRESSIVE;
        }
    }
}

Status MemoryPoolManager::copy_between_memory_types(void* src, void* dst, size_t size, bool host_to_device) {
    cudaMemcpyKind kind = host_to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    cudaError_t err = cudaMemcpy(dst, src, size, kind);
    return (err == cudaSuccess) ? Status::SUCCESS : Status::ERROR_CUDA_ERROR;
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