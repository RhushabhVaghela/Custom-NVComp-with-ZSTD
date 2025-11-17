// ============================================================================
// cuda_zstd_memory_pool.h - GPU Memory Pool Manager
// ============================================================================

#ifndef CUDA_ZSTD_MEMORY_POOL_COMPLEX_H_
#define CUDA_ZSTD_MEMORY_POOL_COMPLEX_H_

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <chrono>

namespace cuda_zstd {
namespace memory {

// ============================================================================
// Fallback and Degradation Configuration
// ============================================================================

enum class DegradationMode {
    NORMAL = 0,           // Full functionality
    CONSERVATIVE = 1,     // Reduced pool sizes, prefer smaller allocations
    AGGRESSIVE = 2,       // Minimal pool sizes, aggressive fallback
    EMERGENCY = 3         // Host memory fallback only
};

enum class AllocationStrategy {
    AUTO_ADAPTIVE = 0,    // Automatically select best strategy
    PREFER_GPU = 1,       // Always prefer GPU, fallback to host
    PREFER_HOST = 2,      // Always prefer host memory
    BALANCED = 3,         // Balance between GPU and host based on pressure
    PERFORMANCE_FIRST = 4 // Maximize performance, aggressive GPU usage
};

struct AllocationContext {
    size_t requested_size;
    size_t min_acceptable_size;
    cudaStream_t stream;
    AllocationStrategy strategy;
    uint64_t timestamp;
    std::string operation_id;
    
    AllocationContext(size_t size, AllocationStrategy strat = AllocationStrategy::AUTO_ADAPTIVE)
        : requested_size(size), min_acceptable_size(size / 10), stream(nullptr),
          strategy(strat), timestamp(std::chrono::steady_clock::now().time_since_epoch().count()),
          operation_id("alloc_" + std::to_string(timestamp)) {}
};

struct ResourceState {
    size_t available_gpu_memory;
    size_t available_host_memory;
    size_t current_gpu_usage;
    size_t current_host_usage;
    size_t total_system_memory;
    float gpu_utilization;
    float host_utilization;
    size_t active_allocations;
    size_t fragmentation_ratio;
    
    float get_memory_efficiency() const {
        return total_system_memory > 0 ?
               static_cast<float>(current_gpu_usage + current_host_usage) / total_system_memory : 0.0f;
    }
};

struct FallbackConfig {
    bool enable_host_memory_fallback = true;
    bool enable_progressive_degradation = true;
    bool enable_chunk_reduction = true;
    bool enable_rollback_protection = true;
    bool enable_adaptive_strategies = true;
    bool enable_resource_aware_allocation = true;
    bool enable_progressive_enhancement = true;
    
    size_t emergency_threshold_mb = 100;  // Switch to emergency mode
    size_t host_memory_limit_mb = 1024;   // Max host memory to use
    size_t adaptive_threshold_mb = 500;   // Threshold for adaptive strategies
    float degradation_factor = 0.5f;      // Reduce allocation size by this factor
    float enhancement_factor = 1.2f;      // Increase allocation size for enhancement
    int max_retry_attempts = 3;           // Max retries for failed allocations
    
    // Smart allocation parameters
    float fragmentation_tolerance = 0.3f; // Max acceptable fragmentation
    float performance_vs_reliability = 0.7f; // Balance factor (0.0 = reliability, 1.0 = performance)
    size_t min_allocation_unit = 4096;    // Minimum allocation unit size
    
    // Progressive enhancement parameters
    bool enable_feature_scaling = true;
    bool enable_dynamic_pool_sizing = true;
    size_t enhancement_check_interval_ms = 5000; // Check for enhancement every 5 seconds
};

// ============================================================================
// Fallback Allocation Result
// ============================================================================

struct FallbackAllocation {
    void* ptr = nullptr;
    void* host_ptr = nullptr;  // Additional host pointer for dual allocations
    size_t allocated_size = 0;
    size_t requested_size = 0;
    size_t effective_size = 0; // Actual usable size after any transformations
    bool is_host_memory = false;
    bool is_degraded = false;
    bool is_enhanced = false;   // Whether allocation was enhanced beyond request
    bool is_dual_memory = false; // Whether both GPU and host memory were allocated
    AllocationStrategy strategy_used = AllocationStrategy::AUTO_ADAPTIVE;
    Status status = Status::SUCCESS;
    std::string allocation_path; // Track which path was used for allocation
    uint64_t allocation_time_ns = 0; // Track allocation latency
    
    bool is_valid() const { return ptr != nullptr && allocated_size > 0; }
    
    // Get the appropriate pointer based on memory type preference
    void* get_pointer(bool prefer_host = false) const {
        if (prefer_host && host_ptr) return host_ptr;
        return ptr;
    }
    
    // Calculate efficiency ratio
    double get_efficiency_ratio() const {
        return requested_size > 0 ? static_cast<double>(effective_size) / requested_size : 0.0;
    }
};

// ============================================================================
// Pool Statistics
// ============================================================================

struct PoolStats {
    uint64_t total_allocations = 0;
    uint64_t total_deallocations = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t pool_grows = 0;
    uint64_t fallback_allocations = 0;
    uint64_t host_memory_allocations = 0;
    uint64_t degraded_allocations = 0;
    uint64_t allocation_failures = 0;
    uint64_t rollback_operations = 0;
    
    size_t peak_memory_usage = 0;
    size_t current_memory_usage = 0;
    size_t total_pool_capacity = 0;
    size_t host_memory_usage = 0;
    
    DegradationMode current_mode = DegradationMode::NORMAL;
    
    double get_hit_rate() const {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
    }
    
    double get_fallback_rate() const {
        uint64_t total = total_allocations;
        return total > 0 ? static_cast<double>(fallback_allocations) / total : 0.0;
    }
    
    double get_degradation_rate() const {
        uint64_t total = total_allocations;
        return total > 0 ? static_cast<double>(degraded_allocations) / total : 0.0;
    }
};

// ============================================================================
// Pool Entry with Fallback Support
// ============================================================================

struct PoolEntry {
    void* ptr = nullptr;
    void* host_ptr = nullptr;  // Host fallback pointer
    size_t size = 0;
    size_t host_size = 0;
    bool in_use = false;
    bool is_host_fallback = false;
    bool is_degraded = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    
    PoolEntry() = default;
    PoolEntry(void* p, size_t s) : ptr(p), size(s), in_use(false),
                                   stream(nullptr), ready_event(nullptr) {}
    
    // Cleanup both GPU and host memory
    ~PoolEntry() {
        if (ptr && !is_host_fallback) {
            cudaFree(ptr);
        }
        if (host_ptr) {
            free(host_ptr);
        }
        if (ready_event) {
            cudaEventDestroy(ready_event);
        }
    }
};

// ============================================================================
// Memory Pool Manager with Fallback Support
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
    
    // Allocation interface with fallback support
    void* allocate(size_t size, cudaStream_t stream = 0);
    Status deallocate(void* ptr);
    
    // Fallback-aware allocation
    FallbackAllocation allocate_with_fallback(size_t requested_size, cudaStream_t stream = 0);
    
    // Progressive allocation (degrade size until successful)
    FallbackAllocation allocate_progressive(size_t min_size, size_t max_size, cudaStream_t stream = 0);
    
    // Async allocation with stream synchronization
    void* allocate_async(size_t size, cudaStream_t stream);
    
    // Pool management
    Status prewarm(size_t total_memory);
    Status prewarm_by_sizes(const std::vector<size_t>& allocation_sizes);
    Status defragment();
    void clear();
    
    // Fallback and degradation management
    void set_fallback_config(const FallbackConfig& config);
    const FallbackConfig& get_fallback_config() const;
    
    void set_degradation_mode(DegradationMode mode);
    DegradationMode get_degradation_mode() const;
    
    // Emergency operations
    Status emergency_clear();
    Status switch_to_host_memory_mode();
    
    // Statistics
    PoolStats get_statistics() const;
    void reset_statistics();
    void print_statistics() const;
    
    // Configuration
    void set_growth_factor(float factor);
    void enable_defragmentation(bool enable);
    void set_max_pool_size(size_t max_size);
    
    // Memory pressure monitoring
    size_t get_available_gpu_memory() const;
    size_t get_memory_pressure_percentage() const;
    bool is_memory_pressure_high() const;
    
    // Smart allocation interface
    FallbackAllocation allocate_smart(const AllocationContext& context);
    FallbackAllocation allocate_with_strategy(size_t size, AllocationStrategy strategy, cudaStream_t stream = 0);
    FallbackAllocation allocate_dual_memory(size_t size, cudaStream_t stream = 0);
    FallbackAllocation enhance_allocation(void* existing_ptr, size_t additional_size);
    
    // Rollback and recovery
    struct RollbackContext {
        size_t rollback_size;
        void* original_ptr;
        void* backup_ptr;
        std::string operation_id;
        uint64_t timestamp;
    };
    
    Status execute_rollback(RollbackContext& context);
    Status recover_from_failure(const std::string& failure_context);
    RollbackContext create_rollback_context();
    Status complete_rollback_operation(RollbackContext& context);
    
    // Progressive enhancement
    Status attempt_progressive_enhancement();
    Status downgrade_allocation_if_needed();
    bool is_enhancement_possible() const;
    enum class EnhancementState {
        DISABLED = 0,
        ENABLED = 1,
        ENHANCING = 2,
        ENHANCED = 3,
        DOWNGRADED = 4
    };
    
    EnhancementState get_enhancement_state() const;
    
    // Resource-aware allocation
    ResourceState get_current_resource_state() const;
    AllocationStrategy select_optimal_strategy(const ResourceState& state, size_t size) const;
    size_t calculate_optimal_allocation_size(const ResourceState& state, size_t requested_size) const;
    Status perform_resource_balance();
    
    // Advanced statistics
    void get_detailed_statistics(PoolStats& stats) const;
    double get_average_allocation_latency() const;
    double get_fragmentation_ratio() const;
    
    // State management
    void capture_system_state();
    Status restore_system_state();
    void log_allocation_decision(const std::string& decision, const AllocationContext& context);
    
    // Additional member variables needed for enhanced functionality
    mutable std::chrono::steady_clock::time_point last_resource_update_;
    mutable std::chrono::steady_clock::time_point last_enhancement_check_;
    ResourceState cached_resource_state_;
    ProgressiveEnhancementState enhancement_state_;
    ResourceAwareAllocationManager resource_manager_;
    
    // Enhanced allocation tracking
    std::unordered_map<void*, EnhancedAllocationState> allocation_states_;
    mutable std::mutex allocation_states_mutex_;
    
    // Additional statistics tracking
    mutable std::atomic<uint64_t> enhanced_allocations_{0};
    mutable std::atomic<uint64_t> dual_memory_allocations_{0};
    mutable std::atomic<uint64_t> enhancement_operations_{0};
    mutable std::atomic<uint64_t> allocation_latency_count_{0};
    mutable std::atomic<uint64_t> total_allocation_latency_ns_{0};
    mutable std::atomic<uint64_t> deallocation_latency_count_{0};
    mutable std::atomic<uint64_t> total_deallocation_latency_ns_{0};
    
    // Enhanced functionality methods
    FallbackAllocation allocate_with_resource_awareness(const AllocationContext& context);
    FallbackAllocation allocate_adaptive(size_t size, cudaStream_t stream);
    FallbackAllocation allocate_performance_optimized(size_t size, cudaStream_t stream);
    FallbackAllocation allocate_reliability_first(size_t size, cudaStream_t stream);
    FallbackAllocation attempt_allocation_enhancement(size_t base_size, size_t target_size, cudaStream_t stream);
    Status evaluate_enhancement_opportunity();
    Status execute_enhancement_plan(const std::vector<size_t>& enhancement_sizes);
    void update_resource_state();
    void update_allocation_latency(uint64_t latency_ns);
    void update_deallocation_latency(uint64_t latency_ns);
    void update_fragmentation_metrics();
    void log_allocation_path(const std::string& path, const FallbackAllocation& result);
    bool should_downgrade_due_to_pressure() const;
    size_t estimate_memory_requirements(const ResourceState& state, size_t requested_size) const;
    bool is_allocation_feasible(const ResourceState& state, size_t size) const;
    Status perform_health_check();
    void optimize_memory_distribution();
    void track_allocation_state(const AllocationState& state);
    Status restore_consistent_state();
    Status cleanup_failed_allocation_state(RollbackContext& context);
    Status rollback_partial_allocation(RollbackContext& context, const std::string& reason);
    void log_fallback_event(const std::string& event_type, const std::string& details);

    // Utility to check whether a returned pointer is a GPU device pointer
    // (useful to detect host-fallback entries returned from pool).
    bool is_device_pointer(void* ptr) const;
    
    // Options for host fallback handling
    // TODO: consider making these non-static config members in the future
    static bool disable_host_fallback_env();
    static bool auto_migrate_host_env();

    // Migrate a block of host memory into device memory. Returns device
    // pointer on success or nullptr on failure. The 'host_ptr' will be
    // freed on success.
    void* migrate_host_to_device(void* host_ptr, size_t size, cudaStream_t stream = 0);
    
private:
    // Pool storage - one pool per size class
    std::vector<PoolEntry> pools_[NUM_POOL_SIZES];
    mutable std::timed_mutex pool_mutexes_[NUM_POOL_SIZES];
    
    // Pool size thresholds
    static constexpr size_t POOL_SIZES[NUM_POOL_SIZES] = {
        SIZE_4KB, SIZE_16KB, SIZE_64KB, SIZE_256KB, SIZE_1MB, SIZE_4MB
    };
    
    // Configuration
    float growth_factor_ = 1.5f;
    bool defrag_enabled_ = true;
    size_t max_pool_size_ = 1024ULL * 1024 * 1024 * 2;  // 2GB default max
    
    // Fallback and degradation configuration
    FallbackConfig fallback_config_;
    DegradationMode current_mode_ = DegradationMode::NORMAL;
    mutable std::timed_mutex mode_mutex_;
    
    // Memory pressure tracking
    mutable std::atomic<size_t> host_memory_usage_{0};
    size_t last_memory_check_ = 0;
    mutable std::chrono::steady_clock::time_point last_pressure_update_;
    
    // Statistics (atomic for thread safety)
    mutable std::atomic<uint64_t> total_allocations_{0};
    mutable std::atomic<uint64_t> total_deallocations_{0};
    mutable std::atomic<uint64_t> cache_hits_{0};
    mutable std::atomic<uint64_t> cache_misses_{0};
    mutable std::atomic<uint64_t> pool_grows_{0};
    mutable std::atomic<uint64_t> fallback_allocations_{0};
    mutable std::atomic<uint64_t> host_memory_allocations_{0};
    mutable std::atomic<uint64_t> degraded_allocations_{0};
    mutable std::atomic<uint64_t> allocation_failures_{0};
    mutable std::atomic<uint64_t> rollback_operations_{0};
    mutable std::atomic<size_t> peak_memory_usage_{0};
    mutable std::atomic<size_t> current_memory_usage_{0};
    
    // Helper functions
    int get_pool_index(size_t size) const;
    size_t round_up_to_pool_size(size_t size) const;
    Status grow_pool(int pool_idx, size_t min_entries = 1);
    PoolEntry* find_free_entry(int pool_idx, cudaStream_t stream);
    
    // Fallback allocation strategies
    FallbackAllocation allocate_with_cuda_fallback(size_t size, cudaStream_t stream);
    FallbackAllocation allocate_host_memory(size_t size);
    FallbackAllocation allocate_degraded(size_t size, cudaStream_t stream);
    
    // Progressive allocation helpers
    size_t calculate_degraded_size(size_t original_size) const;
    bool should_use_progressive_allocation(size_t requested_size) const;
    bool is_emergency_mode() const;
    
    // Memory pressure monitoring
    size_t get_available_gpu_memory_impl() const;
    bool is_memory_pressure_critical() const;
    void update_degradation_mode();
    void trigger_rollback_protection();
    
    // Enhanced allocation with fallback
    void* allocate_from_cuda(size_t size);
    void* allocate_from_host(size_t size);
    Status copy_between_memory_types(void* src, void* dst, size_t size, bool host_to_device);
    
    void update_peak_usage(size_t current_usage);
};

// ============================================================================
// Global Pool Instance (Optional Singleton)
// ============================================================================

MemoryPoolManager& get_global_pool();
void destroy_global_pool();

} // namespace memory
} // namespace cuda_zstd

#endif // CUDA_ZSTD_MEMORY_POOL_COMPLEX_H_