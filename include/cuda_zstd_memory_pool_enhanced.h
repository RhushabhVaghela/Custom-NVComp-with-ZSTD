// ============================================================================
// cuda_zstd_memory_pool_enhanced.h - Enhanced Memory Pool Structures
// ============================================================================

#ifndef CUDA_ZSTD_MEMORY_POOL_ENHANCED_H_
#define CUDA_ZSTD_MEMORY_POOL_ENHANCED_H_

#include "cuda_zstd_memory_pool.h"
#include <unordered_map>
#include <queue>
#include <memory>

namespace cuda_zstd {
namespace memory {

// ============================================================================
// Enhanced Allocation State Tracking
// ============================================================================

struct EnhancedAllocationState {
    void* gpu_ptr = nullptr;
    void* host_ptr = nullptr;
    size_t gpu_size = 0;
    size_t host_size = 0;
    size_t requested_size = 0;
    size_t effective_size = 0;
    
    // Allocation metadata
    AllocationStrategy strategy = AllocationStrategy::AUTO_ADAPTIVE;
    DegradationMode degradation_mode = DegradationMode::NORMAL;
    std::string operation_id;
    uint64_t allocation_timestamp = 0;
    uint64_t allocation_latency_ns = 0;
    
    // State flags
    bool is_host_fallback = false;
    bool is_degraded = false;
    bool is_enhanced = false;
    bool is_dual_memory = false;
    bool is_pinned = false; // For host memory
    
    // Dependencies and relationships
    std::vector<void*> related_allocations;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    
    // Performance tracking
    uint64_t last_access_time = 0;
    uint64_t access_count = 0;
    double performance_score = 1.0;
    
    // Cleanup tracking
    bool cleanup_scheduled = false;
    std::chrono::steady_clock::time_point cleanup_deadline;
    
    EnhancedAllocationState() = default;
    
    EnhancedAllocationState(void* gpu_p, size_t gpu_s, void* host_p = nullptr, size_t host_s = 0)
        : gpu_ptr(gpu_p), host_ptr(host_p), gpu_size(gpu_s), host_size(host_s),
          allocation_timestamp(std::chrono::steady_clock::now().time_since_epoch().count()),
          last_access_time(allocation_timestamp) {}
    
    ~EnhancedAllocationState() {
        // Note: Actual cleanup should be handled by the pool manager
        // This destructor is for tracking purposes only
    }
    
    bool is_valid() const {
        return (gpu_ptr != nullptr && gpu_size > 0) || (host_ptr != nullptr && host_size > 0);
    }
    
    size_t total_size() const {
        return gpu_size + host_size;
    }
    
    void record_access() {
        access_count++;
        last_access_time = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    double get_utilization_ratio() const {
        return requested_size > 0 ? static_cast<double>(effective_size) / requested_size : 0.0;
    }
};

// ============================================================================
// Advanced Rollback Context
// ============================================================================

struct AdvancedRollbackContext {
    enum class RollbackType {
        PARTIAL_ALLOCATION,
        SYSTEM_FAILURE,
        MEMORY_PRESSURE,
        DEADLOCK_AVOIDANCE,
        PERFORMANCE_DEGRADATION
    };
    
    // Context identification
    std::string rollback_id;
    RollbackType type;
    std::string trigger_reason;
    uint64_t timestamp;
    
    // Allocation state snapshots
    std::vector<EnhancedAllocationState> active_allocations;
    std::vector<PoolEntry> pool_state_before;
    std::vector<PoolEntry> pool_state_after;
    
    // Rollback operations to perform
    struct RollbackOperation {
        enum class OperationType {
            DEALLOCATE_GPU = 0,
            DEALLOCATE_HOST = 1,
            RESTORE_POOL_ENTRY = 2,
            UPDATE_STATISTICS = 3,
            NOTIFY_CALLBACK = 4
        };
        
        OperationType type;
        void* target_ptr;
        size_t size;
        std::string description;
        std::function<Status()> execute;
        std::chrono::steady_clock::time_point scheduled_time;
    };
    
    std::vector<RollbackOperation> operations;
    
    // Recovery strategy
    struct RecoveryPlan {
        bool requires_immediate_action = false;
        size_t estimated_recovery_time_ms = 0;
        std::vector<std::string> recovery_steps;
        AllocationStrategy recommended_strategy = AllocationStrategy::AUTO_ADAPTIVE;
        size_t memory_to_free = 0;
    };
    
    RecoveryPlan recovery_plan;
    
    // Status tracking
    Status overall_status = Status::SUCCESS;
    bool rollback_completed = false;
    bool recovery_initiated = false;
    uint64_t rollback_start_time = 0;
    uint64_t rollback_end_time = 0;
    
    // Metrics
    size_t total_memory_affected = 0;
    size_t operations_completed = 0;
    size_t operations_failed = 0;
    double rollback_efficiency = 0.0;
    
    AdvancedRollbackContext(RollbackType rb_type, const std::string& reason)
        : type(rb_type), trigger_reason(reason),
          timestamp(std::chrono::steady_clock::now().time_since_epoch().count()) {
        rollback_id = "rb_" + std::to_string(timestamp);
    }
    
    void add_operation(RollbackOperation op) {
        operations.push_back(std::move(op));
        if (op.type == RollbackOperation::OperationType::DEALLOCATE_GPU ||
            op.type == RollbackOperation::OperationType::DEALLOCATE_HOST) {
            total_memory_affected += op.size;
        }
    }
    
    size_t get_operation_count() const { return operations.size(); }
    
    double get_completion_ratio() const {
        size_t total = operations_completed + operations_failed;
        return total > 0 ? static_cast<double>(operations_completed) / total : 0.0;
    }
    
    uint64_t get_duration_ns() const {
        return rollback_completed ? rollback_end_time - rollback_start_time : 0;
    }
};

// ============================================================================
// Progressive Enhancement State
// ============================================================================

struct ProgressiveEnhancementState {
    struct EnhancementLevel {
        size_t level_id;
        size_t target_size;
        float performance_gain;
        float reliability_cost;
        bool is_available;
        std::string description;
        std::vector<std::string> requirements;
        
        EnhancementLevel(size_t id, size_t target, float gain, float cost, const std::string& desc)
            : level_id(id), target_size(target), performance_gain(gain), 
              reliability_cost(cost), is_available(false), description(desc) {}
    };
    
    struct EnhancementCandidate {
        void* base_allocation;
        size_t current_size;
        size_t target_size;
        double expected_improvement;
        std::vector<EnhancementLevel> available_levels;
        std::chrono::steady_clock::time_point evaluation_time;
        
        EnhancementCandidate(void* ptr, size_t current, size_t target)
            : base_allocation(ptr), current_size(current), target_size(target),
              evaluation_time(std::chrono::steady_clock::now()) {}
    };
    
    // Enhancement configuration
    bool enhancement_enabled = true;
    bool auto_enhancement = true;
    size_t current_level = 0;
    size_t max_level = 5;
    uint64_t last_enhancement_time = 0;
    uint64_t enhancement_cooldown_ms = 1000;
    
    // Available enhancement levels
    std::vector<EnhancementLevel> enhancement_levels;
    
    // Current candidates
    std::vector<EnhancementCandidate> candidates;
    std::queue<std::pair<void*, size_t>> enhancement_queue;
    
    // Enhancement history
    struct EnhancementRecord {
        void* allocation;
        size_t original_size;
        size_t enhanced_size;
        uint64_t timestamp;
        Status result;
        std::string enhancement_path;
        double actual_improvement;
    };
    
    std::vector<EnhancementRecord> enhancement_history;
    
    // Performance tracking
    struct PerformanceMetrics {
        double average_improvement = 0.0;
        double success_rate = 0.0;
        uint64_t total_attempts = 0;
        uint64_t successful_enhancements = 0;
        uint64_t failed_enhancements = 0;
        double average_enhancement_time_ms = 0.0;
    };
    
    PerformanceMetrics metrics;
    
    ProgressiveEnhancementState() {
        initialize_enhancement_levels();
    }
    
    void initialize_enhancement_levels() {
        enhancement_levels.clear();
        
        // Level 1: Small improvement
        enhancement_levels.emplace_back(1, cuda_zstd::memory::MemoryPoolManager::SIZE_16KB, 1.1f, 0.05f,
            "10% performance improvement with minimal reliability cost");
        
        // Level 2: Medium improvement
        enhancement_levels.emplace_back(2, cuda_zstd::memory::MemoryPoolManager::SIZE_64KB, 1.25f, 0.1f,
            "25% performance improvement with low reliability cost");
        
        // Level 3: Significant improvement
        enhancement_levels.emplace_back(3, cuda_zstd::memory::MemoryPoolManager::SIZE_256KB, 1.5f, 0.2f,
            "50% performance improvement with moderate reliability cost");
        
        // Level 4: Major improvement
        enhancement_levels.emplace_back(4, cuda_zstd::memory::MemoryPoolManager::SIZE_1MB, 2.0f, 0.35f,
            "2x performance improvement with higher reliability cost");
        
        // Level 5: Maximum improvement
        enhancement_levels.emplace_back(5, cuda_zstd::memory::MemoryPoolManager::SIZE_4MB, 3.0f, 0.5f,
            "3x performance improvement with significant reliability cost");
    }
    
    bool can_attempt_enhancement() const {
        if (!enhancement_enabled || !auto_enhancement) return false;
        
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        return (now - last_enhancement_time) >= enhancement_cooldown_ms * 1000000ULL;
    }
    
    void record_enhancement_attempt(const EnhancementRecord& record) {
        enhancement_history.push_back(record);
        metrics.total_attempts++;
        
        if (record.result == Status::SUCCESS) {
            metrics.successful_enhancements++;
            metrics.success_rate = static_cast<double>(metrics.successful_enhancements) / metrics.total_attempts;
        } else {
            metrics.failed_enhancements++;
        }
        
        // Keep only last 50 records
        if (enhancement_history.size() > 50) {
            enhancement_history.erase(enhancement_history.begin());
        }
    }
    
    void update_performance_metrics() {
        if (enhancement_history.empty()) return;
        
        double total_improvement = 0.0;
        uint64_t successful_count = 0;
        
        for (const auto& record : enhancement_history) {
            if (record.result == Status::SUCCESS) {
                total_improvement += record.actual_improvement;
                successful_count++;
            }
        }
        
        if (successful_count > 0) {
            metrics.average_improvement = total_improvement / successful_count;
        }
        
        metrics.success_rate = static_cast<double>(metrics.successful_enhancements) / metrics.total_attempts;
    }
    
    std::vector<EnhancementLevel> get_available_levels() const {
        std::vector<EnhancementLevel> available;
        for (const auto& level : enhancement_levels) {
            if (level.is_available && level.level_id <= current_level) {
                available.push_back(level);
            }
        }
        return available;
    }
    
    double get_current_enhancement_score() const {
        return metrics.success_rate * metrics.average_improvement;
    }
};

// ============================================================================
// Resource-Aware Allocation Manager
// ============================================================================

class ResourceAwareAllocationManager {
public:
    struct AllocationRequest {
        size_t requested_size;
        AllocationStrategy strategy;
        DegradationMode min_acceptable_mode;
        uint64_t priority;
        std::string requestor_id;
        std::chrono::steady_clock::time_point deadline;
        
        AllocationRequest(size_t size, AllocationStrategy strat = AllocationStrategy::AUTO_ADAPTIVE)
            : requested_size(size), strategy(strat), min_acceptable_mode(DegradationMode::EMERGENCY),
              priority(1), deadline(std::chrono::steady_clock::now() + std::chrono::seconds(30)) {}
    };
    
    struct AllocationResult {
        FallbackAllocation allocation;
        std::vector<std::string> decision_path;
        double confidence_score;
        std::vector<std::string> alternative_options;
        uint64_t evaluation_time_ns;
        
        bool is_successful() const { return allocation.is_valid(); }
    };
    
private:
    // Resource state tracking
    ResourceState current_state_;
    std::unordered_map<std::string, EnhancedAllocationState> active_allocations_;
    std::queue<AllocationRequest> pending_requests_;
    
    // Decision matrices
    std::unordered_map<AllocationStrategy, double> strategy_weights_;
    std::unordered_map<DegradationMode, double> mode_thresholds_;
    
    // Performance models
    struct PerformanceModel {
        double gpu_allocation_cost;
        double host_allocation_cost;
        double transfer_cost_per_mb;
        double fragmentation_penalty;
        double reliability_factor;
    };
    
    PerformanceModel performance_model_;
    
    // Prediction models
    struct ResourcePrediction {
        size_t predicted_gpu_memory;
        size_t predicted_host_memory;
        double predicted_fragmentation;
        std::chrono::steady_clock::time_point prediction_time;
        double confidence;
    };
    
    std::vector<ResourcePrediction> predictions_;
    
public:
    ResourceAwareAllocationManager();
    
    // Core allocation interface
    AllocationResult evaluate_allocation(const AllocationRequest& request);
    FallbackAllocation allocate_with_optimization(const AllocationRequest& request);
    
    // Resource state management
    void update_resource_state(const ResourceState& state);
    ResourcePrediction predict_resource_state(std::chrono::steady_clock::time_point when);
    bool is_allocation_feasible(const AllocationRequest& request);
    
    // Strategy optimization
    AllocationStrategy select_optimal_strategy(const AllocationRequest& request);
    std::vector<AllocationStrategy> get_strategy_ranking(const AllocationRequest& request);
    void update_strategy_weights(const AllocationResult& result);
    
    // Performance modeling
    void calibrate_performance_model();
    double estimate_allocation_cost(AllocationStrategy strategy, size_t size);
    double calculate_overall_score(AllocationStrategy strategy, const AllocationRequest& request);
    
    // Advanced features
    std::vector<FallbackAllocation> allocate_multiple(const std::vector<AllocationRequest>& requests);
    Status rebalance_allocations();
    void optimize_resource_distribution();
    
    // Metrics and monitoring
    double get_average_decision_time() const;
    double get_strategy_success_rate(AllocationStrategy strategy) const;
    size_t get_pending_request_count() const;
    
private:
    // Internal decision algorithms
    AllocationStrategy decision_tree_algorithm(const AllocationRequest& request);
    AllocationStrategy optimization_algorithm(const AllocationRequest& request);
    AllocationStrategy heuristic_algorithm(const AllocationRequest& request);
    
    // State update methods
    void update_performance_model(const AllocationResult& result);
    void update_resource_predictions();
    void cleanup_expired_predictions();
    
    // Utility methods
    double calculate_fragmentation_cost(size_t size);
    double calculate_reliability_penalty(AllocationStrategy strategy);
    double normalize_score(double raw_score);
};

} // namespace memory
} // namespace cuda_zstd

#endif // CUDA_ZSTD_MEMORY_POOL_ENHANCED_H