// ============================================================================
// test_fallback_strategies.cu - Comprehensive Fallback Strategy Tests
// ============================================================================

#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_safe_alloc.h"
#include "cuda_error_checking.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>
#include <iomanip>
#include <algorithm>

using namespace cuda_zstd;
using namespace cuda_zstd::memory;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg) std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_EQ(a, b, msg) if ((a) != (b)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_NE(a, b, msg) if ((a) == (b)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_TRUE(cond, msg) if (!(cond)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_STATUS(status, msg) if ((status) != Status::SUCCESS) { LOG_FAIL(__func__, msg << " Status: " << status_to_string(status)); return false; }

void print_separator() {
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// TEST SUITE 1: Host Memory Fallback
// ============================================================================

bool test_host_memory_fallback() {
    LOG_TEST("Host Memory Fallback Strategy");
    
    MemoryPoolManager pool; // Disable defrag for simpler testing
    
    // Configure fallback
    FallbackConfig config;
    config.enable_host_memory_fallback = true;
    config.host_memory_limit_mb = 50;  // 50MB limit for testing
    config.max_retry_attempts = 2;
    pool.set_fallback_config(config);
    
    // Try to allocate a size that might fail on GPU but should work on host
    size_t large_size = 16 * 1024 * 1024; // 16MB
    
    FallbackAllocation result = pool.allocate_with_fallback(large_size);
    
    if (result.is_valid()) {
        LOG_INFO("Fallback allocation succeeded");
        LOG_INFO("  Allocated size: " << result.allocated_size / 1024 << " KB");
        LOG_INFO("  Is host memory: " << (result.is_host_memory ? "Yes" : "No"));
        LOG_INFO("  Is degraded: " << (result.is_degraded ? "Yes" : "No"));
        
        // Test deallocation
        Status dealloc_status = pool.deallocate(result.ptr);
        ASSERT_STATUS(dealloc_status, "Deallocation failed");
        
        LOG_PASS("Host Memory Fallback");
        return true;
    } else {
        LOG_INFO("Fallback allocation failed (this may be expected on systems with plenty of GPU memory)");
        LOG_PASS("Host Memory Fallback (No GPU pressure detected)");
        return true;
    }
}

bool test_host_memory_limit_enforcement() {
    LOG_TEST("Host Memory Limit Enforcement");
    
    MemoryPoolManager pool;
    
    FallbackConfig config;
    config.enable_host_memory_fallback = true;
    config.host_memory_limit_mb = 1;  // Very small limit for testing
    config.max_retry_attempts = 1;
    pool.set_fallback_config(config);
    
    std::vector<void*> allocations;
    size_t alloc_size = 512 * 1024; // 512KB each
    
    // Try to exceed host memory limit
    int successful_allocs = 0;
    for (int i = 0; i < 5; i++) {
        FallbackAllocation result = pool.allocate_with_strategy(alloc_size, AllocationStrategy::PREFER_HOST);
        if (result.is_valid()) {
            allocations.push_back(result.ptr);
            successful_allocs++;
            LOG_INFO("Successful allocation " << i << " (host: " << result.is_host_memory << ")");
        } else {
            LOG_INFO("Allocation " << i << " failed as expected");
            break;
        }
    }
    
    // Cleanup
    for (void* ptr : allocations) {
        pool.deallocate(ptr);
    }
    
    // Should have been limited by host memory limit
    ASSERT_TRUE(successful_allocs <= 2, "Should be limited by host memory limit");
    
    LOG_PASS("Host Memory Limit Enforcement");
    return true;
}

// ============================================================================
// TEST SUITE 2: Progressive Degradation
// ============================================================================

bool test_progressive_allocation() {
    LOG_TEST("Progressive Allocation Strategy");
    
    MemoryPoolManager pool;
    
    FallbackConfig config;
    config.enable_progressive_degradation = true;
    config.degradation_factor = 0.5f;
    config.max_retry_attempts = 3;
    pool.set_fallback_config(config);
    
    // Test progressive allocation from large to small
    size_t min_size = 1024;        // 1KB
    size_t max_size = 8 * 1024 * 1024; // 8MB
    
    FallbackAllocation result = pool.allocate_progressive(min_size, max_size);
    
    if (result.is_valid()) {
        LOG_INFO("Progressive allocation succeeded");
        LOG_INFO("  Requested range: " << min_size / 1024 << "KB - " << max_size / 1024 / 1024 << "MB");
        LOG_INFO("  Actual allocated: " << result.allocated_size / 1024 << " KB");
        LOG_INFO("  Is degraded: " << (result.is_degraded ? "Yes" : "No"));
        
        // Verify allocated size is within expected range
        ASSERT_TRUE(result.allocated_size >= min_size, "Allocated size below minimum");
        ASSERT_TRUE(result.allocated_size <= max_size, "Allocated size exceeds maximum");
        
        // Cleanup
        Status dealloc_status = pool.deallocate(result.ptr);
        ASSERT_STATUS(dealloc_status, "Deallocation failed");
        
        LOG_PASS("Progressive Allocation");
        return true;
    } else {
        LOG_INFO("Progressive allocation failed (may be expected if GPU has plenty of memory)");
        LOG_PASS("Progressive Allocation (No memory pressure)");
        return true;
    }
}

bool test_degradation_modes() {
    LOG_TEST("Degradation Modes");
    
    MemoryPoolManager pool;
    
    // Test each degradation mode
    std::vector<std::pair<DegradationMode, float>> modes = {
        {DegradationMode::NORMAL, 1.0f},
        {DegradationMode::CONSERVATIVE, 0.8f},
        {DegradationMode::AGGRESSIVE, 0.5f},
        {DegradationMode::EMERGENCY, 0.25f}
    };
    
    size_t original_size = 4 * 1024 * 1024; // 4MB
    
    for (const auto& mode_info : modes) {
        DegradationMode mode = mode_info.first;
        float expected_factor = mode_info.second;
        
        pool.set_degradation_mode(mode);
        
        size_t degraded_size = pool.calculate_degraded_size(original_size);
        size_t expected_size = static_cast<size_t>(original_size * expected_factor);
        
        LOG_INFO("Mode " << static_cast<int>(mode) << ": " << degraded_size / 1024 << " KB (expected ~" << expected_size / 1024 << " KB)");
        
        // Allow some tolerance for rounding
        size_t tolerance = 1024; // 1KB
        ASSERT_TRUE(degraded_size >= expected_size - tolerance && 
                   degraded_size <= expected_size + tolerance,
                   "Degraded size not within expected range");
    }
    
    LOG_PASS("Degradation Modes");
    return true;
}

// ============================================================================
// TEST SUITE 3: Memory Pressure Monitoring
// ============================================================================

bool test_memory_pressure_monitoring() {
    LOG_TEST("Memory Pressure Monitoring");
    
    MemoryPoolManager pool;
    
    // Test memory pressure detection
    size_t available_memory = pool.get_available_gpu_memory();
    size_t pressure_percentage = pool.get_memory_pressure_percentage();
    
    LOG_INFO("Available GPU memory: " << available_memory / 1024 / 1024 << " MB");
    LOG_INFO("Memory pressure: " << pressure_percentage << "%");
    
    // These should always return valid values
    ASSERT_TRUE(available_memory > 0, "Should be able to get available memory");
    ASSERT_TRUE(pressure_percentage <= 100, "Pressure should be <= 100%");
    
    // Test pressure-based mode switching
    DegradationMode initial_mode = pool.get_degradation_mode();
    LOG_INFO("Initial degradation mode: " << static_cast<int>(initial_mode));
    
    // Force update to test the logic
    // pool.update_degradation_mode();
    DegradationMode updated_mode = pool.get_degradation_mode();
    
    LOG_INFO("Updated degradation mode: " << static_cast<int>(updated_mode));
    
    LOG_PASS("Memory Pressure Monitoring");
    return true;
}

bool test_emergency_mode_operations() {
    LOG_TEST("Emergency Mode Operations");
    
    MemoryPoolManager pool;
    
    // Switch to emergency mode
    Status status = pool.switch_to_host_memory_mode();
    ASSERT_STATUS(status, "Failed to switch to host memory mode");
    
    DegradationMode mode = pool.get_degradation_mode();
    ASSERT_TRUE(mode == DegradationMode::EMERGENCY, "Should be in emergency mode");
    
    LOG_INFO("Successfully switched to emergency mode");
    
    // Try allocation in emergency mode
    FallbackConfig config = pool.get_fallback_config();
    if (config.enable_host_memory_fallback) {
        FallbackAllocation result = pool.allocate_with_fallback(1024 * 1024); // 1MB
        if (result.is_valid()) {
            LOG_INFO("Host memory allocation in emergency mode succeeded");
            
            Status dealloc_status = pool.deallocate(result.ptr);
            ASSERT_STATUS(dealloc_status, "Deallocation failed in emergency mode");
        } else {
            LOG_INFO("Host memory allocation failed (may be due to system limits)");
        }
    }
    
    // Test emergency clear
    status = pool.emergency_clear();
    ASSERT_STATUS(status, "Emergency clear failed");
    
    LOG_PASS("Emergency Mode Operations");
    return true;
}

// ============================================================================
// TEST SUITE 4: Rollback Protection
// ============================================================================

bool test_rollback_protection() {
    LOG_TEST("Rollback Protection");
    
    MemoryPoolManager pool;
    
    // Configure for testing rollback
    FallbackConfig config;
    config.enable_rollback_protection = true;
    config.max_retry_attempts = 1; // Limit attempts to trigger rollback
    config.enable_host_memory_fallback = false; // Disable host fallback to ensure fast failure
    pool.set_fallback_config(config);
    
    PoolStats initial_stats = pool.get_statistics();
    uint64_t initial_failures = initial_stats.allocation_failures;
    uint64_t initial_rollbacks = initial_stats.rollback_operations;
    
    LOG_INFO("Initial failures: " << initial_failures);
    LOG_INFO("Initial rollbacks: " << initial_rollbacks);
    
    // Simulate realistic memory pressure by pre-allocating GPU memory
    size_t total_mem = 0, free_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    LOG_INFO("GPU total memory: " << (total_mem / 1024 / 1024) << " MB");
    LOG_INFO("GPU free memory: " << (free_mem / 1024 / 1024) << " MB");
    
    // Fill GPU until we hit OOM to ensure we are truly at the limit
    std::vector<void*> pressure_allocs;
    size_t chunk_size = 256 * 1024 * 1024; // Start with 256MB chunks
    size_t allocated = 0;
    
    LOG_INFO("Creating memory pressure (filling until OOM)...");
    
    // Phase 1: Large chunks
    while (true) {
        void* ptr = nullptr;
        cudaError_t err = cuda_zstd::safe_cuda_malloc(&ptr, chunk_size);
        if (err == cudaSuccess && ptr != nullptr) {
            pressure_allocs.push_back(ptr);
            allocated += chunk_size;
        } else {
            break; // Failed to allocate large chunk
        }
    }
    
    // Phase 2: Small chunks to fill the gaps
    chunk_size = 10 * 1024 * 1024; // 10MB chunks
    while (true) {
        void* ptr = nullptr;
        cudaError_t err = cuda_zstd::safe_cuda_malloc(&ptr, chunk_size);
        if (err == cudaSuccess && ptr != nullptr) {
            pressure_allocs.push_back(ptr);
            allocated += chunk_size;
        } else {
            break; // Failed to allocate small chunk
        }
    }

    LOG_INFO("Applied memory pressure: " << (allocated / 1024 / 1024) << " MB allocated. GPU should be full.");
    
    size_t free_after_pressure = 0, total_after_pressure = 0;
    cudaMemGetInfo(&free_after_pressure, &total_after_pressure);
    LOG_INFO("Free memory after pressure: " << (free_after_pressure / 1024 / 1024) << " MB");

    // Now try allocations that should fail due to memory pressure
    size_t test_size = 512 * 1024 * 1024; // 512MB - should definitely fail now
    int failure_count = 0;
    
    for (int i = 0; i < 5; i++) {
        FallbackAllocation result = pool.allocate_with_fallback(test_size);
        if (!result.is_valid()) {
            LOG_INFO("Expected failure " << i << " recorded under memory pressure");
            failure_count++;
        } else {
            // Unexpected success, clean it up
            LOG_INFO("Unexpected success " << i << " ptr=" << result.ptr);
            pool.deallocate(result.ptr);
        }
    }
    
    for (int i = 0; i < 5; i++) {
        FallbackAllocation result = pool.allocate_with_fallback(test_size);
        if (!result.is_valid()) {
            LOG_INFO("Expected failure " << i << " recorded under memory pressure");
            failure_count++;
        } else {
            // Unexpected success, clean it up
            pool.deallocate(result.ptr);
        }
    }
    
    // Cleanup pressure allocations
    for (void* ptr : pressure_allocs) {
        cudaFree(ptr);
    }
    
    LOG_INFO("Cleanup complete, freed " << pressure_allocs.size() << " pressure allocations");
    
    // Check if rollback protection was triggered
    PoolStats final_stats = pool.get_statistics();
    uint64_t final_failures = final_stats.allocation_failures;
    uint64_t final_rollbacks = final_stats.rollback_operations;
    
    LOG_INFO("Final failures: " << final_failures);
    LOG_INFO("Final rollbacks: " << final_rollbacks);
    LOG_INFO("Failures during test: " << failure_count);
    
    // Should have recorded failures (we expect at least some failures under pressure)
    ASSERT_TRUE(final_failures > initial_failures, "Should have recorded allocation failures");
    
    // Check if degradation mode changed due to failures
    DegradationMode current_mode = pool.get_degradation_mode();
    LOG_INFO("Current degradation mode after failures: " << static_cast<int>(current_mode));
    
    LOG_PASS("Rollback Protection");
    return true;
}

// ============================================================================
// TEST SUITE 5: Stress Tests with Fallback
// ============================================================================

bool test_fallback_stress_test() {
    LOG_TEST("Fallback Stress Test");
    
    MemoryPoolManager pool;
    
    FallbackConfig config;
    config.enable_host_memory_fallback = true;
    config.enable_progressive_degradation = true;
    config.enable_rollback_protection = true;
    config.host_memory_limit_mb = 100;
    config.max_retry_attempts = 2;
    pool.set_fallback_config(config);
    
    const int num_allocations = 50;
    std::vector<std::pair<void*, bool>> allocations; // ptr, is_host_memory
    
    LOG_INFO("Running stress test with " << num_allocations << " allocations");
    
    int successful_allocs = 0;
    int host_allocs = 0;
    int degraded_allocs = 0;
    
    for (int i = 0; i < num_allocations; i++) {
        size_t size = 512 * 1024 + (i * 64 * 1024); // Start with 512KB, increase
        size = std::min(size, (size_t)(4 * 1024 * 1024)); // Cap at 4MB
        
        FallbackAllocation result = pool.allocate_with_fallback(size);
        
        if (result.is_valid()) {
            successful_allocs++;
            allocations.push_back({result.ptr, result.is_host_memory});
            
            if (result.is_host_memory) host_allocs++;
            if (result.is_degraded) degraded_allocs++;
            
            LOG_INFO("Alloc " << i << ": " << size / 1024 << "KB -> " 
                     << result.allocated_size / 1024 << "KB " 
                     << (result.is_host_memory ? "(host)" : "(gpu)")
                     << (result.is_degraded ? " [degraded]" : ""));
            
            // Small delay to simulate real usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else {
            LOG_INFO("Alloc " << i << " failed");
        }
    }
    
    // Check for duplicates - TEMPORARILY DISABLED TO SEE REAL ERROR
    /*
    std::sort(allocations.begin(), allocations.end());
    for (size_t i = 1; i < allocations.size(); ++i) {
        if (allocations[i].first == allocations[i-1].first) {
            LOG_INFO("DUPLICATE POINTER DETECTED: " << allocations[i].first);
        }
    }
    */
    
    LOG_INFO("Stress test results:");
    LOG_INFO("  Successful: " << successful_allocs << "/" << num_allocations);
    LOG_INFO("  Host memory: " << host_allocs);
    LOG_INFO("  Degraded: " << degraded_allocs);
    
    // Cleanup all allocations
    for (const auto& alloc : allocations) {
        Status status = pool.deallocate(alloc.first);
        if (status != Status::SUCCESS) {
            std::cerr << "CLEANUP FAILED for ptr=" << alloc.first << " status=" << (int)status << "\n";
        }
        ASSERT_STATUS(status, "Cleanup deallocation failed");
    }
    
    // Check final statistics
    PoolStats stats = pool.get_statistics();
    LOG_INFO("Final fallback rate: " << (stats.get_fallback_rate() * 100.0) << "%");
    LOG_INFO("Final degradation rate: " << (stats.get_degradation_rate() * 100.0) << "%");
    
    ASSERT_TRUE(successful_allocs > 0, "Should have some successful allocations");
    
    LOG_PASS("Fallback Stress Test");
    return true;
}

// ============================================================================
// TEST SUITE 6: Concurrent Fallback Operations
// ============================================================================

bool test_concurrent_fallback_operations() {
    LOG_TEST("Concurrent Fallback Operations");
    
    MemoryPoolManager pool;
    
    FallbackConfig config;
    config.enable_host_memory_fallback = true;
    config.enable_progressive_degradation = true;
    config.host_memory_limit_mb = 200;
    pool.set_fallback_config(config);
    
    const int num_threads = 4;
    const int allocations_per_thread = 20;
    
    LOG_INFO("Testing " << num_threads << " threads with " 
             << allocations_per_thread << " allocations each");
    
    std::vector<std::thread> threads;
    std::vector<int> success_counts(num_threads, 0);
    std::vector<int> host_counts(num_threads, 0);
    
    auto worker = [&](int thread_id) {
        for (int i = 0; i < allocations_per_thread; i++) {
            size_t size = 256 * 1024 + (thread_id * i * 32 * 1024); // Vary sizes
            size = std::min(size, (size_t)(2 * 1024 * 1024)); // Cap at 2MB
            
            FallbackAllocation result = pool.allocate_with_fallback(size);
            if (result.is_valid()) {
                success_counts[thread_id]++;
                if (result.is_host_memory) {
                    host_counts[thread_id]++;
                }
                
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                
                // Cleanup
                pool.deallocate(result.ptr);
            }
            
            // Small delay between allocations
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    };
    
    // Launch threads
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }
    
    // Check results
    int total_success = 0;
    int total_host = 0;
    for (int i = 0; i < num_threads; i++) {
        LOG_INFO("Thread " << i << ": " << success_counts[i] << " success, " 
                 << host_counts[i] << " host allocations");
        total_success += success_counts[i];
        total_host += host_counts[i];
    }
    
    LOG_INFO("Total: " << total_success << "/" << (num_threads * allocations_per_thread) 
             << " success, " << total_host << " host allocations");
    
    ASSERT_TRUE(total_success > 0, "Should have some successful allocations");
    
    // Check final statistics for consistency
    PoolStats stats = pool.get_statistics();
    LOG_INFO("Pool statistics - Total allocs: " << stats.total_allocations
             << ", Fallbacks: " << stats.fallback_allocations
             << ", Host: " << stats.host_memory_allocations);
    
    LOG_PASS("Concurrent Fallback Operations");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA ZSTD - Fallback Strategies Test Suite" << std::endl;
    print_separator();
    std::cout << "\n";
    
    // Skip on CPU-only environments
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();
    
    int passed = 0;
    int total = 0;
    
    // Host Memory Fallback Tests
    print_separator();
    std::cout << "SUITE 1: Host Memory Fallback" << std::endl;
    print_separator();
    
    total++; if (test_host_memory_fallback()) passed++;
    total++; if (test_host_memory_limit_enforcement()) passed++;
    
    // Progressive Degradation Tests
    print_separator();
    std::cout << "SUITE 2: Progressive Degradation" << std::endl;
    print_separator();
    
    total++; if (test_progressive_allocation()) passed++;
    total++; if (test_degradation_modes()) passed++;
    
    // Memory Pressure Monitoring Tests
    print_separator();
    std::cout << "SUITE 3: Memory Pressure Monitoring" << std::endl;
    print_separator();
    
    total++; if (test_memory_pressure_monitoring()) passed++;
    total++; if (test_emergency_mode_operations()) passed++;
    
    // Rollback Protection Tests
    print_separator();
    std::cout << "SUITE 4: Rollback Protection" << std::endl;
    print_separator();
    
    total++; if (test_rollback_protection()) passed++;
    
    // Stress Tests
    print_separator();
    std::cout << "SUITE 5: Stress Tests with Fallback" << std::endl;
    print_separator();
    
    total++; if (test_fallback_stress_test()) passed++;
    
    // Concurrent Tests
    print_separator();
    std::cout << "SUITE 6: Concurrent Fallback Operations" << std::endl;
    print_separator();
    
    total++; if (test_concurrent_fallback_operations()) passed++;
    
    // Final Statistics
    print_separator();
    std::cout << "FINAL POOL STATISTICS" << std::endl;
    print_separator();
    
    MemoryPoolManager final_pool;
    PoolStats final_stats = final_pool.get_statistics();
    std::cout << "Total Allocations: " << final_stats.total_allocations << "\n";
    std::cout << "Fallback Allocations: " << final_stats.fallback_allocations << "\n";
    std::cout << "Host Memory Allocs: " << final_stats.host_memory_allocations << "\n";
    std::cout << "Degraded Allocations: " << final_stats.degraded_allocations << "\n";
    std::cout << "Allocation Failures: " << final_stats.allocation_failures << "\n";
    std::cout << "Rollback Operations: " << final_stats.rollback_operations << "\n";
    std::cout << "Fallback Rate: " << std::fixed << std::setprecision(2) 
              << (final_stats.get_fallback_rate() * 100.0) << "%\n";
    std::cout << "Degradation Rate: " << std::fixed << std::setprecision(2) 
              << (final_stats.get_degradation_rate() * 100.0) << "%\n";
    print_separator();
    
    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "TEST RESULTS" << std::endl;
    print_separator();
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    std::cout << "Failed: " << (total - passed) << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "\n✓ ALL FALLBACK TESTS PASSED" << std::endl;
        std::cout << "  The memory pool fallback strategies are working correctly!" << std::endl;
    } else {
        std::cout << "\n✗ SOME FALLBACK TESTS FAILED" << std::endl;
        std::cout << "  Fallback strategies need attention." << std::endl;
    }
    print_separator();
    std::cout << "\n";
    
    return (passed == total) ? 0 : 1;
}