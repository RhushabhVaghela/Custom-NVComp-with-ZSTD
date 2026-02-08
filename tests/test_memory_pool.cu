// ============================================================================
// test_memory_pool.cu - Comprehensive Memory Pool Manager Tests
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_safe_alloc.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::memory;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg)                                                    \
  std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_EQ(a, b, msg)                                                   \
  if ((a) != (b)) {                                                            \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_NE(a, b, msg)                                                   \
  if ((a) == (b)) {                                                            \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_STATUS(status, msg)                                             \
  if ((status) != cuda_zstd::Status::SUCCESS) {                                \
    LOG_FAIL(__func__, msg << " Status: " << status_to_string(status));        \
    return false;                                                              \
  }

void print_separator() {
  std::cout << "========================================" << std::endl;
}

// ============================================================================
// TEST SUITE 1: Basic Functionality
// ============================================================================

bool test_basic_allocation_deallocation() {
  LOG_TEST("Basic Allocation and Deallocation");

  try {
    MemoryPoolManager pool; // Disable defrag for simpler testing

    // Test all pool sizes
    std::vector<size_t> test_sizes = {
        MemoryPoolManager::SIZE_4KB,  MemoryPoolManager::SIZE_16KB,
        MemoryPoolManager::SIZE_64KB, MemoryPoolManager::SIZE_256KB,
        MemoryPoolManager::SIZE_1MB,  MemoryPoolManager::SIZE_4MB};

    std::vector<void *> ptrs;

    for (size_t size : test_sizes) {
      void *ptr = pool.allocate(size);
      ASSERT_NE(ptr, nullptr, "Failed to allocate " << size << " bytes");
      ptrs.push_back(ptr);

      LOG_INFO("Allocated " << size / 1024 << " KB at " << ptr);
    }

    // Verify all allocations
    ASSERT_EQ(ptrs.size(), test_sizes.size(), "Allocation count mismatch");

    // Deallocate all
    for (void *ptr : ptrs) {
      cuda_zstd::Status status = pool.deallocate(ptr);
      ASSERT_STATUS(status, "Deallocation failed");
    }

    // Get statistics
    PoolStats stats = pool.get_statistics();
    LOG_INFO("Total allocations: " << stats.total_allocations);
    LOG_INFO("Total deallocations: " << stats.total_deallocations);
    LOG_INFO("Cache hits: " << stats.cache_hits);
    LOG_INFO("Cache misses: " << stats.cache_misses);

    ASSERT_EQ(stats.total_allocations, test_sizes.size(),
              "Allocation stats mismatch");
    ASSERT_EQ(stats.total_deallocations, test_sizes.size(),
              "Deallocation stats mismatch");

    LOG_PASS("Basic Allocation and Deallocation");
    return true;

  } catch (const std::exception &e) {
    LOG_FAIL("test_basic_allocation_deallocation",
             std::string("Exception: ") + e.what());
    return false;
  } catch (...) {
    LOG_FAIL("test_basic_allocation_deallocation", "Unknown exception");
    return false;
  }
}

bool test_pool_reuse() {
  LOG_TEST("Pool Reuse (Cache Hit Testing)");

  try {
    MemoryPoolManager pool;
    const size_t test_size = MemoryPoolManager::SIZE_64KB;

    // Allocate
    void *ptr1 = pool.allocate(test_size);
    ASSERT_NE(ptr1, nullptr, "First allocation failed");

    // Deallocate
    cuda_zstd::Status status = pool.deallocate(ptr1);
    ASSERT_STATUS(status, "Deallocation failed");

    PoolStats stats1 = pool.get_statistics();
    uint64_t cache_hits_before = stats1.cache_hits;

    // Allocate again - should reuse the same pool entry
    void *ptr2 = pool.allocate(test_size);
    ASSERT_NE(ptr2, nullptr, "Second allocation failed");

    PoolStats stats2 = pool.get_statistics();
    uint64_t cache_hits_after = stats2.cache_hits;

    LOG_INFO("Cache hits before: " << cache_hits_before);
    LOG_INFO("Cache hits after: " << cache_hits_after);
    LOG_INFO("Hit rate: " << std::fixed << std::setprecision(1)
                          << (stats2.get_hit_rate() * 100) << "%");

    ASSERT_TRUE(cache_hits_after > cache_hits_before,
                "Expected cache hit on reuse");

    pool.deallocate(ptr2);

    LOG_PASS("Pool Reuse");
    return true;

  } catch (const std::exception &e) {
    LOG_FAIL("test_pool_reuse", std::string("Exception: ") + e.what());
    return false;
  } catch (...) {
    LOG_FAIL("test_pool_reuse", "Unknown exception");
    return false;
  }
}

bool test_pool_growth() {
  LOG_TEST("Pool Growth When Exhausted");

  MemoryPoolManager pool;
  const size_t test_size = MemoryPoolManager::SIZE_16KB;
  const int num_allocations = 100;

  std::vector<void *> ptrs;
  PoolStats initial_stats = pool.get_statistics();
  uint64_t initial_grows = initial_stats.pool_grows;

  LOG_INFO("Allocating " << num_allocations << " blocks of " << test_size / 1024
                         << " KB");

  for (int i = 0; i < num_allocations; i++) {
    void *ptr = pool.allocate(test_size);
    ASSERT_NE(ptr, nullptr, "Allocation " << i << " failed");
    ptrs.push_back(ptr);
  }

  PoolStats final_stats = pool.get_statistics();
  uint64_t final_grows = final_stats.pool_grows;

  LOG_INFO("Pool grows: " << (final_grows - initial_grows));
  LOG_INFO("Total pool capacity: "
           << final_stats.total_pool_capacity / (1024 * 1024) << " MB");
  LOG_INFO("Current memory usage: "
           << final_stats.current_memory_usage / (1024 * 1024) << " MB");
  LOG_INFO("Peak memory usage: "
           << final_stats.peak_memory_usage / (1024 * 1024) << " MB");

  ASSERT_TRUE(final_grows > initial_grows, "Expected pool to grow");

  // Cleanup
  for (void *ptr : ptrs) {
    pool.deallocate(ptr);
  }

  LOG_PASS("Pool Growth");
  return true;
}

bool test_concurrent_allocations() {
  LOG_TEST("Thread-Safe Concurrent Allocations");

  MemoryPoolManager pool;
  const int num_threads = 4;
  const int allocations_per_thread = 50;
  const size_t test_size = MemoryPoolManager::SIZE_64KB;

  LOG_INFO("Testing with " << num_threads << " threads");
  LOG_INFO("Each allocating " << allocations_per_thread << " blocks");

  std::vector<std::thread> threads;
  std::vector<std::vector<void *>> thread_ptrs(num_threads);
  std::atomic<int> success_count{0};

  auto worker = [&](int thread_id) {
    for (int i = 0; i < allocations_per_thread; i++) {
      void *ptr = pool.allocate(test_size);
      if (ptr != nullptr) {
        thread_ptrs[thread_id].push_back(ptr);
        success_count++;
      }
      // Small delay to increase contention
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  };

  // Launch threads
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker, i);
  }

  // Wait for completion
  for (auto &t : threads) {
    t.join();
  }

  LOG_INFO("Successful allocations: " << success_count);
  ASSERT_EQ(success_count, num_threads * allocations_per_thread,
            "Some allocations failed");

  // Deallocate from all threads
  for (int i = 0; i < num_threads; i++) {
    for (void *ptr : thread_ptrs[i]) {
      pool.deallocate(ptr);
    }
  }

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Total allocations: " << stats.total_allocations);
  LOG_INFO("Total deallocations: " << stats.total_deallocations);

  LOG_PASS("Concurrent Allocations");
  return true;
}

bool test_stream_based_allocation() {
  LOG_TEST("Stream-Based Async Allocations");

  MemoryPoolManager pool;
  const int num_streams = 4;
  const size_t test_size = MemoryPoolManager::SIZE_256KB;

  std::vector<cudaStream_t> streams(num_streams);
  std::vector<void *> ptrs(num_streams);

  // Create streams
  for (int i = 0; i < num_streams; i++) {
    cudaError_t err = cudaStreamCreate(&streams[i]);
    ASSERT_EQ(err, cudaSuccess, "cudaStreamCreate failed");
  }

  LOG_INFO("Allocating with " << num_streams << " different streams");

  // Allocate on different streams
  for (int i = 0; i < num_streams; i++) {
    ptrs[i] = pool.allocate_async(test_size, streams[i]);
    ASSERT_NE(ptrs[i], nullptr,
              "Async allocation on stream " << i << " failed");
    LOG_INFO("Stream " << i << ": allocated " << test_size / 1024 << " KB");
  }

  // Synchronize all streams
  for (int i = 0; i < num_streams; i++) {
    cudaError_t err = cudaStreamSynchronize(streams[i]);
    ASSERT_EQ(err, cudaSuccess, "cudaStreamSynchronize failed");
  }

  // Deallocate
  for (int i = 0; i < num_streams; i++) {
    pool.deallocate(ptrs[i]);
    cudaError_t err = cudaStreamDestroy(streams[i]);
    ASSERT_EQ(err, cudaSuccess, "cudaStreamDestroy failed");
  }

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Total async allocations: " << stats.total_allocations);

  LOG_PASS("Stream-Based Allocations");
  return true;
}

// ============================================================================
// TEST SUITE 2: Performance Tests
// ============================================================================

bool test_allocation_overhead() {
  LOG_TEST("Allocation Overhead Comparison");

  const int num_iterations = 1000;
  const size_t test_size = MemoryPoolManager::SIZE_1MB;

  LOG_INFO("Comparing " << num_iterations << " allocations/deallocations");
  LOG_INFO("Block size: " << test_size / 1024 << " KB");

  // Test with pool
  MemoryPoolManager pool;
  auto pool_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; i++) {
    void *ptr = pool.allocate(test_size);
    pool.deallocate(ptr);
  }

  auto pool_end = std::chrono::high_resolution_clock::now();
  double pool_time =
      std::chrono::duration<double, std::milli>(pool_end - pool_start).count();

  // Test with direct cudaMalloc
  auto direct_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; i++) {
    void *ptr;
    cudaError_t err = cuda_zstd::safe_cuda_malloc(&ptr, test_size);
    ASSERT_EQ(err, cudaSuccess, "cudaMalloc failed");
    err = cudaFree(ptr);
    ASSERT_EQ(err, cudaSuccess, "cudaFree failed");
  }

  auto direct_end = std::chrono::high_resolution_clock::now();
  double direct_time =
      std::chrono::duration<double, std::milli>(direct_end - direct_start)
          .count();

  LOG_INFO("Pool time: " << std::fixed << std::setprecision(2) << pool_time
                         << " ms");
  LOG_INFO("Direct cudaMalloc time: " << std::fixed << std::setprecision(2)
                                      << direct_time << " ms");
  LOG_INFO("Speedup: " << std::fixed << std::setprecision(2)
                       << (direct_time / pool_time) << "x");

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Cache hit rate: " << std::fixed << std::setprecision(1)
                              << (stats.get_hit_rate() * 100) << "%");

  // Pool should be significantly faster due to reuse
  ASSERT_TRUE(pool_time < direct_time,
              "Pool should be faster than direct allocation");

  LOG_PASS("Allocation Overhead");
  return true;
}

bool test_cache_hit_rate() {
  LOG_TEST("Cache Hit Rate Measurement");

  MemoryPoolManager pool;
  const size_t test_size = MemoryPoolManager::SIZE_64KB;
  const int num_cycles = 100;

  LOG_INFO("Running " << num_cycles << " allocation/deallocation cycles");

  // Warm up the pool
  std::vector<void *> warmup_ptrs;
  for (int i = 0; i < 10; i++) {
    warmup_ptrs.push_back(pool.allocate(test_size));
  }
  for (void *ptr : warmup_ptrs) {
    pool.deallocate(ptr);
  }

  pool.reset_statistics();

  // Run test cycles
  for (int i = 0; i < num_cycles; i++) {
    void *ptr = pool.allocate(test_size);
    pool.deallocate(ptr);
  }

  PoolStats stats = pool.get_statistics();
  double hit_rate = stats.get_hit_rate() * 100;

  LOG_INFO("Cache hits: " << stats.cache_hits);
  LOG_INFO("Cache misses: " << stats.cache_misses);
  LOG_INFO("Hit rate: " << std::fixed << std::setprecision(1) << hit_rate
                        << "%");

  // After warmup, hit rate should be very high
  ASSERT_TRUE(hit_rate > 90.0, "Hit rate should exceed 90% after warmup");

  LOG_PASS("Cache Hit Rate");
  return true;
}

bool test_peak_memory_tracking() {
  LOG_TEST("Peak Memory Usage Tracking");

  MemoryPoolManager pool;
  std::vector<void *> ptrs;

  // Allocate increasing amounts
  std::vector<size_t> sizes = {
      MemoryPoolManager::SIZE_4KB, MemoryPoolManager::SIZE_16KB,
      MemoryPoolManager::SIZE_64KB, MemoryPoolManager::SIZE_256KB,
      MemoryPoolManager::SIZE_1MB};

  size_t expected_peak = 0;
  for (size_t size : sizes) {
    void *ptr = pool.allocate(size);
    ptrs.push_back(ptr);
    expected_peak += size;

    PoolStats stats = pool.get_statistics();
    LOG_INFO("Allocated " << size / 1024 << " KB, current: "
                          << stats.current_memory_usage / 1024 << " KB, peak: "
                          << stats.peak_memory_usage / 1024 << " KB");
  }

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Final peak memory: " << stats.peak_memory_usage / (1024 * 1024)
                                 << " MB");

  ASSERT_TRUE(stats.peak_memory_usage >= stats.current_memory_usage,
              "Peak should be >= current");

  // Deallocate half
  for (size_t i = 0; i < ptrs.size() / 2; i++) {
    pool.deallocate(ptrs[i]);
  }

  PoolStats stats2 = pool.get_statistics();
  LOG_INFO("After partial deallocation - current: "
           << stats2.current_memory_usage / 1024
           << " KB, peak: " << stats2.peak_memory_usage / 1024 << " KB");

  ASSERT_EQ(stats2.peak_memory_usage, stats.peak_memory_usage,
            "Peak should remain the same after deallocation");

  // Cleanup remaining
  for (size_t i = ptrs.size() / 2; i < ptrs.size(); i++) {
    pool.deallocate(ptrs[i]);
  }

  LOG_PASS("Peak Memory Tracking");
  return true;
}

// ============================================================================
// TEST SUITE 3: Stress Tests
// ============================================================================

bool test_many_allocations() {
  LOG_TEST("Stress Test: 10000+ Allocations");

  MemoryPoolManager pool;
  const int num_allocations = 10000;
  std::vector<void *> ptrs;
  ptrs.reserve(num_allocations);

  LOG_INFO("Allocating " << num_allocations << " random-sized blocks");

  auto start = std::chrono::high_resolution_clock::now();

  // Random allocation pattern
  for (int i = 0; i < num_allocations; i++) {
    size_t size =
        MemoryPoolManager::SIZE_4KB * (1 << (i % 4)); // 4KB, 16KB, 64KB, 256KB
    void *ptr = pool.allocate(size);
    if (ptr != nullptr) {
      ptrs.push_back(ptr);
    }

    if ((i + 1) % 2000 == 0) {
      LOG_INFO("Progress: " << (i + 1) << "/" << num_allocations);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  LOG_INFO("Allocated " << ptrs.size() << " blocks in " << std::fixed
                        << std::setprecision(2) << elapsed << " ms");
  LOG_INFO("Average: " << std::fixed << std::setprecision(3)
                       << (elapsed / num_allocations) << " ms per allocation");

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Peak memory: " << stats.peak_memory_usage / (1024 * 1024) << " MB");
  LOG_INFO("Cache hit rate: " << std::fixed << std::setprecision(1)
                              << (stats.get_hit_rate() * 100) << "%");

  // Deallocate all
  for (void *ptr : ptrs) {
    pool.deallocate(ptr);
  }

  LOG_PASS("Many Allocations");
  return true;
}

bool test_random_allocation_pattern() {
  LOG_TEST("Random Allocation/Deallocation Pattern");

  MemoryPoolManager pool;
  const int num_operations = 5000;
  std::vector<void *> active_ptrs;

  LOG_INFO("Running " << num_operations << " random operations");

  // Seed for reproducibility
  srand(42);

  for (int i = 0; i < num_operations; i++) {
    bool allocate = (rand() % 2 == 0) || active_ptrs.empty();

    if (allocate) {
      size_t size = MemoryPoolManager::SIZE_4KB * (1 << (rand() % 6));
      void *ptr = pool.allocate(size);
      if (ptr != nullptr) {
        active_ptrs.push_back(ptr);
      }
    } else {
      // Deallocate random pointer
      int idx = rand() % active_ptrs.size();
      pool.deallocate(active_ptrs[idx]);
      active_ptrs.erase(active_ptrs.begin() + idx);
    }

    if ((i + 1) % 1000 == 0) {
      PoolStats stats = pool.get_statistics();
      LOG_INFO("After " << (i + 1) << " ops: " << active_ptrs.size()
                        << " active, hit rate: " << std::fixed
                        << std::setprecision(1) << (stats.get_hit_rate() * 100)
                        << "%");
    }
  }

  // Cleanup remaining
  for (void *ptr : active_ptrs) {
    pool.deallocate(ptr);
  }

  PoolStats stats = pool.get_statistics();
  LOG_INFO("Final stats - Allocations: " << stats.total_allocations
                                         << ", Deallocations: "
                                         << stats.total_deallocations);

  LOG_PASS("Random Allocation Pattern");
  return true;
}

bool test_memory_leak_detection() {
  LOG_TEST("Memory Leak Detection");

  MemoryPoolManager pool;

  PoolStats initial = pool.get_statistics();

  // Allocate and deallocate many times
  for (int cycle = 0; cycle < 10; cycle++) {
    std::vector<void *> ptrs;
    for (int i = 0; i < 100; i++) {
      ptrs.push_back(pool.allocate(MemoryPoolManager::SIZE_64KB));
    }
    for (void *ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  PoolStats final = pool.get_statistics();

  LOG_INFO("Initial current memory: " << initial.current_memory_usage / 1024
                                      << " KB");
  LOG_INFO("Final current memory: " << final.current_memory_usage / 1024
                                    << " KB");
  LOG_INFO("Allocations: " << final.total_allocations);
  LOG_INFO("Deallocations: " << final.total_deallocations);

  // All allocations should be deallocated
  ASSERT_EQ(final.total_allocations, final.total_deallocations,
            "Allocation/Deallocation mismatch indicates leak");

  LOG_PASS("Memory Leak Detection");
  return true;
}

bool test_out_of_memory_handling() {
  LOG_TEST("Out-of-Memory Handling");

  MemoryPoolManager pool;
  pool.set_max_pool_size(100 * 1024 * 1024); // 100MB limit

  LOG_INFO("Pool limit set to 100 MB");

  std::vector<void *> ptrs;
  const size_t large_size = 10 * 1024 * 1024; // 10MB blocks

  // Try to allocate beyond limit
  size_t total_allocated = 0;
  for (int i = 0; i < 20; i++) {
    void *ptr = pool.allocate(large_size);
    if (ptr != nullptr) {
      ptrs.push_back(ptr);
      total_allocated += large_size;
      LOG_INFO("Allocated block "
               << i << " (" << total_allocated / (1024 * 1024) << " MB total)");
      // Check pool statistics after each allocation
      PoolStats stats = pool.get_statistics();
      ASSERT_TRUE(stats.current_memory_usage <= 100 * 1024 * 1024,
                  "Exceeded pool memory limit");
    } else {
      LOG_INFO("Allocation " << i << " failed (expected - reached limit)");
      break;
    }
  }

  PoolStats stats = pool.get_statistics();
  ASSERT_TRUE(stats.current_memory_usage <= 100 * 1024 * 1024,
              "Pool usage exceeded limit after allocations");

  // Cleanup
  for (void *ptr : ptrs) {
    pool.deallocate(ptr);
  }

  LOG_PASS("Out-of-Memory Handling");
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << "\n";
  print_separator();
  std::cout << "CUDA ZSTD - Memory Pool Manager Test Suite" << std::endl;
  print_separator();
  std::cout << "\n";

  int passed = 0;
  int total = 0;

  SKIP_IF_NO_CUDA_RET(0);
  check_cuda_device();

  // Basic Functionality Tests
  print_separator();
  std::cout << "SUITE 1: Basic Functionality" << std::endl;
  print_separator();

  // Allow running a single test for debugging via env var TEST_NAME
  const char *debug_test = getenv("TEST_NAME");
  auto run_or_skip = [&](bool (*fn)(), const char *name) {
    total++;
    if (!debug_test || strcmp(debug_test, name) == 0) {
      if (fn())
        passed++;
    } else {
      // Skip test if debug set and doesn't match
      std::cout << "[SKIP] " << name << std::endl;
    }
  };

  run_or_skip(test_basic_allocation_deallocation,
              "test_basic_allocation_deallocation");
  run_or_skip(test_pool_reuse, "test_pool_reuse");
  run_or_skip(test_pool_growth, "test_pool_growth");
  run_or_skip(test_concurrent_allocations, "test_concurrent_allocations");
  run_or_skip(test_stream_based_allocation, "test_stream_based_allocation");

  // Performance Tests
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 2: Performance Tests" << std::endl;
  print_separator();

  total++;
  if (test_allocation_overhead())
    passed++;
  total++;
  if (test_cache_hit_rate())
    passed++;
  total++;
  if (test_peak_memory_tracking())
    passed++;

  // Stress Tests
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 3: Stress Tests" << std::endl;
  print_separator();

  total++;
  if (test_many_allocations())
    passed++;
  total++;
  if (test_random_allocation_pattern())
    passed++;
  total++;
  if (test_memory_leak_detection())
    passed++;
  total++;
  if (test_out_of_memory_handling())
    passed++;

  // Summary
  std::cout << "\n";
  print_separator();
  std::cout << "TEST RESULTS" << std::endl;
  print_separator();
  std::cout << "Passed: " << passed << "/" << total << std::endl;
  std::cout << "Failed: " << (total - passed) << "/" << total << std::endl;

  if (passed == total) {
    std::cout << "\n✓ ALL TESTS PASSED" << std::endl;
  } else {
    std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
  }
  print_separator();
  std::cout << "\n";

  return (passed == total) ? 0 : 1;
}
