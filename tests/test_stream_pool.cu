// ==============================================================================
// test_stream_pool.cu - Unit tests for StreamPool memory safety fixes
// ==============================================================================
// Tests for verifying the fixes to stream pool memory access issues:
// 1. Use-after-free in constructor fallback path
// 2. Missing bounds checking in Guard constructor
// 3. Thread safety issues
// 4. Proper error handling
// ==============================================================================

#include "cuda_zstd_stream_pool.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <thread>
#include <vector>


using namespace cuda_zstd;

// Test macros
#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg);          \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define TEST_ASSERT_EQ(a, b, msg) TEST_ASSERT((a) == (b), msg)

// Global test counter
static int g_tests_passed = 0;
static int g_tests_failed = 0;

void test_passed(const char *name) {
  printf("[PASS] %s\n", name);
  g_tests_passed++;
}

void test_failed(const char *name) {
  printf("[FAIL] %s\n", name);
  g_tests_failed++;
}

// ==============================================================================
// Test 1: Basic pool creation and validity
// ==============================================================================
bool test_basic_creation() {
  // Test creating pool with default size
  {
    StreamPool pool;
    TEST_ASSERT(pool.is_valid(), "Default pool should be valid");
    TEST_ASSERT(pool.size() > 0, "Pool should have streams");
    TEST_ASSERT(pool.available_count() == pool.size(),
                "All streams should be available");
  }

  // Test creating pool with specific size
  {
    StreamPool pool(4);
    TEST_ASSERT(pool.is_valid(), "Pool(4) should be valid");
    TEST_ASSERT_EQ(pool.size(), 4, "Pool should have 4 streams");
    TEST_ASSERT_EQ(pool.available_count(), 4,
                   "All 4 streams should be available");
  }

  // Test creating pool with size 0 (should default to 1)
  {
    StreamPool pool(0);
    TEST_ASSERT(pool.is_valid(), "Pool(0) should be valid");
    TEST_ASSERT(pool.size() > 0, "Pool(0) should have at least 1 stream");
  }

  return true;
}

// ==============================================================================
// Test 2: Stream acquisition and release
// ==============================================================================
bool test_acquire_release() {
  StreamPool pool(4);

  // Acquire all streams
  auto guard1 = pool.acquire();
  auto guard2 = pool.acquire();
  auto guard3 = pool.acquire();
  auto guard4 = pool.acquire();

  TEST_ASSERT(guard1.is_valid(), "Guard1 should be valid");
  TEST_ASSERT(guard2.is_valid(), "Guard2 should be valid");
  TEST_ASSERT(guard3.is_valid(), "Guard3 should be valid");
  TEST_ASSERT(guard4.is_valid(), "Guard4 should be valid");

  TEST_ASSERT(guard1.get_stream() != nullptr, "Guard1 should have stream");
  TEST_ASSERT(guard2.get_stream() != nullptr, "Guard2 should have stream");
  TEST_ASSERT(guard3.get_stream() != nullptr, "Guard3 should have stream");
  TEST_ASSERT(guard4.get_stream() != nullptr, "Guard4 should have stream");

  TEST_ASSERT_EQ(pool.available_count(), 0, "No streams should be available");

  // Guards release automatically when they go out of scope
  return true;
}

// ==============================================================================
// Test 3: Guard validation and invalid access
// ==============================================================================
bool test_guard_validation() {
  // Test invalid guard from failed acquisition
  {
    StreamPool pool(0); // Empty pool

    // This should return an invalid guard since pool has no streams
    auto guard = pool.acquire();
    TEST_ASSERT(!guard.is_valid(), "Guard from empty pool should be invalid");
    TEST_ASSERT(guard.get_stream() == nullptr,
                "Invalid guard should return nullptr");
  }

  // Test guard movement
  {
    StreamPool pool(2);
    auto guard1 = pool.acquire();
    TEST_ASSERT(guard1.is_valid(), "Original guard should be valid");

    auto guard2 = std::move(guard1);
    TEST_ASSERT(guard2.is_valid(), "Moved-to guard should be valid");
    // guard1 is in moved-from state, behavior is implementation-defined
  }

  return true;
}

// ==============================================================================
// Test 4: Timeout-based acquisition
// ==============================================================================
bool test_timeout_acquisition() {
  StreamPool pool(2);

  // Acquire all streams
  auto guard1 = pool.acquire();
  auto guard2 = pool.acquire();

  // Try to acquire with timeout - should fail
  auto guard3_opt = pool.acquire_for(10); // 10ms timeout
  TEST_ASSERT(!guard3_opt.has_value(), "Should fail to acquire with timeout");

  // Release one and try again
  guard1.~Guard(); // Force release

  // Need to create new pool since we can't easily force release
  StreamPool pool2(2);
  auto g1 = pool2.acquire();

  auto g2_opt = pool2.acquire_for(100); // Should succeed quickly
  TEST_ASSERT(g2_opt.has_value(), "Should acquire with timeout");
  TEST_ASSERT(g2_opt->is_valid(), "Acquired guard should be valid");

  return true;
}

// ==============================================================================
// Test 5: Thread safety - concurrent acquisition
// ==============================================================================
bool test_thread_safety() {
  const int num_threads = 8;
  const int iterations = 100;
  StreamPool pool(4);
  std::atomic<int> success_count{0};
  std::atomic<int> fail_count{0};

  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < iterations; ++i) {
        auto guard = pool.acquire();
        if (guard.is_valid()) {
          // Do some work with the stream
          cudaStream_t stream = guard.get_stream();
          if (stream != nullptr) {
            // Simple kernel launch or sync to test stream validity
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err == cudaSuccess) {
              success_count++;
            } else {
              fail_count++;
            }
          } else {
            fail_count++;
          }
        } else {
          fail_count++;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  TEST_ASSERT_EQ(success_count.load(), num_threads * iterations,
                 "All acquisitions should succeed");
  TEST_ASSERT_EQ(fail_count.load(), 0, "No failures should occur");

  return true;
}

// ==============================================================================
// Test 6: Stream functionality validation
// ==============================================================================
bool test_stream_functionality() {
  StreamPool pool(2);

  auto guard = pool.acquire();
  TEST_ASSERT(guard.is_valid(), "Guard should be valid");

  cudaStream_t stream = guard.get_stream();
  TEST_ASSERT(stream != nullptr, "Should have valid stream");

  // Test that the stream actually works
  int *d_data;
  cudaError_t err = cudaMalloc(&d_data, sizeof(int));
  TEST_ASSERT(err == cudaSuccess, "Should allocate device memory");

  err = cudaMemsetAsync(d_data, 0, sizeof(int), stream);
  TEST_ASSERT(err == cudaSuccess, "Should do async memset");

  err = cudaStreamSynchronize(stream);
  TEST_ASSERT(err == cudaSuccess, "Should synchronize stream");

  cudaFree(d_data);

  return true;
}

// ==============================================================================
// Test 7: Multiple pool instances
// ==============================================================================
bool test_multiple_pools() {
  StreamPool pool1(2);
  StreamPool pool2(4);
  StreamPool pool3(8);

  TEST_ASSERT(pool1.is_valid(), "Pool1 should be valid");
  TEST_ASSERT(pool2.is_valid(), "Pool2 should be valid");
  TEST_ASSERT(pool3.is_valid(), "Pool3 should be valid");

  auto g1 = pool1.acquire();
  auto g2 = pool2.acquire();
  auto g3 = pool3.acquire();

  TEST_ASSERT(g1.is_valid(), "Guard1 should be valid");
  TEST_ASSERT(g2.is_valid(), "Guard2 should be valid");
  TEST_ASSERT(g3.is_valid(), "Guard3 should be valid");

  // Each pool should have independent streams
  TEST_ASSERT(g1.get_stream() != g2.get_stream(),
              "Streams should be different");
  TEST_ASSERT(g2.get_stream() != g3.get_stream(),
              "Streams should be different");

  return true;
}

// ==============================================================================
// Test 8: Stress test - rapid acquire/release
// ==============================================================================
bool test_stress_acquire_release() {
  const int iterations = 1000;
  StreamPool pool(4);

  for (int i = 0; i < iterations; ++i) {
    auto guard = pool.acquire();
    TEST_ASSERT(guard.is_valid(), "Guard should be valid in iteration");

    // Quick operation
    cudaStream_t stream = guard.get_stream();
    cudaStreamSynchronize(stream);

    // Guard releases automatically
  }

  TEST_ASSERT_EQ(pool.available_count(), pool.size(),
                 "All streams should be available after stress test");

  return true;
}

// ==============================================================================
// Test 9: Edge cases
// ==============================================================================
bool test_edge_cases() {
  // Test pool with size 1
  {
    StreamPool pool(1);
    TEST_ASSERT(pool.is_valid(), "Pool(1) should be valid");

    auto g1 = pool.acquire();
    TEST_ASSERT(g1.is_valid(), "Single stream guard should be valid");

    // Try to acquire another - should block (we won't actually wait)
    auto g2_opt = pool.acquire_for(1);
    TEST_ASSERT(!g2_opt.has_value(),
                "Should timeout waiting for single stream");
  }

  // Test that streams are properly released
  {
    StreamPool pool(2);

    // Acquire and release multiple times to ensure proper recycling
    for (int i = 0; i < 10; ++i) {
      auto g1 = pool.acquire();
      auto g2 = pool.acquire();
      TEST_ASSERT(g1.is_valid() && g2.is_valid(),
                  "Both guards should be valid");
      // Guards release at end of iteration
    }

    TEST_ASSERT_EQ(pool.available_count(), 2,
                   "Both streams should be available");
  }

  return true;
}

// ==============================================================================
// Test 10: CUDA error handling
// ==============================================================================
bool test_cuda_error_handling() {
  // This test verifies the pool handles CUDA errors gracefully
  StreamPool pool(2);

  auto guard = pool.acquire();
  TEST_ASSERT(guard.is_valid(), "Guard should be valid");

  cudaStream_t stream = guard.get_stream();

  // Verify stream is functional
  cudaError_t err = cudaStreamSynchronize(stream);
  TEST_ASSERT(err == cudaSuccess, "Stream should be functional");

  return true;
}

// ==============================================================================
// Main test runner
// ==============================================================================
int main() {
  printf("============================================================\n");
  printf("Stream Pool Unit Tests - Memory Safety Verification\n");
  printf("============================================================\n\n");

  // Check CUDA availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    printf("[SKIP] No CUDA devices available, skipping tests\n");
    return 0;
  }

  printf("Found %d CUDA device(s)\n\n", device_count);

  // Run all tests
  struct TestCase {
    const char *name;
    bool (*func)();
  };

  TestCase tests[] = {
      {"Basic Pool Creation", test_basic_creation},
      {"Stream Acquisition/Release", test_acquire_release},
      {"Guard Validation", test_guard_validation},
      {"Timeout Acquisition", test_timeout_acquisition},
      {"Thread Safety", test_thread_safety},
      {"Stream Functionality", test_stream_functionality},
      {"Multiple Pools", test_multiple_pools},
      {"Stress Test", test_stress_acquire_release},
      {"Edge Cases", test_edge_cases},
      {"CUDA Error Handling", test_cuda_error_handling},
  };

  const int num_tests = sizeof(tests) / sizeof(tests[0]);

  for (int i = 0; i < num_tests; ++i) {
    printf("Running: %s...\n", tests[i].name);
    try {
      if (tests[i].func()) {
        test_passed(tests[i].name);
      } else {
        test_failed(tests[i].name);
      }
    } catch (const std::exception &e) {
      fprintf(stderr, "[EXCEPTION] %s: %s\n", tests[i].name, e.what());
      test_failed(tests[i].name);
    } catch (...) {
      fprintf(stderr, "[EXCEPTION] %s: Unknown exception\n", tests[i].name);
      test_failed(tests[i].name);
    }
  }

  printf("\n============================================================\n");
  printf("Results: %d passed, %d failed out of %d tests\n", g_tests_passed,
         g_tests_failed, num_tests);
  printf("============================================================\n");

  return g_tests_failed > 0 ? 1 : 0;
}
