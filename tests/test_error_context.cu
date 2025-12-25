// ==============================================================================
// test_error_context.cu - Dedicated tests for error_context.cpp
//
// Tests:
// 1. get_error_mutex - Thread safety of mutex retrieval
// 2. last_error - ErrorContext operations
// ==============================================================================

#include "error_context.h"
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using namespace cuda_zstd::error_handling;

// ==============================================================================
// Test 1: get_error_mutex - Concurrent access
// ==============================================================================
bool test_get_error_mutex_thread_safety() {
  std::cout << "[TEST] get_error_mutex thread safety..." << std::flush;

  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;

  // Spawn multiple threads to access the mutex concurrently
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 100; ++j) {
        std::mutex &m = get_error_mutex();
        m.lock();
        // Critical section (simulated work)
        volatile int x = 0;
        for (int k = 0; k < 100; ++k)
          x++;
        m.unlock();
      }
      success_count++;
    });
  }

  for (auto &t : threads)
    t.join();

  if (success_count == 8) {
    std::cout << " PASSED" << std::endl;
    return true;
  } else {
    std::cerr << " FAILED (only " << success_count << " threads completed)"
              << std::endl;
    return false;
  }
}

// ==============================================================================
// Test 2: ErrorContext - Set/Get operations
// ==============================================================================
bool test_error_context_operations() {
  std::cout << "[TEST] ErrorContext operations..." << std::flush;

  // Reset last_error
  last_error.status = cuda_zstd::Status::SUCCESS;
  last_error.message = nullptr;
  last_error.file = nullptr;
  last_error.line = 0;

  // Set an error (use string literals since fields are const char*)
  last_error.status = cuda_zstd::Status::ERROR_BUFFER_TOO_SMALL;
  last_error.message = "Test error message";
  last_error.file = "test_file.cu";
  last_error.line = 42;

  // Verify
  bool pass = true;
  if (last_error.status != cuda_zstd::Status::ERROR_BUFFER_TOO_SMALL) {
    std::cerr << " FAILED (status mismatch)" << std::endl;
    pass = false;
  }
  if (last_error.line != 42) {
    std::cerr << " FAILED (line mismatch)" << std::endl;
    pass = false;
  }

  if (pass)
    std::cout << " PASSED" << std::endl;
  return pass;
}

// ==============================================================================
// Main
// ==============================================================================
int main() {
  std::cout << "=== error_context.cpp Dedicated Tests ===" << std::endl;

  int passed = 0, failed = 0;

  if (test_get_error_mutex_thread_safety())
    passed++;
  else
    failed++;
  if (test_error_context_operations())
    passed++;
  else
    failed++;

  std::cout << "\n=== Results: " << passed << " passed, " << failed
            << " failed ===" << std::endl;
  return failed == 0 ? 0 : 1;
}
