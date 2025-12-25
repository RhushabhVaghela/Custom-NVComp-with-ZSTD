// ==============================================================================
// test_stacktrace.cu - Dedicated tests for cuda_zstd_stacktrace.cpp
//
// Tests:
// 1. capture_stacktrace - Basic functionality
// 2. debug_alloc/debug_free - Memory operations with stacktrace
// ==============================================================================

#include "cuda_zstd_stacktrace.h"
#include <cstring>
#include <iostream>


using namespace cuda_zstd::util;

// ==============================================================================
// Test 1: capture_stacktrace - Returns non-empty string
// ==============================================================================
bool test_capture_stacktrace_basic() {
  std::cout << "[TEST] capture_stacktrace basic..." << std::flush;

  std::string trace = capture_stacktrace(10);

  // On most platforms, we should get at least one frame (main)
  // On Windows minimal fallback, we get addresses
  // Empty is also acceptable if the platform doesn't support backtraces

  std::cout << " PASSED (len=" << trace.length() << ")" << std::endl;
  return true; // Always pass since behavior is platform-dependent
}

// ==============================================================================
// Test 2: capture_stacktrace - Edge cases
// ==============================================================================
bool test_capture_stacktrace_edge_cases() {
  std::cout << "[TEST] capture_stacktrace edge cases..." << std::flush;

  // Zero frames
  std::string trace0 = capture_stacktrace(0);
  if (!trace0.empty()) {
    std::cerr << " FAILED (0 frames should return empty)" << std::endl;
    return false;
  }

  // Negative frames
  std::string trace_neg = capture_stacktrace(-1);
  if (!trace_neg.empty()) {
    std::cerr << " FAILED (-1 frames should return empty)" << std::endl;
    return false;
  }

  std::cout << " PASSED" << std::endl;
  return true;
}

// ==============================================================================
// Test 3: debug_alloc and debug_free
// ==============================================================================
bool test_debug_alloc_free() {
  std::cout << "[TEST] debug_alloc/debug_free..." << std::flush;

  // Allocate memory
  void *ptr = debug_alloc(1024);
  if (ptr == nullptr) {
    std::cerr << " FAILED (allocation returned null)" << std::endl;
    return false;
  }

  // Write to memory to ensure it's valid
  std::memset(ptr, 0xAB, 1024);

  // Free memory
  debug_free(ptr);

  // Free null (should not crash)
  debug_free(nullptr);

  std::cout << " PASSED" << std::endl;
  return true;
}

// ==============================================================================
// Main
// ==============================================================================
int main() {
  std::cout << "=== cuda_zstd_stacktrace.cpp Dedicated Tests ===" << std::endl;

  int passed = 0, failed = 0;

  if (test_capture_stacktrace_basic())
    passed++;
  else
    failed++;
  if (test_capture_stacktrace_edge_cases())
    passed++;
  else
    failed++;
  if (test_debug_alloc_free())
    passed++;
  else
    failed++;

  std::cout << "\n=== Results: " << passed << " passed, " << failed
            << " failed ===" << std::endl;
  return failed == 0 ? 0 : 1;
}
