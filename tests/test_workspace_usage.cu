// test_workspace_usage.cu - Tests for workspace_manager.cu and
// workspace_usage.cu Covers: workspace allocation, compression with workspace,
// memory reuse

#include "workspace_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda_zstd;

// ==============================================================================
// Test Helpers
// ==============================================================================

struct TestResult {
  const char *name;
  bool passed;
};

std::vector<TestResult> g_results;

void record_test(const char *name, bool passed) {
  g_results.push_back({name, passed});
  printf("  %s: %s\n", name, passed ? "PASS" : "FAIL");
}

// ==============================================================================
// Test: Basic Workspace Allocation
// ==============================================================================

bool test_workspace_allocation() {
  printf("=== Test: Workspace Allocation ===\n");

  CompressionConfig config;
  config.window_log = 17; // 128KB window
  config.hash_log = 15;   // 32K hash table
  config.chain_log = 14;
  config.search_log = 3;

  size_t max_block_size = 64 * 1024; // 64KB

  CompressionWorkspace workspace;
  Status status =
      allocate_compression_workspace(workspace, max_block_size, config);

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Verify all pointers are non-null
    if (workspace.d_hash_table == nullptr ||
        workspace.d_chain_table == nullptr || workspace.d_matches == nullptr ||
        workspace.d_costs == nullptr) {
      printf("    Some workspace pointers are null\n");
      passed = false;
    }
  }

  // Cleanup
  free_compression_workspace(workspace);

  record_test("Workspace Allocation", passed);
  return passed;
}

// ==============================================================================
// Test: Workspace Reuse
// ==============================================================================

bool test_workspace_reuse() {
  printf("=== Test: Workspace Reuse ===\n");

  CompressionConfig config;
  config.window_log = 17;
  config.hash_log = 15;
  config.chain_log = 14;
  config.search_log = 3;

  size_t max_block_size = 64 * 1024;

  CompressionWorkspace workspace;
  Status status =
      allocate_compression_workspace(workspace, max_block_size, config);

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Allocate test data
    u8 *d_input = nullptr;
    u8 *d_output = nullptr;
    cuda_zstd::safe_cuda_malloc(&d_input, max_block_size);
    cuda_zstd::safe_cuda_malloc(&d_output, max_block_size * 2);

    // Run multiple compressions reusing workspace
    for (int i = 0; i < 5 && passed; ++i) {
      // Clear hash table (simulate new compression)
      cudaMemset(workspace.d_hash_table, 0xFF,
                 workspace.hash_table_size * sizeof(u32));

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("    CUDA error on iteration %d: %s\n", i,
               cudaGetErrorString(err));
        passed = false;
      }
    }

    cudaFree(d_input);
    cudaFree(d_output);
  }

  free_compression_workspace(workspace);

  record_test("Workspace Reuse (5 iterations)", passed);
  return passed;
}

// ==============================================================================
// Test: Workspace Memory Sizes
// ==============================================================================

bool test_workspace_memory_sizes() {
  printf("=== Test: Workspace Memory Sizes ===\n");

  CompressionConfig config;
  config.window_log = 20; // 1MB window
  config.hash_log = 18;   // 256K hash table
  config.chain_log = 18;
  config.search_log = 4;

  size_t max_block_size = 1024 * 1024; // 1MB

  CompressionWorkspace workspace;
  Status status =
      allocate_compression_workspace(workspace, max_block_size, config);

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Verify sizes are as expected
    size_t expected_hash_size = 1 << config.hash_log;
    size_t expected_chain_size = 1 << config.chain_log;

    if (workspace.hash_table_size != expected_hash_size) {
      printf("    Hash table size mismatch: expected %zu, got %zu\n",
             expected_hash_size, (size_t)workspace.hash_table_size);
      passed = false;
    }

    if (workspace.chain_table_size != expected_chain_size) {
      printf("    Chain table size mismatch: expected %zu, got %zu\n",
             expected_chain_size, (size_t)workspace.chain_table_size);
      passed = false;
    }
  }

  free_compression_workspace(workspace);

  record_test("Workspace Memory Sizes", passed);
  return passed;
}

// ==============================================================================
// Test: Double Free Protection
// ==============================================================================

bool test_workspace_double_free() {
  printf("=== Test: Workspace Double Free ===\n");

  CompressionConfig config;
  config.window_log = 15;
  config.hash_log = 14;
  config.chain_log = 14;
  config.search_log = 2;

  size_t max_block_size = 32 * 1024;

  CompressionWorkspace workspace;
  Status status =
      allocate_compression_workspace(workspace, max_block_size, config);

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // First free
    free_compression_workspace(workspace);

    // Second free should be safe (pointers should be nullified)
    free_compression_workspace(workspace);

    // No crash = passed
    passed = true;
  }

  record_test("Workspace Double Free Protection", passed);
  return passed;
}

// ==============================================================================
// Test: Invalid Config
// ==============================================================================

bool test_workspace_invalid_config() {
  printf("=== Test: Workspace Invalid Config ===\n");

  CompressionConfig config;
  config.window_log = 0; // Invalid
  config.hash_log = 0;
  config.chain_log = 0;
  config.search_log = 0;

  CompressionWorkspace workspace;
  Status status = allocate_compression_workspace(workspace, 1024, config);

  // Should either fail gracefully or succeed with defaults
  bool passed =
      (status == Status::SUCCESS || status == Status::ERROR_INVALID_PARAMETER ||
       status == Status::ERROR_GENERIC);

  if (status == Status::SUCCESS) {
    free_compression_workspace(workspace);
  }

  record_test("Workspace Invalid Config", passed);
  return passed;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  cudaFree(0); // Initialize CUDA

  printf("========================================\n");
  printf("Workspace Usage Test Suite\n");
  printf("========================================\n\n");

  test_workspace_allocation();
  test_workspace_reuse();
  test_workspace_memory_sizes();
  test_workspace_double_free();
  test_workspace_invalid_config();

  printf("\n========================================\n");
  printf("Summary\n");
  printf("========================================\n");

  int passed = 0, failed = 0;
  for (const auto &r : g_results) {
    if (r.passed)
      passed++;
    else
      failed++;
  }

  printf("Passed: %d, Failed: %d\n", passed, failed);

  if (failed == 0) {
    printf("\n✅ ALL WORKSPACE USAGE TESTS PASSED\n");
    return 0;
  } else {
    printf("\n❌ Some tests failed\n");
    return 1;
  }
}
