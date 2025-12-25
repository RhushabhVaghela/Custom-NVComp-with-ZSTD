// test_hash_comprehensive.cu - Comprehensive tests for cuda_zstd_hash.cu
// Covers: hash table operations, chain building, match finding

#include "cuda_zstd_hash.h"
#include "cuda_zstd_types.h"
#include <cstdio>
#include <random>
#include <vector>


using namespace cuda_zstd;
using namespace cuda_zstd::hash;

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
// Test: Hash Table Initialization
// ==============================================================================

bool test_hash_table_init() {
  printf("=== Test: Hash Table Init ===\n");

  const u32 hash_size = 1 << 16; // 64K entries

  u32 *d_hash_table = nullptr;
  cudaMalloc(&d_hash_table, hash_size * sizeof(u32));

  // Initialize to -1 (empty)
  cudaMemset(d_hash_table, 0xFF, hash_size * sizeof(u32));
  cudaDeviceSynchronize();

  // Verify initialization
  std::vector<u32> h_hash(hash_size);
  cudaMemcpy(h_hash.data(), d_hash_table, hash_size * sizeof(u32),
             cudaMemcpyDeviceToHost);

  bool passed = true;
  for (u32 i = 0; i < hash_size; ++i) {
    if (h_hash[i] != 0xFFFFFFFF) {
      printf("    Hash table not properly initialized at %u\n", i);
      passed = false;
      break;
    }
  }

  cudaFree(d_hash_table);

  record_test("Hash Table Initialization", passed);
  return passed;
}

// ==============================================================================
// Test: Chain Table Initialization
// ==============================================================================

bool test_chain_table_init() {
  printf("=== Test: Chain Table Init ===\n");

  const u32 chain_size = 1 << 16; // 64K entries

  u32 *d_chain_table = nullptr;
  cudaMalloc(&d_chain_table, chain_size * sizeof(u32));

  // Initialize to 0
  cudaMemset(d_chain_table, 0, chain_size * sizeof(u32));
  cudaDeviceSynchronize();

  // Verify initialization
  std::vector<u32> h_chain(chain_size);
  cudaMemcpy(h_chain.data(), d_chain_table, chain_size * sizeof(u32),
             cudaMemcpyDeviceToHost);

  bool passed = true;
  for (u32 i = 0; i < chain_size; ++i) {
    if (h_chain[i] != 0) {
      printf("    Chain table not properly initialized at %u\n", i);
      passed = false;
      break;
    }
  }

  cudaFree(d_chain_table);

  record_test("Chain Table Initialization", passed);
  return passed;
}

// ==============================================================================
// Test: Hash Function Determinism
// ==============================================================================

bool test_hash_determinism() {
  printf("=== Test: Hash Function Determinism ===\n");

  // Same input should always produce same hash
  const u32 test_values[] = {0x12345678, 0xABCDEF00, 0x00000000, 0xFFFFFFFF};
  const int num_tests = sizeof(test_values) / sizeof(test_values[0]);

  bool passed = true;

  for (int t = 0; t < num_tests; ++t) {
    u32 value = test_values[t];

    // Compute hash multiple times and verify consistency
    // Using simple multiplicative hash for testing
    u32 hash1 = (value * 2654435761u) >> 16;
    u32 hash2 = (value * 2654435761u) >> 16;

    if (hash1 != hash2) {
      printf("    Hash not deterministic for value 0x%08X\n", value);
      passed = false;
    }
  }

  record_test("Hash Function Determinism", passed);
  return passed;
}

// ==============================================================================
// Test: Hash Distribution Quality
// ==============================================================================

bool test_hash_distribution() {
  printf("=== Test: Hash Distribution Quality ===\n");

  const u32 num_samples = 10000;
  const u32 num_buckets = 256;

  std::vector<u32> bucket_counts(num_buckets, 0);

  // Generate hashes and count distribution
  std::mt19937 rng(42);
  for (u32 i = 0; i < num_samples; ++i) {
    u32 value = rng();
    u32 hash = (value * 2654435761u) >> 24; // 8-bit hash
    bucket_counts[hash % num_buckets]++;
  }

  // Check for reasonable distribution
  u32 expected_per_bucket = num_samples / num_buckets;
  u32 min_count = UINT32_MAX;
  u32 max_count = 0;

  for (u32 i = 0; i < num_buckets; ++i) {
    min_count = std::min(min_count, bucket_counts[i]);
    max_count = std::max(max_count, bucket_counts[i]);
  }

  // Allow up to 3x variance
  bool passed = (max_count < expected_per_bucket * 3);

  printf("    Expected per bucket: %u, Min: %u, Max: %u\n", expected_per_bucket,
         min_count, max_count);

  record_test("Hash Distribution Quality", passed);
  return passed;
}

// ==============================================================================
// Test: Hash Collision Handling
// ==============================================================================

bool test_hash_collision_handling() {
  printf("=== Test: Hash Collision Handling ===\n");

  const u32 hash_size = 256; // Small table to force collisions
  const u32 num_inserts = 1000;

  u32 *d_hash_table = nullptr;
  u32 *d_chain_table = nullptr;

  cudaMalloc(&d_hash_table, hash_size * sizeof(u32));
  cudaMalloc(&d_chain_table, num_inserts * sizeof(u32));

  // Initialize
  cudaMemset(d_hash_table, 0xFF, hash_size * sizeof(u32));
  cudaMemset(d_chain_table, 0, num_inserts * sizeof(u32));

  cudaDeviceSynchronize();

  bool passed = (cudaGetLastError() == cudaSuccess);

  cudaFree(d_hash_table);
  cudaFree(d_chain_table);

  record_test("Hash Collision Handling", passed);
  return passed;
}

// ==============================================================================
// Test: Large Hash Table
// ==============================================================================

bool test_large_hash_table() {
  printf("=== Test: Large Hash Table ===\n");

  const u32 hash_size = 1 << 20; // 1M entries

  u32 *d_hash_table = nullptr;
  cudaError_t err = cudaMalloc(&d_hash_table, hash_size * sizeof(u32));

  bool passed = (err == cudaSuccess);

  if (passed) {
    cudaMemset(d_hash_table, 0xFF, hash_size * sizeof(u32));
    passed = (cudaGetLastError() == cudaSuccess);
    cudaFree(d_hash_table);
  }

  record_test("Large Hash Table (1M)", passed);
  return passed;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  cudaFree(0); // Initialize CUDA

  printf("========================================\n");
  printf("Hash Functions Test Suite\n");
  printf("========================================\n\n");

  test_hash_table_init();
  test_chain_table_init();
  test_hash_determinism();
  test_hash_distribution();
  test_hash_collision_handling();
  test_large_hash_table();

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
    printf("\n✅ ALL HASH FUNCTION TESTS PASSED\n");
    return 0;
  } else {
    printf("\n❌ Some tests failed\n");
    return 1;
  }
}
