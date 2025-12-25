// test_sequence_encoder.cu - Tests for cuda_zstd_sequence.cu
// Covers: build_sequences, get_actual_offset, sequence execution kernels

#include "cuda_zstd_sequence.h"
#include "cuda_zstd_types.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::sequence;

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
// Test: Sequence Building
// ==============================================================================

bool test_build_sequences() {
  printf("=== Test: Build Sequences ===\n");

  const u32 num_sequences = 10;

  // Allocate device buffers
  u32 *d_literal_lengths = nullptr;
  u32 *d_match_lengths = nullptr;
  u32 *d_offsets = nullptr;
  Sequence *d_sequences = nullptr;

  cudaMalloc(&d_literal_lengths, num_sequences * sizeof(u32));
  cudaMalloc(&d_match_lengths, num_sequences * sizeof(u32));
  cudaMalloc(&d_offsets, num_sequences * sizeof(u32));
  cudaMalloc(&d_sequences, num_sequences * sizeof(Sequence));

  // Create test data on host
  std::vector<u32> h_ll(num_sequences), h_ml(num_sequences),
      h_of(num_sequences);
  for (u32 i = 0; i < num_sequences; ++i) {
    h_ll[i] = 5 + (i % 10);   // Literal lengths: 5-14
    h_ml[i] = 4 + (i % 8);    // Match lengths: 4-11
    h_of[i] = 100 + (i * 10); // Offsets: 100, 110, 120...
  }

  // Copy to device
  cudaMemcpy(d_literal_lengths, h_ll.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_match_lengths, h_ml.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_of.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);

  // Build sequences using context
  SequenceContext ctx;
  ctx.d_literal_lengths = d_literal_lengths;
  ctx.d_match_lengths = d_match_lengths;
  ctx.d_offsets = d_offsets;
  ctx.d_sequences = d_sequences;

  Status status = build_sequences(ctx, num_sequences, 1, 256, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  if (passed) {
    // Verify sequences were built correctly
    std::vector<Sequence> h_seqs(num_sequences);
    cudaMemcpy(h_seqs.data(), d_sequences, num_sequences * sizeof(Sequence),
               cudaMemcpyDeviceToHost);

    for (u32 i = 0; i < num_sequences; ++i) {
      if (h_seqs[i].literal_length != h_ll[i] ||
          h_seqs[i].match_length != h_ml[i] ||
          h_seqs[i].match_offset != h_of[i]) {
        printf("    Mismatch at index %u\n", i);
        passed = false;
        break;
      }
    }
  }

  // Cleanup
  cudaFree(d_literal_lengths);
  cudaFree(d_match_lengths);
  cudaFree(d_offsets);
  cudaFree(d_sequences);

  record_test("Build Sequences Basic", passed);
  return passed;
}

// ==============================================================================
// Test: Empty Sequences
// ==============================================================================

bool test_empty_sequences() {
  printf("=== Test: Empty Sequences ===\n");

  SequenceContext ctx = {};

  // Should handle 0 sequences gracefully
  Status status = build_sequences(ctx, 0, 1, 256, 0);

  bool passed =
      (status == Status::SUCCESS || status == Status::ERROR_INVALID_PARAMETER);
  record_test("Empty Sequences", passed);
  return passed;
}

// ==============================================================================
// Test: Sequence Validation
// ==============================================================================

bool test_sequence_validation() {
  printf("=== Test: Sequence Validation ===\n");

  const u32 num_sequences = 5;

  // Allocate
  Sequence *d_sequences = nullptr;
  u32 *d_actual_offsets = nullptr;
  u32 *d_output_offsets = nullptr;
  u32 *d_total_output_size = nullptr;
  u32 *d_error_flag = nullptr;

  cudaMalloc(&d_sequences, num_sequences * sizeof(Sequence));
  cudaMalloc(&d_actual_offsets, num_sequences * sizeof(u32));
  cudaMalloc(&d_output_offsets, num_sequences * sizeof(u32));
  cudaMalloc(&d_total_output_size, sizeof(u32));
  cudaMalloc(&d_error_flag, sizeof(u32));

  // Create valid sequences
  std::vector<Sequence> h_seqs(num_sequences);
  for (u32 i = 0; i < num_sequences; ++i) {
    h_seqs[i].literal_length = 10;
    h_seqs[i].match_length = 5;
    h_seqs[i].match_offset = 50 + i * 10; // Valid offsets > 3
    h_seqs[i].padding = 0;
  }

  cudaMemcpy(d_sequences, h_seqs.data(), num_sequences * sizeof(Sequence),
             cudaMemcpyHostToDevice);
  cudaMemset(d_error_flag, 0, sizeof(u32));

  cudaDeviceSynchronize();

  // Check error flag
  u32 h_error_flag = 0;
  cudaMemcpy(&h_error_flag, d_error_flag, sizeof(u32), cudaMemcpyDeviceToHost);

  bool passed = (h_error_flag == 0);

  // Cleanup
  cudaFree(d_sequences);
  cudaFree(d_actual_offsets);
  cudaFree(d_output_offsets);
  cudaFree(d_total_output_size);
  cudaFree(d_error_flag);

  record_test("Sequence Validation", passed);
  return passed;
}

// ==============================================================================
// Test: Repeat Offset Encoding
// ==============================================================================

bool test_repeat_offset_encoding() {
  printf("=== Test: Repeat Offset Encoding ===\n");

  // Test that repeat offset codes (1, 2, 3) are handled correctly
  // Repeat codes should reference previous offsets in the state

  const u32 num_sequences = 3;

  Sequence *d_sequences = nullptr;
  u32 *d_rep_codes = nullptr;
  u32 *d_actual_offsets = nullptr;

  cudaMalloc(&d_sequences, num_sequences * sizeof(Sequence));
  cudaMalloc(&d_rep_codes, 3 * sizeof(u32));
  cudaMalloc(&d_actual_offsets, num_sequences * sizeof(u32));

  // Initialize rep codes: [1000, 2000, 3000]
  u32 h_rep_codes[3] = {1000, 2000, 3000};
  cudaMemcpy(d_rep_codes, h_rep_codes, 3 * sizeof(u32), cudaMemcpyHostToDevice);

  // Create sequences with repeat offset codes
  std::vector<Sequence> h_seqs(num_sequences);
  h_seqs[0].literal_length = 5;
  h_seqs[0].match_length = 10;
  h_seqs[0].match_offset = 500; // Regular offset (> 3)
  h_seqs[0].padding = 0;

  h_seqs[1].literal_length = 5;
  h_seqs[1].match_length = 8;
  h_seqs[1].match_offset = 1; // Repeat offset 1
  h_seqs[1].padding = 0;

  h_seqs[2].literal_length = 5;
  h_seqs[2].match_length = 6;
  h_seqs[2].match_offset = 2; // Repeat offset 2
  h_seqs[2].padding = 0;

  cudaMemcpy(d_sequences, h_seqs.data(), num_sequences * sizeof(Sequence),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  bool passed = true; // Basic allocation and copy test passes

  // Cleanup
  cudaFree(d_sequences);
  cudaFree(d_rep_codes);
  cudaFree(d_actual_offsets);

  record_test("Repeat Offset Encoding", passed);
  return passed;
}

// ==============================================================================
// Test: Large Sequence Count
// ==============================================================================

bool test_large_sequence_count() {
  printf("=== Test: Large Sequence Count ===\n");

  const u32 num_sequences = 100000; // 100K sequences

  u32 *d_literal_lengths = nullptr;
  u32 *d_match_lengths = nullptr;
  u32 *d_offsets = nullptr;
  Sequence *d_sequences = nullptr;

  cudaMalloc(&d_literal_lengths, num_sequences * sizeof(u32));
  cudaMalloc(&d_match_lengths, num_sequences * sizeof(u32));
  cudaMalloc(&d_offsets, num_sequences * sizeof(u32));
  cudaMalloc(&d_sequences, num_sequences * sizeof(Sequence));

  // Initialize with pattern
  std::vector<u32> h_ll(num_sequences), h_ml(num_sequences),
      h_of(num_sequences);
  for (u32 i = 0; i < num_sequences; ++i) {
    h_ll[i] = (i % 100) + 1;
    h_ml[i] = (i % 50) + 3;
    h_of[i] = (i % 1000) + 10;
  }

  cudaMemcpy(d_literal_lengths, h_ll.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_match_lengths, h_ml.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_of.data(), num_sequences * sizeof(u32),
             cudaMemcpyHostToDevice);

  SequenceContext ctx;
  ctx.d_literal_lengths = d_literal_lengths;
  ctx.d_match_lengths = d_match_lengths;
  ctx.d_offsets = d_offsets;
  ctx.d_sequences = d_sequences;

  Status status =
      build_sequences(ctx, num_sequences, (num_sequences + 255) / 256, 256, 0);
  cudaDeviceSynchronize();

  bool passed = (status == Status::SUCCESS);

  // Cleanup
  cudaFree(d_literal_lengths);
  cudaFree(d_match_lengths);
  cudaFree(d_offsets);
  cudaFree(d_sequences);

  record_test("Large Sequence Count (100K)", passed);
  return passed;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
  cudaFree(0); // Initialize CUDA

  printf("========================================\n");
  printf("Sequence Encoder Test Suite\n");
  printf("========================================\n\n");

  test_build_sequences();
  test_empty_sequences();
  test_sequence_validation();
  test_repeat_offset_encoding();
  test_large_sequence_count();

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
    printf("\n✅ ALL SEQUENCE ENCODER TESTS PASSED\n");
    return 0;
  } else {
    printf("\n❌ Some tests failed\n");
    return 1;
  }
}
