// test_batch_fse.cu - Batch Parallel FSE Encoder Tests
// Tests verified sequential encoder running on multiple independent blocks

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

void fill_random(std::vector<byte_t> &buffer, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)dist(rng);
  }
}

void fill_compressible(std::vector<byte_t> &buffer, unsigned int seed = 123) {
  std::mt19937 rng(seed);
  std::discrete_distribution<int> dist({30, 25, 20, 10, 5, 5, 3, 2});
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = (byte_t)(dist(rng) * 32);
  }
}

bool verify_batch_roundtrip(u32 num_blocks, u32 block_size, bool compressible) {
  printf("  Testing %u blocks x %uKB (%s)...\n", num_blocks, block_size / 1024,
         compressible ? "compressible" : "random");

  // Allocate host data
  std::vector<std::vector<byte_t>> h_inputs(num_blocks);
  std::vector<byte_t *> d_inputs_ptrs(num_blocks);
  std::vector<byte_t *> d_outputs_ptrs(num_blocks);
  std::vector<u32> input_sizes(num_blocks);
  std::vector<u32> output_sizes(num_blocks);

  // Generate input data
  for (u32 i = 0; i < num_blocks; i++) {
    h_inputs[i].resize(block_size);
    if (compressible) {
      fill_compressible(h_inputs[i], 100 + i);
    } else {
      fill_random(h_inputs[i], 100 + i);
    }
    input_sizes[i] = block_size;
  }

  // Allocate device memory
  for (u32 i = 0; i < num_blocks; i++) {
    CUDA_CHECK(cudaMalloc(&d_inputs_ptrs[i], block_size));
    CUDA_CHECK(cudaMalloc(&d_outputs_ptrs[i], block_size * 2));
    CUDA_CHECK(cudaMemcpy(d_inputs_ptrs[i], h_inputs[i].data(), block_size,
                          cudaMemcpyHostToDevice));
  }

  // NOTE: encode_fse_batch expects HOST arrays of device pointers
  // Pass d_inputs_ptrs.data() and d_outputs_ptrs.data() directly

  // Encode batch
  Status status = encode_fse_batch((const byte_t **)d_inputs_ptrs.data(),
                                   input_sizes.data(), d_outputs_ptrs.data(),
                                   output_sizes.data(), num_blocks, 0);

  if (status != Status::SUCCESS) {
    printf("  ❌ Batch encode failed: %d\n", (int)status);
    // Cleanup
    for (u32 i = 0; i < num_blocks; i++) {
      cudaFree(d_inputs_ptrs[i]);
      cudaFree(d_outputs_ptrs[i]);
    }
    return false;
  }

  // Verify each block via decode
  bool all_passed = true;
  for (u32 i = 0; i < num_blocks && i < 3; i++) { // Check first 3 blocks
    byte_t *d_decoded;
    CUDA_CHECK(cudaMalloc(&d_decoded, block_size));

    u32 decoded_size = 0;
    Status decode_status = decode_fse(d_outputs_ptrs[i], output_sizes[i],
                                      d_decoded, &decoded_size, 0);

    if (decode_status != Status::SUCCESS) {
      printf("  ❌ Block %u decode failed\n", i);
      all_passed = false;
    } else if (decoded_size != block_size) {
      printf("  ❌ Block %u size mismatch: %u vs %u\n", i, decoded_size,
             block_size);
      all_passed = false;
    } else {
      // Compare content
      std::vector<byte_t> h_decoded(block_size);
      CUDA_CHECK(cudaMemcpy(h_decoded.data(), d_decoded, block_size,
                            cudaMemcpyDeviceToHost));

      bool match = true;
      for (u32 j = 0; j < block_size && match; j++) {
        if (h_decoded[j] != h_inputs[i][j]) {
          printf("  ❌ Block %u content mismatch at byte %u\n", i, j);
          match = false;
          all_passed = false;
        }
      }
    }
    cudaFree(d_decoded);
  }

  // Cleanup
  for (u32 i = 0; i < num_blocks; i++) {
    cudaFree(d_inputs_ptrs[i]);
    cudaFree(d_outputs_ptrs[i]);
  }

  return all_passed;
}

// =============================================================================
// TESTS
// =============================================================================

bool test_batch_single_block() {
  printf("=== Test: Batch Single Block ===\n");
  bool ok = verify_batch_roundtrip(1, 64 * 1024, false);
  if (ok)
    printf("  ✅ Passed\n\n");
  return ok;
}

bool test_batch_small() {
  printf("=== Test: Batch Small (10 blocks x 64KB) ===\n");
  bool ok = verify_batch_roundtrip(10, 64 * 1024, false);
  if (ok)
    printf("  ✅ Passed\n\n");
  return ok;
}

bool test_batch_medium() {
  printf("=== Test: Batch Medium (100 blocks x 256KB) ===\n");
  bool ok = verify_batch_roundtrip(100, 256 * 1024, false);
  if (ok)
    printf("  ✅ Passed\n\n");
  return ok;
}

bool test_batch_large() {
  printf("=== Test: Batch Large (50 blocks x 256KB = 12.5MB) ===\n");
  bool ok = verify_batch_roundtrip(50, 256 * 1024, false);
  if (ok)
    printf("  ✅ Passed\n\n");
  return ok;
}

bool test_batch_compressible() {
  printf("=== Test: Batch Compressible Data (20x256KB) ===\n");
  bool ok = verify_batch_roundtrip(20, 256 * 1024, true);
  if (ok)
    printf("  ✅ Passed\n\n");
  return ok;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
  printf("\n========================================\n");
  printf("  Batch FSE Encoder Test Suite\n");
  printf("========================================\n\n");

  int passed = 0;
  int failed = 0;

  if (test_batch_single_block())
    passed++;
  else
    failed++;
  if (test_batch_small())
    passed++;
  else
    failed++;
  if (test_batch_medium())
    passed++;
  else
    failed++;
  if (test_batch_large())
    passed++;
  else
    failed++;
  if (test_batch_compressible())
    passed++;
  else
    failed++;

  printf("========================================\n");
  printf("  Results: %d passed, %d failed\n", passed, failed);
  printf("========================================\n");

  return failed > 0 ? 1 : 0;
}
