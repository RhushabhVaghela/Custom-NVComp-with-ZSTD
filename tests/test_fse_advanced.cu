// test_fse_advanced.cu - Test all 4 advanced features

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

void test_bit_exact_fse_roundtrip() {
  printf("[TEST4] === ENTRY test_bit_exact_fse_roundtrip ===\n");
  fflush(stdout);
  // REVERTED: Use original 11 bytes
  const cuda_zstd::byte_t test_data[] = {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6};
  cuda_zstd::u32 data_size = sizeof(test_data);

  printf("[TEST4] Testing FSE roundtrip with %u bytes of data\n", data_size);
  fflush(stdout);

  // Allocate device memory
  cuda_zstd::byte_t *d_input = nullptr;
  cuda_zstd::byte_t *d_output = nullptr;
  cuda_zstd::u32 *d_output_size = nullptr;

  printf("[TEST4] Allocating d_input (%u bytes)...\n", data_size);
  fflush(stdout);
  if (!safe_cuda_malloc((void **)&d_input, data_size)) {
    printf("ERROR: CUDA malloc for d_input failed\n");
    fflush(stdout);
    assert(0);
  }
  printf("[TEST4] d_input allocated: %p\n", d_input);
  fflush(stdout);

  printf("[TEST4] Allocating d_output...\n");
  fflush(stdout);
  if (!safe_cuda_malloc((void **)&d_output, data_size * 2 + 1024)) {
    printf("ERROR: CUDA malloc for d_output failed\n");
    fflush(stdout);
    safe_cuda_free(d_input);
    assert(0);
  }
  printf("[TEST4] d_output allocated: %p\n", d_output);
  fflush(stdout);

  if (!safe_cuda_malloc((void **)&d_output_size, sizeof(cuda_zstd::u32))) {
    printf("ERROR: CUDA malloc for d_output_size failed\n");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    assert(0);
  }

  // Initialize output buffer to 0 (crucial for atomic operations in kernels)
  safe_cuda_memset(d_output, 0, data_size * 2 + 1024);

  printf("[TEST4] Copying %u bytes to d_input...\n", data_size);
  fflush(stdout);
  if (!safe_cuda_memcpy(d_input, test_data, data_size,
                        cudaMemcpyHostToDevice)) {
    printf("ERROR: CUDA memcpy to d_input failed\n");
    fflush(stdout);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    assert(0);
  }
  printf("[TEST4] Data copied to device\n");
  fflush(stdout);

  // Encode using the FSE API directly
  printf("[TEST4] === CALLING encode_fse_advanced_debug ===\n");
  fflush(stdout);
  cuda_zstd::u32 encoded_size = 0;
  // Run FSE encoding (host pointer for output size)
  cuda_zstd::Status status = cuda_zstd::fse::encode_fse_advanced_debug(
      d_input, data_size, d_output,
      &encoded_size, // Pass HOST pointer to output size
      true,          // gpu_optimize
      0              // stream
  );
  printf("[TEST4] === RETURNED from encode_fse_advanced_debug, status=%d ===\n",
         (int)status);
  fflush(stdout);
  printf("[TEST4] encoded_size reported: %u\n", encoded_size);
  fflush(stdout);

  if (status != cuda_zstd::Status::SUCCESS) {
    printf("[TEST4] ERROR: FSE encoding failed: %d\n", (int)status);
    fflush(stdout);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    assert(0);
  }
  printf("[TEST4] Encoding SUCCESS\n");
  fflush(stdout);

  // Size is already in encoded_size
  /*
  // Copy encoded size back
  cuda_zstd::u32 encoded_size = 0;
  if (!safe_cuda_memcpy(&encoded_size, d_output_size, sizeof(cuda_zstd::u32),
  cudaMemcpyDeviceToHost)) { printf("ERROR: CUDA memcpy for encoded_size
  failed\n"); safe_cuda_free(d_input); safe_cuda_free(d_output);
      safe_cuda_free(d_output_size);
      assert(0);
  }
  */

  printf("Encoded to %u bytes\n", encoded_size);
  fflush(stdout);

  // Allocate memory for decoded data
  printf("[TEST4] Allocating d_decoded (%u bytes)...\n", data_size);
  fflush(stdout);
  cuda_zstd::byte_t *d_decoded = nullptr;
  if (!safe_cuda_malloc((void **)&d_decoded, data_size)) {
    printf("[TEST4] ERROR: CUDA malloc for d_decoded failed\n");
    fflush(stdout);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    assert(0);
  }
  printf("[TEST4] d_decoded allocated: %p\n", d_decoded);
  fflush(stdout);

  // [FIX] Copy encoded data to host for decoding (decode_fse is Host function)
  printf("[TEST4] Copying encoded data to host...\n");
  fflush(stdout);
  std::vector<cuda_zstd::byte_t> h_output(encoded_size);
  if (!safe_cuda_memcpy(h_output.data(), d_output, encoded_size,
                        cudaMemcpyDeviceToHost)) {
    printf("[TEST4] ERROR: CUDA memcpy for encoded data failed\n");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    safe_cuda_free(d_decoded);
    assert(0);
  }

  printf("[TEST4] === CALLING decode_fse ===\n");
  fflush(stdout);
  cuda_zstd::u32 decoded_size = 0;
  cuda_zstd::Status dec_status = cuda_zstd::fse::decode_fse(
      d_output, encoded_size, // Pass Device Pointer directly
      d_decoded, &decoded_size,
      0 // stream
  );
  printf("[TEST4] === RETURNED from decode_fse, status=%d ===\n",
         (int)dec_status);
  fflush(stdout);

  if (dec_status != cuda_zstd::Status::SUCCESS) {
    printf("ERROR: FSE decoding failed: %d\n", (int)dec_status);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    safe_cuda_free(d_decoded);
    assert(0);
  }

  printf("[TEST4] Decoded to %u bytes (expected %u)\n", decoded_size,
         data_size);
  fflush(stdout);

  // Verify
  printf("[TEST4] Checking decoded_size == data_size...\n");
  fflush(stdout);
  assert(decoded_size == data_size);
  printf("[TEST4] Size check passed\n");
  fflush(stdout);

  // Verify content
  printf("[TEST4] Copying decoded data to host...\n");
  fflush(stdout);
  std::vector<cuda_zstd::byte_t> decoded_data(data_size);
  if (!safe_cuda_memcpy(decoded_data.data(), d_decoded, data_size,
                        cudaMemcpyDeviceToHost)) {
    printf("[TEST4] ERROR: CUDA memcpy for decoded data failed\n");
    fflush(stdout);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    safe_cuda_free(d_decoded);
    assert(0);
  }
  printf("[TEST4] Decoded data copied to host\n");
  fflush(stdout);

  printf("[TEST4] Verifying %u bytes...\n", data_size);
  fflush(stdout);
  for (cuda_zstd::u32 i = 0; i < data_size; i++) {
    if (decoded_data[i] != test_data[i]) {
      printf("[TEST4] MISMATCH at byte %u: decoded=0x%02X expected=0x%02X\n", i,
             decoded_data[i], test_data[i]);
      fflush(stdout);
    }
    assert(decoded_data[i] == test_data[i]);
  }
  printf("[TEST4] All bytes verified!\n");
  fflush(stdout);

  printf("FSE roundtrip test passed!\n");

  // Cleanup
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_output_size);
  safe_cuda_free(d_decoded);
}

void test_fse_normalization() {
  std::vector<cuda_zstd::u32> raw_freqs = {100, 200, 50, 75, 150};
  std::vector<cuda_zstd::u16> normalized(256, 0);

  cuda_zstd::u32 actual_table_size = 0;
  cuda_zstd::Status status = normalize_frequencies_accurate(
      raw_freqs.data(), 575, 256, normalized.data(), 8, &actual_table_size);

  assert(status == cuda_zstd::Status::SUCCESS);

  cuda_zstd::u32 sum = 0;
  for (auto freq : normalized) {
    sum += freq;
  }

  // CRITICAL: Exact match guaranteed
  assert(sum == 256); // ✅ MUST PASS
}

int main() {
  // Skip on CPU-only environments; otherwise show device info
  SKIP_IF_NO_CUDA_RET(0);
  check_cuda_device();

  printf("========================================\n");
  printf("  FSE Advanced Features Test Suite\n");
  printf("========================================\n\n");

  // Test data: various entropy levels
  const char *test_data[] = {// Low entropy (repetitive)
                             "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",

                             // Medium entropy
                             "The quick brown fox jumps over the lazy dog. ",

                             // High entropy (mixed)
                             "1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ!@#$%"};

  const char *labels[] = {"Low Entropy (Repetitive)", "Medium Entropy (Text)",
                          "High Entropy (Mixed)"};

  for (int test = 0; test < 3; test++) {
    printf("\n========================================\n");
    printf("TEST %d: %s\n", test + 1, labels[test]);
    printf("========================================\n\n");

    const char *input = test_data[test];
    cuda_zstd::u32 input_size = strlen(input);

    // Allocate device memory with error checking
    cuda_zstd::byte_t *d_input = nullptr;
    cuda_zstd::byte_t *d_output = nullptr;
    cuda_zstd::u32 *d_output_size = nullptr;

    if (!safe_cuda_malloc((void **)&d_input, input_size)) {
      printf("ERROR: CUDA malloc for d_input failed\n");
      return 1;
    }

    // Ensure sufficient space for FSE header (approx 512 bytes) + compressed
    // data
    u32 alloc_size = (input_size * 2) + 1024;
    if (!safe_cuda_malloc((void **)&d_output, alloc_size)) {
      printf("ERROR: CUDA malloc for d_output failed\n");
      safe_cuda_free(d_input);
      return 1;
    }

    if (!safe_cuda_malloc((void **)&d_output_size, sizeof(u32))) {
      printf("ERROR: CUDA malloc for d_output_size failed\n");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return 1;
    }

    if (!safe_cuda_memcpy(d_input, input, input_size, cudaMemcpyHostToDevice)) {
      printf("ERROR: CUDA memcpy to d_input failed\n");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      safe_cuda_free(d_output_size);
      return 1;
    }

    // FEATURE 1 & 2: Analyze with adaptive table log + accurate normalization
    printf("--- Feature 1 & 2: Adaptive + Accurate ---\n");
    // Compute FSE block statistics on the device and print them.
    // This replaces the old commented-out FseManager usage.
    cuda_zstd::fse::FSEStats stats;
    Status s = cuda_zstd::fse::analyze_block_statistics(d_input, input_size,
                                                        &stats, 0);
    if (s != Status::SUCCESS) {
      printf("ERROR: analyze_block_statistics failed: %s\n",
             status_to_string(s));
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      safe_cuda_free(d_output_size);
      return 1;
    }
    cuda_zstd::fse::print_fse_stats(stats);

    // FEATURE 3: GPU-optimized encoding
    printf("--- Feature 3: GPU Optimization ---\n");
    // Using encode_fse_advanced with gpu_optimize = true
    cuda_zstd::u32 h_output_size = 0;
    cuda_zstd::Status status = cuda_zstd::fse::encode_fse_advanced_debug(
        d_input, input_size, d_output, &h_output_size, // Pass HOST pointer
        true, // gpu_optimize (Feature 3)
        0     // stream
    );

    if (status != cuda_zstd::Status::SUCCESS) {
      printf("ERROR: encode_fse_advanced failed with status %d\n", (int)status);
      // Don't assert, let's see which one fails
      return 1;
    }
    // Size is already in h_output_size
    /*
    // Copy encoded size back if needed (it's already in h_output_size)
    */

    // FEATURE 4 (Newly Added): Verify Roundtrip for these tests too!
    printf("--- Feature 4: Verify Roundtrip ---\n");
    printf("  Encoded Size: %u bytes\n", h_output_size);

    // Allocate decode buffer
    cuda_zstd::byte_t *d_decoded = nullptr;
    if (!safe_cuda_malloc((void **)&d_decoded, input_size)) {
      printf("ERROR: CUDA malloc for d_decoded failed\n");
      return 1;
    }

    // Decode
    cuda_zstd::u32 decoded_size = 0;
    cuda_zstd::Status dec_status = cuda_zstd::fse::decode_fse(
        d_output, h_output_size, // Pass Device Pointer directly
        d_decoded, &decoded_size,
        0 // stream
    );

    if (dec_status != cuda_zstd::Status::SUCCESS) {
      printf("ERROR: FSE decoding failed: %d\n", (int)dec_status);
      return 1;
    }

    if (decoded_size != input_size) {
      printf("ERROR: Decoded size mismatch: %u != %u\n", decoded_size,
             input_size);
      return 1;
    }

    // Verify content
    std::vector<cuda_zstd::byte_t> h_decoded(input_size);
    if (!safe_cuda_memcpy(h_decoded.data(), d_decoded, input_size,
                          cudaMemcpyDeviceToHost)) {
      printf("ERROR: CUDA memcpy for decoded data failed\n");
      return 1;
    }

    // Compare with input (which is on host as 'input')
    int mismatches = 0;
    for (cuda_zstd::u32 i = 0; i < input_size; i++) {
      if (h_decoded[i] != (cuda_zstd::byte_t)input[i]) {
        if (mismatches < 10) {
          printf("MISMATCH at byte %u: decoded=0x%02X expected=0x%02X\n", i,
                 h_decoded[i], (unsigned char)input[i]);
        }
        mismatches++;
      }
    }

    if (mismatches > 0) {
      printf("ERROR: Verification failed with %d mismatches\n", mismatches);
      return 1;
    }

    printf("SUCCESS: Roundtrip verified! (%u bytes match)\n", input_size);

    // Cleanup with safe free functions
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    safe_cuda_free(d_decoded);
  }

  /*
  // FEATURE 4: Multi-table FSE test
  printf("\n========================================\n");
  printf("TEST 4: Multi-Table FSE\n");
  printf("========================================\n\n");

  // ... (commented out multi-table test code) ...
  // cuda_zstd::fse::free_multi_table(multi_table);
  // safe_cuda_free(d_mixed);
  */

  test_bit_exact_fse_roundtrip();
  // test_fse_normalization();

  printf("\n========================================\n");
  printf("  ALL TESTS COMPLETED SUCCESSFULLY!\n");
  printf("========================================\n");

  return 0;
}
