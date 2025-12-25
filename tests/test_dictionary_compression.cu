// ============================================================================
// test_dictionary_compression.cu - Unit tests for dictionary compression
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_utils.h"
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::dictionary;

// Helper to create repetitive sample data (good for dictionaries)
std::vector<byte_t> create_sample(size_t size, const char *prefix) {
  std::vector<byte_t> sample(size);
  size_t prefix_len = strlen(prefix);
  for (size_t i = 0; i < size; ++i) {
    if (i % 128 < prefix_len) { // Interleave prefix
      sample[i] = (byte_t)prefix[i % prefix_len];
    } else {
      sample[i] = (byte_t)(i % 256);
    }
  }
  return sample;
}

void print_test_header(const char *title) {
  std::cout << "\n--- " << title << " ---\n";
}

int main() {
  // Skip when no CUDA device is available on the machine (useful for CI).
  SKIP_IF_NO_CUDA_RET(0);
  std::cout << "\n==================================================\n";
  std::cout << "  Test: Dictionary Compression Feature\n";
  std::cout << "==================================================\n";

  // 1. Setup: Train a dictionary
  print_test_header("Dictionary Training");
  const size_t dict_capacity = 16 * 1024; // 16 KB
  std::vector<std::vector<byte_t>> h_samples;
  std::vector<const void *> h_sample_ptrs;
  std::vector<size_t> h_sample_sizes;

  h_samples.push_back(create_sample(1024 * 32, "REPEATING_SEQUENCE_"));
  h_samples.push_back(create_sample(1024 * 48, "REPEATING_SEQUENCE_"));
  for (const auto &s : h_samples) {
    h_sample_ptrs.push_back(static_cast<const void *>(s.data()));
    h_sample_sizes.push_back(s.size());
  }

  Dictionary gpu_dict;
  DictionaryManager::allocate_dictionary_gpu(gpu_dict, dict_capacity, 0);

  CoverParams params;
  params.k = 1024;
  params.d = 8;

  Status status = DictionaryTrainer::train_dictionary(
      h_sample_ptrs, h_sample_sizes, gpu_dict, dict_capacity, &params, 0);
  cudaDeviceSynchronize();

  if (status != Status::SUCCESS || gpu_dict.raw_size == 0) {
    std::cerr
        << "  \033[1;31m\u2717 FAILED\033[0m: Dictionary training failed.\n";
    return 1;
  }
  std::cout << "  \033[1;32m\u2713 PASSED\033[0m: Dictionary trained. Size: "
            << gpu_dict.raw_size
            << " bytes, ID: " << gpu_dict.header.dictionary_id << "\n";

  // 2. Prepare data for tests
  const size_t test_data_size = 128 * 1024;
  std::vector<byte_t> h_test_data =
      create_sample(test_data_size, "REPEATING_SEQUENCE_");

  byte_t *d_input, *d_compressed, *d_decompressed, *d_temp;
  cudaError_t err;
  err = cudaMalloc(&d_input, test_data_size);
  assert(err == cudaSuccess);
  // Calculate max compressed size manually: input_size + input_size/256 + 128
  size_t max_compressed = test_data_size + test_data_size / 256 + 128;
  err = cudaMalloc(&d_compressed, max_compressed);
  assert(err == cudaSuccess);
  err = cudaMalloc(&d_decompressed, test_data_size);
  assert(err == cudaSuccess);
  err = cudaMemcpy(d_input, h_test_data.data(), test_data_size,
                   cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  auto manager = create_manager(5);
  fprintf(stderr, "[DEBUG] manager created\n");
  fflush(stderr);
  manager->set_dictionary(gpu_dict);
  fprintf(stderr, "[DEBUG] dictionary set\n");
  fflush(stderr);
  size_t temp_size = manager->get_compress_temp_size(test_data_size);
  fprintf(stderr, "[DEBUG] temp_size=%zu\n", temp_size);
  fflush(stderr);
  err = cudaMalloc(&d_temp, temp_size);
  assert(err == cudaSuccess);
  fprintf(stderr, "[DEBUG] d_temp allocated\n");
  fflush(stderr);

  // ========================================================================
  // Test 1: Correctness Test (Round-trip)
  // ========================================================================
  print_test_header("Correctness Test (Round-trip)");
  size_t compressed_size = max_compressed;
  fprintf(stderr, "[TEST] Calling compress (Round-trip)...\n");
  fflush(stderr);
  status =
      manager->compress(d_input, test_data_size, d_compressed, &compressed_size,
                        d_temp, temp_size, nullptr, 0, 0);
  fprintf(stderr, "[TEST] compress returned status=%d\n", (int)status);
  fflush(stderr);
  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: compress failed with status " << (int)status
              << "\n";
    return 1;
  }

  size_t decompressed_size = test_data_size;
  fprintf(stderr, "[TEST] Calling decompress...\n");
  fflush(stderr);
  status = manager->decompress(d_compressed, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size, 0);
  fprintf(stderr, "[TEST] decompress returned status=%d\n", (int)status);
  fflush(stderr);
  assert(status == Status::SUCCESS);
  cudaError_t sync_err = cudaDeviceSynchronize();
  fprintf(stderr, "[TEST] Device sync after decompress: %d\n", (int)sync_err);
  fflush(stderr);
  assert(sync_err == cudaSuccess);

  std::vector<byte_t> h_decompressed_data(test_data_size);
  err = cudaMemcpy(h_decompressed_data.data(), d_decompressed, test_data_size,
                   cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);

  if (decompressed_size == test_data_size &&
      memcmp(h_test_data.data(), h_decompressed_data.data(), test_data_size) ==
          0) {
    std::cout << "  \033[1;32m\u2713 PASSED\033[0m: Decompressed data matches "
                 "original.\n";
  } else {
    std::cerr
        << "  \033[1;31m\u2717 FAILED\033[0m: Round-trip data mismatch.\n";
    return 1;
  }

  // ========================================================================
  // Test 2: Compression Ratio Test
  // ========================================================================
  print_test_header("Compression Ratio Test");
  manager->clear_dictionary();
  size_t size_no_dict = 0;
  manager->compress(d_input, test_data_size, d_compressed, &size_no_dict,
                    d_temp, temp_size, nullptr, 0, 0);
  sync_err = cudaDeviceSynchronize();
  assert(sync_err == cudaSuccess);
  std::cout << "  Size without dictionary: " << size_no_dict << " bytes\n";

  manager->set_dictionary(gpu_dict);
  size_t size_with_dict = 0;
  manager->compress(d_input, test_data_size, d_compressed, &size_with_dict,
                    d_temp, temp_size, nullptr, 0, 0);
  sync_err = cudaDeviceSynchronize();
  assert(sync_err == cudaSuccess);
  std::cout << "  Size with dictionary:    " << size_with_dict << " bytes\n";

  if (size_with_dict < size_no_dict * 0.8) { // Expect at least 20% improvement
    std::cout << "  \033[1;32m\u2713 PASSED\033[0m: Dictionary compression is "
                 "significantly smaller.\n";
  } else {
    std::cerr << "  \033[1;31m\u2717 FAILED\033[0m: Dictionary compression "
                 "ratio not met.\n";
    return 1;
  }

  // ========================================================================
  // Test 3: Dictionary ID Test
  // ========================================================================
  print_test_header("Dictionary ID Test");

  // Dictionary ID verification (simplified - actual implementation may vary)
  if (gpu_dict.header.dictionary_id > 0) {
    std::cout << "  \033[1;32m\u2713 PASSED\033[0m: Dictionary ID is set ("
              << gpu_dict.header.dictionary_id << ").\n";
  } else {
    std::cerr << "  \033[1;31m\u2717 FAILED\033[0m: Dictionary ID not set.\n";
    return 1;
  }

  // ========================================================================
  // Test 4: Negative Test (Wrong Dictionary)
  // ========================================================================
  print_test_header("Negative Test (Wrong Dictionary)");
  Dictionary wrong_dict;
  DictionaryManager::allocate_dictionary_gpu(wrong_dict, 1024, 0);
  wrong_dict.header.dictionary_id = 99999; // Set a bogus ID
  wrong_dict.raw_size = 1024;
  err = cudaMemset(wrong_dict.raw_content, 0, 1024);
  assert(err == cudaSuccess);

  manager->set_dictionary(wrong_dict);
  status = manager->decompress(d_compressed, size_with_dict, d_decompressed,
                               &decompressed_size, d_temp, temp_size, 0);
  sync_err = cudaDeviceSynchronize();
  assert(sync_err == cudaSuccess);

  if (status == Status::ERROR_DICTIONARY_MISMATCH) {
    std::cout << "  \033[1;32m\u2713 PASSED\033[0m: Decompression failed as "
                 "expected with wrong dictionary.\n";
  } else {
    std::cerr << "  \033[1;31m\u2717 FAILED\033[0m: Decompression did not fail "
                 "with expected error. Status: "
              << (int)status << "\n";
    return 1;
  }

  // Cleanup - ensure all async operations complete first
  cudaDeviceSynchronize();

  // Explicitly reset manager before freeing GPU memory it may reference
  manager.reset();

  // Sync again after manager destruction
  cudaDeviceSynchronize();
  cudaGetLastError(); // Clear any stale errors

  DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
  DictionaryManager::free_dictionary_gpu(wrong_dict, 0);
  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(d_temp);

  std::cout << "\n\033[1;32mAll dictionary compression tests passed "
               "successfully!\033[0m\n\n";
  return 0;
}
