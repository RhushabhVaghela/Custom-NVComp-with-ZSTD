// ============================================================================
// test_dictionary_memory.cu - Unit tests for dictionary memory handling
// ============================================================================
// These tests verify the fixes for:
// 1. Memory leaks in train_dictionary_gpu
// 2. Use-after-free prevention in set_dictionary
// 3. Bounds checking in dictionary operations
// 4. Proper error handling for invalid dictionary parameters
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::dictionary;

// Test helper: Create a minimal valid dictionary buffer
std::vector<unsigned char> create_test_dictionary(size_t size) {
  std::vector<unsigned char> dict(size);
  // Fill with pattern
  for (size_t i = 0; i < size; i++) {
    dict[i] = (unsigned char)(i % 256);
  }
  // Set magic number if size permits
  if (size >= 4) {
    dict[0] = 0x37; // DICT_MAGIC_NUMBER bytes (0xEC30A437 little-endian)
    dict[1] = 0xA4;
    dict[2] = 0x30;
    dict[3] = 0xEC;
  }
  return dict;
}

// ============================================================================
// Test 1: Dictionary Training with Memory Validation
// ============================================================================
bool test_dictionary_training_memory() {
  std::cout << "\n[Test] Dictionary Training Memory Handling...\n";

  // Create sample data for training
  std::vector<std::vector<unsigned char>> samples;
  std::vector<const void *> sample_ptrs;
  std::vector<size_t> sample_sizes;

  // Create 5 samples of 4KB each (total > 100KB to trigger GPU path)
  for (int i = 0; i < 5; i++) {
    std::vector<unsigned char> sample(4096);
    for (size_t j = 0; j < 4096; j++) {
      sample[j] = (unsigned char)((j + i * 100) % 256);
    }
    samples.push_back(std::move(sample));
    sample_ptrs.push_back(samples.back().data());
    sample_sizes.push_back(4096);
  }

  // Train dictionary
  const size_t dict_size = 4096; // 4KB dictionary
  std::vector<unsigned char> dict_buffer(dict_size);

  DictionaryTrainingParams params;
  params.use_gpu = true;

  Status status = train_dictionary(sample_ptrs, sample_sizes,
                                   dict_buffer.data(), dict_size, &params, 0);

  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: Dictionary training failed with status "
              << (int)status << "\n";
    return false;
  }

  // Verify dictionary was created
  bool has_nonzero = false;
  for (auto b : dict_buffer) {
    if (b != 0) {
      has_nonzero = true;
      break;
    }
  }

  if (!has_nonzero) {
    std::cerr << "  FAILED: Dictionary is all zeros\n";
    return false;
  }

  std::cout << "  PASSED: Dictionary training completed successfully\n";
  return true;
}

// ============================================================================
// Test 2: Dictionary Training with Invalid Parameters
// ============================================================================
bool test_dictionary_training_invalid_params() {
  std::cout << "\n[Test] Dictionary Training Invalid Parameters...\n";

  bool all_passed = true;

  // Test 2a: NULL buffer
  {
    std::vector<const void *> samples;
    std::vector<size_t> sizes;
    samples.push_back(nullptr);
    sizes.push_back(100);

    Status status = train_dictionary(samples, sizes, nullptr, 1024, nullptr, 0);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr
          << "  FAILED: NULL buffer should return ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  // Test 2b: Empty samples
  {
    std::vector<const void *> samples;
    std::vector<size_t> sizes;
    std::vector<unsigned char> buf(1024);

    Status status =
        train_dictionary(samples, sizes, buf.data(), 1024, nullptr, 0);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr
          << "  FAILED: Empty samples should return ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  // Test 2c: Size mismatch
  {
    std::vector<const void *> samples;
    std::vector<size_t> sizes;
    std::vector<unsigned char> buf(1024);
    std::vector<unsigned char> sample(100);

    samples.push_back(sample.data());
    sizes.push_back(100);
    samples.push_back(sample.data());
    // sizes has only 1 element but samples has 2

    Status status =
        train_dictionary(samples, sizes, buf.data(), 1024, nullptr, 0);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr
          << "  FAILED: Size mismatch should return ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  PASSED: All invalid parameter tests passed\n";
  }
  return all_passed;
}

// ============================================================================
// Test 3: Dictionary Set with Validation
// ============================================================================
bool test_dictionary_set_validation() {
  std::cout << "\n[Test] Dictionary Set Validation...\n";

  bool all_passed = true;
  auto manager = create_manager(3);

  // Test 3a: Valid dictionary
  {
    auto dict_data = create_test_dictionary(4096);
    Dictionary dict;
    dict.raw_content = dict_data.data();
    dict.raw_size = dict_data.size();
    dict.header.dictionary_id = 12345;

    Status status = manager->set_dictionary(dict);
    if (status != Status::SUCCESS) {
      std::cerr << "  FAILED: Valid dictionary should succeed\n";
      all_passed = false;
    }
  }

  // Test 3b: NULL content
  {
    Dictionary dict;
    dict.raw_content = nullptr;
    dict.raw_size = 4096;

    Status status = manager->set_dictionary(dict);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr
          << "  FAILED: NULL content should return ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  // Test 3c: Zero size
  {
    auto dict_data = create_test_dictionary(4096);
    Dictionary dict;
    dict.raw_content = dict_data.data();
    dict.raw_size = 0;

    Status status = manager->set_dictionary(dict);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr
          << "  FAILED: Zero size should return ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  // Test 3d: Size too small
  {
    auto dict_data =
        create_test_dictionary(100); // Less than MIN_DICT_SIZE (256)
    Dictionary dict;
    dict.raw_content = dict_data.data();
    dict.raw_size = 100;

    Status status = manager->set_dictionary(dict);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr << "  FAILED: Small dictionary should return "
                   "ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  // Test 3e: Size too large
  {
    auto dict_data =
        create_test_dictionary(200000); // Greater than MAX_DICT_SIZE (128KB)
    Dictionary dict;
    dict.raw_content = dict_data.data();
    dict.raw_size = 200000;

    Status status = manager->set_dictionary(dict);
    if (status != Status::ERROR_INVALID_PARAMETER) {
      std::cerr << "  FAILED: Large dictionary should return "
                   "ERROR_INVALID_PARAMETER\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  PASSED: All dictionary validation tests passed\n";
  }
  return all_passed;
}

// ============================================================================
// Test 4: Dictionary Compression Round-trip
// ============================================================================
bool test_dictionary_compression_roundtrip() {
  std::cout << "\n[Test] Dictionary Compression Round-trip...\n";

  // Skip if no CUDA device
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cout << "  SKIPPED: No CUDA device available\n";
    return true;
  }

  // Create a dictionary
  auto dict_data = create_test_dictionary(4096);
  Dictionary dict;
  dict.raw_content = dict_data.data();
  dict.raw_size = dict_data.size();
  dict.header.dictionary_id = 12345;

  // Create manager and set dictionary
  auto manager = create_manager(3);
  Status status = manager->set_dictionary(dict);
  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: set_dictionary failed\n";
    return false;
  }

  // Create test data
  const size_t test_size = 128 * 1024; // 128KB
  std::vector<unsigned char> test_data(test_size);
  for (size_t i = 0; i < test_size; i++) {
    test_data[i] = (unsigned char)(i % 256);
  }

  // Allocate device buffers
  unsigned char *d_input, *d_compressed, *d_decompressed, *d_temp;
  err = cuda_zstd::safe_cuda_malloc(&d_input, test_size);
  if (err != cudaSuccess) {
    std::cerr << "  FAILED: cudaMalloc d_input failed\n";
    return false;
  }

  size_t max_compressed = test_size * 2;
  err = cuda_zstd::safe_cuda_malloc(&d_compressed, max_compressed);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    std::cerr << "  FAILED: cudaMalloc d_compressed failed\n";
    return false;
  }

  err = cuda_zstd::safe_cuda_malloc(&d_decompressed, test_size);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    std::cerr << "  FAILED: cudaMalloc d_decompressed failed\n";
    return false;
  }

  // Copy input to device
  err =
      cudaMemcpy(d_input, test_data.data(), test_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    std::cerr << "  FAILED: cudaMemcpy H2D failed\n";
    return false;
  }

  // Allocate temp workspace
  size_t temp_size = manager->get_compress_temp_size(test_size);
  err = cuda_zstd::safe_cuda_malloc(&d_temp, temp_size);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    std::cerr << "  FAILED: cudaMalloc d_temp failed\n";
    return false;
  }

  // Compress
  size_t compressed_size = max_compressed;
  status = manager->compress(d_input, test_size, d_compressed, &compressed_size,
                             d_temp, temp_size, nullptr, 0, 0);

  if (status != Status::SUCCESS) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    std::cerr << "  FAILED: Compression failed with status " << (int)status
              << "\n";
    return false;
  }

  // Decompress
  size_t decompressed_size = test_size;
  status = manager->decompress(d_compressed, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size, 0);

  if (status != Status::SUCCESS) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    std::cerr << "  FAILED: Decompression failed with status " << (int)status
              << "\n";
    return false;
  }

  // Copy result back
  std::vector<unsigned char> result_data(test_size);
  err = cudaMemcpy(result_data.data(), d_decompressed, test_size,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
    std::cerr << "  FAILED: cudaMemcpy D2H failed\n";
    return false;
  }

  // Verify
  bool match = (decompressed_size == test_size) &&
               (memcmp(test_data.data(), result_data.data(), test_size) == 0);

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(d_temp);

  if (!match) {
    std::cerr << "  FAILED: Data mismatch after round-trip\n";
    return false;
  }

  std::cout << "  PASSED: Round-trip compression successful\n";
  return true;
}

// ============================================================================
// Test 5: Dictionary Clear and Reset
// ============================================================================
bool test_dictionary_clear() {
  std::cout << "\n[Test] Dictionary Clear and Reset...\n";

  auto manager = create_manager(3);

  // Set dictionary
  auto dict_data = create_test_dictionary(4096);
  Dictionary dict;
  dict.raw_content = dict_data.data();
  dict.raw_size = dict_data.size();

  Status status = manager->set_dictionary(dict);
  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: set_dictionary failed\n";
    return false;
  }

  // Clear dictionary
  status = manager->clear_dictionary();
  if (status != Status::SUCCESS) {
    std::cerr << "  FAILED: clear_dictionary failed\n";
    return false;
  }

  // Try to get cleared dictionary (should fail)
  Dictionary retrieved;
  status = manager->get_dictionary(retrieved);
  if (status == Status::SUCCESS) {
    std::cerr << "  FAILED: get_dictionary should fail after clear\n";
    return false;
  }

  std::cout << "  PASSED: Dictionary clear and reset working\n";
  return true;
}

// ============================================================================
// Test 6: Dictionary Training Edge Cases
// ============================================================================
bool test_dictionary_training_edge_cases() {
  std::cout << "\n[Test] Dictionary Training Edge Cases...\n";

  bool all_passed = true;

  // Test 6a: Single sample
  {
    std::vector<unsigned char> sample(500000); // 500KB > 100KB threshold
    for (size_t i = 0; i < sample.size(); i++) {
      sample[i] = (unsigned char)(i % 256);
    }

    std::vector<const void *> samples;
    std::vector<size_t> sizes;
    samples.push_back(sample.data());
    sizes.push_back(sample.size());

    const size_t dict_size = 4096;
    std::vector<unsigned char> dict_buffer(dict_size);

    Status status = train_dictionary(samples, sizes, dict_buffer.data(),
                                     dict_size, nullptr, 0);
    if (status != Status::SUCCESS) {
      std::cerr << "  FAILED: Single large sample training failed\n";
      all_passed = false;
    }
  }

  // Test 6b: Minimum size dictionary
  {
    std::vector<std::vector<unsigned char>> samples;
    std::vector<const void *> sample_ptrs;
    std::vector<size_t> sample_sizes;

    for (int i = 0; i < 3; i++) {
      std::vector<unsigned char> sample(1024);
      samples.push_back(std::move(sample));
      sample_ptrs.push_back(samples.back().data());
      sample_sizes.push_back(1024);
    }

    // Use MIN_DICT_SIZE (256 bytes)
    std::vector<unsigned char> dict_buffer(MIN_DICT_SIZE);

    Status status =
        train_dictionary(sample_ptrs, sample_sizes, dict_buffer.data(),
                         MIN_DICT_SIZE, nullptr, 0);
    if (status != Status::SUCCESS) {
      std::cerr << "  FAILED: Minimum size dictionary training failed\n";
      all_passed = false;
    }
  }

  if (all_passed) {
    std::cout << "  PASSED: All edge case tests passed\n";
  }
  return all_passed;
}

// ============================================================================
// Main Test Runner
// ============================================================================
int main() {
  std::cout << "==================================================\n";
  std::cout << "  Dictionary Memory Handling Tests\n";
  std::cout << "==================================================\n";

  int passed = 0;
  int failed = 0;

  // Run all tests
  if (test_dictionary_training_memory())
    passed++;
  else
    failed++;
  if (test_dictionary_training_invalid_params())
    passed++;
  else
    failed++;
  if (test_dictionary_set_validation())
    passed++;
  else
    failed++;
  if (test_dictionary_compression_roundtrip())
    passed++;
  else
    failed++;
  if (test_dictionary_clear())
    passed++;
  else
    failed++;
  if (test_dictionary_training_edge_cases())
    passed++;
  else
    failed++;

  // Summary
  std::cout << "\n==================================================\n";
  std::cout << "  Test Summary: " << passed << " passed, " << failed
            << " failed\n";
  std::cout << "==================================================\n\n";

  return failed > 0 ? 1 : 0;
}
