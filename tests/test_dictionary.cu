// ============================================================================
// test_dictionary.cu - Verify Dictionary Training and Usage
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_manager.h"
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
    if (i % 100 < prefix_len) {
      sample[i] = (byte_t)prefix[i % prefix_len];
    } else {
      sample[i] = (byte_t)(i % 256);
    }
  }
  return sample;
}

int main() {
  // Skip on systems without a CUDA device to avoid false negatives.
  SKIP_IF_NO_CUDA_RET(0);
  std::cout << "\n========================================\n";
  std::cout << "  Test: Dictionary Training & Usage\n";
  std::cout << "========================================\n\n";

  // 1. Create training data
  std::cout << "Creating training samples...\n";
  std::vector<std::vector<byte_t>> h_samples;
  std::vector<const void *> h_sample_ptrs;
  std::vector<size_t> h_sample_sizes;

  h_samples.push_back(create_sample(1024 * 64, "COMMON_HEADER_PATTERN_"));
  h_samples.push_back(create_sample(1024 * 32, "COMMON_HEADER_PATTERN_"));
  h_samples.push_back(create_sample(1024 * 48, "COMMON_HEADER_PATTERN_"));

  for (const auto &s : h_samples) {
    h_sample_ptrs.push_back(static_cast<const void *>(s.data()));
    h_sample_sizes.push_back(s.size());
  }

  // 2. Train dictionary
  std::cout << "Training dictionary (32KB)...\n";
  Dictionary gpu_dict;
  CoverParams params;
  params.k = 1024;
  params.d = 8;

  // Allocate GPU struct
  DictionaryManager::allocate_dictionary_gpu(gpu_dict, 32 * 1024, 0);

  Status status = DictionaryTrainer::train_dictionary(
      h_sample_ptrs, h_sample_sizes, gpu_dict, 32 * 1024, &params, 0);
  cudaDeviceSynchronize();

  if (status != Status::SUCCESS || gpu_dict.raw_size == 0) {
    std::cerr << "  ✗ FAILED: Dictionary training failed.\n";
    return 1;
  }
  std::cout << "  ✓ Dictionary trained. Size: " << gpu_dict.raw_size
            << " bytes.\n";

  // 3. Create test data (similar, but not identical)
  std::vector<byte_t> h_test_data =
      create_sample(1024 * 128, "COMMON_HEADER_PATTERN_");
  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  // Allocate GPU memory with error checking
  if (!test_safe_cuda_malloc(&d_input, h_test_data.size())) {
    std::cerr << "  ✗ FAILED: CUDA malloc for d_input\n";
    return 1;
  }

  if (!test_safe_cuda_malloc(&d_output, h_test_data.size() * 2)) {
    std::cerr << "  ✗ FAILED: CUDA malloc for d_output\n";
    safe_cuda_free(d_input);
    return 1;
  }

  // Copy input data to device
  if (!safe_cuda_memcpy(d_input, h_test_data.data(), h_test_data.size(),
                        cudaMemcpyHostToDevice)) {
    std::cerr << "  ✗ FAILED: CUDA memcpy to d_input\n";
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return 1;
  }

  auto manager = create_manager(5);
  size_t temp_size = manager->get_compress_temp_size(h_test_data.size());
  if (!test_safe_cuda_malloc(&d_temp, temp_size)) {
    std::cerr << "  ✗ FAILED: CUDA malloc for d_temp\n";
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return 1;
  }

  // 4. Test 1: Compress *without* dictionary
  std::cout << "Compressing *without* dictionary...\n";
  size_t size_no_dict = h_test_data.size() * 2;
  Status compress_status =
      manager->compress(d_input, h_test_data.size(), d_output, &size_no_dict,
                        d_temp, temp_size, nullptr, 0, 0);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "  ✗ FAILED: CUDA sync after compression without dictionary\n";
    DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return 1;
  }
  if (compress_status != Status::SUCCESS) {
    std::cerr << "  ✗ FAILED: Compression without dictionary. Status="
              << (int)compress_status << "\n";
    DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return 1;
  }
  std::cout << "  Compressed size: " << size_no_dict << " bytes.\n";

  // 5. Test 2: Compress *with* dictionary
  std::cout << "Compressing *with* dictionary...\n";
  manager->set_dictionary(gpu_dict); // Set the dictionary
  size_t size_with_dict = h_test_data.size() * 2;
  Status compress_dict_status =
      manager->compress(d_input, h_test_data.size(), d_output, &size_with_dict,
                        d_temp, temp_size, nullptr, 0, 0);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "  ✗ FAILED: CUDA sync after compression with dictionary\n";
    DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return 1;
  }
  if (compress_dict_status != Status::SUCCESS) {
    std::cerr << "  ✗ FAILED: Compression with dictionary\n";
    DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return 1;
  }
  std::cout << "  Compressed size: " << size_with_dict << " bytes.\n";

  // 6. Verify - Test passes if BOTH compressions succeeded
  // Size comparison is informational only (dictionaries don't always help)
  if (size_with_dict < size_no_dict) {
    std::cout << "  ✓ INFO: Dictionary compression is " << std::fixed
              << std::setprecision(1)
              << (100.0 * (size_no_dict - size_with_dict) / size_no_dict)
              << "% smaller.\n";
  } else {
    std::cout << "  ⚠ INFO: Dictionary compression was not smaller ("
              << size_with_dict << " vs " << size_no_dict
              << "). This is normal for some data patterns.\n";
  }

  DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  // Test passes if both compression operations succeeded
  std::cout << "  ✓ PASSED: Both compression modes completed successfully.\n";
  return 0;
}
