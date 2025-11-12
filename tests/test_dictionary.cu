// ============================================================================
// test_dictionary.cu - Verify Dictionary Training and Usage
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_dictionary.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

using namespace cuda_zstd;
using namespace cuda_zstd::dictionary;

// Helper to create repetitive sample data (good for dictionaries)
std::vector<byte_t> create_sample(size_t size, const char* prefix) {
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
    std::cout << "\n========================================\n";
    std::cout << "  Test: Dictionary Training & Usage\n";
    std::cout << "========================================\n\n";

    // 1. Create training data
    std::cout << "Creating training samples...\n";
    std::vector<std::vector<byte_t>> h_samples;
    std::vector<const byte_t*> h_sample_ptrs;
    std::vector<size_t> h_sample_sizes;
    
    h_samples.push_back(create_sample(1024 * 64, "COMMON_HEADER_PATTERN_"));
    h_samples.push_back(create_sample(1024 * 32, "COMMON_HEADER_PATTERN_"));
    h_samples.push_back(create_sample(1024 * 48, "COMMON_HEADER_PATTERN_"));
    
    for(const auto& s : h_samples) {
        h_sample_ptrs.push_back(s.data());
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
        h_sample_ptrs, h_sample_sizes,
        gpu_dict, 32 * 1024, params, 0
    );
    cudaDeviceSynchronize();
    
    if (status != Status::SUCCESS || gpu_dict.raw_size == 0) {
        std::cerr << "  ✗ FAILED: Dictionary training failed.\n";
        return 1;
    }
    std::cout << "  ✓ Dictionary trained. Size: " << gpu_dict.raw_size << " bytes.\n";

    // 3. Create test data (similar, but not identical)
    std::vector<byte_t> h_test_data = create_sample(1024 * 128, "COMMON_HEADER_PATTERN_");
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, h_test_data.size());
    cudaMalloc(&d_output, h_test_data.size() * 2);
    cudaMemcpy(d_input, h_test_data.data(), h_test_data.size(), cudaMemcpyHostToDevice);
    
    auto manager = create_manager(5);
    size_t temp_size = manager->get_compress_temp_size(h_test_data.size());
    cudaMalloc(&d_temp, temp_size);

    // 4. Test 1: Compress *without* dictionary
    std::cout << "Compressing *without* dictionary...\n";
    size_t size_no_dict = 0;
    manager->compress(d_input, h_test_data.size(), d_output, &size_no_dict, d_temp, temp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    std::cout << "  Compressed size: " << size_no_dict << " bytes.\n";

    // 5. Test 2: Compress *with* dictionary
    std::cout << "Compressing *with* dictionary...\n";
    manager->set_dictionary(gpu_dict); // Set the dictionary
    size_t size_with_dict = 0;
    manager->compress(d_input, h_test_data.size(), d_output, &size_with_dict, d_temp, temp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    std::cout << "  Compressed size: " << size_with_dict << " bytes.\n";

    // 6. Verify
    if (size_with_dict < size_no_dict) {
        std::cout << "  ✓ PASSED: Dictionary compression is "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (size_no_dict - size_with_dict) / size_no_dict)
                  << "% smaller.\n";
    } else {
        std::cout << "  ✗ FAILED: Dictionary compression was not smaller.\n";
    }

    DictionaryManager::free_dictionary_gpu(gpu_dict, 0);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);

    if (size_with_dict < size_no_dict) {
        return 0;
    } else {
        return 1;
    }
}