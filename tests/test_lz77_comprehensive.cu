// ============================================================================
// test_lz77_comprehensive.cu - Comprehensive LZ77 Correctness & Robustness
// ============================================================================

#include "cuda_zstd_nvcomp.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

using namespace cuda_zstd::nvcomp_v5;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

// Helper to check Status
#ifdef CHECK_STATUS
#undef CHECK_STATUS
#endif
#define CHECK_STATUS(call) \
    do { \
        cuda_zstd::Status status = call; \
        if (status != cuda_zstd::Status::SUCCESS) { \
            std::cerr << "Status error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << (int)status << std::endl; \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// Data Generators
// ============================================================================

void gen_random(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint16_t> dist(0, 255);
    for(size_t i=0; i<size; ++i) data[i] = (uint8_t)dist(rng);
}

void gen_repetitive(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    for(size_t i=0; i<size; ++i) data[i] = 'A';
}

void gen_periodic(std::vector<uint8_t>& data, size_t size, int period) {
    data.resize(size);
    for(size_t i=0; i<size; ++i) data[i] = (uint8_t)(i % period);
}

void gen_long_matches(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // Fill with random
    gen_random(data, size);
    // Insert long matches
    size_t match_len = 10000;
    if (size > match_len * 2) {
        // Copy first 10000 bytes to the second half
        std::copy(data.begin(), data.begin() + match_len, data.begin() + size/2);
    }
}

void gen_distant_matches(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    gen_random(data, size);
    // Copy a small chunk from beginning to end (testing window limits)
    size_t chunk = 128;
    if (size > chunk) {
        std::copy(data.begin(), data.begin() + chunk, data.end() - chunk);
    }
}

// ============================================================================
// Test Runner
// ============================================================================

void run_test(const std::string& name, const std::vector<uint8_t>& input, bool expect_compression) {
    std::cout << "Test: " << name << " (" << input.size() << " bytes)... ";
    
    void* d_input;
    void* d_output;
    void* d_decomp;
    void* d_temp;
    
    size_t input_size = input.size();
    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));
    
    NvcompV5Options opts;
    opts.level = 3;
    NvcompV5BatchManager manager(opts);
    
    size_t max_out = manager.get_max_compressed_chunk_size(input_size);
    CHECK_CUDA(cudaMalloc(&d_output, max_out));
    
    size_t temp_size = manager.get_compress_temp_size(&input_size, 1);
    CHECK_CUDA(cudaMalloc(&d_temp, temp_size));
    
    void* h_uncompressed_ptrs[] = { d_input };
    void* h_compressed_ptrs[] = { d_output };
    
    void** d_uncompressed_ptrs_array;
    void** d_compressed_ptrs_array;
    CHECK_CUDA(cudaMalloc(&d_uncompressed_ptrs_array, sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_compressed_ptrs_array, sizeof(void*)));
    CHECK_CUDA(cudaMemcpy(d_uncompressed_ptrs_array, h_uncompressed_ptrs, sizeof(void*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_compressed_ptrs_array, h_compressed_ptrs, sizeof(void*), cudaMemcpyHostToDevice));
    
    size_t d_compressed_size_val = max_out;
    size_t* d_compressed_size;
    CHECK_CUDA(cudaMalloc(&d_compressed_size, sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_compressed_size, &d_compressed_size_val, sizeof(size_t), cudaMemcpyHostToDevice));
    
    // Compress
    CHECK_STATUS(manager.compress_async(
        (const void* const*)d_uncompressed_ptrs_array, &input_size, 1,
        d_compressed_ptrs_array, d_compressed_size,
        d_temp, temp_size
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    size_t compressed_size;
    CHECK_CUDA(cudaMemcpy(&compressed_size, d_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    // Verify compression ratio
    double ratio = (double)input_size / compressed_size;
    if (expect_compression && ratio < 1.1) {
        std::cout << "[WARNING] Expected compression but ratio is " << ratio << " ";
    }
    
    // Decompress
    CHECK_CUDA(cudaMalloc(&d_decomp, input_size));
    void* h_decomp_ptrs[] = { d_decomp };
    void** d_decomp_ptrs_array;
    CHECK_CUDA(cudaMalloc(&d_decomp_ptrs_array, sizeof(void*)));
    CHECK_CUDA(cudaMemcpy(d_decomp_ptrs_array, h_decomp_ptrs, sizeof(void*), cudaMemcpyHostToDevice));
    
    size_t d_decomp_size_val = input_size; // Capacity
    size_t* d_decomp_size;
    CHECK_CUDA(cudaMalloc(&d_decomp_size, sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_decomp_size, &d_decomp_size_val, sizeof(size_t), cudaMemcpyHostToDevice));
    
    size_t decomp_temp_size = manager.get_decompress_temp_size(&compressed_size, 1);
    if (decomp_temp_size > temp_size) {
        CHECK_CUDA(cudaFree(d_temp));
        CHECK_CUDA(cudaMalloc(&d_temp, decomp_temp_size));
    }
    
    CHECK_STATUS(manager.decompress_async(
        (const void* const*)d_compressed_ptrs_array, d_compressed_size, 1,
        d_decomp_ptrs_array, d_decomp_size,
        d_temp, decomp_temp_size
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Verify
    std::vector<uint8_t> output(input_size);
    CHECK_CUDA(cudaMemcpy(output.data(), d_decomp, input_size, cudaMemcpyDeviceToHost));
    
    if (output != input) {
        std::cerr << "FAILED: Data mismatch!" << std::endl;
        exit(1);
    }
    
    std::cout << "✅ Passed (Ratio: " << ratio << ")" << std::endl;
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_decomp));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_compressed_size));
    CHECK_CUDA(cudaFree(d_decomp_size));
    CHECK_CUDA(cudaFree(d_uncompressed_ptrs_array));
    CHECK_CUDA(cudaFree(d_compressed_ptrs_array));
    CHECK_CUDA(cudaFree(d_decomp_ptrs_array));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Comprehensive LZ77 Testing" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<uint8_t> data;
    
    // 1. Random Data (Should not compress)
    gen_random(data, 1024 * 1024);
    run_test("Random 1MB", data, false);
    
    // 2. Repetitive Data (Should compress extremely well)
    gen_repetitive(data, 1024 * 1024);
    run_test("Repetitive 1MB", data, true);
    
    // 3. Periodic Data (Should compress well)
    gen_periodic(data, 1024 * 1024, 100);
    run_test("Periodic 1MB", data, true);
    
    // 4. Long Matches
    gen_long_matches(data, 1024 * 1024);
    run_test("Long Matches 1MB", data, true);
    
    // 5. Distant Matches (Window size test)
    // ZSTD default window is usually large enough for 1MB
    gen_distant_matches(data, 1024 * 1024);
    run_test("Distant Matches 1MB", data, true);
    
    // 6. Small Input
    gen_random(data, 100);
    run_test("Small Random 100B", data, false);
    
    // 7. Empty Input (Edge case)
    // Note: NVCOMP API might handle 0 size differently, but let's try
    // run_test("Empty Input", {}, false); // Skip for now, might need special handling
    
    std::cout << "========================================" << std::endl;
    std::cout << "All LZ77 tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
