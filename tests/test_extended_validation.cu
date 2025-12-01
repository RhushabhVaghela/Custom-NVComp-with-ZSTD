// Extended Validation Tests for Parallel Compression
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace cuda_zstd;

bool test_compression(const char* test_name, size_t input_size, bool use_random = false) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Input Size: " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Generate input data
    std::vector<uint8_t> h_input(input_size);
    if (use_random) {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(0, 255);
        for (size_t i = 0; i < input_size; ++i) {
            h_input[i] = dist(rng);
        }
    } else {
        for (size_t i = 0; i < input_size; ++i) {
            h_input[i] = (uint8_t)(i % 256);
        }
    }
    
    // Initialize manager
    CompressionConfig config = CompressionConfig::from_level(3);
    ZstdBatchManager manager(config);
    
    // Allocate GPU buffers
    void* d_input;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess) {
        std::cerr << "Failed to allocate d_input" << std::endl;
        return false;
    }
    if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }
    
    size_t max_compressed_size = manager.get_max_compressed_size(input_size);
    void* d_compressed;
    if (cudaMalloc(&d_compressed, max_compressed_size) != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }
    
    size_t temp_size = manager.get_compress_temp_size(input_size);
    void* d_temp;
    if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        return false;
    }
    
    // Compress
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t compressed_size = max_compressed_size;
    Status status = manager.compress(
        d_input, input_size,
        d_compressed, &compressed_size,
        d_temp, temp_size,
        nullptr, 0,
        0
    );
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    if (status != Status::SUCCESS) {
        std::cerr << "Compression Failed: " << (int)status << std::endl;
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        return false;
    }
    
    double ratio = (double)input_size / compressed_size;
    std::cout << "Compressed Size: " << compressed_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Compression Ratio: " << ratio << std::endl;
    
    // Decompress
    void* d_decompressed;
    if (cudaMalloc(&d_decompressed, input_size) != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        return false;
    }
    
    size_t decompressed_size = input_size;
    status = manager.decompress(
        d_compressed, compressed_size,
        d_decompressed, &decompressed_size,
        d_temp, temp_size,
        0
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    
    if (status != Status::SUCCESS) {
        std::cerr << "Decompression Failed: " << (int)status << std::endl;
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        cudaFree(d_decompressed);
        return false;
    }
    
    // Verify
    std::vector<uint8_t> h_output(input_size);
    if (cudaMemcpy(h_output.data(), d_decompressed, input_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        cudaFree(d_decompressed);
        return false;
    }
    
    bool match = (h_input == h_output);
    
    // Timing
    auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compress_end - start).count();
    auto decompress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decompress_end - compress_end).count();
    
    std::cout << "Compress Time: " << compress_ms << " ms" << std::endl;
    std::cout << "Decompress Time: " << decompress_ms << " ms" << std::endl;
    std::cout << "Compress Throughput: " << (input_size / (1024.0 * 1024.0)) / (compress_ms / 1000.0) << " MB/s" << std::endl;
    
    if (match) {
        std::cout << "✅ Verification PASSED" << std::endl;
    } else {
        std::cerr << "❌ Verification FAILED" << std::endl;
        for (size_t i = 0; i < input_size; ++i) {
            if (h_input[i] != h_output[i]) {
                std::cerr << "First mismatch at index " << i << ": input=" << (int)h_input[i] 
                         << ", output=" << (int)h_output[i] << std::endl;
                break;
            }
        }
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    cudaFree(d_decompressed);
    
    return match;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   CUDA-ZSTD Extended Validation Test Suite" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Test 1: 2MB (small, below original test size)
    total++;
    if (test_compression("Test 1: 2MB Sequential Pattern", 2 * 1024 * 1024, false)) {
        passed++;
    }
    
    // Test 2: 10MB (original test)
    total++;
    if (test_compression("Test 2: 10MB Sequential Pattern", 10 * 1024 * 1024, false)) {
        passed++;
    }
    
    // Test 3: 50MB (larger scale)
    total++;
    if (test_compression("Test 3: 50MB Sequential Pattern", 50 * 1024 * 1024, false)) {
        passed++;
    }
    
    // Test 4: 2MB Random (incompressible)
    total++;
    if (test_compression("Test 4: 2MB Random Data", 2 * 1024 * 1024, true)) {
        passed++;
    }
    
    // Summary
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
