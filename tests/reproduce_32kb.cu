#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_zstd_manager.h"

using namespace cuda_zstd;

void generate_cuberoot_data(std::vector<uint8_t>& data, size_t size, int k) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t)(std::cbrt(i) * k);
    }
}

int main() {
    size_t input_size = 256 * 1024; // 256 KB
    u32 block_size = 128 * 1024;    // 128 KB
    
    std::cout << "Reproduction Test: 256KB Input, 128KB Blocks\n";
    std::cout << "------------------------------------------\n";
    
    // Generate data
    std::vector<uint8_t> h_input(input_size);
    generate_cuberoot_data(h_input, input_size, 400); // Sqrt_K400 equivalent
    
    // Setup Manager
    CompressionConfig config;
    config.block_size = block_size;
    
    ZstdBatchManager manager(config);
    
    // Allocate Device Memory
    void* d_input;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess) {
        std::cout << "cudaMalloc input failed\n";
        return 1;
    }
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    size_t max_compressed = manager.get_max_compressed_size(input_size);
    size_t temp_size = manager.get_compress_temp_size(input_size);
    
    void* d_compressed;
    void* d_temp;
    cudaMalloc(&d_compressed, max_compressed);
    cudaMalloc(&d_temp, temp_size);
    
    // Run Compress
    size_t compressed_size = max_compressed;
    std::cout << "Running compress...\n";
    
    Status status = manager.compress(d_input, input_size, d_compressed, &compressed_size,
                                    d_temp, temp_size, nullptr, 0, 0);
    
    cudaDeviceSynchronize();
    
    if (status != Status::SUCCESS) {
        std::cout << "FAILED with error: " << (int)status << "\n";
        return 1;
    } else {
        std::cout << "SUCCESS!\n";
    }
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    
    return 0;
}
