// Debug Single Test: Find why 25MB fails
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace cuda_zstd;

int main() {
    std::cout << "=== DEBUG: Testing Sqrt_K400 @ 25MB ===\n\n";
    
    size_t input_size = 25 * 1024 * 1024;  // 25 MB
    u32 block_size = (u32)(std::sqrt((double)input_size) * 400.0);  // ~2000 KB
    
    std::cout << "Input: " << (input_size / (1024*1024)) << " MB\n";
    std::cout << "Block: " << (block_size / 1024) << " KB\n\n";
    
    // Generate test data
    std::cout << "1. Generating test data...\n";
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        h_input[i] = (uint8_t)(i % 256);
    }
    std::cout << "   OK\n\n";
    
    // Setup config
    std::cout << "2. Creating compression config...\n";
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = block_size;
    std::cout << "   Config: level=" << config.level << ", block=" << config.block_size << "\n\n";
    
    // Create manager
    std::cout << "3. Creating ZstdBatchManager...\n";
    ZstdBatchManager manager(config);
    std::cout << "   OK\n\n";
    
    // Allocate GPU memory
    std::cout << "4. Allocating GPU memory...\n";
    void* d_input;
    cudaError_t err = cudaMalloc(&d_input, input_size);
    std::cout << "   d_input: " << (err == cudaSuccess ? "OK" : cudaGetErrorString(err)) << "\n";
    if (err != cudaSuccess) return 1;
    
    err = cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    std::cout << "   memcpy: " << (err == cudaSuccess ? "OK" : cudaGetErrorString(err)) << "\n\n";
    if (err != cudaSuccess) { cudaFree(d_input); return 1; }
    
    // Get size requirements
    std::cout << "5. Getting size requirements...\n";
    size_t max_compressed = manager.get_max_compressed_size(input_size);
    size_t temp_size = manager.get_compress_temp_size(input_size);
    std::cout << "   Max compressed: " << (max_compressed / (1024*1024)) << " MB\n";
    std::cout << "   Temp workspace: " << (temp_size / (1024*1024)) << " MB\n";
    std::cout << "   Total GPU mem: " << ((input_size + max_compressed + temp_size) / (1024*1024)) << " MB\n\n";
    
    void* d_compressed;
    err = cudaMalloc(&d_compressed, max_compressed);
    std::cout << "   d_compressed: " << (err == cudaSuccess ? "OK" : cudaGetErrorString(err)) << "\n";
    if (err != cudaSuccess) { cudaFree(d_input); return 1; }
    
    void* d_temp;
    err = cudaMalloc(&d_temp, temp_size);
    std::cout << "   d_temp: " << (err == cudaSuccess ? "OK" : cudaGetErrorString(err)) << "\n\n";
    if (err != cudaSuccess) { cudaFree(d_input); cudaFree(d_compressed); return 1; }
    
    // Try compression
    std::cout << "6. Running compression (this is where it likely fails)...\n";
    size_t compressed_size = max_compressed;
    
    Status status = manager.compress(
        d_input, input_size,
        d_compressed, &compressed_size,
        d_temp, temp_size,
        nullptr, 0, 0
    );
    
    std::cout << "   Status: " << (int)status;
    if (status == Status::SUCCESS) {
        std::cout << " (SUCCESS)\n";
        std::cout << "   Compressed size: " << (compressed_size / 1024) << " KB\n";
    } else {
        std::cout << " (FAILED - Status code " << (int)status << ")\n";
        
        // Try to get more error info
        ErrorContext last_err = get_last_error();
        std::cout << "   Error status: " << (int)last_err.status << "\n";
        if (last_err.cuda_error != cudaSuccess) {
            std::cout << "   CUDA error: " << cudaGetErrorString(last_err.cuda_error) << "\n";
        }
    }
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    std::cout << "   Sync: " << (sync_err == cudaSuccess ? "OK" : cudaGetErrorString(sync_err)) << "\n\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    
    std::cout << (status == Status::SUCCESS ? "=== TEST PASSED ===\n" : "=== TEST FAILED ===\n");
    
    return (status == Status::SUCCESS) ? 0 : 1;
}
