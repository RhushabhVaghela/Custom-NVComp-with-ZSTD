// Test for Parallel Compression (GPU Path)
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

using namespace cuda_zstd;

int main() {
    std::cout << "=== Parallel Compression Test (GPU Path) ===\n";

    // 1. Setup Input Data (Large enough to trigger GPU path > 1MB)
    size_t input_size = 10 * 1024 * 1024; // 10 MB
    std::cout << "Input Size: " << input_size / (1024.0 * 1024.0) << " MB\n";
    
    std::vector<uint8_t> h_input(input_size);
    // Generate compressible data (repetitive pattern)
    for (size_t i = 0; i < input_size; ++i) {
        h_input[i] = (uint8_t)(i % 256); // Simple pattern
    }

    // 2. Initialize Manager
    CompressionConfig config = CompressionConfig::from_level(3);
    ZstdBatchManager manager(config);
    
    // 3. Allocate Buffers
    void* d_input;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess) return 1;
    if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    
    size_t max_compressed_size = manager.get_max_compressed_size(input_size);
    void* d_compressed;
    if (cudaMalloc(&d_compressed, max_compressed_size) != cudaSuccess) return 1;
    
    size_t compressed_size = max_compressed_size;
    
    size_t temp_size = manager.get_compress_temp_size(input_size);
    void* d_temp;
    if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) return 1;
    
    std::cout << "Allocated Workspace: " << temp_size / (1024.0 * 1024.0) << " MB\n";
    
    // 4. Compress
    std::cout << "Compressing...\n";
    Status status = manager.compress(
        d_input, input_size,
        d_compressed, &compressed_size,
        d_temp, temp_size,
        nullptr, 0, // No dictionary
        0 // Default stream
    );
    
    if (status != Status::SUCCESS) {
        std::cerr << "Compression Failed: " << (int)status << "\n";
        return 1;
    }
    
    std::cout << "Compression Successful! Size: " << compressed_size << " bytes\n";
    std::cout << "Ratio: " << (double)input_size / compressed_size << "\n";
    
    // 5. Decompress
    std::cout << "Decompressing...\n";
    void* d_decompressed;
    if (cudaMalloc(&d_decompressed, input_size) != cudaSuccess) return 1;
    size_t decompressed_size = input_size;
    
    status = manager.decompress(
        d_compressed, compressed_size,
        d_decompressed, &decompressed_size,
        d_temp, temp_size,
        0 // Default stream
    );
    
    if (status != Status::SUCCESS) {
        std::cerr << "Decompression Failed: " << (int)status << "\n";
        return 1;
    }
    
    std::vector<uint8_t> h_output(input_size);
    if (cudaMemcpy(h_output.data(), d_decompressed, input_size, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    
    if (h_input == h_output) {
        std::cout << "Verification PASSED: Data matches exactly.\n";
    } else {
        std::cerr << "Verification FAILED: Data mismatch!\n";
        for (size_t i = 0; i < input_size; ++i) {
            if (h_input[i] != h_output[i]) {
                std::cerr << "First mismatch at index " << i << ": input=" << (int)h_input[i] << ", output=" << (int)h_output[i] << "\n";
                size_t block_idx = i / (128 * 1024);
                std::cerr << "Mismatch in Block " << block_idx << "\n";
                break;
            }
        }
        return 1;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    cudaFree(d_decompressed);
    
    return 0;
}
