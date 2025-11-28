#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_zstd_manager.h"

int main() {
    std::cout << "=== Minimal Compression Test ===" << std::endl;
    
    // Create a minimal input
    const size_t input_size = 1024;
    std::vector<uint8_t> input(input_size, 'A');
    
    std::cout << "Creating manager..." << std::endl;
    auto manager = cuda_zstd::create_manager();
    
    std::cout << "Getting workspace size..." << std::endl;
    size_t workspace_size = manager->get_compress_temp_size(input_size);
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
    
    std::cout << "Allocating workspace..." << std::endl;
    void* workspace = nullptr;
    cudaError_t err = cudaMalloc(&workspace, workspace_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "Allocating output buffer..." << std::endl;
    size_t max_compressed_size = manager->get_max_compressed_size(input_size);
    void* compressed = nullptr;
    err = cudaMalloc(&compressed, max_compressed_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc compressed failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(workspace);
        return 1;
    }
    
    std::cout << "Calling compress..." << std::endl;
    size_t compressed_size = 0;
    auto status = manager->compress(
        input.data(),
        input_size,
        compressed,
        &compressed_size,
        workspace,
        workspace_size,
        nullptr, // no dict
        0,
        0 // default stream
    );
    
    std::cout << "Compress status: " << static_cast<int>(status) << std::endl;
    std::cout << "Compressed size: " << compressed_size << std::endl;
    
    cudaFree(workspace);
    cudaFree(compressed);
    
    std::cout << "=== Test Complete ===" << std::endl;
    return (status == cuda_zstd::Status::SUCCESS) ? 0 : 1;
}
