// Test: Run SAME test twice to isolate state corruption
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>

using namespace cuda_zstd;

bool compress_test(int iteration) {
    size_t input_size = 10 * 1024 * 1024; // 10 MB
    u32 block_size = 1264 * 1024; // Sqrt_K400 formula result
    
    std::cout << "\n=== Iteration " << iteration << " ===" << std::endl;
    
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) h_input[i] = i % 256;
    
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = block_size;
    ZstdBatchManager mgr(config);
    
    void *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in, input_size);
    cudaMemcpy(d_in, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    size_t max_out = mgr.get_max_compressed_size(input_size);
    size_t tmp_size = mgr.get_compress_temp_size(input_size);
    
    cudaMalloc(&d_out, max_out);
    cudaMalloc(&d_tmp, tmp_size);
    
    size_t out_size = max_out;
    Status status = mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    
    std::cout << "Status: " << (int)status << " (" << (status == Status::SUCCESS ? "SUCCESS" : "FAILED") << ")" << std::endl;
    std::cout << "Compressed: " << out_size << " bytes" << std::endl;
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    std::cout << "CUDA error state: " << cudaGetErrorString(err) << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    
    return status == Status::SUCCESS;
}

int main() {
    std::cout << "Testing: Run Sqrt_K400@10MB twice to find state corruption\n";
    
    bool test1 = compress_test(1);
    bool test2 = compress_test(2);
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Test 1: " << (test1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 2: " << (test2 ? "PASS" : "FAIL") << std::endl;
    
    if (test1 && !test2) {
        std::cout << "\n**CONFIRMED**: State corruption after first compression!" << std::endl;
    } else if (test1 && test2) {
        std::cout << "\n**INTERESTING**: Both passed - issue is formula/size specific" << std::endl;
    }
    
    return 0;
}
