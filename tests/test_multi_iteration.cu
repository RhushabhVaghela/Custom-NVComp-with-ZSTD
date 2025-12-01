#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>

using namespace cuda_zstd;

bool run_test(int iteration, size_t input_size, u32 block_size) {
    std::cout << "Iteration " << iteration << ": " << (input_size / (1024 * 1024)) << "MB, block=" << (block_size / 1024) << "KB" << std::endl;
    
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
    
    bool success = (status == Status::SUCCESS);
    std::cout << "  Result: " << (success ? "PASS" : "FAIL") << " (" << out_size << " bytes)" << std::endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    
    return success;
}

int main() {
    std::cout << "Multi-Iteration Compression Test\n";
    std::cout << "=================================\n\n";
    
    int passed = 0;
    int failed = 0;
    
    // Test various sizes and block sizes
    struct TestCase { size_t size; u32 block; };
    TestCase tests[] = {
        {10*1024*1024, 1264*1024},  // Sqrt_K400 @ 10MB
        {10*1024*1024, 2*1024*1024}, // 2MB blocks @ 10MB
        {25*1024*1024, 2*1024*1024}, // 2MB blocks @ 25MB
        {50*1024*1024, 4*1024*1024}, // 4MB blocks @ 50MB
        {10*1024*1024, 1264*1024},  // Repeat first test
    };
    
    for (int i = 0; i < 5; i++) {
        if (run_test(i+1, tests[i].size, tests[i].block)) {
            passed++;
        } else {
            failed++;
        }
    }
    
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success Rate: " << (passed * 100 / (passed + failed)) << "%" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
