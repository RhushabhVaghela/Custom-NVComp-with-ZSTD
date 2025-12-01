#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace cuda_zstd;
using namespace std::chrono;

int main() {
    size_t input_size = 10 * 1024 * 1024; // 10 MB
    u32 block_size = 1264 * 1024;
    
    std::cout << "Performance Test: Single 10MB Compression\n";
    std::cout << "=========================================\n\n";
    
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) h_input[i] = i % 256;
    
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = block_size;
    
    auto start = high_resolution_clock::now();
    ZstdBatchManager mgr(config);
    auto after_manager = high_resolution_clock::now();
    
    void *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in, input_size);
    cudaMemcpy(d_in, h_input.data(), input_size, cudaMemcpyHostToDevice);
    auto after_upload = high_resolution_clock::now();
    
    size_t max_out = mgr.get_max_compressed_size(input_size);
    size_t tmp_size = mgr.get_compress_temp_size(input_size);
    
    cudaMalloc(&d_out, max_out);
    cudaMalloc(&d_tmp, tmp_size);
    auto after_alloc = high_resolution_clock::now();
    
    size_t out_size = max_out;
    auto compress_start = high_resolution_clock::now();
    Status status = mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    auto compress_end = high_resolution_clock::now();
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    
    // Calculate durations
    auto manager_time = duration_cast<milliseconds>(after_manager - start).count();
    auto upload_time = duration_cast<milliseconds>(after_upload - after_manager).count();
    auto alloc_time = duration_cast<milliseconds>(after_alloc - after_upload).count();
    auto compress_time = duration_cast<milliseconds>(compress_end - compress_start).count();
    auto total_time = duration_cast<milliseconds>(compress_end - start).count();
    
    std::cout << "Timings:\n";
    std::cout << "  Manager creation: " << manager_time << " ms\n";
    std::cout << "  Data upload:      " << upload_time << " ms\n";
    std::cout << "  Buffer alloc:     " << alloc_time << " ms\n";
    std::cout << "  Compression:      " << compress_time << " ms\n";
    std::cout << "  Total:            " << total_time << " ms\n\n";
    
    double throughput = (input_size / (1024.0 * 1024.0)) / (compress_time / 1000.0);
    std::cout << "Throughput: " << throughput << " MB/s\n";
    std::cout << "Result: " << (status == Status::SUCCESS ? "PASS" : "FAIL") << "\n";
    std::cout << "Compressed size: " << out_size << " bytes\n";
    
    return (status == Status::SUCCESS) ? 0 : 1;
}
