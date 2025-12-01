#include "cuda_zstd_manager.h"
#include <iostream>
#include <chrono>

using namespace cuda_zstd;

int main() {
    size_t input_size = 10 * 1024 * 1024;
    
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) h_input[i] = i % 256;
    
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = 1264 * 1024;
    ZstdBatchManager mgr(config);
    
    void *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in, input_size);
    cudaMemcpy(d_in, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    size_t max_out = mgr.get_max_compressed_size(input_size);
    size_t tmp_size = mgr.get_compress_temp_size(input_size);
    cudaMalloc(&d_out, max_out);
    cudaMalloc(&d_tmp, tmp_size);
    
    // Warmup
    size_t out_size = max_out;
    mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    out_size = max_out;
    Status status = mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double throughput_mbps = (input_size / (1024.0 * 1024.0)) / (duration_ms / 1000.0);
    
    printf("\\n=== PERFORMANCE RESULTS ===\\n");
    printf("Input size:    10 MB\\n");
    printf("Compress time: %ld ms\\n", duration_ms);
    printf("Throughput:    %.2f MB/s\\n", throughput_mbps);
    printf("Result:        %s\\n", status == Status::SUCCESS ? "PASS" : "FAIL");
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    
    return (status == Status::SUCCESS) ? 0 : 1;
}
