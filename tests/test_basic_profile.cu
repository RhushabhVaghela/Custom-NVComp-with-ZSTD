// Minimal profiling test for CUDA Zstandard library
#include <cuda_runtime.h>
#include "cuda_error_checking.h"
#include <iostream>
#include <vector>
#include <cstdint>

using byte_t = unsigned char;
using u32 = uint32_t;

// Simple kernel to simulate compression work
__global__ void dummy_compress_kernel(const byte_t* input, byte_t* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simulate some work
        output[idx] = input[idx] ^ 0x55;
    }
}

int main() {
    std::cout << "CUDA Zstandard Profiling Test\n";
    std::cout << "==============================\n\n";
    
    // Check for CUDA device
    int deviceCount;
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
    
    // Test data size: 10 MB
    const size_t data_size = 10 * 1024 * 1024;
    std::cout << "Processing " << data_size / (1024*1024) << " MB of data\n\n";
    
    // Allocate host memory
    std::vector<byte_t> h_input(data_size);
    std::vector<byte_t> h_output(data_size);
    
    // Initialize input with pattern
    for (size_t i = 0; i < data_size; ++i) {
        h_input[i] = static_cast<byte_t>(i % 256);
    }
    
    // Allocate device memory
    byte_t *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Copy to device
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float h2d_time;
    cudaEventElapsedTime(&h2d_time, start, stop);
    std::cout << "Host->Device transfer: " << h2d_time << " ms\n";
    
    // Launch kernel (simulate compression)
    const int threads = 256;
    const int blocks = (data_size + threads - 1) / threads;
    
    cudaEventRecord(start);
    dummy_compress_kernel<<<blocks, threads>>>(d_input, d_output, data_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    std::cout << "Kernel execution: " << kernel_time << " ms\n";
    
    // Copy back
    cudaEventRecord(start);
    cudaMemcpy(h_output.data(), d_output, data_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float d2h_time;
    cudaEventElapsedTime(&d2h_time, start, stop);
    std::cout << "Device->Host transfer: " << d2h_time << " ms\n";
    
    float total_time = h2d_time + kernel_time + d2h_time;
    float throughput = (data_size / (1024.0 * 1024.0)) / (total_time / 1000.0);
    
    std::cout << "\nTotal time: " << total_time << " ms\n";
    std::cout << "Throughput: " << throughput << " MB/s\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nBasic profiling test completed successfully.\n";
    std::cout << "For detailed profiling, run with: nvprof ./profile_test\n";
    std::cout << "Or: nsys profile --stats=true ./profile_test\n";
    
    return 0;
}