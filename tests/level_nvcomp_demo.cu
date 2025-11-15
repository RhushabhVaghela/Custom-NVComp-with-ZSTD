// ============================================================================
// level_nvcomp_demo.cu - Level-Based Compression Examples
// ============================================================================

#include "cuda_zstd_nvcomp.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <numeric>

using namespace cuda_zstd;
using namespace cuda_zstd::nvcomp_v5;

// ============================================================================
// Example 1: Performance vs. Ratio Trade-off
// ============================================================================

void example_performance_tradeoff() {
    std::cout << "=== Example 1: Performance vs. Ratio Trade-off ===\n";
    
    size_t data_size = 10 * 1024 * 1024; // 10 MB
    std::vector<uint8_t> input_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        input_data[i] = static_cast<uint8_t>((i * 31) % 256);
    }
    
    void* d_input;
    cudaMalloc(&d_input, data_size);
    cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
    
    std::cout << "\nInput size: " << data_size / (1024*1024) << " MB\n";
    std::cout << "\nLevel | Speed (MB/s) | Compressed Size (KB) | Category\n";
    std::cout << "------+--------------+----------------------+-----------\n";
    
    const size_t chunk_sizes[] = { data_size };
    const void* d_uncompressed_ptrs[] = { d_input };
    
    for (int level : {1, 3, 5, 9, 15, 22}) {
        NvcompV5Options opts;
        opts.level = level;
        NvcompV5BatchManager manager(opts);
        
        size_t max_compressed = manager.get_max_compressed_chunk_size(data_size);
        void* d_output;
        cudaMalloc(&d_output, max_compressed);
        
        size_t temp_size = manager.get_compress_temp_size(chunk_sizes, 1);
        void* d_temp;
        cudaMalloc(&d_temp, temp_size);
        
        void* d_compressed_ptrs[] = { d_output };
        size_t compressed_size;

        auto start = std::chrono::high_resolution_clock::now();
        Status status = manager.compress_async(
            d_uncompressed_ptrs, chunk_sizes, 1,
            d_compressed_ptrs, &compressed_size,
            d_temp, temp_size
        );
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == Status::SUCCESS) {
            double time_s = std::chrono::duration<double>(end - start).count();
            double speed_mbps = (data_size / (1024.0 * 1024.0)) / time_s;
            
            printf("%5d | %12.1f | %20.2f | %-10s\n",
                   level, speed_mbps, compressed_size / 1024.0, "");
        }
        
        cudaFree(d_output);
        cudaFree(d_temp);
    }
    
    cudaFree(d_input);
}

// ============================================================================
// Example 2: Batch Compression with NVCOMP API
// ============================================================================

void example_batch_compression_nvcomp() {
    std::cout << "\n=== Example 2: Batch Compression with NVCOMP API ===\n";
    
    const int batch_size = 8;
    const size_t chunk_size = 128 * 1024;
    
    std::cout << "Batch size: " << batch_size << " items\n";
    std::cout << "Item size: " << chunk_size / 1024 << " KB\n\n";
    
    NvcompV5Options opts;
    opts.level = 7;
    NvcompV5BatchManager manager(opts);
    
    std::vector<void*> d_inputs(batch_size);
    std::vector<void*> d_outputs(batch_size);
    std::vector<size_t> h_input_sizes(batch_size, chunk_size);
    
    size_t max_compressed = manager.get_max_compressed_chunk_size(chunk_size);
    
    for (int i = 0; i < batch_size; ++i) {
        cudaMalloc(&d_inputs[i], chunk_size);
        cudaMalloc(&d_outputs[i], max_compressed);
        
        std::vector<uint8_t> data(chunk_size);
        for (size_t j = 0; j < chunk_size; ++j) {
            data[j] = static_cast<uint8_t>((i * 1000 + j) % 256);
        }
        cudaMemcpy(d_inputs[i], data.data(), chunk_size, cudaMemcpyHostToDevice);
    }
    
    void** d_input_ptrs;
    void** d_output_ptrs;
    size_t* d_output_sizes;
    cudaMalloc(&d_input_ptrs, batch_size * sizeof(void*));
    cudaMalloc(&d_output_ptrs, batch_size * sizeof(void*));
    cudaMalloc(&d_output_sizes, batch_size * sizeof(size_t));
    cudaMemcpy(d_input_ptrs, d_inputs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_ptrs, d_outputs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t temp_size = manager.get_compress_temp_size(h_input_sizes.data(), batch_size);
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    manager.compress_async(
        (const void* const*)d_input_ptrs, h_input_sizes.data(), batch_size,
        d_output_ptrs, d_output_sizes,
        d_temp, temp_size
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::vector<size_t> h_output_sizes(batch_size);
    cudaMemcpy(h_output_sizes.data(), d_output_sizes, batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);
    
    size_t total_input = batch_size * chunk_size;
    size_t total_output = 0;
    for(size_t s : h_output_sizes) total_output += s;
    
    std::cout << "Results:\n";
    std::cout << "  Total input: " << total_input / 1024 << " KB\n";
    std::cout << "  Total output: " << total_output / 1024 << " KB\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    
    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
    }
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
}

// ============================================================================
// Example 3: Metadata Validation
// ============================================================================

void example_metadata_validation() {
    std::cout << "\n=== Example 3: Metadata Validation ===\n\n";
    
    int test_level = 7;
    NvcompV5Options opts;
    opts.level = test_level;
    NvcompV5BatchManager manager(opts);
    std::cout << "Compressing one block with level: " << test_level << "\n";

    size_t data_size = 64 * 1024;
    std::vector<uint8_t> input_data(data_size, 'a');
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);

    const size_t chunk_sizes[] = { data_size };
    size_t temp_size = manager.get_compress_temp_size(chunk_sizes, 1);
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);

    const void* d_uncompressed_ptrs[] = { d_input };
    void* d_compressed_ptrs[] = { d_output };
    size_t compressed_size;

    manager.compress_async(
        d_uncompressed_ptrs, chunk_sizes, 1,
        d_compressed_ptrs, &compressed_size,
        d_temp, temp_size
    );
    cudaDeviceSynchronize();
    
    std::cout << "Compressed to " << compressed_size << " bytes.\n";

    std::cout << "Extracting metadata...\n";
    NvcompV5Metadata metadata;
    Status status = get_metadata(d_output, compressed_size, metadata);

    if (status != Status::SUCCESS) {
        std::cout << "  ERROR: Metadata extraction failed!\n";
    } else {
        std::cout << "  Metadata read successfully:\n";
        std::cout << "    Uncompressed Size: " << metadata.uncompressed_size << "\n";
        std::cout << "    Compression Level: " << metadata.compression_level << "\n";
        
        if (metadata.compression_level == test_level) {
            std::cout << "  ✓ PASSED: Compression level was correctly saved and read.\n";
        } else {
            std::cout << "  ✗ FAILED: Expected level " << test_level
                      << ", but read " << metadata.compression_level << ".\n";
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "CUDA Zstandard - NVCOMP API Examples\n";
    std::cout << "========================================\n\n";
    
    try {
        example_performance_tradeoff();
        example_batch_compression_nvcomp();
        example_metadata_validation();
        
        std::cout << "\n=== All Examples Complete ===\n\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
