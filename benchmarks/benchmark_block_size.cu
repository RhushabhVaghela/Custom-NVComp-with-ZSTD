// Block Size Benchmarking Tool
// MODIFIED for RTX 5080 (16GB VRAM) - Safe memory limits

#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

// Hardware-safe constants for RTX 5080 (16GB VRAM)
#define MAX_INPUT_SIZE (256ULL * 1024 * 1024)          // Max 256MB
#define MAX_BLOCK_SIZE (2ULL * 1024 * 1024)            // Max 2MB block size
#define MAX_VRAM_USAGE (8ULL * 1024 * 1024 * 1024)     // 8GB VRAM limit per test

using namespace cuda_zstd;

struct BenchmarkResult {
    size_t input_size;
    u32 block_size;
    u32 num_blocks;
    double compress_time_ms;
    double throughput_mbps;
    bool success;
};

// Memory safety check
bool check_memory_safety(size_t input_size, u32 block_size) {
    size_t estimated_memory = input_size * 4; // 4x overhead worst case
    if (estimated_memory > MAX_VRAM_USAGE) {
        return false;
    }
    if (block_size > MAX_BLOCK_SIZE) {
        return false;
    }
    return true;
}

bool benchmark_block_size(size_t input_size, u32 block_size, BenchmarkResult& result) {
    // Memory safety check
    if (!check_memory_safety(input_size, block_size)) {
        result.success = false;
        return false;
    }
    
    result.input_size = input_size;
    result.block_size = block_size;
    result.num_blocks = (input_size + block_size - 1) / block_size;
    result.success = false;
    
    // Generate test data
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        h_input[i] = (uint8_t)(i % 256);
    }
    
    // Setup
    CompressionConfig config = CompressionConfig::from_level(3);
    ZstdBatchManager manager(config);
    
    void* d_input;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess) return false;
    if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }
    
    size_t max_compressed_size = manager.get_max_compressed_size(input_size);
    void* d_compressed;
    if (cudaMalloc(&d_compressed, max_compressed_size) != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }
    
    size_t temp_size = manager.get_compress_temp_size(input_size);
    void* d_temp;
    if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_compressed);
        return false;
    }
    
    // Warmup
    size_t compressed_size = max_compressed_size;
    manager.compress(d_input, input_size, d_compressed, &compressed_size, 
                    d_temp, temp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    
    // Benchmark (3 runs, take average)
    const int NUM_RUNS = 3;
    double total_time = 0.0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        compressed_size = max_compressed_size;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        Status status = manager.compress(
            d_input, input_size,
            d_compressed, &compressed_size,
            d_temp, temp_size,
            nullptr, 0, 0
        );
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status != Status::SUCCESS) {
            cudaFree(d_input);
            cudaFree(d_compressed);
            cudaFree(d_temp);
            return false;
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0; // Convert to ms
    }

    result.compress_time_ms = total_time / NUM_RUNS;
    result.throughput_mbps = (input_size / (1024.0 * 1024.0)) / (result.compress_time_ms / 1000.0);
    result.success = true;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    
    return true;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Block Size Benchmark (RTX 5080 Safe Mode)\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Hardware Limits:\n";
    std::cout << "  Max Input Size: " << (MAX_INPUT_SIZE/1024/1024) << " MB\n";
    std::cout << "  Max Block Size: " << (MAX_BLOCK_SIZE/1024) << " KB\n";
    std::cout << "  Max VRAM Usage: " << (MAX_VRAM_USAGE/1024/1024/1024) << " GB\n\n";
    
    // Reduced input sizes for safety
    std::vector<size_t> input_sizes = {
        1 * 1024 * 1024,      // 1 MB
        4 * 1024 * 1024,      // 4 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
        128 * 1024 * 1024     // 128 MB
    };
    
    // Reduced block sizes for safety
    std::vector<u32> block_sizes = {
        128 * 1024,      // 128 KB
        256 * 1024,      // 256 KB
        512 * 1024,      // 512 KB
        1 * 1024 * 1024, // 1 MB
        2 * 1024 * 1024  // 2 MB (max)
    };
    
    std::vector<BenchmarkResult> all_results;
    
    // Run benchmarks
    for (size_t input_size : input_sizes) {
        std::cout << "\n=== Input Size: " << (input_size / (1024.0 * 1024.0)) << " MB ===\n";
        std::cout << std::setw(12) << "Block Size"
                  << std::setw(10) << "Blocks"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Status\n";
        std::cout << std::string(59, '-') << "\n";
        
        for (u32 block_size : block_sizes) {
            // Skip if memory would exceed limits
            if (!check_memory_safety(input_size, block_size)) {
                std::cout << std::setw(9) << (block_size / 1024) << " KB"
                          << std::setw(10) << "N/A"
                          << std::setw(12) << "SKIPPED"
                          << std::setw(15) << "(safety)"
                          << std::setw(10) << "-\n";
                continue;
            }
            
            BenchmarkResult result;
            bool success = benchmark_block_size(input_size, block_size, result);
            
            if (success) {
                all_results.push_back(result);
                
                std::cout << std::setw(9) << (block_size / 1024) << " KB"
                          << std::setw(10) << result.num_blocks
                          << std::setw(12) << std::fixed << std::setprecision(2) << result.compress_time_ms
                          << std::setw(12) << std::fixed << std::setprecision(2) << result.throughput_mbps << " MB/s"
                          << std::setw(10) << "OK\n";
            } else {
                std::cout << std::setw(9) << (block_size / 1024) << " KB"
                          << std::setw(10) << "N/A"
                          << std::setw(12) << "FAILED"
                          << std::setw(15) << "-"
                          << std::setw(10) << "X\n";
            }
        }
    }
    
    // Analysis: Find optimal block size for each input
    std::cout << "\n========================================\n";
    std::cout << "  Optimal Block Sizes\n";
    std::cout << "========================================\n\n";
    
    for (size_t input_size : input_sizes) {
        double best_throughput = 0.0;
        BenchmarkResult best_result;
        
        for (const auto& result : all_results) {
            if (result.input_size == input_size && result.throughput_mbps > best_throughput) {
                best_throughput = result.throughput_mbps;
                best_result = result;
            }
        }
        
        if (best_throughput > 0) {
            std::cout << std::setw(6) << (input_size / (1024 * 1024)) << " MB"
                      << " -> Best: " << std::setw(6) << (best_result.block_size / 1024) << " KB"
                      << " (" << std::setw(2) << best_result.num_blocks << " blocks, "
                      << std::fixed << std::setprecision(2) << best_result.throughput_mbps << " MB/s)\n";
        }
    }
    
    // Export CSV for analysis
    std::cout << "\n========================================\n";
    std::cout << "  CSV Export (copy to spreadsheet)\n";
    std::cout << "========================================\n\n";
    std::cout << "InputMB,BlockKB,NumBlocks,TimeMS,ThroughputMBPS\n";
    for (const auto& r : all_results) {
        std::cout << (r.input_size / (1024 * 1024)) << ","
                  << (r.block_size / 1024) << ","
                  << r.num_blocks << ","
                  << std::fixed << std::setprecision(3) << r.compress_time_ms << ","
                  << std::fixed << std::setprecision(3) << r.throughput_mbps << "\n";
    }
    
    return 0;
}
