// Limited Benchmark: Testing Known-Working Range (10MB - 100MB)
// Purpose: Validate library works before expanding to full range
// MODIFIED for RTX 5080 (16GB VRAM) - Safe memory limits

#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm> // For std::clamp, std::min
#include <functional> // For std::function

// Hardware-safe constants for RTX 5080 (16GB VRAM)
#define MAX_INPUT_SIZE (256ULL * 1024 * 1024)          // Max 256MB
#define MAX_BLOCK_SIZE (2ULL * 1024 * 1024)            // Max 2MB block size
#define MAX_VRAM_USAGE (8ULL * 1024 * 1024 * 1024)     // 8GB VRAM limit per test

using namespace cuda_zstd;

// Formula definitions
namespace formulas {
    u32 sqrt_k400(size_t input_size) {
        return (u32)(std::sqrt((double)input_size) * 400.0);
    }
    
    u32 logarithmic(size_t input_size) {
        double base = 512.0 * 1024.0;
        double power = std::pow(input_size / (1024.0 * 1024.0), 0.25);
        return (u32)(base * power);
    }
    
    u32 linear_128blocks(size_t input_size) {
        return (u32)(input_size / 128);
    }
    
    u32 cuberoot_k150(size_t input_size) {
        return (u32)(std::cbrt((double)input_size) * 150.0);
    }
    
    u32 piecewise(size_t input_size) {
        if (input_size < 10 * 1024 * 1024) return 2 * 1024 * 1024;
        if (input_size < 100 * 1024 * 1024) return 4 * 1024 * 1024;
        return 4 * 1024 * 1024; // Capped for safety
    }
    
    u32 hybrid(size_t input_size) {
        u32 ideal = (u32)(std::sqrt((double)input_size) * 400.0);
        size_t target_blocks = input_size / ideal;
        target_blocks = std::clamp(target_blocks, (size_t)64, (size_t)256);
        u32 block_size = (u32)(input_size / target_blocks);
        u32 power = (u32)std::ceil(std::log2(block_size));
        u32 result = (u32)(1 << power);
        if (result > MAX_BLOCK_SIZE) result = MAX_BLOCK_SIZE;
        return result;
    }
}

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

struct Result {
    std::string formula;
    size_t input_mb;
    u32 block_size;
    double time_ms;
    double throughput_mbps;
    Status status;
};

bool test_config(const std::string& formula_name, size_t input_size, u32 block_size, Result& result) {
    // Memory safety check
    if (!check_memory_safety(input_size, block_size)) {
        result.status = Status::ERROR_INVALID_PARAMETER;
        return false;
    }
    
    // Generate test data
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        h_input[i] = (uint8_t)(i % 256);
    }
    
    // Setup  
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = std::min(block_size, (u32)input_size);
    
    ZstdBatchManager manager(config);
    
    void* d_input;
    if (cudaMalloc(&d_input, input_size) != cudaSuccess) {
        std::cout << " [FAIL] cudaMalloc input failed\n";
        return false;
    }
    if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << " [FAIL] cudaMemcpy input failed\n";
        cudaFree(d_input);
        return false;
    }
    
    size_t max_compressed = manager.get_max_compressed_size(input_size);
    size_t temp_size = manager.get_compress_temp_size(input_size);
    
    void* d_compressed;
    void* d_temp;
    if (cudaMalloc(&d_compressed, max_compressed) != cudaSuccess) {
        std::cout << " [FAIL] cudaMalloc compressed failed\n";
        cudaFree(d_input);
        return false;
    }
    if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
        std::cout << " [FAIL] cudaMalloc temp failed\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        return false;
    }
    
    // Warmup
    size_t compressed_size = max_compressed;
    Status warmup_status = manager.compress(d_input, input_size, d_compressed, &compressed_size,
                                           d_temp, temp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    
    if (warmup_status != Status::SUCCESS) {
        std::cout << " [FAIL] Warmup failed: " << (int)warmup_status << "\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_temp);
        return false;
    }
    
    // Benchmark (3 runs)
    const int NUM_RUNS = 3;
    double total_time = 0.0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        compressed_size = max_compressed;
        
        auto start = std::chrono::high_resolution_clock::now();
        Status status = manager.compress(d_input, input_size, d_compressed, &compressed_size,
                                        d_temp, temp_size, nullptr, 0, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status != Status::SUCCESS) {
            result.status = status;
            cudaFree(d_input);
            cudaFree(d_compressed);
            cudaFree(d_temp);
            return false;
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count() / 1000.0;
    }
    
    result.formula = formula_name;
    result.input_mb = input_size / (1024 * 1024);
    result.block_size = block_size;
    result.time_ms = total_time / NUM_RUNS;
    result.throughput_mbps = (input_size / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);
    result.status = Status::SUCCESS;
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    
    return true;
}

int main() {
    std::cout << "  Limited Benchmark: 10MB-100MB Range (RTX 5080 Safe Mode)\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Hardware Limits:\n";
    std::cout << "  Max Input Size: " << (MAX_INPUT_SIZE/1024/1024) << " MB\n";
    std::cout << "  Max Block Size: " << (MAX_BLOCK_SIZE/1024) << " KB\n";
    std::cout << "  Max VRAM Usage: " << (MAX_VRAM_USAGE/1024/1024/1024) << " GB\n\n";
    
    std::vector<std::pair<std::string, std::function<u32(size_t)>>> formulas = {
        {"Sqrt_K400", formulas::sqrt_k400},
        {"Logarithmic", formulas::logarithmic},
        {"CubeRoot_K150", formulas::cuberoot_k150},
        {"Piecewise", formulas::piecewise},
        {"Hybrid", formulas::hybrid}
    };
    
    // Limited size range: known to work and safe for RTX 5080
    std::vector<size_t> input_sizes = {
        10 * 1024 * 1024,   // 10 MB  
        25 * 1024 * 1024,   // 25 MB
        50 * 1024 * 1024,   // 50 MB
        100 * 1024 * 1024   // 100 MB
    };
    
    std::vector<Result> all_results;
    int test_num = 0;
    int total_tests = formulas.size() * input_sizes.size();
    
    // Run tests
    for (const auto& [fname, ffunc] : formulas) {
        for (size_t input_size : input_sizes) {
            test_num++;
            u32 block_size = ffunc(input_size);
            
            // Safety check
            if (!check_memory_safety(input_size, block_size)) {
                std::cout << "[" << test_num << "/" << total_tests << "] "
                          << fname << " @ " << (input_size / (1024 * 1024)) << "MB "
                          << "(block=" << (block_size / 1024) << "KB)... SKIPPED (safety)\n";
                continue;
            }
            
            std::cout << "[" << test_num << "/" << total_tests << "] "
                      << fname << " @ " << (input_size / (1024 * 1024)) << "MB "
                      << "(block=" << (block_size / 1024) << "KB)... " << std::flush;
            
            Result result;
            bool success = test_config(fname, input_size, block_size, result);
            
            if (success) {
                all_results.push_back(result);
                std::cout << "OK (" << std::fixed << std::setprecision(2) 
                          << result.throughput_mbps << " MB/s)\n";
            } else {
                std::cout << "FAILED (Error: " << (int)result.status << ")\n";
            }
        }
    }
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Results Summary\n";
    std::cout << "========================================\n";
    std::cout << "Successful tests: " << all_results.size() << "/" << total_tests << "\n\n";
    
    // CSV Export
    std::cout << "Formula,InputMB,BlockKB,TimeMS,ThroughputMBPS\n";
    for (const auto& r : all_results) {
        std::cout << r.formula << ","
                  << r.input_mb << ","
                  << (r.block_size / 1024) << ","
                  << std::fixed << std::setprecision(3) << r.time_ms << ","
                  << std::fixed << std::setprecision(2) << r.throughput_mbps << "\n";
    }
    
    return 0;
}
