/**
 * benchmark_streaming_comparison.cu - Benchmark comparing streaming modes
 * 
 * Compares basic streaming vs streaming with window history.
 * Tests different chunk sizes and measures throughput and compression ratios.
 */

#include "cuda_zstd_manager.h"
#include "cuda_error_checking.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<size_t> chunk_sizes = {16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024};
    int num_iterations = 10;
    int num_chunks = 20;
};

struct BenchmarkResult {
    size_t chunk_size;
    double avg_time_ms;
    double throughput_mb_s;
    double compression_ratio;
    size_t total_input;
    size_t total_output;
};

// Generate compressible test data
void generate_compressible_data(std::vector<uint8_t>& data, size_t size, unsigned int seed = 42) {
    data.resize(size);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t)dist(rng);
    }
    
    size_t pos = 0;
    while (pos < size - 100) {
        if (dist(rng) < 76) {
            size_t len = 10 + (dist(rng) % 100);
            size_t offset = 1 + (dist(rng) % 2000);
            if (pos >= offset && pos + len < size) {
                for (size_t i = 0; i < len; ++i) {
                    data[pos + i] = data[pos - offset + i];
                }
                pos += len;
            } else {
                pos++;
            }
        } else {
            pos++;
        }
    }
}

// Run benchmark for basic streaming
BenchmarkResult run_benchmark_basic(size_t chunk_size, int num_chunks, int iterations) {
    BenchmarkResult result;
    result.chunk_size = chunk_size;
    result.total_input = chunk_size * num_chunks;
    result.total_output = 0;
    
    std::vector<std::vector<uint8_t>> chunks(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
        generate_compressible_data(chunks[i], chunk_size, 42 + i);
    }
    
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&d_input, chunk_size), BenchmarkResult{});
    CUDA_CHECK_RET(cudaMalloc(&d_output, chunk_size * 2), BenchmarkResult{});

    // Warmup
    ZstdStreamingManager manager;
    manager.init_compression(0, chunk_size * 2);
    CUDA_CHECK_RET(cudaMemcpy(d_input, chunks[0].data(), chunk_size, cudaMemcpyHostToDevice), BenchmarkResult{});
    size_t out_size = 0;
    manager.compress_chunk(d_input, chunk_size, d_output, &out_size, false, 0);
    
    double total_time_ms = 0;
    size_t total_compressed = 0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        ZstdStreamingManager mgr;
        mgr.init_compression(0, chunk_size * 2);
        
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_chunks; ++i) {
            CUDA_CHECK_RET(cudaMemcpy(d_input, chunks[i].data(), chunk_size, cudaMemcpyHostToDevice), BenchmarkResult{});
            size_t compressed = 0;
            mgr.compress_chunk(d_input, chunk_size, d_output, &compressed, (i == num_chunks - 1), 0);
            if (iter == 0) total_compressed += compressed;
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    result.avg_time_ms = total_time_ms / iterations;
    result.throughput_mb_s = (result.total_input / (1024.0 * 1024.0)) / (result.avg_time_ms / 1000.0);
    result.total_output = total_compressed;
    result.compression_ratio = (double)result.total_input / result.total_output;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// Run benchmark with history
BenchmarkResult run_benchmark_with_history(size_t chunk_size, int num_chunks, int iterations) {
    BenchmarkResult result;
    result.chunk_size = chunk_size;
    result.total_input = chunk_size * num_chunks;
    result.total_output = 0;
    
    std::vector<std::vector<uint8_t>> chunks(num_chunks);
    std::vector<uint8_t> base_pattern;
    generate_compressible_data(base_pattern, chunk_size, 123);
    
    for (int i = 0; i < num_chunks; ++i) {
        chunks[i] = base_pattern;
        for (size_t j = 0; j < 50; ++j) {
            chunks[i][j * 200] = (uint8_t)(i * 5);
        }
    }
    
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&d_input, chunk_size), BenchmarkResult{});
    CUDA_CHECK_RET(cudaMalloc(&d_output, chunk_size * 2), BenchmarkResult{});

    // Warmup
    ZstdStreamingManager manager;
    manager.init_compression_with_history(0, chunk_size * 2);
    CUDA_CHECK_RET(cudaMemcpy(d_input, chunks[0].data(), chunk_size, cudaMemcpyHostToDevice), BenchmarkResult{});
    size_t out_size = 0;
    manager.compress_chunk_with_history(d_input, chunk_size, d_output, &out_size, false, 0);
    
    double total_time_ms = 0;
    size_t total_compressed = 0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        ZstdStreamingManager mgr;
        mgr.init_compression_with_history(0, chunk_size * 2);
        
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_chunks; ++i) {
            CUDA_CHECK_RET(cudaMemcpy(d_input, chunks[i].data(), chunk_size, cudaMemcpyHostToDevice), BenchmarkResult{});
            size_t compressed = 0;
            mgr.compress_chunk_with_history(d_input, chunk_size, d_output, &compressed, (i == num_chunks - 1), 0);
            if (iter == 0) total_compressed += compressed;
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    result.avg_time_ms = total_time_ms / iterations;
    result.throughput_mb_s = (result.total_input / (1024.0 * 1024.0)) / (result.avg_time_ms / 1000.0);
    result.total_output = total_compressed;
    result.compression_ratio = (double)result.total_input / result.total_output;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// Print results
void print_results(const std::vector<BenchmarkResult>& basic_results,
                   const std::vector<BenchmarkResult>& history_results) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "STREAMING COMPARISON BENCHMARK" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << std::left << std::setw(12) << "Chunk"
              << std::setw(15) << "Mode"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput"
              << std::setw(15) << "Ratio"
              << std::setw(15) << "Savings"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    for (size_t i = 0; i < basic_results.size(); ++i) {
        const auto& basic = basic_results[i];
        const auto& history = history_results[i];
        
        double savings = 100.0 * (1.0 - 1.0 / history.compression_ratio);
        
        std::cout << std::left << std::setw(12) << (std::to_string(basic.chunk_size / 1024) + "KB")
                  << std::setw(15) << "Basic"
                  << std::setw(15) << std::fixed << std::setprecision(2) << basic.avg_time_ms
                  << std::setw(18) << std::fixed << std::setprecision(1) << basic.throughput_mb_s
                  << std::setw(15) << std::fixed << std::setprecision(2) << basic.compression_ratio
                  << std::setw(15) << std::fixed << std::setprecision(1) 
                  << (100.0 * (1.0 - 1.0 / basic.compression_ratio)) << "%"
                  << std::endl;
        
        std::cout << std::left << std::setw(12) << ""
                  << std::setw(15) << "With History"
                  << std::setw(15) << std::fixed << std::setprecision(2) << history.avg_time_ms
                  << std::setw(18) << std::fixed << std::setprecision(1) << history.throughput_mb_s
                  << std::setw(15) << std::fixed << std::setprecision(2) << history.compression_ratio
                  << std::setw(15) << std::fixed << std::setprecision(1) << savings << "%"
                  << std::endl;
        
        double ratio_improvement = 100.0 * (history.compression_ratio - basic.compression_ratio) / basic.compression_ratio;
        std::cout << std::left << std::setw(12) << ""
                  << std::setw(15) << "Improvement"
                  << std::setw(15) << ""
                  << std::setw(18) << ""
                  << std::setw(15) << std::fixed << std::setprecision(1) << ratio_improvement << "%"
                  << std::setw(15) << ""
                  << std::endl;
        
        if (i < basic_results.size() - 1) {
            std::cout << std::string(100, '-') << std::endl;
        }
    }
    
    std::cout << std::string(100, '=') << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Streaming Compression Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << std::endl;
    
    BenchmarkConfig config;
    
    if (argc > 1) config.num_iterations = std::atoi(argv[1]);
    if (argc > 2) config.num_chunks = std::atoi(argv[2]);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Iterations: " << config.num_iterations << std::endl;
    std::cout << "  Chunks per test: " << config.num_chunks << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchmarkResult> basic_results;
    std::vector<BenchmarkResult> history_results;
    
    std::cout << "Running benchmarks..." << std::endl;
    
    for (size_t chunk_size : config.chunk_sizes) {
        std::cout << "Testing " << (chunk_size / 1024) << "KB chunks..." << std::flush;
        
        auto basic = run_benchmark_basic(chunk_size, config.num_chunks, config.num_iterations);
        auto history = run_benchmark_with_history(chunk_size, config.num_chunks, config.num_iterations);
        
        basic_results.push_back(basic);
        history_results.push_back(history);
        
        std::cout << " Done" << std::endl;
    }
    
    print_results(basic_results, history_results);
    
    double avg_throughput = 0;
    for (const auto& r : history_results) {
        avg_throughput += r.throughput_mb_s;
    }
    avg_throughput /= history_results.size();
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Average throughput: " << std::fixed << std::setprecision(1) 
              << avg_throughput << " MB/s" << std::endl;
    
    return 0;
}