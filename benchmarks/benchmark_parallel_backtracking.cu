#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include "cuda_zstd_lz77.h"
#include "cuda_zstd_utils.h"

// Helper to generate compressible data
void generate_compressible_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    
    // Fill with random data first
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t)dist(rng);
    }
    
    // Inject repeated patterns to make it compressible
    size_t pos = 0;
    while (pos < size - 100) {
        if (dist(rng) < 128) { // 50% chance of pattern
            size_t len = 10 + (dist(rng) % 50);
            size_t offset = 1 + (dist(rng) % 1000);
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

struct BenchmarkResult {
    std::string name;
    size_t input_size;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    double throughput_mb_s;
};

void run_benchmark(size_t size_mb, std::vector<BenchmarkResult>& results) {
    size_t input_size = size_mb * 1024 * 1024;
    std::cout << "\nRunning Benchmark: " << size_mb << " MB" << std::endl;
    
    std::vector<uint8_t> h_input;
    generate_compressible_data(h_input, input_size);
    
    // Allocate device memory
    uint8_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    
    // Setup LZ77
    LZ77Metadata metadata;
    size_t num_sequences = 0;
    
    // --- CPU Benchmark ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // Note: We are only benchmarking the backtracking part effectively if we isolate it,
    // but here we run the full LZ77 pipeline to get overall comparison or just the backtracking if exposed.
    // For this benchmark, let's assume we want to measure the `lz77_compress_parallel` vs a CPU equivalent.
    // However, since we don't have a pure CPU implementation of the *exact* same logic exposed easily,
    // we will rely on the `test_parallel_backtracking` logic which compares the backtracking step.
    // BUT, `lz77_compress_parallel` does EVERYTHING (matches -> backtracking -> sequences).
    // To measure JUST backtracking speedup, we need to instrument the code or run the specific stages.
    
    // Let's measure the FULL `lz77_compress_parallel` execution time on GPU
    // and compare it to a simulated CPU baseline (or just report GPU performance).
    // Since we want "CPU vs GPU", we'll use the `run_cpu_backtracking` from the test suite if available,
    // but that's not in the header.
    
    // Instead, let's measure the GPU execution time and throughput.
    // And if possible, run the CPU backtracking logic if we can access it.
    // Given the headers, we might only have access to `lz77_compress_parallel`.
    
    // Actually, let's look at `test_parallel_backtracking.cu` again. It has `run_cpu_backtracking`.
    // We should probably copy that logic or make it shared.
    // For now, I will implement a simplified CPU backtracking here for comparison.
    
    // ... (CPU Backtracking Implementation would go here, but for brevity/correctness, 
    // let's focus on the GPU performance metrics first, as that's the new feature).
    
    // Wait, the user wants "Backtracking speedup (CPU time / GPU time)".
    // I will implement the CPU backtracking loop here.
    
    // 1. Run Pass 1 & 2 (Matches & Costs) - Common to both
    // We need to run this to get the data for backtracking.
    // This requires accessing internal device vectors which is hard from a separate benchmark file
    // without exposing internals.
    
    // ALTERNATIVE: Modify `test_parallel_backtracking.cu` to accept command line args for size
    // and run it as the benchmark. This is much easier and cleaner.
    
    std::cout << "Skipping custom benchmark implementation. Please use test_parallel_backtracking with arguments." << std::endl;
}

int main(int argc, char** argv) {
    // This file is a placeholder. I will modify test_parallel_backtracking.cu instead.
    return 0;
}
