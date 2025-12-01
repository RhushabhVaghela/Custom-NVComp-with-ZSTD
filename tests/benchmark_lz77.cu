#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "lz77_parallel.h"
#include "workspace_manager.h"

using namespace compression;

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

void generate_test_data(std::vector<uint8_t>& data, size_t size) {
    // Generate highly compressible data (repeating pattern)
    const char* pattern = "This is a repeating pattern to ensure LZ77 matches are found. ";
    size_t pattern_len = strlen(pattern);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = pattern[i % pattern_len];
    }
    
    // Add some random noise to prevent 100% compression
    std::mt19937 rng(42);
    std::uniform_int_distribution<> dist(0, 255);
    for (size_t i = 0; i < size / 20; ++i) { // 5% noise
        size_t pos = rng() % size;
        data[pos] = dist(rng);
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

void run_benchmark(size_t size, int iterations) {
    std::cout << "Benchmarking Size: " << size << " bytes" << std::endl;

    // Reduce iterations for very large sizes
    if (size >= 100 * 1024 * 1024) {
        iterations = 3; // 100MB+: only 3 iterations
    } else if (size >= 16 * 1024 * 1024) {
        iterations = 5; // 16MB+: only 5 iterations
    }

    std::vector<uint8_t> h_input(size);
    generate_test_data(h_input, size);

    uint8_t *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate workspace
    CompressionWorkspace workspace;
    CompressionConfig config;
    config.window_log = 15;
    config.hash_log = 17;
    config.chain_log = 17;
    config.search_log = 8;
    
    Status status = allocate_compression_workspace(workspace, size, config);
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to allocate workspace" << std::endl;
        return;
    }

    lz77::LZ77Config lz77_config;
    lz77_config.window_log = config.window_log;
    lz77_config.hash_log = config.hash_log;
    lz77_config.chain_log = config.chain_log;
    lz77_config.search_depth = (1u << config.search_log);
    lz77_config.min_match = 3;

    uint32_t h_num_sequences = 0;

    // Warmup
    lz77::find_matches_parallel(d_input, size, workspace, lz77_config, stream);
    lz77::compute_optimal_parse(d_input, size, workspace, lz77_config, stream);
    lz77::backtrack_sequences(size, workspace, &h_num_sequences, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark each phase separately
    double find_matches_ms = 0, compute_costs_ms = 0, backtrack_ms = 0;
    
    for (int i = 0; i < iterations; ++i) {
        // Pass 1: Find matches
        Timer t1;
        lz77::find_matches_parallel(d_input, size, workspace, lz77_config, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        find_matches_ms += t1.elapsed_ms();

        // Pass 2: Compute optimal parse
        Timer t2;
        lz77::compute_optimal_parse(d_input, size, workspace, lz77_config, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        compute_costs_ms += t2.elapsed_ms();

        // Pass 3: Backtrack sequences
        Timer t3;
        lz77::backtrack_sequences(size, workspace, &h_num_sequences, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        backtrack_ms += t3.elapsed_ms();
    }

    find_matches_ms /= iterations;
    compute_costs_ms /= iterations;
    backtrack_ms /= iterations;
    double total_ms = find_matches_ms + compute_costs_ms + backtrack_ms;

    double backtrack_pct = (backtrack_ms / total_ms) * 100.0;
    double throughput = (size / 1e6) / (total_ms / 1000.0);

    std::cout << "  Pass 1 (Find Matches):  " << find_matches_ms << " ms (" << ((find_matches_ms/total_ms)*100.0) << "%)" << std::endl;
    std::cout << "  Pass 2 (Compute Costs): " << compute_costs_ms << " ms (" << ((compute_costs_ms/total_ms)*100.0) << "%)" << std::endl;
    std::cout << "  Pass 3 (Backtrack):     " << backtrack_ms << " ms (" << backtrack_pct << "%)" << std::endl;
    std::cout << "  Total LZ77 Time:        " << total_ms << " ms" << std::endl;
    std::cout << "  Throughput:             " << throughput << " MB/s" << std::endl;
    std::cout << "  Sequences Found:        " << h_num_sequences << std::endl;
    
    if (backtrack_pct > 15.0) {
        std::cout << "  ⚠️  Backtrack >15%: Consider parallel optimization (Option A)" << std::endl;
    } else if (backtrack_pct > 5.0) {
        std::cout << "  ℹ️  Backtrack 5-15%: CPU offloading may help (Option B)" << std::endl;
    } else {
        std::cout << "  ✓ Backtrack <5%: Current performance acceptable (Option C)" << std::endl;
    }

    free_compression_workspace(workspace);
    cudaFree(d_input);
    cudaStreamDestroy(stream);
}

int main() {
    try {
        std::vector<size_t> sizes = {
            4 * 1024,            // 4KB
            64 * 1024,           // 64KB
            1024 * 1024,         // 1MB
            16 * 1024 * 1024,    // 16MB
            100 * 1024 * 1024,   // 100MB
            200 * 1024 * 1024    // 200MB (SAFE MAX for 16GB VRAM)
            // REMOVED: 500MB requires ~28GB VRAM (input×57 multiplier for workspace)
            // With 16GB VRAM, safe maximum is ~280MB theoretical, 200MB practical
        };

        for (size_t size : sizes) {
            run_benchmark(size, 10);
            std::cout << "--------------------------------" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
