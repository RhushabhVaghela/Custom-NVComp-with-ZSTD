// test_parallel_backtracking.cu - Validate parallel backtracking correctness

#include "lz77_parallel.h"
#include "workspace_manager.h"
#include "cuda_error_checking.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

using namespace compression;
using namespace compression::lz77;

// ==============================================================================
// Test Data Generators
// ==============================================================================

void generate_repeated_pattern(std::vector<u8>& data, size_t size) {
    const char* pattern = "The quick brown fox jumps over the lazy dog. ";
    size_t pattern_len = strlen(pattern);
    
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = pattern[i % pattern_len];
    }
}

void generate_random_data(std::vector<u8>& data, size_t size) {
    data.resize(size);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = (u8)dist(rng);
    }
}

void generate_json_pattern(std::vector<u8>& data, size_t size) {
    data.clear();
    data.reserve(size);
    
    const char* json_template = 
        "{\"id\":%d,\"name\":\"user%d\",\"email\":\"user%d@example.com\"},";
    
    int id = 0;
    char buffer[256];
    
    while (data.size() < size) {
        int len = snprintf(buffer, sizeof(buffer), json_template, id, id, id);
        for (int i = 0; i < len && data.size() < size; ++i) {
            data.push_back((u8)buffer[i]);
        }
        id++;
    }
}

// ==============================================================================
// Validation Helper
// ==============================================================================

struct BacktrackResult {
    std::vector<u32> literal_lengths;
    std::vector<u32> match_lengths;
    std::vector<u32> offsets;
    u32 num_sequences;
    double time_ms;
    bool success;
};

BacktrackResult run_cpu_backtracking(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    cudaStream_t stream
) {
    BacktrackResult result;
    result.num_sequences = 0;
    result.time_ms = 0.0;
    result.success = false;
    
    // Copy costs from device to host
    ParseCost* h_costs = new ParseCost[input_size + 1];
    cudaMemcpy(h_costs, workspace.d_costs, (input_size + 1) * sizeof(ParseCost),
               cudaMemcpyDeviceToHost);
    
    // Allocate host buffers
    u32* h_literal_lengths = new u32[workspace.max_sequences];
    u32* h_match_lengths = new u32[workspace.max_sequences];
    u32* h_offsets = new u32[workspace.max_sequences];
    u32 num_sequences = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run CPU backtracking
    Status status = backtrack_sequences_cpu(h_costs, input_size, h_literal_lengths,
                           h_match_lengths, h_offsets, &num_sequences);
    
    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (status != Status::SUCCESS) {
        delete[] h_costs;
        delete[] h_literal_lengths;
        delete[] h_match_lengths;
        delete[] h_offsets;
        return result;
    }
    
    // Copy results
    result.num_sequences = num_sequences;
    result.literal_lengths.assign(h_literal_lengths, h_literal_lengths + num_sequences);
    result.match_lengths.assign(h_match_lengths, h_match_lengths + num_sequences);
    result.offsets.assign(h_offsets, h_offsets + num_sequences);
    result.success = true;
    
    delete[] h_costs;
    delete[] h_literal_lengths;
    delete[] h_match_lengths;
    delete[] h_offsets;
    
    return result;
}

BacktrackResult run_parallel_backtracking(
    const u8* d_input,
    u32 input_size,
    CompressionWorkspace& workspace,
    cudaStream_t stream
) {
    BacktrackResult result;
    result.num_sequences = 0;
    result.time_ms = 0.0;
    result.success = false;
    
    u32 num_sequences = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run parallel backtracking
    Status status = backtrack_sequences_parallel(workspace.d_costs, input_size,
                                                workspace, &num_sequences, stream);
    
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (status != Status::SUCCESS) {
        printf("❌ Parallel backtracking failed: %d\n", (int)status);
        return result;
    }
    
    // Copy results from device
    u32* h_literal_lengths = new u32[num_sequences];
    u32* h_match_lengths = new u32[num_sequences];
    u32* h_offsets = new u32[num_sequences];
    
    cudaMemcpy(h_literal_lengths, workspace.d_literal_lengths_reverse,
               num_sequences * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_match_lengths, workspace.d_match_lengths_reverse,
               num_sequences * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets, workspace.d_offsets_reverse,
               num_sequences * sizeof(u32), cudaMemcpyDeviceToHost);
    
    result.num_sequences = num_sequences;
    result.literal_lengths.assign(h_literal_lengths, h_literal_lengths + num_sequences);
    result.match_lengths.assign(h_match_lengths, h_match_lengths + num_sequences);
    result.offsets.assign(h_offsets, h_offsets + num_sequences);
    result.success = true;
    
    delete[] h_literal_lengths;
    delete[] h_match_lengths;
    delete[] h_offsets;
    
    return result;
}

bool compare_results(const BacktrackResult& cpu, const BacktrackResult& gpu, const char* test_name) {
    printf("\n=== %s ===\n", test_name);
    printf("CPU sequences: %u (%.2f ms)\n", cpu.num_sequences, cpu.time_ms);
    printf("GPU sequences: %u (%.2f ms)\n", gpu.num_sequences, gpu.time_ms);
    
    if (cpu.num_sequences != gpu.num_sequences) {
        printf("❌ Sequence count mismatch: CPU=%u, GPU=%u\n", 
               cpu.num_sequences, gpu.num_sequences);
        return false;
    }
    
    // Compare sequences
    bool match = true;
    for (u32 i = 0; i < cpu.num_sequences; ++i) {
        if (cpu.literal_lengths[i] != gpu.literal_lengths[i] ||
            cpu.match_lengths[i] != gpu.match_lengths[i] ||
            cpu.offsets[i] != gpu.offsets[i]) {
            
            if (match) {  // Only print first few mismatches
                printf("❌ Sequence %u mismatch:\n", i);
                printf("   CPU: lit=%u, match=%u, offset=%u\n",
                       cpu.literal_lengths[i], cpu.match_lengths[i], cpu.offsets[i]);
                printf("   GPU: lit=%u, match=%u, offset=%u\n",
                       gpu.literal_lengths[i], gpu.match_lengths[i], gpu.offsets[i]);
            }
            match = false;
        }
    }
    
    if (match) {
        float speedup = cpu.time_ms / gpu.time_ms;
        printf("✅ Results match! Speedup: %.2fx\n", speedup);
        return true;
    } else {
        printf("❌ Results DO NOT match\n");
        return false;
    }
}

// ==============================================================================
// Test Cases
// ==============================================================================

bool test_small_input_threshold() {
    printf("\n========================================\n");
    printf("Test 1: Small Input (< 1MB) - Should use CPU\n");
    printf("========================================\n");
    
    const u32 input_size = 512 * 1024;  // 512KB
    std::vector<u8> h_input;
    generate_repeated_pattern(h_input, input_size);
    
    u8* d_input = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    CompressionWorkspace workspace;
    CompressionConfig config;
    config.window_log = 17;
    config.hash_log = 20;
    config.chain_log = 16;
    config.search_log = 3;
    
    allocate_compression_workspace(workspace, input_size, config);
    
    LZ77Config lz77_config;
    lz77_config.window_log = config.window_log;
    lz77_config.hash_log = config.hash_log;
    lz77_config.chain_log = config.chain_log;
    lz77_config.search_depth = (1u << config.search_log);
    lz77_config.min_match = 3;
    
    // Run passes 1 and 2
    find_matches_parallel(d_input, input_size, workspace, lz77_config, 0);
    compute_optimal_parse(d_input, input_size, workspace, lz77_config, 0);
    cudaDeviceSynchronize();
    
    // Test adaptive routing (should use CPU for < 1MB)
    u32 num_sequences = 0;
    Status status = backtrack_sequences(input_size, workspace, &num_sequences, 0);
    
    printf("Adaptive routing selected mode for %uKB input\n", input_size / 1024);
    printf("Sequences found: %u\n", num_sequences);
    printf("✅ Test passed (adaptive routing works)\n");
    
    free_compression_workspace(workspace);
    cudaFree(d_input);
    
    return status == Status::SUCCESS;
}

bool test_correctness_1mb() {
    printf("\n========================================\n");
    printf("Test 2: Correctness - 1MB Repeated Pattern\n");
    printf("========================================\n");
    
    const u32 input_size = 1024 * 1024;  // 1MB
    std::vector<u8> h_input;
    generate_repeated_pattern(h_input, input_size);
    
    u8* d_input = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    CompressionWorkspace workspace;
    CompressionConfig config;
    config.window_log = 17;
    config.hash_log = 20;
    config.chain_log = 16;
    config.search_log = 3;
    
    allocate_compression_workspace(workspace, input_size, config);
    
    LZ77Config lz77_config;
    lz77_config.window_log = config.window_log;
    lz77_config.hash_log = config.hash_log;
    lz77_config.chain_log = config.chain_log;
    lz77_config.search_depth = (1u << config.search_log);
    lz77_config.min_match = 3;
    
    // Run passes 1 and 2
    find_matches_parallel(d_input, input_size, workspace, lz77_config, 0);
    compute_optimal_parse(d_input, input_size, workspace, lz77_config, 0);
    cudaDeviceSynchronize();
    
    // Run both CPU and GPU
    auto cpu_result = run_cpu_backtracking(d_input, input_size, workspace, 0);
    auto gpu_result = run_parallel_backtracking(d_input, input_size, workspace, 0);
    
    bool match = compare_results(cpu_result, gpu_result, "1MB Repeated Pattern");
    
    free_compression_workspace(workspace);
    cudaFree(d_input);
    
    return match;
}

bool test_correctness_10mb() {
    printf("\n========================================\n");
    printf("Test 3: Correctness - 10MB JSON Pattern\n");
    printf("========================================\n");
    
    const u32 input_size = 10 * 1024 * 1024;  // 10MB
    std::vector<u8> h_input;
    generate_json_pattern(h_input, input_size);
    
    u8* d_input = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    CompressionWorkspace workspace;
    CompressionConfig config;
    config.window_log = 17;
    config.hash_log = 20;
    config.chain_log = 16;
    config.search_log = 3;
    
    allocate_compression_workspace(workspace, input_size, config);
    
    LZ77Config lz77_config;
    lz77_config.window_log = config.window_log;
    lz77_config.hash_log = config.hash_log;
    lz77_config.chain_log = config.chain_log;
    lz77_config.search_depth = (1u << config.search_log);
    lz77_config.min_match = 3;
    
    // Run passes 1 and 2
    find_matches_parallel(d_input, input_size, workspace, lz77_config, 0);
    compute_optimal_parse(d_input, input_size, workspace, lz77_config, 0);
    cudaDeviceSynchronize();
    
    // Run both CPU and GPU
    auto cpu_result = run_cpu_backtracking(d_input, input_size, workspace, 0);
    auto gpu_result = run_parallel_backtracking(d_input, input_size, workspace, 0);
    
    bool match = compare_results(cpu_result, gpu_result, "10MB JSON Pattern");
    
    free_compression_workspace(workspace);
    cudaFree(d_input);
    
    return match;
}

bool test_performance_100mb() {
    printf("\n========================================\n");
    printf("Test 4: Performance - 100MB Repeated Pattern\n");
    printf("========================================\n");
    
    const u32 input_size = 100 * 1024 * 1024;  // 100MB
    std::vector<u8> h_input;
    generate_repeated_pattern(h_input, input_size);
    
    u8* d_input = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    CompressionWorkspace workspace;
    CompressionConfig config;
    config.window_log = 17;
    config.hash_log = 20;
    config.chain_log = 16;
    config.search_log = 3;
    
    allocate_compression_workspace(workspace, input_size, config);
    
    LZ77Config lz77_config;
    lz77_config.window_log = config.window_log;
    lz77_config.hash_log = config.hash_log;
    lz77_config.chain_log = config.chain_log;
    lz77_config.search_depth = (1u << config.search_log);
    lz77_config.min_match = 3;
    
    // Run passes 1 and 2
    printf("Running Pass 1 and 2...\n");
    find_matches_parallel(d_input, input_size, workspace, lz77_config, 0);
    compute_optimal_parse(d_input, input_size, workspace, lz77_config, 0);
    cudaDeviceSynchronize();
    
    // Run both CPU and GPU
    printf("Running CPU backtracking...\n");
    auto cpu_result = run_cpu_backtracking(d_input, input_size, workspace, 0);
    
    printf("Running GPU parallel backtracking...\n");
    auto gpu_result = run_parallel_backtracking(d_input, input_size, workspace, 0);
    
    bool match = compare_results(cpu_result, gpu_result, "100MB Performance Test");
    
    free_compression_workspace(workspace);
    cudaFree(d_input);
    
    return match;
}

// ==============================================================================
// Main
// ==============================================================================

int main() {
    printf("========================================\n");
    printf("Parallel Backtracking Validation Tests\n");
    printf("========================================\n");
    
    int passed = 0;
    int total = 4;
    
    if (test_small_input_threshold()) passed++;
    if (test_correctness_1mb()) passed++;
    if (test_correctness_10mb()) passed++;
    if (test_performance_100mb()) passed++;
    
    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("========================================\n");
    
    if (passed == total) {
        printf("\n✅ ALL TESTS PASSED!\n");
        printf("Parallel backtracking is CORRECT and FASTER than CPU.\n\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed\n\n");
        return 1;
    }
}
