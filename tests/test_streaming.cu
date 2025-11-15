// ============================================================================
// test_streaming.cu - Comprehensive Streaming Compression Tests
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include "cuda_error_checking.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

using namespace cuda_zstd;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg) std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_EQ(a, b, msg) if ((a) != (b)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_NE(a, b, msg) if ((a) == (b)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_TRUE(cond, msg) if (!(cond)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_STATUS(status, msg) if ((status) != Status::SUCCESS) { LOG_FAIL(__func__, msg << " Status: " << status_to_string(status)); return false; }

// ============================================================================
// Helper Functions
// ============================================================================

void print_separator() {
    std::cout << "========================================" << std::endl;
}

void generate_test_data(std::vector<uint8_t>& data, size_t size, const char* pattern) {
    data.resize(size);
    if (strcmp(pattern, "repetitive") == 0) {
        // Highly compressible - repeated pattern
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<uint8_t>(i % 16);
        }
    } else if (strcmp(pattern, "random") == 0) {
        // Low compressibility - pseudo-random
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<uint8_t>((i * 1103515245 + 12345) & 0xFF);
        }
    } else if (strcmp(pattern, "text") == 0) {
        // Medium compressibility - text-like
        const char* text = "The quick brown fox jumps over the lazy dog. ";
        size_t text_len = strlen(text);
        for (size_t i = 0; i < size; i++) {
            data[i] = text[i % text_len];
        }
    } else {
        // Sequential
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<uint8_t>(i & 0xFF);
        }
    }
}

bool verify_decompressed_data(const uint8_t* original, const uint8_t* decompressed, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (original[i] != decompressed[i]) {
            std::cerr << "  Mismatch at byte " << i << ": expected " 
                      << (int)original[i] << ", got " << (int)decompressed[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// TEST SUITE 1: Basic Streaming Tests
// ============================================================================

bool test_single_chunk_streaming() {
    LOG_TEST("Single Chunk Streaming");
    
    const size_t chunk_size = 64 * 1024; // 64KB
    std::vector<uint8_t> h_input(chunk_size);
    generate_test_data(h_input, chunk_size, "repetitive");
    
    LOG_INFO("Input size: " << chunk_size << " bytes");
    
    // Allocate GPU memory with error checking
    void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr;
    
    if (!safe_cuda_malloc(&d_input, chunk_size)) {
        LOG_FAIL("test_single_chunk_streaming", "CUDA malloc for d_input failed");
        return false;
    }
    
    if (!safe_cuda_malloc(&d_compressed, chunk_size * 2)) {
        LOG_FAIL("test_single_chunk_streaming", "CUDA malloc for d_compressed failed");
        safe_cuda_free(d_input);
        return false;
    }
    
    if (!safe_cuda_malloc(&d_output, chunk_size)) {
        LOG_FAIL("test_single_chunk_streaming", "CUDA malloc for d_output failed");
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        return false;
    }
    
    // Copy input data to device
    if (!safe_cuda_memcpy(d_input, h_input.data(), chunk_size, cudaMemcpyHostToDevice)) {
        LOG_FAIL("test_single_chunk_streaming", "CUDA memcpy to d_input failed");
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        safe_cuda_free(d_output);
        return false;
    }
    
    // Create streaming manager
    ZstdStreamingManager manager(CompressionConfig{.level = 3});
    
    // Initialize compression
    Status status = manager.init_compression();
    ASSERT_STATUS(status, "init_compression failed");
    
    // Compress single chunk
    size_t compressed_size;
    status = manager.compress_chunk(d_input, chunk_size, d_compressed, &compressed_size, true);
    ASSERT_STATUS(status, "compress_chunk failed");
    
    LOG_INFO("Compressed size: " << compressed_size << " bytes");
    LOG_INFO("Compression ratio: " << std::fixed << std::setprecision(2)
             << get_compression_ratio(chunk_size, compressed_size) << ":1");
    
    // Initialize decompression
    status = manager.init_decompression();
    ASSERT_STATUS(status, "init_decompression failed");
    
    // Decompress
    size_t decompressed_size;
    bool is_last;
    status = manager.decompress_chunk(d_compressed, compressed_size, d_output,
                                       &decompressed_size, &is_last);
    ASSERT_STATUS(status, "decompress_chunk failed");
    ASSERT_EQ(decompressed_size, chunk_size, "Decompressed size mismatch");
    ASSERT_TRUE(is_last, "Expected last chunk flag");
    
    // Verify data
    std::vector<uint8_t> h_output(chunk_size);
    if (!safe_cuda_memcpy(h_output.data(), d_output, chunk_size, cudaMemcpyDeviceToHost)) {
        LOG_FAIL("test_single_chunk_streaming", "CUDA memcpy from d_output failed");
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        safe_cuda_free(d_output);
        return false;
    }
    
    ASSERT_TRUE(verify_decompressed_data(h_input.data(), h_output.data(), chunk_size),
                "Data verification failed");
    
    // Cleanup with safe free functions
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    
    LOG_PASS("Single Chunk Streaming");
    return true;
}

bool test_multi_chunk_streaming() {
    LOG_TEST("Multi-Chunk Streaming (5 chunks)");
    
    const size_t chunk_size = 16 * 1024; // 16KB per chunk
    const int num_chunks = 5;
    const size_t total_size = chunk_size * num_chunks;
    
    LOG_INFO("Chunk size: " << chunk_size << " bytes");
    LOG_INFO("Number of chunks: " << num_chunks);
    LOG_INFO("Total size: " << total_size << " bytes");
    
    // Generate test data
    std::vector<uint8_t> h_input(total_size);
    generate_test_data(h_input, total_size, "text");
    
    // Allocate GPU memory
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, chunk_size) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, chunk_size * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, chunk_size) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 5});
    manager.init_compression();
    
    // Compress chunks
    std::vector<std::vector<uint8_t>> compressed_chunks;
    size_t total_compressed = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        size_t offset = i * chunk_size;
        cudaMemcpy(d_input, h_input.data() + offset, chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        bool is_last = (i == num_chunks - 1);
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, is_last);
        ASSERT_STATUS(status, "compress_chunk " << i << " failed");
        
        std::vector<uint8_t> chunk_data(compressed_size);
        cudaMemcpy(chunk_data.data(), d_compressed, compressed_size, cudaMemcpyDeviceToHost);
        compressed_chunks.push_back(chunk_data);
        total_compressed += compressed_size;
        
        LOG_INFO("Chunk " << i << ": " << chunk_size << " -> " << compressed_size
                 << " bytes (ratio: " << std::fixed << std::setprecision(2)
                 << get_compression_ratio(chunk_size, compressed_size) << ":1)");
    }
    
    LOG_INFO("Total compressed: " << total_compressed << " bytes");
    LOG_INFO("Overall ratio: " << std::fixed << std::setprecision(2)
             << get_compression_ratio(total_size, total_compressed) << ":1");
    
    // Decompress chunks
    manager.init_decompression();
    std::vector<uint8_t> h_output(total_size);
    
    for (int i = 0; i < num_chunks; i++) {
        cudaMemcpy(d_compressed, compressed_chunks[i].data(),
                   compressed_chunks[i].size(), cudaMemcpyHostToDevice);
        
        size_t decompressed_size;
        bool is_last;
        Status status = manager.decompress_chunk(d_compressed, compressed_chunks[i].size(),
                                                  d_output, &decompressed_size, &is_last);
        ASSERT_STATUS(status, "decompress_chunk " << i << " failed");
        ASSERT_EQ(decompressed_size, chunk_size, "Chunk " << i << " size mismatch");
        
        cudaMemcpy(h_output.data() + (i * chunk_size), d_output, 
                   chunk_size, cudaMemcpyDeviceToHost);
    }
    
    // Verify
    ASSERT_TRUE(verify_decompressed_data(h_input.data(), h_output.data(), total_size),
                "Multi-chunk data verification failed");
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Multi-Chunk Streaming");
    return true;
}

bool test_variable_chunk_sizes() {
    LOG_TEST("Variable Chunk Sizes");
    
    std::vector<size_t> chunk_sizes = {1024, 4096, 16384, 65536, 1024}; // 1KB, 4KB, 16KB, 64KB, 1KB
    size_t total_size = 0;
    for (auto size : chunk_sizes) total_size += size;
    
    LOG_INFO("Testing with " << chunk_sizes.size() << " chunks of varying sizes");
    LOG_INFO("Total size: " << total_size << " bytes");
    
    // Generate continuous data
    std::vector<uint8_t> h_input(total_size);
    generate_test_data(h_input, total_size, "sequential");
    
    // Allocate GPU memory (use max chunk size)
    size_t max_chunk = *std::max_element(chunk_sizes.begin(), chunk_sizes.end());
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, max_chunk) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, max_chunk * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, max_chunk) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 7});
    manager.init_compression();
    
    // Compress variable chunks
    std::vector<std::vector<uint8_t>> compressed_chunks;
    size_t offset = 0;
    size_t total_compressed = 0;
    
    for (size_t i = 0; i < chunk_sizes.size(); i++) {
        size_t chunk_size = chunk_sizes[i];
        cudaMemcpy(d_input, h_input.data() + offset, chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        bool is_last = (i == chunk_sizes.size() - 1);
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, is_last);
        ASSERT_STATUS(status, "Variable chunk " << i << " compression failed");
        
        std::vector<uint8_t> chunk_data(compressed_size);
        cudaMemcpy(chunk_data.data(), d_compressed, compressed_size, cudaMemcpyDeviceToHost);
        compressed_chunks.push_back(chunk_data);
        total_compressed += compressed_size;
        
        LOG_INFO("Chunk " << i << ": " << chunk_size << " -> " << compressed_size << " bytes");
        offset += chunk_size;
    }
    
    LOG_INFO("Total: " << total_size << " -> " << total_compressed << " bytes");
    
    // Decompress and verify
    manager.init_decompression();
    std::vector<uint8_t> h_output(total_size);
    offset = 0;
    
    for (size_t i = 0; i < chunk_sizes.size(); i++) {
        cudaMemcpy(d_compressed, compressed_chunks[i].data(),
                   compressed_chunks[i].size(), cudaMemcpyHostToDevice);
        
        size_t decompressed_size;
        bool is_last;
        Status status = manager.decompress_chunk(d_compressed, compressed_chunks[i].size(),
                                                  d_output, &decompressed_size, &is_last);
        ASSERT_STATUS(status, "Variable chunk " << i << " decompression failed");
        ASSERT_EQ(decompressed_size, chunk_sizes[i], "Size mismatch for chunk " << i);
        
        cudaMemcpy(h_output.data() + offset, d_output, chunk_sizes[i], cudaMemcpyDeviceToHost);
        offset += chunk_sizes[i];
    }
    
    ASSERT_TRUE(verify_decompressed_data(h_input.data(), h_output.data(), total_size),
                "Variable chunk verification failed");
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Variable Chunk Sizes");
    return true;
}

// ============================================================================
// TEST SUITE 2: Window Management Tests
// ============================================================================

bool test_cross_chunk_matching() {
    LOG_TEST("Cross-Chunk LZ77 Matching");
    
    const size_t chunk_size = 8192; // 8KB chunks
    const int num_chunks = 3;
    
    // Create data with patterns that span chunks
    std::vector<uint8_t> h_input(chunk_size * num_chunks);
    const char* pattern = "This pattern repeats across chunks! ";
    size_t pattern_len = strlen(pattern);
    
    for (size_t i = 0; i < h_input.size(); i++) {
        h_input[i] = pattern[i % pattern_len];
    }
    
    LOG_INFO("Testing cross-chunk pattern matching");
    LOG_INFO("Pattern: \"" << pattern << "\"");
    LOG_INFO("Chunk size: " << chunk_size << " bytes");
    
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, chunk_size) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, chunk_size * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, chunk_size) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 9});
    manager.init_compression();
    
    std::vector<std::vector<uint8_t>> compressed_chunks;
    
    for (int i = 0; i < num_chunks; i++) {
        cudaMemcpy(d_input, h_input.data() + (i * chunk_size), chunk_size,
                   cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, i == num_chunks - 1);
        ASSERT_STATUS(status, "Cross-chunk compression failed at chunk " << i);
        
        std::vector<uint8_t> chunk_data(compressed_size);
        cudaMemcpy(chunk_data.data(), d_compressed, compressed_size, cudaMemcpyDeviceToHost);
        compressed_chunks.push_back(chunk_data);
        
        LOG_INFO("Chunk " << i << ": " << compressed_size << " bytes (ratio: "
                 << std::fixed << std::setprecision(2)
                 << get_compression_ratio(chunk_size, compressed_size) << ":1)");
    }
    
    // Verify decompression
    manager.init_decompression();
    std::vector<uint8_t> h_output(chunk_size * num_chunks);
    
    for (int i = 0; i < num_chunks; i++) {
        cudaMemcpy(d_compressed, compressed_chunks[i].data(),
                   compressed_chunks[i].size(), cudaMemcpyHostToDevice);
        
        size_t decompressed_size;
        bool is_last;
        Status status = manager.decompress_chunk(d_compressed, compressed_chunks[i].size(),
                                                  d_output, &decompressed_size, &is_last);
        ASSERT_STATUS(status, "Cross-chunk decompression failed at chunk " << i);
        
        cudaMemcpy(h_output.data() + (i * chunk_size), d_output, 
                   decompressed_size, cudaMemcpyDeviceToHost);
    }
    
    ASSERT_TRUE(verify_decompressed_data(h_input.data(), h_output.data(), 
                                         chunk_size * num_chunks),
                "Cross-chunk data verification failed");
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Cross-Chunk LZ77 Matching");
    return true;
}

// ============================================================================
// TEST SUITE 3: Edge Cases
// ============================================================================

bool test_very_small_chunks() {
    LOG_TEST("Very Small Chunks (<100 bytes)");
    
    std::vector<size_t> tiny_sizes = {1, 10, 50, 99};
    
    for (auto size : tiny_sizes) {
        LOG_INFO("Testing chunk size: " << size << " bytes");
        
        std::vector<uint8_t> h_input(size);
        for (size_t i = 0; i < size; i++) h_input[i] = static_cast<uint8_t>(i);
        
        void *d_input, *d_compressed, *d_output;
        ASSERT_TRUE(cudaMalloc(&d_input, 1024) == cudaSuccess, "cudaMalloc d_input failed");
        ASSERT_TRUE(cudaMalloc(&d_compressed, 1024) == cudaSuccess, "cudaMalloc d_compressed failed");
        ASSERT_TRUE(cudaMalloc(&d_output, 1024) == cudaSuccess, "cudaMalloc d_output failed");
        cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
        
        ZstdStreamingManager manager(CompressionConfig{.level = 1});
        manager.init_compression();
        
        size_t compressed_size;
        Status status = manager.compress_chunk(d_input, size, d_compressed,
                                               &compressed_size, true);
        ASSERT_STATUS(status, "Tiny chunk compression failed for size " << size);
        
        LOG_INFO("  Compressed to " << compressed_size << " bytes");
        
        // Decompress
        manager.init_decompression();
        size_t decompressed_size;
        bool is_last;
        status = manager.decompress_chunk(d_compressed, compressed_size, d_output,
                                          &decompressed_size, &is_last);
        ASSERT_STATUS(status, "Tiny chunk decompression failed for size " << size);
        ASSERT_EQ(decompressed_size, size, "Size mismatch for tiny chunk");
        
        std::vector<uint8_t> h_output(size);
        cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);
        ASSERT_TRUE(verify_decompressed_data(h_input.data(), h_output.data(), size),
                    "Tiny chunk verification failed");
        
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_output);
    }
    
    LOG_PASS("Very Small Chunks");
    return true;
}

bool test_large_streaming() {
    LOG_TEST("Large File Streaming (10MB in 100 chunks)");
    
    const size_t chunk_size = 100 * 1024; // 100KB
    const int num_chunks = 100; // Total 10MB
    const size_t total_size = chunk_size * num_chunks;
    
    LOG_INFO("Total size: " << total_size / (1024 * 1024) << " MB");
    LOG_INFO("Chunk size: " << chunk_size / 1024 << " KB");
    LOG_INFO("Number of chunks: " << num_chunks);
    
    // Generate large test data
    std::vector<uint8_t> h_input(chunk_size); // Reuse buffer
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, chunk_size) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, chunk_size * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, chunk_size) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 3});
    manager.init_compression();
    
    size_t total_compressed = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        // Generate chunk data
        generate_test_data(h_input, chunk_size, (i % 3 == 0) ? "text" : "repetitive");
        cudaMemcpy(d_input, h_input.data(), chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, i == num_chunks - 1);
        ASSERT_STATUS(status, "Large chunk " << i << " compression failed");
        
        total_compressed += compressed_size;
        
        if (i % 20 == 0) {
            LOG_INFO("Progress: " << i << "/" << num_chunks << " chunks, "
                     << total_compressed / (1024 * 1024) << " MB compressed");
        }
    }
    
    LOG_INFO("Total compressed: " << total_compressed / (1024 * 1024) << " MB");
    LOG_INFO("Compression ratio: " << std::fixed << std::setprecision(2)
             << get_compression_ratio(total_size, total_compressed) << ":1");
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Large File Streaming");
    return true;
}

bool test_incompressible_data_streaming() {
    LOG_TEST("Incompressible Data Streaming");
    
    const size_t chunk_size = 32 * 1024;
    const int num_chunks = 5;
    
    LOG_INFO("Testing random (incompressible) data");
    
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, chunk_size) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, chunk_size * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, chunk_size) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 5});
    manager.init_compression();
    
    std::vector<uint8_t> h_input(chunk_size);
    size_t total_compressed = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        generate_test_data(h_input, chunk_size, "random");
        cudaMemcpy(d_input, h_input.data(), chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, i == num_chunks - 1);
        ASSERT_STATUS(status, "Incompressible chunk " << i << " failed");
        
        total_compressed += compressed_size;
        float ratio = get_compression_ratio(chunk_size, compressed_size);
        
        LOG_INFO("Chunk " << i << ": ratio " << std::fixed << std::setprecision(3) << ratio << ":1"
                 << " (expected ~1.0 for random data)");
    }
    
    LOG_INFO("Overall ratio: " << std::fixed << std::setprecision(3)
             << get_compression_ratio(chunk_size * num_chunks, total_compressed) << ":1");
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Incompressible Data Streaming");
    return true;
}

bool test_mixed_compressibility() {
    LOG_TEST("Mixed Compressibility Across Chunks");
    
    const size_t chunk_size = 16 * 1024;
    const char* patterns[] = {"repetitive", "random", "text", "sequential", "repetitive"};
    const int num_chunks = 5;
    
    LOG_INFO("Testing alternating data patterns");
    
    void *d_input, *d_compressed, *d_output;
    ASSERT_TRUE(cudaMalloc(&d_input, chunk_size) == cudaSuccess, "cudaMalloc d_input failed");
    ASSERT_TRUE(cudaMalloc(&d_compressed, chunk_size * 2) == cudaSuccess, "cudaMalloc d_compressed failed");
    ASSERT_TRUE(cudaMalloc(&d_output, chunk_size) == cudaSuccess, "cudaMalloc d_output failed");
    
    ZstdStreamingManager manager(CompressionConfig{.level = 5});
    manager.init_compression();
    
    std::vector<uint8_t> h_input(chunk_size);
    
    for (int i = 0; i < num_chunks; i++) {
        generate_test_data(h_input, chunk_size, patterns[i]);
        cudaMemcpy(d_input, h_input.data(), chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size;
        Status status = manager.compress_chunk(d_input, chunk_size, d_compressed,
                                               &compressed_size, i == num_chunks - 1);
        ASSERT_STATUS(status, "Mixed chunk " << i << " failed");
        
        LOG_INFO("Chunk " << i << " (" << patterns[i] << "): "
                 << chunk_size << " -> " << compressed_size << " bytes (ratio: "
                 << std::fixed << std::setprecision(2)
                 << get_compression_ratio(chunk_size, compressed_size) << ":1)");
    }
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    LOG_PASS("Mixed Compressibility");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA ZSTD - Streaming Compression Test Suite" << std::endl;
    print_separator();
    std::cout << "\n";
    
    int passed = 0;
    int total = 0;
    
    // Skip on CPU-only environments; otherwise print device info
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();
    
    // Basic Streaming Tests
    print_separator();
    std::cout << "SUITE 1: Basic Streaming Tests" << std::endl;
    print_separator();
    
    total++; if (test_single_chunk_streaming()) passed++;
    total++; if (test_multi_chunk_streaming()) passed++;
    total++; if (test_variable_chunk_sizes()) passed++;
    
    // Window Management Tests
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 2: Window Management Tests" << std::endl;
    print_separator();
    
    total++; if (test_cross_chunk_matching()) passed++;
    
    // Edge Cases
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 3: Edge Cases" << std::endl;
    print_separator();
    
    total++; if (test_very_small_chunks()) passed++;
    total++; if (test_large_streaming()) passed++;
    total++; if (test_incompressible_data_streaming()) passed++;
    total++; if (test_mixed_compressibility()) passed++;
    
    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "TEST RESULTS" << std::endl;
    print_separator();
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    std::cout << "Failed: " << (total - passed) << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "\n✓ ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
    }
    print_separator();
    std::cout << "\n";
    
    return (passed == total) ? 0 : 1;
}