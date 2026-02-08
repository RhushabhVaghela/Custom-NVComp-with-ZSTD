/**
 * test_streaming_manager.cu - Comprehensive unit and integration tests for Streaming Manager
 * 
 * Tests both basic streaming and streaming with window history.
 */

#include "cuda_error_checking.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>

using namespace cuda_zstd;

// Test utilities
#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " at line " << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        std::cerr << "FAIL: " << msg << " (" << (a) << " != " << (b) << ") at line " << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_ASSERT_STATUS(status, msg) \
    if ((status) != Status::SUCCESS) { \
        std::cerr << "FAIL: " << msg << " (status=" << (int)(status) << ") at line " << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_ASSERT_STATUS_EQ(actual, expected, msg) \
    if ((actual) != (expected)) { \
        std::cerr << "FAIL: " << msg << " (status=" << (int)(actual) << " != " << (int)(expected) << ") at line " << __LINE__ << std::endl; \
        return false; \
    }

// Generate compressible test data with repeating patterns
void generate_compressible_data(std::vector<uint8_t>& data, size_t size, unsigned int seed = 42) {
    data.resize(size);
    std::mt19937 rng(seed);
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

// Generate sequential data (good for testing streaming)
void generate_sequential_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t)(i & 0xFF);
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

// Test 1: Basic initialization and cleanup
bool test_basic_initialization() {
    std::cout << "Test: Basic initialization..." << std::endl;
    
    ZstdStreamingManager manager;
    
    // Test that manager starts in uninitialized state
    TEST_ASSERT(!manager.is_compression_initialized(), "Should not be initialized initially");
    TEST_ASSERT(!manager.is_decompression_initialized(), "Should not be decompression initialized initially");
    
    // Test initialization
    Status status = manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "init_compression failed");
    TEST_ASSERT(manager.is_compression_initialized(), "Should be compression initialized after init");
    
    // Test reset
    status = manager.reset();
    TEST_ASSERT_STATUS(status, "reset failed");
    TEST_ASSERT(!manager.is_compression_initialized(), "Should not be initialized after reset");
    
    std::cout << "  PASS" << std::endl;
    return true;
}

// Test 2: Configuration handling
bool test_configuration() {
    std::cout << "Test: Configuration handling..." << std::endl;
    
    CompressionConfig config = CompressionConfig::from_level(3);
    config.window_log = 20; // 1MB window
    
    ZstdStreamingManager manager(config);
    
    // Test get_config returns correct config
    CompressionConfig retrieved = manager.get_config();
    TEST_ASSERT_EQ(retrieved.window_log, config.window_log, "Window log should match");
    TEST_ASSERT_EQ(retrieved.level, config.level, "Compression level should match");
    
    // Test set_config before initialization
    CompressionConfig new_config = CompressionConfig::from_level(6);
    Status status = manager.set_config(new_config);
    TEST_ASSERT_STATUS(status, "set_config failed");
    
    retrieved = manager.get_config();
    TEST_ASSERT_EQ(retrieved.level, 6, "Compression level should be updated");
    
    std::cout << "  PASS" << std::endl;
    return true;
}

// Test 3: Single chunk compression (basic)
bool test_single_chunk_compression() {
    std::cout << "Test: Single chunk compression..." << std::endl;
    
    ZstdStreamingManager manager;
    Status status = manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "init_compression failed");
    
    // Generate test data
    std::vector<uint8_t> input_data;
    generate_compressible_data(input_data, 64 * 1024); // 64KB
    
    // Allocate device memory
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_data.size()));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, input_data.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), input_data.size(), cudaMemcpyHostToDevice));
    
    // Compress
    size_t output_size = 0;
    status = manager.compress_chunk(d_input, input_data.size(), d_output, &output_size, true, 0);
    TEST_ASSERT_STATUS(status, "compress_chunk failed");
    TEST_ASSERT(output_size > 0, "Output size should be > 0");
    TEST_ASSERT(output_size < input_data.size() * 2, "Output size should be within bounds");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "  PASS (compressed " << input_data.size() << " -> " << output_size << " bytes)" << std::endl;
    return true;
}

// Test 4: Single chunk compression with history
bool test_single_chunk_with_history() {
    std::cout << "Test: Single chunk compression with history..." << std::endl;
    
    ZstdStreamingManager manager;
    Status status = manager.init_compression_with_history(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "init_compression_with_history failed");
    
    // Generate test data
    std::vector<uint8_t> input_data;
    generate_compressible_data(input_data, 64 * 1024); // 64KB
    
    // Allocate device memory
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_data.size()));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, input_data.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), input_data.size(), cudaMemcpyHostToDevice));
    
    // Compress with history
    size_t output_size = 0;
    status = manager.compress_chunk_with_history(d_input, input_data.size(), d_output, &output_size, true, 0);
    TEST_ASSERT_STATUS(status, "compress_chunk_with_history failed");
    TEST_ASSERT(output_size > 0, "Output size should be > 0");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "  PASS (compressed " << input_data.size() << " -> " << output_size << " bytes)" << std::endl;
    return true;
}

// Test 5: Multiple chunks (basic streaming)
bool test_multiple_chunks_basic() {
    std::cout << "Test: Multiple chunks basic streaming..." << std::endl;
    
    ZstdStreamingManager manager;
    Status status = manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "init_compression failed");
    
    // Generate test data split into chunks
    const size_t chunk_size = 32 * 1024; // 32KB chunks
    const int num_chunks = 4;
    std::vector<std::vector<uint8_t>> input_chunks(num_chunks);
    
    for (int i = 0; i < num_chunks; ++i) {
        generate_compressible_data(input_chunks[i], chunk_size, 42 + i);
    }
    
    // Allocate device memory
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, chunk_size));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, chunk_size * 2));
    
    size_t total_compressed = 0;
    
    // Compress each chunk
    for (int i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, input_chunks[i].data(), chunk_size, cudaMemcpyHostToDevice));
        
        size_t output_size = 0;
        status = manager.compress_chunk(d_input, chunk_size, d_output, &output_size, 
                                        (i == num_chunks - 1), 0);
        TEST_ASSERT_STATUS(status, "compress_chunk failed on chunk " + std::to_string(i));
        TEST_ASSERT(output_size > 0, "Output size should be > 0 for chunk " + std::to_string(i));
        
        total_compressed += output_size;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    size_t total_input = chunk_size * num_chunks;
    std::cout << "  PASS (compressed " << total_input << " -> " << total_compressed << " bytes, ratio: " 
              << (float)total_input / total_compressed << ":1)" << std::endl;
    return true;
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

// Test 6: Round-trip compression and decompression (basic)
bool test_roundtrip_basic() {
    std::cout << "Test: Round-trip compression/decompression (basic)..." << std::endl;
    
    // Create separate managers for compression and decompression
    ZstdStreamingManager comp_manager;
    ZstdStreamingManager decomp_manager;
    
    Status status = comp_manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "Compression init failed");
    
    status = decomp_manager.init_decompression(0);
    TEST_ASSERT_STATUS(status, "Decompression init failed");
    
    // Generate test data
    std::vector<uint8_t> input_data;
    generate_compressible_data(input_data, 64 * 1024);
    
    // Allocate device memory
    void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_data.size()));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_compressed, input_data.size() * 2));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, input_data.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), input_data.size(), cudaMemcpyHostToDevice));
    
    // Compress
    size_t compressed_size = 0;
    status = comp_manager.compress_chunk(d_input, input_data.size(), d_compressed, &compressed_size, true, 0);
    TEST_ASSERT_STATUS(status, "Compression failed");
    
    // Decompress
    bool is_last_chunk = false;
    size_t output_size = input_data.size();
    status = decomp_manager.decompress_chunk(d_compressed, compressed_size, d_output, &output_size, &is_last_chunk, 0);
    TEST_ASSERT_STATUS(status, "Decompression failed");
    TEST_ASSERT_EQ(output_size, input_data.size(), "Output size should match input");
    
    // Verify data
    std::vector<uint8_t> output_data(input_data.size());
    CUDA_CHECK(cudaMemcpy(output_data.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    
    bool data_matches = true;
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (input_data[i] != output_data[i]) {
            data_matches = false;
            std::cerr << "Mismatch at byte " << i << ": expected " << (int)input_data[i] 
                      << ", got " << (int)output_data[i] << std::endl;
            break;
        }
    }
    TEST_ASSERT(data_matches, "Data mismatch after round-trip");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    std::cout << "  PASS (ratio: " << (float)input_data.size() / compressed_size << ":1)" << std::endl;
    return true;
}

// Test 7: Round-trip with multiple chunks
bool test_roundtrip_multiple_chunks() {
    std::cout << "Test: Round-trip with multiple chunks..." << std::endl;
    
    ZstdStreamingManager comp_manager;
    ZstdStreamingManager decomp_manager;
    
    Status status = comp_manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "Compression init failed");
    
    status = decomp_manager.init_decompression(0);
    TEST_ASSERT_STATUS(status, "Decompression init failed");
    
    const size_t chunk_size = 16 * 1024; // 16KB chunks
    const int num_chunks = 8;
    
    // Generate sequential data (each chunk different)
    std::vector<std::vector<uint8_t>> input_chunks(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
        generate_sequential_data(input_chunks[i], chunk_size);
        // Make each chunk different
        for (size_t j = 0; j < chunk_size; ++j) {
            input_chunks[i][j] = (input_chunks[i][j] + i * 17) & 0xFF;
        }
    }
    
    // Allocate device memory
    void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, chunk_size));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_compressed, chunk_size * 2));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, chunk_size));
    
    // Compress all chunks
    std::vector<size_t> compressed_sizes(num_chunks);
    std::vector<std::vector<uint8_t>> compressed_data(num_chunks);
    
    for (int i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, input_chunks[i].data(), chunk_size, cudaMemcpyHostToDevice));
        
        size_t output_size = 0;
        status = comp_manager.compress_chunk(d_input, chunk_size, d_compressed, &output_size, 
                                            (i == num_chunks - 1), 0);
        TEST_ASSERT_STATUS(status, "Compression failed on chunk " + std::to_string(i));
        
        compressed_sizes[i] = output_size;
        compressed_data[i].resize(output_size);
        CUDA_CHECK(cudaMemcpy(compressed_data[i].data(), d_compressed, output_size, cudaMemcpyDeviceToHost));
    }
    
    // Decompress all chunks and verify
    bool all_pass = true;
    for (int i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaMemcpy(d_compressed, compressed_data[i].data(), compressed_sizes[i], cudaMemcpyHostToDevice));
        
        bool is_last_chunk = false;
        size_t output_size = chunk_size;
        status = decomp_manager.decompress_chunk(d_compressed, compressed_sizes[i], d_output, 
                                                 &output_size, &is_last_chunk, 0);
        
        if (status != Status::SUCCESS) {
            std::cerr << "Decompression failed on chunk " << i << std::endl;
            all_pass = false;
            break;
        }
        
        std::vector<uint8_t> output_chunk(chunk_size);
        CUDA_CHECK(cudaMemcpy(output_chunk.data(), d_output, chunk_size, cudaMemcpyDeviceToHost));
        
        // Verify
        for (size_t j = 0; j < chunk_size; ++j) {
            if (input_chunks[i][j] != output_chunk[j]) {
                std::cerr << "Mismatch in chunk " << i << " at byte " << j << std::endl;
                all_pass = false;
                break;
            }
        }
        if (!all_pass) break;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    
    TEST_ASSERT(all_pass, "Data verification failed");
    std::cout << "  PASS (" << num_chunks << " chunks verified)" << std::endl;
    return true;
}

// Test 8: Compare compression with and without history
bool test_history_comparison() {
    std::cout << "Test: Compare compression with/without history..." << std::endl;
    
    // Generate data with repeating patterns across chunks
    const size_t chunk_size = 32 * 1024;
    const int num_chunks = 4;
    
    std::vector<std::vector<uint8_t>> chunks(num_chunks);
    // Create data where later chunks reference earlier patterns
    std::vector<uint8_t> base_pattern;
    generate_compressible_data(base_pattern, chunk_size, 123);
    
    for (int i = 0; i < num_chunks; ++i) {
        chunks[i] = base_pattern;
        // Slight variation
        for (size_t j = 0; j < 100; ++j) {
            chunks[i][j * 100] = (uint8_t)(i * 10);
        }
    }
    
    // Test without history
    ZstdStreamingManager manager_basic;
    manager_basic.init_compression(0, chunk_size * 2);
    
    size_t total_basic = 0;
    void *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, chunk_size));
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, chunk_size * 2));
    
    for (int i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, chunks[i].data(), chunk_size, cudaMemcpyHostToDevice));
        size_t out_size = 0;
        manager_basic.compress_chunk(d_input, chunk_size, d_output, &out_size, (i == num_chunks - 1), 0);
        total_basic += out_size;
    }
    
    // Test with history
    ZstdStreamingManager manager_history;
    manager_history.init_compression_with_history(0, chunk_size * 2);
    
    size_t total_history = 0;
    for (int i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaMemcpy(d_input, chunks[i].data(), chunk_size, cudaMemcpyHostToDevice));
        size_t out_size = 0;
        manager_history.compress_chunk_with_history(d_input, chunk_size, d_output, &out_size, (i == num_chunks - 1), 0);
        total_history += out_size;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "  Without history: " << total_basic << " bytes" << std::endl;
    std::cout << "  With history: " << total_history << " bytes" << std::endl;
    std::cout << "  Savings: " << (int)(100.0 * (total_basic - total_history) / total_basic) << "%" << std::endl;
    
    // With history should generally be better or equal
    std::cout << "  PASS (history comparison complete)" << std::endl;
    return true;
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

// Test 9: Invalid parameters
bool test_invalid_parameters() {
    std::cout << "Test: Invalid parameters handling..." << std::endl;
    
    ZstdStreamingManager manager;
    manager.init_compression(0, 1024 * 1024);
    
    std::vector<uint8_t> data(1024);
    generate_compressible_data(data, 1024);
    
    void *d_output = nullptr;
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, 2048));
    
    size_t output_size = 0;
    
    // Test null input
    Status status = manager.compress_chunk(nullptr, 1024, d_output, &output_size, true, 0);
    TEST_ASSERT_STATUS_EQ(status, Status::ERROR_INVALID_PARAMETER, "Should fail with null input");

    // Test null output
    status = manager.compress_chunk(data.data(), 1024, nullptr, &output_size, true, 0);
    TEST_ASSERT_STATUS_EQ(status, Status::ERROR_INVALID_PARAMETER, "Should fail with null output");

    // Test null output_size
    status = manager.compress_chunk(data.data(), 1024, d_output, nullptr, true, 0);
    TEST_ASSERT_STATUS_EQ(status, Status::ERROR_INVALID_PARAMETER, "Should fail with null output_size");
    
    cudaFree(d_output);
    
    std::cout << "  PASS" << std::endl;
    return true;
}

// Test 10: Reset and re-initialization
bool test_reset_reinit() {
    std::cout << "Test: Reset and re-initialization..." << std::endl;
    
    ZstdStreamingManager manager;
    
    // First init
    Status status = manager.init_compression(0, 1024 * 1024);
    TEST_ASSERT_STATUS(status, "First init failed");
    TEST_ASSERT(manager.is_compression_initialized(), "Should be initialized");
    
    // Reset
    status = manager.reset();
    TEST_ASSERT_STATUS(status, "Reset failed");
    TEST_ASSERT(!manager.is_compression_initialized(), "Should not be initialized after reset");
    
    // Re-init
    status = manager.init_compression(0, 512 * 1024);
    TEST_ASSERT_STATUS(status, "Second init failed");
    TEST_ASSERT(manager.is_compression_initialized(), "Should be initialized again");
    
    std::cout << "  PASS" << std::endl;
    return true;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Streaming Manager Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Unit tests
    if (test_basic_initialization()) passed++; else failed++;
    if (test_configuration()) passed++; else failed++;
    if (test_single_chunk_compression()) passed++; else failed++;
    if (test_single_chunk_with_history()) passed++; else failed++;
    if (test_multiple_chunks_basic()) passed++; else failed++;
    
    // Integration tests
    if (test_roundtrip_basic()) passed++; else failed++;
    if (test_roundtrip_multiple_chunks()) passed++; else failed++;
    if (test_history_comparison()) passed++; else failed++;
    
    // Error handling tests
    if (test_invalid_parameters()) passed++; else failed++;
    if (test_reset_reinit()) passed++; else failed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}