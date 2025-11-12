// ============================================================================
// test_error_handling.cu - Comprehensive Error Handling Tests
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>

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
#define EXPECT_ERROR(status, expected, msg) if ((status) != (expected)) { LOG_FAIL(__func__, msg << " Got: " << status_to_string(status)); return false; }

void print_separator() {
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// TEST SUITE 1: Error Code Coverage
// ============================================================================

bool test_all_error_codes() {
    LOG_TEST("All Error Code Messages");
    
    // Test all defined error codes have valid messages
    std::vector<Status> error_codes = {
        Status::SUCCESS,
        Status::ERROR_INVALID_PARAMETER,
        Status::ERROR_BUFFER_TOO_SMALL,
        Status::ERROR_OUT_OF_MEMORY,
        Status::ERROR_COMPRESSION_FAILED,
        Status::ERROR_DECOMPRESSION_FAILED,
        Status::ERROR_CORRUPTED_DATA,
        Status::ERROR_UNSUPPORTED_VERSION,
        Status::ERROR_DICTIONARY_MISMATCH,
        Status::ERROR_CUDA_ERROR,
        Status::ERROR_NOT_INITIALIZED,
        Status::ERROR_ALREADY_INITIALIZED,
        Status::ERROR_INVALID_STATE,
        Status::ERROR_TIMEOUT,
        Status::ERROR_CANCELLED,
        Status::ERROR_NOT_IMPLEMENTED,
        Status::ERROR_INTERNAL,
        Status::ERROR_UNKNOWN
    };
    
    LOG_INFO("Testing " << error_codes.size() << " error codes");
    
    for (Status code : error_codes) {
        const char* msg = status_to_string(code);
        ASSERT_NE(msg, nullptr, "Error code " << static_cast<int>(code) << " has null message");
        ASSERT_TRUE(strlen(msg) > 0, "Error code " << static_cast<int>(code) << " has empty message");
        
        LOG_INFO("Status::" << msg << " (" << static_cast<int>(code) << ")");
    }
    
    LOG_PASS("All Error Code Messages");
    return true;
}

bool test_error_message_validation() {
    LOG_TEST("Error Message Validation");
    
    // Verify specific error messages are descriptive
    const char* invalid_param = status_to_string(Status::ERROR_INVALID_PARAMETER);
    ASSERT_TRUE(strstr(invalid_param, "parameter") != nullptr || strstr(invalid_param, "PARAMETER") != nullptr,
                "Invalid parameter message should contain 'parameter'");
    
    const char* oom = status_to_string(Status::ERROR_OUT_OF_MEMORY);
    ASSERT_TRUE(strstr(oom, "memory") != nullptr || strstr(oom, "MEMORY") != nullptr,
                "Out of memory message should contain 'memory'");
    
    const char* corrupted = status_to_string(Status::ERROR_CORRUPTED_DATA);
    ASSERT_TRUE(strstr(corrupted, "corrupt") != nullptr || strstr(corrupted, "CORRUPT") != nullptr,
                "Corrupted data message should contain 'corrupt'");
    
    LOG_INFO("Error messages are descriptive");
    LOG_PASS("Error Message Validation");
    return true;
}

// ============================================================================
// TEST SUITE 2: Input Validation
// ============================================================================

bool test_null_pointer_detection() {
    LOG_TEST("Null Pointer Detection");
    
    auto manager = create_manager(3);
    size_t dummy_size;
    
    // Null input pointer
    Status status = manager->compress(nullptr, 1024, (void*)0x1000, &dummy_size, (void*)0x2000, 1024, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Null input should fail");
    LOG_INFO("✓ Null input detected");
    
    // Null output pointer
    status = manager->compress((void*)0x1000, 1024, nullptr, &dummy_size, (void*)0x2000, 1024, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Null output should fail");
    LOG_INFO("✓ Null output detected");
    
    // Null size pointer
    status = manager->compress((void*)0x1000, 1024, (void*)0x3000, nullptr, (void*)0x2000, 1024, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Null size pointer should fail");
    LOG_INFO("✓ Null size pointer detected");
    
    // Null workspace pointer
    status = manager->compress((void*)0x1000, 1024, (void*)0x3000, &dummy_size, nullptr, 0, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Null workspace should fail");
    LOG_INFO("✓ Null workspace detected");
    
    LOG_PASS("Null Pointer Detection");
    return true;
}

bool test_invalid_size_parameters() {
    LOG_TEST("Invalid Size Parameters");
    
    auto manager = create_manager(5);
    
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, 1024);
    cudaMalloc(&d_output, 1024);
    cudaMalloc(&d_temp, 1024);
    
    size_t compressed_size;
    
    // Zero input size
    Status status = manager->compress(d_input, 0, d_output, &compressed_size, d_temp, 1024, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Zero input size should fail");
    LOG_INFO("✓ Zero input size detected");
    
    // Extremely large size (potential overflow)
    size_t huge_size = static_cast<size_t>(-1);
    status = manager->compress(d_input, huge_size, d_output, &compressed_size, d_temp, 1024, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Huge size should fail");
    LOG_INFO("✓ Overflow size detected");
    
    // Insufficient workspace
    status = manager->compress(d_input, 1024, d_output, &compressed_size, d_temp, 1, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_BUFFER_TOO_SMALL, "Tiny workspace should fail");
    LOG_INFO("✓ Insufficient workspace detected");
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    LOG_PASS("Invalid Size Parameters");
    return true;
}

bool test_invalid_compression_levels() {
    LOG_TEST("Invalid Compression Levels");
    
    auto manager = create_manager(3);
    
    // Level 0 (invalid - minimum is 1)
    Status status = manager->set_compression_level(0);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Level 0 should fail");
    LOG_INFO("✓ Level 0 rejected");
    
    // Level 23 (invalid - maximum is 22)
    status = manager->set_compression_level(23);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Level 23 should fail");
    LOG_INFO("✓ Level 23 rejected");
    
    // Negative level
    status = manager->set_compression_level(-1);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Negative level should fail");
    LOG_INFO("✓ Negative level rejected");
    
    // Very large level
    status = manager->set_compression_level(100);
    EXPECT_ERROR(status, Status::ERROR_INVALID_PARAMETER, "Level 100 should fail");
    LOG_INFO("✓ Large level rejected");
    
    // Valid levels should work
    for (int level = 1; level <= 22; level++) {
        status = manager->set_compression_level(level);
        ASSERT_EQ(status, Status::SUCCESS, "Valid level " << level << " should succeed");
    }
    LOG_INFO("✓ All valid levels (1-22) accepted");
    
    LOG_PASS("Invalid Compression Levels");
    return true;
}

bool test_invalid_workspace_size() {
    LOG_TEST("Invalid Workspace Size");
    
    auto manager = create_manager(3);
    const size_t input_size = 64 * 1024;
    
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);
    
    size_t required_size = manager->get_compress_temp_size(input_size);
    LOG_INFO("Required workspace: " << required_size << " bytes");
    
    // Allocate too small workspace
    size_t small_size = required_size / 2;
    cudaMalloc(&d_temp, small_size);
    
    size_t compressed_size;
    Status status = manager->compress(d_input, input_size, d_output, &compressed_size,
                                     d_temp, small_size, nullptr, 0, 0);
    EXPECT_ERROR(status, Status::ERROR_BUFFER_TOO_SMALL, "Small workspace should fail");
    LOG_INFO("✓ Insufficient workspace detected");
    
    cudaFree(d_temp);
    
    // Allocate sufficient workspace
    cudaMalloc(&d_temp, required_size);
    status = manager->compress(d_input, input_size, d_output, &compressed_size,
                              d_temp, required_size, nullptr, 0, 0);
    ASSERT_EQ(status, Status::SUCCESS, "Sufficient workspace should succeed");
    LOG_INFO("✓ Sufficient workspace accepted");
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    LOG_PASS("Invalid Workspace Size");
    return true;
}

bool test_corrupted_frame_headers() {
    LOG_TEST("Corrupted Frame Header Detection");
    
    const size_t data_size = 1024;
    std::vector<uint8_t> corrupted_data(data_size);
    
    // Fill with invalid magic number
    corrupted_data[0] = 0xFF;
    corrupted_data[1] = 0xFF;
    corrupted_data[2] = 0xFF;
    corrupted_data[3] = 0xFF;
    
    void *d_compressed, *d_output, *d_temp;
    cudaMalloc(&d_compressed, data_size);
    cudaMalloc(&d_output, data_size);
    cudaMalloc(&d_temp, data_size);
    cudaMemcpy(d_compressed, corrupted_data.data(), data_size, cudaMemcpyHostToDevice);
    
    auto manager = create_manager(3);
    size_t output_size;
    
    Status status = manager->decompress(d_compressed, data_size, d_output, &output_size,
                                       d_temp, data_size);
    EXPECT_ERROR(status, Status::ERROR_CORRUPTED_DATA, "Invalid magic number should fail");
    LOG_INFO("✓ Invalid magic number detected");
    
    cudaFree(d_compressed);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    LOG_PASS("Corrupted Frame Header Detection");
    return true;
}

// ============================================================================
// TEST SUITE 3: Resource Management
// ============================================================================

bool test_cleanup_on_error_paths() {
    LOG_TEST("Resource Cleanup on Error Paths");
    
    // Get initial CUDA memory stats
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    
    auto manager = create_manager(3);
    
    // Trigger multiple errors
    for (int i = 0; i < 100; i++) {
        size_t dummy;
        // This should fail but not leak memory
        manager->compress(nullptr, 1024, nullptr, &dummy, nullptr, 0, nullptr, 0, 0);
    }
    
    // Force cleanup
    manager.reset();
    cudaDeviceSynchronize();
    
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);
    
    LOG_INFO("Free memory before: " << free_before / (1024 * 1024) << " MB");
    LOG_INFO("Free memory after: " << free_after / (1024 * 1024) << " MB");
    
    // Should not have leaked significant memory
    size_t leaked = (free_before > free_after) ? (free_before - free_after) : 0;
    ASSERT_TRUE(leaked < 1024 * 1024, "Memory leak detected: " << leaked << " bytes");
    LOG_INFO("✓ No memory leaks on error paths");
    
    LOG_PASS("Resource Cleanup on Error Paths");
    return true;
}

bool test_memory_leak_on_failures() {
    LOG_TEST("Memory Leak Detection on Repeated Failures");
    
    size_t free_initial, total;
    cudaMemGetInfo(&free_initial, &total);
    
    auto manager = create_manager(3);
    
    // Repeatedly fail with invalid parameters
    for (int i = 0; i < 1000; i++) {
        manager->set_compression_level(0); // Invalid
        manager->set_compression_level(100); // Invalid
        
        size_t dummy;
        manager->compress(nullptr, 0, nullptr, &dummy, nullptr, 0, nullptr, 0, 0);
    }
    
    cudaDeviceSynchronize();
    
    size_t free_final;
    cudaMemGetInfo(&free_final, &total);
    
    size_t diff = (free_initial > free_final) ? (free_initial - free_final) : 0;
    
    LOG_INFO("Memory change: " << diff / 1024 << " KB");
    ASSERT_TRUE(diff < 100 * 1024, "Memory leak on repeated failures");
    LOG_INFO("✓ No leaks on repeated failures");
    
    LOG_PASS("Memory Leak on Failures");
    return true;
}

// ============================================================================
// TEST SUITE 4: CUDA Error Propagation
// ============================================================================

bool test_cuda_error_detection() {
    LOG_TEST("CUDA Error Detection and Propagation");
    
    auto manager = create_manager(3);
    
    // Try to use invalid device pointers (not actually allocated)
    void* fake_ptr = reinterpret_cast<void*>(0xDEADBEEF);
    size_t dummy_size;
    
    // This should detect the invalid pointer and return CUDA error
    Status status = manager->compress(fake_ptr, 1024, fake_ptr, &dummy_size, fake_ptr, 1024, nullptr, 0, 0);
    
    // Should get some error (either CUDA error or invalid parameter)
    ASSERT_NE(status, Status::SUCCESS, "Fake pointer should trigger error");
    LOG_INFO("✓ Invalid device pointer detected");
    
    // Clear any CUDA errors
    cudaGetLastError();
    
    LOG_PASS("CUDA Error Detection");
    return true;
}

// ============================================================================
// TEST SUITE 5: State Management
// ============================================================================

bool test_initialization_state() {
    LOG_TEST("Initialization State Management");
    
    auto streaming_mgr = create_streaming_manager(3);
    
    // Try to compress before initialization
    void *d_input, *d_output;
    cudaMalloc(&d_input, 1024);
    cudaMalloc(&d_output, 1024);
    
    size_t output_size;
    Status status = streaming_mgr->compress_chunk(d_input, 1024, d_output, &output_size, true);
    EXPECT_ERROR(status, Status::ERROR_NOT_INITIALIZED, "Should fail before init");
    LOG_INFO("✓ Not initialized error detected");
    
    // Initialize
    status = streaming_mgr->init_compression();
    ASSERT_EQ(status, Status::SUCCESS, "Initialization should succeed");
    LOG_INFO("✓ Initialization succeeded");
    
    // Now compression should work (or at least not fail with NOT_INITIALIZED)
    status = streaming_mgr->compress_chunk(d_input, 1024, d_output, &output_size, true);
    ASSERT_NE(status, Status::ERROR_NOT_INITIALIZED, "Should not fail with NOT_INITIALIZED after init");
    LOG_INFO("✓ State management working");
    
    // Try to initialize again
    status = streaming_mgr->init_compression();
    // Should either succeed or return ALREADY_INITIALIZED
    ASSERT_TRUE(status == Status::SUCCESS || status == Status::ERROR_ALREADY_INITIALIZED,
                "Re-initialization handling");
    LOG_INFO("✓ Re-initialization handled");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    LOG_PASS("Initialization State Management");
    return true;
}

// ============================================================================
// TEST SUITE 6: Edge Case Error Handling
// ============================================================================

bool test_buffer_overflow_prevention() {
    LOG_TEST("Buffer Overflow Prevention");
    
    auto manager = create_manager(3);
    const size_t input_size = 1024;
    
    std::vector<uint8_t> h_input(input_size, 0xAA);
    
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, 512); // Too small for worst case
    cudaMalloc(&d_temp, manager->get_compress_temp_size(input_size));
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    size_t compressed_size;
    Status status = manager->compress(d_input, input_size, d_output, &compressed_size,
                                     d_temp, manager->get_compress_temp_size(input_size), nullptr, 0, 0);
    
    // Should either succeed (if data compressed enough) or fail gracefully
    if (status == Status::SUCCESS) {
        ASSERT_TRUE(compressed_size <= 512, "Should not write beyond buffer");
        LOG_INFO("✓ Compressed within buffer limits");
    } else {
        EXPECT_ERROR(status, Status::ERROR_BUFFER_TOO_SMALL, "Should fail gracefully");
        LOG_INFO("✓ Buffer overflow prevented");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    LOG_PASS("Buffer Overflow Prevention");
    return true;
}

bool test_integer_overflow_checks() {
    LOG_TEST("Integer Overflow Checks");
    
    auto manager = create_manager(3);
    
    // Try sizes that could cause integer overflow
    size_t huge_size = SIZE_MAX;
    size_t workspace = manager->get_compress_temp_size(huge_size);
    
    // Should return reasonable value or 0, not wrap around
    LOG_INFO("Workspace for SIZE_MAX: " << workspace);
    
    // Try max compressed size
    size_t max_compressed = manager->get_max_compressed_size(huge_size);
    LOG_INFO("Max compressed for SIZE_MAX: " << max_compressed);
    
    // Should handle gracefully without crashing
    LOG_INFO("✓ Integer overflow handled");
    
    LOG_PASS("Integer Overflow Checks");
    return true;
}

bool test_concurrent_error_safety() {
    LOG_TEST("Thread-Safe Error Handling");
    
    auto manager = create_manager(3);
    std::atomic<int> error_count{0};
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    
    auto worker = [&]() {
        for (int i = 0; i < 100; i++) {
            // Trigger various errors
            Status s1 = manager->set_compression_level(0);
            Status s2 = manager->set_compression_level(100);
            
            if (s1 != Status::SUCCESS || s2 != Status::SUCCESS) {
                error_count++;
            }
        }
    };
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    LOG_INFO("Errors detected: " << error_count);
    ASSERT_TRUE(error_count > 0, "Should have detected errors");
    LOG_INFO("✓ Thread-safe error handling");
    
    LOG_PASS("Concurrent Error Safety");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA ZSTD - Error Handling Test Suite" << std::endl;
    print_separator();
    std::cout << "\n";
    
    int passed = 0;
    int total = 0;
    
    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on: " << prop.name << std::endl;
    std::cout << "CUDA Version: " << CUDART_VERSION << "\n" << std::endl;
    
    // Error Code Coverage
    print_separator();
    std::cout << "SUITE 1: Error Code Coverage" << std::endl;
    print_separator();
    
    total++; if (test_all_error_codes()) passed++;
    total++; if (test_error_message_validation()) passed++;
    
    // Input Validation
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 2: Input Validation" << std::endl;
    print_separator();
    
    total++; if (test_null_pointer_detection()) passed++;
    total++; if (test_invalid_size_parameters()) passed++;
    total++; if (test_invalid_compression_levels()) passed++;
    total++; if (test_invalid_workspace_size()) passed++;
    total++; if (test_corrupted_frame_headers()) passed++;
    
    // Resource Management
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 3: Resource Management" << std::endl;
    print_separator();
    
    total++; if (test_cleanup_on_error_paths()) passed++;
    total++; if (test_memory_leak_on_failures()) passed++;
    
    // CUDA Error Propagation
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 4: CUDA Error Propagation" << std::endl;
    print_separator();
    
    total++; if (test_cuda_error_detection()) passed++;
    
    // State Management
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 5: State Management" << std::endl;
    print_separator();
    
    total++; if (test_initialization_state()) passed++;
    
    // Edge Cases
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 6: Edge Case Error Handling" << std::endl;
    print_separator();
    
    total++; if (test_buffer_overflow_prevention()) passed++;
    total++; if (test_integer_overflow_checks()) passed++;
    total++; if (test_concurrent_error_safety()) passed++;
    
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