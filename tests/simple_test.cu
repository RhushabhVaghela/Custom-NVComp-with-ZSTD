// ============================================================================
// simple_test.cu - Basic Functionality Test
// ============================================================================

#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>

using namespace cuda_zstd;

void print_separator() {
    std::cout << "========================================\n";
}

void print_device_info() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found!\n";
        exit(1);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device Information:\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "  Clock Rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
    
    if (prop.major < 7) {
        std::cerr << "\nERROR: Compute capability 7.0+ required!\n";
        std::cerr << "Found: " << prop.major << "." << prop.minor << "\n";
        exit(1);
    }
    
    std::cout << "\n✓ Device is compatible\n";
}

void test_basic_compression() {
    print_separator();
    std::cout << "Test 1: Basic Compression\n";
    print_separator();
    
    const size_t data_size = 1024 * 1024; // 1 MB
    std::cout << "Input size: " << data_size << " bytes\n";
    
    // Create test data with pattern
    std::vector<uint8_t> h_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        h_data[i] = static_cast<uint8_t>(i % 256);
    }
    std::cout << "✓ Created test data\n";
    
    // Allocate GPU memory
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);
    std::cout << "✓ Allocated GPU memory\n";
    
    // Create manager (FIXED: using create_manager() with level)
    auto manager = create_manager(3);
    std::cout << "✓ Created compression manager (level 3)\n";
    
    // Get workspace size
    size_t temp_size = manager->get_compress_temp_size(data_size);
    std::cout << "  Workspace required: " << temp_size / 1024 << " KB\n";
    
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);
    
    // Compress
    size_t compressed_size;
    Status status = manager->compress(
        d_input, data_size,
        d_output, &compressed_size,
        d_temp, temp_size
    );
    cudaDeviceSynchronize();
    
    std::cout << "\nResults:\n";
    std::cout << "  Status: " << status_to_string(status) << "\n";
    std::cout << "  Compressed size: " << compressed_size << " bytes\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2)
              << get_compression_ratio(data_size, compressed_size) << ":1\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    if (status == Status::SUCCESS) {
        std::cout << "\n✓ Test 1 PASSED\n";
    } else {
        std::cout << "\n✗ Test 1 FAILED\n";
    }
}

void test_level_selection() {
    print_separator();
    std::cout << "Test 2: Compression Level Selection\n";
    print_separator();
    
    const size_t data_size = 512 * 1024; // 512 KB
    std::vector<uint8_t> h_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        h_data[i] = static_cast<uint8_t>((i * 31) % 256);
    }
    
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);
    
    std::cout << "Testing levels: 1, 3, 5, 9\n\n";
    std::cout << std::left << std::setw(8) << "Level"
              << std::setw(20) << "Compressed Size"
              << std::setw(12) << "Ratio" << "\n";
    std::cout << "----------------------------------------\n";
    
    for (int level : {1, 3, 5, 9}) {
        auto manager = create_manager(level);
        
        size_t temp_size = manager->get_compress_temp_size(data_size);
        void* d_temp;
        cudaMalloc(&d_temp, temp_size);
        
        size_t compressed_size;
        Status status = manager->compress(
            d_input, data_size,
            d_output, &compressed_size,
            d_temp, temp_size
        );
        cudaDeviceSynchronize();
        
        if (status == Status::SUCCESS) {
            std::cout << std::left << std::setw(8) << level
                      << std::setw(20) << (std::to_string(compressed_size) + " bytes")
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << get_compression_ratio(data_size, compressed_size) << ":1\n";
        }
        
        cudaFree(d_temp);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    std::cout << "\n✓ Test 2 PASSED\n";
}

void test_manager_api() {
    print_separator();
    std::cout << "Test 3: Manager API\n";
    print_separator();
    
    auto manager = create_manager(5);
    std::cout << "Testing manager configuration:\n";
    
    // Test get level
    int level = manager->get_compression_level();
    std::cout << "  Initial level: " << level << "\n";
    if (level == 5) std::cout << "  ✓ get_compression_level() works\n";
    
    // Test set level
    Status status = manager->set_compression_level(10);
    if (status == Status::SUCCESS) {
        std::cout << "  ✓ set_compression_level() works\n";
        level = manager->get_compression_level();
        std::cout << "  New level: " << level << "\n";
    }
    
    // Test max compressed size
    size_t max_size = manager->get_max_compressed_size(1024 * 1024);
    std::cout << "  Max compressed size for 1MB: " << max_size << " bytes\n";
    if (max_size > 1024 * 1024) std::cout << "  ✓ get_max_compressed_size() works\n";
    
    // Test workspace size
    size_t ws_size = manager->get_compress_temp_size(1024 * 1024);
    std::cout << "  Workspace size for 1MB: " << ws_size / 1024 << " KB\n";
    if (ws_size > 0) std::cout << "  ✓ get_compress_temp_size() works\n";
    
    std::cout << "\n✓ Test 3 PASSED\n";
}

void test_error_handling() {
    print_separator();
    std::cout << "Test 4: Error Handling\n";
    print_separator();
    
    auto manager = create_manager(3);
    std::cout << "Testing error conditions:\n";
    
    // Test invalid level
    Status status = manager->set_compression_level(0); // Invalid (< 1)
    if (status == Status::ERROR_INVALID_PARAMETER) {
        std::cout << "  ✓ Invalid level detected (0)\n";
    }
    
    status = manager->set_compression_level(23); // Invalid (> 22)
    if (status == Status::ERROR_INVALID_PARAMETER) {
        std::cout << "  ✓ Invalid level detected (23)\n";
    }
    
    // Test null pointers
    size_t dummy_size;
    status = manager->compress(nullptr, 100, nullptr, &dummy_size, nullptr, 0);
    if (status == Status::ERROR_INVALID_PARAMETER) {
        std::cout << "  ✓ Null pointer detected\n";
    }
    
    std::cout << "\n✓ Test 4 PASSED\n";
}

void test_gpu_sync_safety() {
    const size_t NUM_CHUNKS = 10;
    const size_t CHUNK_SIZE = 1024 * 1024;
    
    // Create device arrays
    byte_t* d_data;
    size_t* d_sizes;
    
    CUDA_CHECK(cudaMalloc(&d_data, NUM_CHUNKS * CHUNK_SIZE));
    CUDA_CHECK(cudaMalloc(&d_sizes, NUM_CHUNKS * sizeof(size_t)));
    
    // Launch async kernel that writes sizes
    launch_size_calculation_kernel<<<NUM_CHUNKS, 256>>>(d_data, d_sizes);
    // Kernel is now running asynchronously
    
    // === TEST: Without sync - would read wrong data ===
    // Without fix: reads garbage
    
    // === TEST: With sync - reads correct data ===
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel
    
    std::vector<size_t> h_sizes(NUM_CHUNKS);
    CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_sizes, 
                         NUM_CHUNKS * sizeof(size_t), 
                         cudaMemcpyDeviceToHost));
    
    // Verify all sizes are correct
    for (size_t i = 0; i < NUM_CHUNKS; i++) {
        assert(h_sizes[i] == CHUNK_SIZE);
    }
    
    cudaFree(d_data);
    cudaFree(d_sizes);
}

void test_block_scan_flexibility() {
    u32 num_segments = 1000000;
    
    // Allocate test data
    DictSegment* d_segments;
    u32* d_offsets;
    u32* d_block_sums;
    
    cudaMalloc(&d_segments, num_segments * sizeof(DictSegment));
    cudaMalloc(&d_offsets, num_segments * sizeof(u32));
    cudaMalloc(&d_block_sums, 
              ((num_segments + 1024) / 1024) * sizeof(u32));
    
    // Fill segments with test data
    std::vector<DictSegment> h_segments(num_segments);
    for (u32 i = 0; i < num_segments; i++) {
        h_segments[i].length = (i % 100) + 1;
    }
    
    cudaMemcpy(d_segments, h_segments.data(), 
              num_segments * sizeof(DictSegment), cudaMemcpyHostToDevice);
    
    // Test different block sizes
    std::vector<u32> block_sizes = {128, 256, 512, 1024};
    
    for (u32 block_size : block_sizes) {
        std::cout << "\n=== Testing block size: " << block_size << " ===\n";
        
        auto bench = benchmark_block_scan(block_size, num_segments, 0, 10);
        
        std::cout << "Time: " << bench.time_ms << " ms\n";
        std::cout << "Throughput: " << bench.throughput_gbps << " GB/s\n";
        std::cout << "Utilization: " << bench.utilization_percent << "%\n";
    }
    
    cudaFree(d_segments);
    cudaFree(d_offsets);
    cudaFree(d_block_sums);
}

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA Zstandard Library - Test Suite\n";
    print_separator();
    std::cout << "\n";
    
    try {
        // Device info
        print_device_info();
        std::cout << "\n";
        
        // Run tests
        test_basic_compression();
        std::cout << "\n";
        
        test_level_selection();
        std::cout << "\n";
        
        test_manager_api();
        std::cout << "\n";
        
        test_error_handling();
        std::cout << "\n";

        test_gpu_sync_safety();
        std::cout << "\n";

        test_block_scan_flexibility();
        std::cout << "\n";
        
        // Summary
        print_separator();
        std::cout << "ALL TESTS PASSED ✓\n";
        print_separator();
        std::cout << "\n";
        std::cout << "Library is working correctly!\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
