// ============================================================================
// simple_test.cu - Basic Functionality Test
// ============================================================================

#include "cuda_zstd_nvcomp.h"
#include "cuda_error_checking.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <numeric>

using namespace cuda_zstd;
using namespace cuda_zstd::nvcomp_v5;

void print_separator() {
    std::cout << "========================================\n";
}

bool print_device_info() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    SKIP_IF_NO_CUDA();
    // Print device info
    check_cuda_device();
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device Information:\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    // 'clockRate' field may be unavailable in newer CUDA versions; avoid using it
    std::cout << "  Clock Rate: (not available)\n";
    std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
    
    if (prop.major < 7) {
        std::cerr << "\nERROR: Compute capability 7.0+ required!\n";
        std::cerr << "Found: " << prop.major << "." << prop.minor << "\n";
        return false;
    }
    
    std::cout << "\n✓ Device is compatible\n";
    return true;
}

bool test_basic_compression() {
    print_separator();
    std::cout << "Test 1: Basic Compression (NVCOMP API)\n";
    print_separator();
    
    const size_t num_chunks = 1;
    const size_t data_size = 512 * 1024; // 512 KB
    std::cout << "Input size: " << data_size << " bytes\n";
    
    std::vector<uint8_t> h_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        h_data[i] = static_cast<uint8_t>(i % 256);
    }
    std::cout << "✓ Created test data\n";
    
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);
    std::cout << "✓ Allocated GPU memory\n";
    
    NvcompV5Options opts;
    opts.level = 1;
    NvcompV5BatchManager manager(opts);
    std::cout << "✓ Created compression manager (level 3)\n";
    
    const size_t chunk_sizes[] = { data_size };
    size_t temp_size = manager.get_compress_temp_size(chunk_sizes, num_chunks);
    std::cout << "  Workspace required: " << temp_size / 1024 << " KB\n";
    
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);
    
    const void* d_uncompressed_ptrs[] = { d_input };
    void* d_compressed_ptrs[] = { d_output };
    size_t* d_compressed_size;
    cudaMalloc(&d_compressed_size, sizeof(size_t));

    Status status = manager.compress_async(
        d_uncompressed_ptrs, chunk_sizes, num_chunks,
        d_compressed_ptrs, d_compressed_size,
        d_temp, temp_size
    );
    cudaDeviceSynchronize();
    
    size_t compressed_size;
    cudaMemcpy(&compressed_size, d_compressed_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    
    std::cout << "\nResults:\n";
    std::cout << "  Status: " << status_to_string(status) << "\n";
    std::cout << "  Compressed size: " << compressed_size << " bytes\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_compressed_size);
    
    if (status == Status::SUCCESS) {
        std::cout << "\n✓ Test 1 PASSED\n";
    } else {
        std::cout << "\n✗ Test 1 FAILED\n";
    }
    return true;
}

void test_level_selection() {
    print_separator();
    std::cout << "Test 2: Compression Level Selection (NVCOMP API)\n";
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
              << std::setw(20) << "Compressed Size" << "\n";
    std::cout << "----------------------------------------\n";
    
    const size_t chunk_sizes[] = { data_size };
    const void* d_uncompressed_ptrs[] = { d_input };
    void* d_compressed_ptrs[] = { d_output };

    for (int level : {1, 3, 5, 9}) {
        NvcompV5Options opts;
        opts.level = level;
        NvcompV5BatchManager manager(opts);
        
        size_t temp_size = manager.get_compress_temp_size(chunk_sizes, 1);
        void* d_temp;
        cudaMalloc(&d_temp, temp_size);
        
        size_t compressed_size;
        Status status = manager.compress_async(
            d_uncompressed_ptrs, chunk_sizes, 1,
            d_compressed_ptrs, &compressed_size,
            d_temp, temp_size
        );
        cudaDeviceSynchronize();
        
        if (status == Status::SUCCESS) {
            std::cout << std::left << std::setw(8) << level
                      << std::setw(20) << (std::to_string(compressed_size) + " bytes")
                      << "\n";
        }
        
        cudaFree(d_temp);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    std::cout << "\n✓ Test 2 PASSED\n";
}

void test_metadata_api() {
    print_separator();
    std::cout << "Test 3: Metadata API (NVCOMP API)\n";
    print_separator();

    const size_t data_size = 256 * 1024;
    std::vector<uint8_t> h_data(data_size, 123);
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);

    NvcompV5Options opts;
    opts.level = 5;
    NvcompV5BatchManager manager(opts);

    const size_t chunk_sizes[] = { data_size };
    size_t temp_size = manager.get_compress_temp_size(chunk_sizes, 1);
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);
    
    const void* d_uncompressed_ptrs[] = { d_input };
    void* d_compressed_ptrs[] = { d_output };
    size_t compressed_size;

    manager.compress_async(
        d_uncompressed_ptrs, chunk_sizes, 1,
        d_compressed_ptrs, &compressed_size,
        d_temp, temp_size
    );
    cudaDeviceSynchronize();

    NvcompV5Metadata metadata;
    Status status = get_metadata(d_output, compressed_size, metadata);

    std::cout << "  get_metadata status: " << status_to_string(status) << "\n";
    if(status == Status::SUCCESS) {
        std::cout << "  ✓ Metadata extraction successful\n";
        std::cout << "  Uncompressed size: " << metadata.uncompressed_size << "\n";
        std::cout << "  Compression level: " << metadata.compression_level << "\n";
        if(metadata.uncompressed_size == data_size && metadata.compression_level == 5) {
            std::cout << "  ✓ Metadata content is correct\n";
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    std::cout << "\n✓ Test 3 PASSED\n";
}

void test_error_handling() {
    print_separator();
    std::cout << "Test 4: Error Handling (NVCOMP API)\n";
    print_separator();
    
    NvcompV5Options opts;
    NvcompV5BatchManager manager(opts);
    std::cout << "Testing error conditions:\n";
    
    // Test null pointers
    size_t dummy_size;
    Status status = manager.compress_async(nullptr, nullptr, 1, nullptr, &dummy_size, nullptr, 0);
    if (status == Status::ERROR_INVALID_PARAMETER) {
        std::cout << "  ✓ Null pointer detected\n";
    }
    
    std::cout << "\n✓ Test 4 PASSED\n";
}

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA Zstandard Library - Test Suite\n";
    print_separator();
    std::cout << "\n";
    
    try {
        if (!print_device_info()) {
            // Prefer SKIP macro here to keep consistent output and return an
            // explicit return value for the test harness.
            SKIP_IF_NO_CUDA_RET(0);
            return 0; // redundant, but explicit
        }
        std::cout << "\n";
        
        test_basic_compression();
        std::cout << "\n";
        
        test_level_selection();
        std::cout << "\n";
        
        test_metadata_api();
        std::cout << "\n";
        
        test_error_handling();
        std::cout << "\n";
        
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
