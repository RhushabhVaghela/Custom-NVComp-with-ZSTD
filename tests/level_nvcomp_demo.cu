// ============================================================================
// level_nvcomp_demo.cu - Level-Based Compression Examples
// ============================================================================

#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

using namespace cuda_zstd;

// ============================================================================
// Example 1: Basic Level Selection
// ============================================================================

void example_level_selection() {
    std::cout << "=== Example 1: Level-Based Compression ===\n";
    
    // Prepare test data
    size_t data_size = 1024 * 1024; // 1 MB
    std::vector<uint8_t> input_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        input_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Allocate GPU memory
    void *d_input, *d_output;
    cudaMalloc(&d_input, data_size);
    cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
    
    // Test different compression levels
    std::cout << "\nCompression level comparison:\n";
    std::cout << "Level | Compressed Size | Ratio | Time (ms)\n";
    std::cout << "------+-----------------+-------+-----------\n";
    
    for (int level : {1, 3, 5, 7, 9, 11, 15, 19, 22}) {
        // Create manager with specific level (FIXED: using create_manager(level))
        auto manager = create_manager(level);
        
        // Allocate output buffer
        size_t max_compressed = manager->get_max_compressed_size(data_size);
        cudaMalloc(&d_output, max_compressed);
        
        // Allocate workspace
        size_t temp_size = manager->get_compress_temp_size(data_size);
        void* d_temp;
        cudaMalloc(&d_temp, temp_size);
        
        // Compress with timing
        auto start = std::chrono::high_resolution_clock::now();
        size_t compressed_size = max_compressed;
        Status status = manager->compress(
            d_input, data_size,
            d_output, &compressed_size,
            d_temp, temp_size
        );
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == Status::SUCCESS) {
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double ratio = get_compression_ratio(data_size, compressed_size);
            printf("%5d | %15zu | %5.2f | %9.2f\n",
                   level, compressed_size, ratio, time_ms);
        }
        
        cudaFree(d_output);
        cudaFree(d_temp);
    }
    
    cudaFree(d_input);
}

// ============================================================================
// Example 2: Fine-Grained Configuration
// ============================================================================

void example_fine_grained_config() {
    std::cout << "\n=== Example 2: Fine-Grained Configuration ===\n";
    
    // Method 1: Using set_compression_level()
    std::cout << "\nMethod 1: Direct level setting\n";
    auto manager1 = create_manager(3);
    manager1->set_compression_level(12);
    std::cout << "Configured with level: " << manager1->get_compression_level() << "\n";
    
    // Method 2: Create with specific level
    std::cout << "\nMethod 2: Create with level\n";
    auto manager2 = create_manager(9);
    std::cout << "Created with level: " << manager2->get_compression_level() << "\n";
    
    // Method 3: Using configure() with CompressionConfig
    std::cout << "\nMethod 3: Using CompressionConfig\n";
    auto manager3 = create_manager(5);
    CompressionConfig config;
    config.compression_level = 15;
    config.window_log = 23;
    config.hash_log = 23;
    manager3->configure(config);
    std::cout << "Level: " << manager3->get_compression_level() << "\n";
    std::cout << "Window size: " << (1 << config.window_log) / 1024 << " KB\n";
}

// ============================================================================
// Example 3: Performance Comparison
// ============================================================================

void example_performance_comparison() {
    std::cout << "\n=== Example 3: Performance vs Ratio Trade-off ===\n";
    
    size_t data_size = 10 * 1024 * 1024; // 10 MB
    std::vector<uint8_t> input_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        input_data[i] = static_cast<uint8_t>((i * 31) % 256);
    }
    
    void* d_input;
    cudaMalloc(&d_input, data_size);
    cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);
    
    std::cout << "\nInput size: " << data_size / (1024*1024) << " MB\n";
    std::cout << "\nLevel | Speed (MB/s) | Ratio | Category\n";
    std::cout << "------+--------------+-------+-----------\n";
    
    struct TestConfig {
        int level;
        const char* category;
    };
    
    std::vector<TestConfig> configs = {
        {1, "Fastest"},
        {3, "Fast"},
        {5, "Balanced"},
        {9, "High"},
        {15, "Maximum"},
        {22, "Ultra"}
    };
    
    for (const auto& cfg : configs) {
        auto manager = create_manager(cfg.level);
        
        size_t max_compressed = manager->get_max_compressed_size(data_size);
        void* d_output;
        cudaMalloc(&d_output, max_compressed);
        
        size_t temp_size = manager->get_compress_temp_size(data_size);
        void* d_temp;
        cudaMalloc(&d_temp, temp_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        size_t compressed_size = max_compressed;
        Status status = manager->compress(
            d_input, data_size,
            d_output, &compressed_size,
            d_temp, temp_size
        );
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (status == Status::SUCCESS) {
            double time_s = std::chrono::duration<double>(end - start).count();
            double speed_mbps = (data_size / (1024.0 * 1024.0)) / time_s;
            double ratio = get_compression_ratio(data_size, compressed_size);
            
            printf("%5d | %12.1f | %5.2f | %-10s\n",
                   cfg.level, speed_mbps, ratio, cfg.category);
        }
        
        cudaFree(d_output);
        cudaFree(d_temp);
    }
    
    cudaFree(d_input);
}

// ============================================================================
// Example 4: Adaptive Level Selection
// ============================================================================

void example_adaptive_level() {
    std::cout << "\n=== Example 4: Adaptive Level Selection ===\n";
    
    auto select_level = [](size_t data_size, const char* use_case) -> int {
        if (strcmp(use_case, "real-time") == 0) {
            return 1; // Fastest for real-time
        }
        if (strcmp(use_case, "network") == 0) {
            return 3; // Fast for network transfers
        }
        if (strcmp(use_case, "archive") == 0) {
            return 19; // High compression for archives
        }
        
        // Size-based selection
        if (data_size < 16 * 1024) {
            return 15; // Small data: prioritize ratio
        } else if (data_size < 1024 * 1024) {
            return 9; // Medium data: balanced
        } else if (data_size < 100 * 1024 * 1024) {
            return 5; // Large data: prioritize speed
        } else {
            return 3; // Very large: fast compression
        }
    };
    
    struct Scenario {
        size_t size;
        const char* use_case;
    };
    
    std::vector<Scenario> scenarios = {
        {8 * 1024, "small"},
        {512 * 1024, "medium"},
        {10 * 1024 * 1024, "large"},
        {1024 * 1024, "real-time"},
        {2 * 1024 * 1024, "network"},
        {100 * 1024 * 1024, "archive"}
    };
    
    std::cout << "\nAdaptive level selection:\n";
    std::cout << "Size | Use Case | Selected Level\n";
    std::cout << "-------------+------------+---------------\n";
    
    for (const auto& scenario : scenarios) {
        int level = select_level(scenario.size, scenario.use_case);
        printf("%10zu KB | %-10s | %d\n",
               scenario.size / 1024,
               scenario.use_case,
               level);
    }
}

// ============================================================================
// Example 5: Streaming Compression
// ============================================================================

void example_streaming() {
    std::cout << "\n=== Example 5: Streaming Compression ===\n";
    
    const size_t chunk_size = 256 * 1024; // 256 KB chunks
    const int num_chunks = 10;
    
    auto manager = create_manager(5);
    std::cout << "Using level 5 for streaming\n";
    std::cout << "Chunk size: " << chunk_size / 1024 << " KB\n";
    std::cout << "Number of chunks: " << num_chunks << "\n\n";
    
    // Setup
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, chunk_size);
    
    size_t max_compressed = manager->get_max_compressed_size(chunk_size);
    cudaMalloc(&d_output, max_compressed);
    
    size_t temp_size = manager->get_compress_temp_size(chunk_size);
    cudaMalloc(&d_temp, temp_size);
    
    size_t total_input = 0;
    size_t total_output = 0;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_chunks; ++i) {
        // Generate chunk data
        std::vector<uint8_t> chunk_data(chunk_size);
        for (size_t j = 0; j < chunk_size; ++j) {
            chunk_data[j] = static_cast<uint8_t>((i + j) % 256);
        }
        
        cudaMemcpy(d_input, chunk_data.data(), chunk_size, cudaMemcpyHostToDevice);
        
        size_t compressed_size = max_compressed;
        Status status = manager->compress(
            d_input, chunk_size,
            d_output, &compressed_size,
            d_temp, temp_size
        );
        
        if (status == Status::SUCCESS) {
            total_input += chunk_size;
            total_output += compressed_size;
        }
    }
    
    cudaDeviceSynchronize();
    auto total_end = std::chrono::high_resolution_clock::now();
    
    double time_s = std::chrono::duration<double>(total_end - total_start).count();
    double throughput = (total_input / (1024.0 * 1024.0)) / time_s;
    
    std::cout << "Results:\n";
    std::cout << "  Total input: " << total_input / 1024 << " KB\n";
    std::cout << "  Total output: " << total_output / 1024 << " KB\n";
    std::cout << "  Overall ratio: " << get_compression_ratio(total_input, total_output) << ":1\n";
    std::cout << "  Time: " << time_s * 1000 << " ms\n";
    std::cout << "  Throughput: " << throughput << " MB/s\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

// ============================================================================
// Example 6: Batch Compression
// ============================================================================

void example_batch_compression() {
    std::cout << "\n=== Example 6: Batch Compression ===\n";
    
    const int batch_size = 5;
    const size_t item_size = 128 * 1024; // 128 KB per item
    
    std::cout << "Batch size: " << batch_size << " items\n";
    std::cout << "Item size: " << item_size / 1024 << " KB\n\n";
    
    auto manager = create_manager(7);
    
    // Allocate buffers for batch
    std::vector<void*> d_inputs(batch_size);
    std::vector<void*> d_outputs(batch_size);
    std::vector<void*> d_temps(batch_size);
    
    size_t max_compressed = manager->get_max_compressed_size(item_size);
    size_t temp_size = manager->get_compress_temp_size(item_size);
    
    for (int i = 0; i < batch_size; ++i) {
        cudaMalloc(&d_inputs[i], item_size);
        cudaMalloc(&d_outputs[i], max_compressed);
        cudaMalloc(&d_temps[i], temp_size);
        
        // Generate data
        std::vector<uint8_t> data(item_size);
        for (size_t j = 0; j < item_size; ++j) {
            data[j] = static_cast<uint8_t>((i * 1000 + j) % 256);
        }
        cudaMemcpy(d_inputs[i], data.data(), item_size, cudaMemcpyHostToDevice);
    }
    
    // Compress batch
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<size_t> compressed_sizes(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        compressed_sizes[i] = max_compressed;
        manager->compress(
            d_inputs[i], item_size,
            d_outputs[i], &compressed_sizes[i],
            d_temps[i], temp_size
        );
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Results
    size_t total_input = batch_size * item_size;
    size_t total_output = 0;
    for (size_t sz : compressed_sizes) {
        total_output += sz;
    }
    
    std::cout << "Results:\n";
    std::cout << "  Total input: " << total_input / 1024 << " KB\n";
    std::cout << "  Total output: " << total_output / 1024 << " KB\n";
    std::cout << "  Average ratio: " << get_compression_ratio(total_input, total_output) << ":1\n";
    std::cout << "  Time: " << time_ms << " ms\n";
    std::cout << "  Throughput: " << (total_input / 1024.0 / 1024.0) / (time_ms / 1000.0) << " MB/s\n";
    
    // Cleanup
    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
        cudaFree(d_temps[i]);
    }
}

// ============================================================================
// Example 7: Memory Usage Analysis
// ============================================================================

void example_memory_analysis() {
    std::cout << "\n=== Example 7: Memory Usage Analysis ===\n\n";
    
    std::cout << "Memory requirements by level:\n";
    std::cout << "Level | Workspace (1MB input)\n";
    std::cout << "------+----------------------\n";
    
    size_t test_size = 1024 * 1024;
    
    for (int level : {1, 5, 10, 15, 22}) {
        auto manager = create_manager(level);
        size_t temp_size = manager->get_compress_temp_size(test_size);
        printf("%5d | %10zu KB\n", level, temp_size / 1024);
    }
}

// ============================================================================
// Example 8: Error Handling
// ============================================================================

void example_error_handling() {
    std::cout << "\n=== Example 8: Error Handling ===\n\n";
    
    auto manager = create_manager(5);
    
    std::cout << "Testing error conditions:\n";
    
    // Test 1: Invalid level
    Status status = manager->set_compression_level(0);
    std::cout << "  Invalid level (0): " << status_to_string(status) << "\n";
    
    status = manager->set_compression_level(25);
    std::cout << "  Invalid level (25): " << status_to_string(status) << "\n";
    
    // Test 2: Null pointers
    size_t dummy_size = 0;
    status = manager->compress(nullptr, 100, nullptr, &dummy_size, nullptr, 0);
    std::cout << "  Null input: " << status_to_string(status) << "\n";
    
    // Test 3: Buffer too small
    void* d_input;
    cudaMalloc(&d_input, 1024);
    
    void* d_output;
    size_t output_size = 10; // Too small
    cudaMalloc(&d_output, output_size);
    
    void* d_temp;
    size_t temp_size = 1024;
    cudaMalloc(&d_temp, temp_size);
    
    status = manager->compress(d_input, 1024, d_output, &output_size, d_temp, temp_size);
    std::cout << "  Buffer too small: " << status_to_string(status) << "\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

// ============================================================================
// (NEW) Example 9: Metadata & Skippable Frame Validation
// ============================================================================

void example_metadata_validation() {
    std::cout << "\n=== Example 9: Metadata & Skippable Frame Validation ===\n\n";
    
    // Use a specific, non-default level
    int test_level = 7;
    auto manager = create_manager(test_level);
    std::cout << "Compressing one block with level: " << test_level << "\n";

    // Prepare data
    size_t data_size = 64 * 1024; // 64 KB
    std::vector<uint8_t> input_data(data_size, 'a');
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    size_t temp_size = manager->get_compress_temp_size(data_size);
    cudaMalloc(&d_temp, temp_size);
    cudaMemcpy(d_input, input_data.data(), data_size, cudaMemcpyHostToDevice);

    // Compress
    size_t compressed_size = data_size * 2;
    manager->compress(
        d_input, data_size,
        d_output, &compressed_size,
        d_temp, temp_size, 0
    );
    cudaDeviceSynchronize();
    
    std::cout << "Compressed to " << compressed_size << " bytes.\n";

    // --- Validate Metadata ---
    std::cout << "Extracting metadata...\n";
    NvcompMetadata metadata;
    Status status = extract_metadata(d_output, compressed_size, metadata);

    if (status != Status::SUCCESS) {
        std::cout << "  ERROR: Metadata extraction failed!\n";
    } else {
        std::cout << "  Metadata read successfully:\n";
        std::cout << "    Uncompressed Size: " << metadata.uncompressed_size << "\n";
        std::cout << "    Compression Level: " << metadata.compression_level << "\n";
        
        if (metadata.compression_level == test_level) {
            std::cout << "  ✓ PASSED: Compression level was correctly saved and read.\n";
        } else {
            std::cout << "  ✗ FAILED: Expected level " << test_level 
                      << ", but read " << metadata.compression_level << ".\n";
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}


// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "CUDA Zstandard - Level Control Examples\n";
    std::cout << "========================================\n\n";
    
    try {
        // Run all examples
        example_level_selection();
        example_fine_grained_config();
        example_performance_comparison();
        example_adaptive_level();
        example_streaming();
        example_batch_compression();
        example_memory_analysis();
        example_error_handling();
        example_metadata_validation(); // (NEW) Add new test to main
        
        std::cout << "\n=== All Examples Complete ===\n\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
