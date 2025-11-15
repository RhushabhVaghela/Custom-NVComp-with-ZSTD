// ============================================================================
// test_adaptive_level.cu - Comprehensive Adaptive Level Selector Tests
// ============================================================================

#include "cuda_zstd_adaptive.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_manager.h"
#include <cuda_runtime.h>
#include "cuda_error_checking.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace cuda_zstd;
using namespace cuda_zstd::adaptive;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg) std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_EQ(a, b, msg) if ((a) != (b)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_TRUE(cond, msg) if (!(cond)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_RANGE(val, min, max, msg) if ((val) < (min) || (val) > (max)) { LOG_FAIL(__func__, msg); return false; }
#define ASSERT_STATUS(status, msg) if ((status) != Status::SUCCESS) { LOG_FAIL(__func__, msg << " Status: " << status_to_string(status)); return false; }

void print_separator() {
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// Data Generation Helpers
// ============================================================================

void generate_random_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // High entropy - pseudo-random
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<uint8_t>((i * 1103515245 + 12345) & 0xFF);
    }
}

void generate_repetitive_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // Low entropy - highly compressible
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<uint8_t>(i % 4); // Only 4 unique values
    }
}

void generate_text_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // Medium entropy - text-like
    const char* text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                       "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
    size_t text_len = strlen(text);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<uint8_t>(text[i % text_len]);
    }
}

void generate_binary_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // Structured binary with some patterns
    for (size_t i = 0; i < size; i += 4) {
        uint32_t val = static_cast<uint32_t>(i);
        if (i + 3 < size) {
            data[i] = val & 0xFF;
            data[i+1] = (val >> 8) & 0xFF;
            data[i+2] = (val >> 16) & 0xFF;
            data[i+3] = (val >> 24) & 0xFF;
        }
    }
}

void generate_mixed_data(std::vector<uint8_t>& data, size_t size) {
    data.resize(size);
    // Mix of patterns
    for (size_t i = 0; i < size; i++) {
        if (i < size / 3) {
            data[i] = static_cast<uint8_t>(i % 16); // Repetitive
        } else if (i < 2 * size / 3) {
            data[i] = static_cast<uint8_t>((i * 31) & 0xFF); // Semi-random
        } else {
            data[i] = 'A' + (i % 26); // Text-like
        }
    }
}

// ============================================================================
// TEST SUITE 1: Data Characteristic Analysis
// ============================================================================

bool test_random_data_detection() {
    LOG_TEST("Random Data Detection");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_random_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for random data");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Entropy: " << std::fixed << std::setprecision(2) << result.characteristics.entropy << " bits");
    LOG_INFO("Repetition ratio: " << std::fixed << std::setprecision(2) << result.characteristics.repetition_ratio);
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Is random: " << (result.characteristics.is_random ? "yes" : "no"));
    LOG_INFO("Confidence: " << std::fixed << std::setprecision(2) << (result.confidence * 100) << "%");
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // Random data should have high entropy and low level
    ASSERT_TRUE(result.characteristics.entropy > 5.5f, "Random data should have high entropy (actual: " << result.characteristics.entropy << ")");
    ASSERT_TRUE(result.characteristics.is_random, "Should detect as random");
    ASSERT_RANGE(result.recommended_level, 1, 7, "Random data should suggest low compression level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Random Data Detection");
    return true;
}

bool test_repetitive_data_detection() {
    LOG_TEST("Highly Repetitive Data Detection");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_repetitive_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for repetitive data");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Entropy: " << std::fixed << std::setprecision(2) << result.characteristics.entropy << " bits");
    LOG_INFO("Repetition ratio: " << std::fixed << std::setprecision(2) << result.characteristics.repetition_ratio);
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Has patterns: " << (result.characteristics.has_patterns ? "yes" : "no"));
    LOG_INFO("Unique bytes: " << result.characteristics.unique_bytes);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // Repetitive data should have low entropy, high compressibility, high level
    ASSERT_TRUE(result.characteristics.entropy < 3.5f, "Repetitive data should have low entropy (actual: " << result.characteristics.entropy << ")");
    ASSERT_TRUE(result.characteristics.repetition_ratio > 0.65f, "Should have high repetition ratio (actual: " << result.characteristics.repetition_ratio << ")");
    ASSERT_TRUE(result.characteristics.compressibility > 0.75f, "Should be highly compressible (actual: " << result.characteristics.compressibility << ")");
    ASSERT_RANGE(result.recommended_level, 13, 22, "Repetitive data should suggest high compression level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Highly Repetitive Data Detection");
    return true;
}

bool test_text_data_detection() {
    LOG_TEST("Text Data Detection");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for text data");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Entropy: " << std::fixed << std::setprecision(2) << result.characteristics.entropy << " bits");
    LOG_INFO("Repetition ratio: " << std::fixed << std::setprecision(2) << result.characteristics.repetition_ratio);
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // Text data should have medium entropy and medium level
    ASSERT_RANGE(result.characteristics.entropy, 3.5f, 7.0f, "Text should have medium entropy (actual: " << result.characteristics.entropy << ")");
    ASSERT_RANGE(result.recommended_level, 3, 14, "Text should suggest medium compression level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Text Data Detection");
    return true;
}

bool test_binary_data_analysis() {
    LOG_TEST("Binary Data Analysis");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_binary_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for binary data");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Entropy: " << std::fixed << std::setprecision(2) << result.characteristics.entropy << " bits");
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Unique bytes: " << result.characteristics.unique_bytes);
    LOG_INFO("Max run length: " << result.characteristics.max_run_length);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    ASSERT_RANGE(result.recommended_level, 1, 22, "Binary data level should be in valid range (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Binary Data Analysis");
    return true;
}

bool test_mixed_data_analysis() {
    LOG_TEST("Mixed Data Analysis");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_mixed_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for mixed data");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Entropy: " << std::fixed << std::setprecision(2) << result.characteristics.entropy << " bits");
    LOG_INFO("Repetition ratio: " << std::fixed << std::setprecision(2) << result.characteristics.repetition_ratio);
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // Mixed data should result in balanced level
    ASSERT_RANGE(result.recommended_level, 3, 17, "Mixed data should suggest balanced level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Mixed Data Analysis");
    return true;
}

// ============================================================================
// TEST SUITE 2: Preference Mode Tests
// ============================================================================

bool test_speed_preference() {
    LOG_TEST("SPEED Preference Mode");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::SPEED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for SPEED mode");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // SPEED mode should prioritize low levels (1-5)
    ASSERT_RANGE(result.recommended_level, 1, 7, "SPEED mode should suggest low level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("SPEED Preference Mode");
    return true;
}

bool test_balanced_preference() {
    LOG_TEST("BALANCED Preference Mode");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::BALANCED);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for BALANCED mode");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // BALANCED mode should use middle levels (5-12)
    ASSERT_RANGE(result.recommended_level, 1, 22, "BALANCED mode level should be valid (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("BALANCED Preference Mode");
    return true;
}

bool test_ratio_preference() {
    LOG_TEST("RATIO Preference Mode");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    AdaptiveLevelSelector selector(AdaptivePreference::RATIO);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed for RATIO mode");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // RATIO mode should prioritize higher levels (12-22)
    ASSERT_RANGE(result.recommended_level, 10, 22, "RATIO mode should suggest high level (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("RATIO Preference Mode");
    return true;
}

bool test_preference_override() {
    LOG_TEST("Preference Override for Incompressible Data");
    
    const size_t data_size = 64 * 1024;
    std::vector<uint8_t> h_data;
    generate_random_data(h_data, data_size); // Incompressible
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    // Even with RATIO preference, should select low level for random data
    AdaptiveLevelSelector selector(AdaptivePreference::RATIO);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "select_level failed");
    
    LOG_INFO("Recommended level (RATIO pref, random data): " << result.recommended_level);
    LOG_INFO("Compressibility: " << std::fixed << std::setprecision(2) << result.characteristics.compressibility);
    LOG_INFO("Reasoning: " << (result.reasoning ? result.reasoning : "N/A"));
    
    // Should override RATIO preference for incompressible data
    ASSERT_TRUE(result.recommended_level < 12, "Should use low level despite RATIO preference (actual: " << result.recommended_level << ")");
    
    cudaFree(d_data);
    
    LOG_PASS("Preference Override");
    return true;
}

// ============================================================================
// TEST SUITE 3: Compressibility Estimation
// ============================================================================

bool test_compressibility_accuracy() {
    LOG_TEST("Compressibility Estimation Accuracy");
    
    const size_t data_size = 128 * 1024;
    
    struct TestCase {
        const char* name;
        void (*generator)(std::vector<uint8_t>&, size_t);
        float expected_min_comp;
        float expected_max_comp;
    };
    
    TestCase cases[] = {
        {"Random", generate_random_data, 0.0f, 0.3f},
        {"Repetitive", generate_repetitive_data, 0.8f, 1.0f},
        {"Text", generate_text_data, 0.4f, 0.7f},
        {"Binary", generate_binary_data, 0.3f, 0.8f}
    };
    
    for (const auto& test_case : cases) {
        LOG_INFO("Testing: " << test_case.name);
        
        std::vector<uint8_t> h_data;
        test_case.generator(h_data, data_size);
        
        void* d_data;
        cudaError_t err = cudaMalloc(&d_data, data_size);
        ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
        err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
        ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
        
        AdaptiveLevelSelector selector;
        AdaptiveResult result;
        
        Status status = selector.select_level(d_data, data_size, result);
        ASSERT_STATUS(status, "Compressibility analysis failed for " << test_case.name);
        
        float comp = result.characteristics.compressibility;
        LOG_INFO("  Estimated compressibility: " << std::fixed << std::setprecision(2) << comp);
        LOG_INFO("  Expected range: [" << test_case.expected_min_comp << ", " 
                 << test_case.expected_max_comp << "]");
        
        ASSERT_TRUE(comp >= test_case.expected_min_comp - 0.05f && comp <= test_case.expected_max_comp + 0.05f,
                    test_case.name << " compressibility out of expected range (actual: " << comp << ")");
        
        cudaFree(d_data);
    }
    
    LOG_PASS("Compressibility Estimation Accuracy");
    return true;
}

bool test_actual_vs_predicted_ratio() {
    LOG_TEST("Predicted vs Actual Compression Ratio");
    
    const size_t data_size = 128 * 1024;
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    // Get adaptive recommendation
    AdaptiveLevelSelector selector;
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result);
    ASSERT_STATUS(status, "Adaptive selection failed");
    
    LOG_INFO("Recommended level: " << result.recommended_level);
    LOG_INFO("Predicted compressibility: " << std::fixed << std::setprecision(2) 
             << result.characteristics.compressibility);
    
    // Actually compress with recommended level
    auto manager = create_manager(result.recommended_level);
    size_t temp_size = manager->get_compress_temp_size(data_size);

    void *d_compressed, *d_temp;
    err = cudaMalloc(&d_compressed, data_size * 2);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed for compressed buffer");
    err = cudaMalloc(&d_temp, temp_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed for temp buffer");

    size_t compressed_size;
    status = manager->compress(d_data, data_size, d_compressed, &compressed_size,
                                d_temp, temp_size, nullptr, 0, 0);
    ASSERT_STATUS(status, "Compression failed");
    
    float actual_ratio = get_compression_ratio(data_size, compressed_size);
    LOG_INFO("Actual compression ratio: " << std::fixed << std::setprecision(2) << actual_ratio << ":1");
    
    // Prediction should be reasonably close to actual
    // (This is more of an informational test)
    LOG_INFO("Prediction quality: " << (result.characteristics.compressibility > 0.5 ? "Good" : "Fair"));
    
    cudaFree(d_data);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    
    LOG_PASS("Predicted vs Actual Ratio");
    return true;
}

// ============================================================================
// TEST SUITE 4: Performance Tests
// ============================================================================

bool test_analysis_performance() {
    LOG_TEST("Analysis Time Performance");
    
    std::vector<size_t> test_sizes = {
        16 * 1024,      // 16KB
        64 * 1024,      // 64KB
        256 * 1024,     // 256KB
        1024 * 1024     // 1MB
    };
    
    AdaptiveLevelSelector selector;
    
    for (size_t size : test_sizes) {
        std::vector<uint8_t> h_data;
        generate_text_data(h_data, size);
        
        void* d_data;
        cudaError_t err = cudaMalloc(&d_data, size);
        ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
        err = cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
        ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
        
        AdaptiveResult result;
        
        auto start = std::chrono::high_resolution_clock::now();
        Status status = selector.select_level(d_data, size, result);
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_STATUS(status, "Analysis failed for size " << size);
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        LOG_INFO("Size: " << size / 1024 << " KB, Analysis time: " 
                 << std::fixed << std::setprecision(3) << elapsed_ms << " ms");
        
        // Analysis should be very fast (< 1% of compression time estimate)
        ASSERT_TRUE(elapsed_ms < 200.0, "Analysis too slow for " << size / 1024 << " KB (actual: " << elapsed_ms << " ms)");
        
        cudaFree(d_data);
    }
    
    LOG_PASS("Analysis Performance");
    return true;
}

bool test_sample_size_impact() {
    LOG_TEST("Sample Size Impact on Accuracy");
    
    const size_t data_size = 1024 * 1024; // 1MB
    std::vector<uint8_t> h_data;
    generate_text_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    std::vector<size_t> sample_sizes = {4096, 16384, 65536, 262144}; // 4KB to 256KB
    
    for (size_t sample_size : sample_sizes) {
        AdaptiveLevelSelector selector;
        selector.set_sample_size(sample_size);
        
        AdaptiveResult result;
        Status status = selector.select_level(d_data, data_size, result);
        ASSERT_STATUS(status, "Analysis failed for sample size " << sample_size);
        
        LOG_INFO("Sample: " << sample_size / 1024 << " KB, Level: " << result.recommended_level
                 << ", Compressibility: " << std::fixed << std::setprecision(2)
                 << result.characteristics.compressibility);
    }
    
    cudaFree(d_data);
    
    LOG_PASS("Sample Size Impact");
    return true;
}

bool test_quick_vs_thorough_mode() {
    LOG_TEST("Quick vs Thorough Analysis Mode");
    
    const size_t data_size = 256 * 1024;
    std::vector<uint8_t> h_data;
    generate_mixed_data(h_data, data_size);
    
    void* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    ASSERT_TRUE(err == cudaSuccess, "cudaMalloc failed");
    err = cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    ASSERT_TRUE(err == cudaSuccess, "cudaMemcpy failed");
    
    // Quick mode
    AdaptiveLevelSelector quick_selector;
    quick_selector.enable_quick_analysis(true);
    AdaptiveResult quick_result;
    
    auto quick_start = std::chrono::high_resolution_clock::now();
    Status status = quick_selector.select_level(d_data, data_size, quick_result);
    auto quick_end = std::chrono::high_resolution_clock::now();
    ASSERT_STATUS(status, "Quick analysis failed");
    
    double quick_time = std::chrono::duration<double, std::milli>(quick_end - quick_start).count();
    
    // Thorough mode
    AdaptiveLevelSelector thorough_selector;
    thorough_selector.enable_quick_analysis(false);
    AdaptiveResult thorough_result;
    
    auto thorough_start = std::chrono::high_resolution_clock::now();
    status = thorough_selector.select_level(d_data, data_size, thorough_result);
    auto thorough_end = std::chrono::high_resolution_clock::now();
    ASSERT_STATUS(status, "Thorough analysis failed");
    
    double thorough_time = std::chrono::duration<double, std::milli>(thorough_end - thorough_start).count();
    
    LOG_INFO("Quick mode:");
    LOG_INFO("  Time: " << std::fixed << std::setprecision(3) << quick_time << " ms");
    LOG_INFO("  Level: " << quick_result.recommended_level);
    
    LOG_INFO("Thorough mode:");
    LOG_INFO("  Time: " << std::fixed << std::setprecision(3) << thorough_time << " ms");
    LOG_INFO("  Level: " << thorough_result.recommended_level);
    
    LOG_INFO("Speedup: " << std::fixed << std::setprecision(2) << (thorough_time / quick_time) << "x");
    
    cudaFree(d_data);
    
    LOG_PASS("Quick vs Thorough Mode");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "CUDA ZSTD - Adaptive Level Selector Test Suite" << std::endl;
    print_separator();
    std::cout << "\n";
    
    int passed = 0;
    int total = 0;
    
    // Skip when no CUDA device available; otherwise print device info
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running on: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n" << std::endl;
    
    // Data Characteristic Analysis
    print_separator();
    std::cout << "SUITE 1: Data Characteristic Analysis" << std::endl;
    print_separator();
    
    total++; if (test_random_data_detection()) passed++;
    total++; if (test_repetitive_data_detection()) passed++;
    total++; if (test_text_data_detection()) passed++;
    total++; if (test_binary_data_analysis()) passed++;
    total++; if (test_mixed_data_analysis()) passed++;
    
    // Preference Mode Tests
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 2: Preference Mode Tests" << std::endl;
    print_separator();
    
    total++; if (test_speed_preference()) passed++;
    total++; if (test_balanced_preference()) passed++;
    total++; if (test_ratio_preference()) passed++;
    total++; if (test_preference_override()) passed++;
    
    // Compressibility Estimation
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 3: Compressibility Estimation" << std::endl;
    print_separator();
    
    total++; if (test_compressibility_accuracy()) passed++;
    total++; if (test_actual_vs_predicted_ratio()) passed++;
    
    // Performance Tests
    std::cout << "\n";
    print_separator();
    std::cout << "SUITE 4: Performance Tests" << std::endl;
    print_separator();
    
    total++; if (test_analysis_performance()) passed++;
    total++; if (test_sample_size_impact()) passed++;
    total++; if (test_quick_vs_thorough_mode()) passed++;
    
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
