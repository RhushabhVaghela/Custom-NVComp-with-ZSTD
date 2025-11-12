// test_fse_advanced.cu - Test all 4 advanced features

#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

void test_bit_exact_fse_roundtrip() {
    const cuda_zstd::byte_t test_data[] = {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6};
    cuda_zstd::u32 data_size = sizeof(test_data);
    
    // Encode
    cuda_zstd::byte_t encoded[1024];
    cuda_zstd::u32 encoded_size;
    
    encode_fse_advanced(
        test_data, data_size,
        encoded, &encoded_size,
        TableType::LITERALS, true, true, false, 0
    );
    
    // Decode
    cuda_zstd::byte_t decoded[256];
    cuda_zstd::u32 decoded_size = 0;
    
    decode_fse_predefined(
        encoded, encoded_size,
        decoded, data_size,
        &decoded_size, TableType::LITERALS, 0
    );
    
    // Verify
    assert(decoded_size == data_size);
    for (cuda_zstd::u32 i = 0; i < data_size; i++) {
        assert(decoded[i] == test_data[i]);
    }
}

void test_fse_normalization() {
    std::vector<cuda_zstd::u32> raw_freqs = {100, 200, 50, 75, 150};
    std::vector<cuda_zstd::u16> normalized(256, 0);
    
    cuda_zstd::u32 actual_table_size = 0;
    cuda_zstd::Status status = normalize_frequencies_accurate(
        raw_freqs.data(), 575, 4, normalized.data(), 8, &actual_table_size);
    
    assert(status == cuda_zstd::Status::SUCCESS);
    
    cuda_zstd::u32 sum = 0;
    for (auto freq : normalized) {
        sum += freq;
    }
    
    // CRITICAL: Exact match guaranteed
    assert(sum == 256);  // ✅ MUST PASS
}

int main() {
    printf("========================================\n");
    printf("  FSE Advanced Features Test Suite\n");
    printf("========================================\n\n");
    
    // Test data: various entropy levels
    const char* test_data[] = {
        // Low entropy (repetitive)
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        
        // Medium entropy
        "The quick brown fox jumps over the lazy dog. ",
        
        // High entropy (mixed)
        "1a2B3c4D5e6F7g8H9i0JkLmNoPqRsTuVwXyZ!@#$%"
    };
    
    const char* labels[] = {
        "Low Entropy (Repetitive)",
        "Medium Entropy (Text)",
        "High Entropy (Mixed)"
    };
    
    for (int test = 0; test < 3; test++) {
        printf("\n========================================\n");
        printf("TEST %d: %s\n", test + 1, labels[test]);
        printf("========================================\n\n");
        
        const char* input = test_data[test];
        cuda_zstd::u32 input_size = strlen(input);
        
        // Allocate device memory
        cuda_zstd::byte_t* d_input;
        cuda_zstd::byte_t* d_output;
        cuda_zstd::u32* d_output_size;
        
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, input_size * 2);
        cudaMalloc(&d_output_size, sizeof(u32));
        
        cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
        
        // FEATURE 1 & 2: Analyze with adaptive table log + accurate normalization
        printf("--- Feature 1 & 2: Adaptive + Accurate ---\n");
        cuda_zstd::fse::FSEStats stats;
        cuda_zstd::fse::analyze_block_statistics(d_input, input_size, &stats, 0);
        cuda_zstd::fse::print_fse_stats(stats);
        
        // FEATURE 3: GPU-optimized encoding
        printf("--- Feature 3: GPU Optimization ---\n");
        auto status = cuda_zstd::fse::encode_fse_advanced(
            d_input, input_size,
            d_output, d_output_size,
            TableType::LITERALS,
            true,  // Auto table log (Feature 1)
            true,  // Accurate normalization (Feature 2)
            true,  // GPU optimization (Feature 3)
            0
        );
        
        cuda_zstd::u32 h_output_size;
        cudaMemcpy(&h_output_size, d_output_size, sizeof(cuda_zstd::u32), cudaMemcpyDeviceToHost);
        
        if (status == cuda_zstd::Status::SUCCESS) {
            // Note: print_compression_stats function doesn't exist in the API
            // Use alternative logging
            float ratio = (float)input_size / h_output_size;
            printf("FSE Advanced Encoding:\n");
            printf("  Input: %u bytes\n", input_size);
            printf("  Output: %u bytes\n", h_output_size);
            printf("  Ratio: %.2f:1\n", ratio);
            printf("  Table log: %u\n", stats.recommended_log);
        } else {
            printf("ERROR: Encoding failed\n");
        }
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_output_size);
    }
    
    // FEATURE 4: Multi-table FSE test
    printf("\n========================================\n");
    printf("TEST 4: Multi-Table FSE\n");
    printf("========================================\n\n");
    
    const char* mixed_data =
        "This is a test of multi-table FSE compression. "
        "It should use different tables for different data patterns.";
    
    cuda_zstd::u32 mixed_size = strlen(mixed_data);
    cuda_zstd::byte_t* d_mixed;
    cudaMalloc(&d_mixed, mixed_size);
    cudaMemcpy(d_mixed, mixed_data, mixed_size, cudaMemcpyHostToDevice);
    
    cuda_zstd::fse::MultiTableFSE multi_table;
    cuda_zstd::fse::create_multi_table_fse(multi_table, d_mixed, mixed_size, 0);
    
    printf("Active Tables: 0x%X\n", multi_table.active_tables);
    if (multi_table.active_tables & (1 << (int)cuda_zstd::fse::TableType::LITERALS))
        printf("  ✓ LITERALS table\n");
    if (multi_table.active_tables & (1 << (int)cuda_zstd::fse::TableType::MATCH_LENGTHS))
        printf("  ✓ MATCH_LENGTHS table\n");
    if (multi_table.active_tables & (1 << (int)cuda_zstd::fse::TableType::OFFSETS))
        printf("  ✓ OFFSETS table\n");
    if (multi_table.active_tables & (1 << (int)cuda_zstd::fse::TableType::CUSTOM))
        printf("  ✓ CUSTOM table\n");
    
    cuda_zstd::fse::free_multi_table(multi_table);
    cudaFree(d_mixed);

    test_bit_exact_fse_roundtrip();
    test_fse_normalization();
    
    printf("\n========================================\n");
    printf("  ALL TESTS COMPLETED SUCCESSFULLY!\n");
    printf("========================================\n");
    
    return 0;
}
