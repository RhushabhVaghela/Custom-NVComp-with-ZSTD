// ============================================================================
// test_roundtrip.cu - Verify Compress -> Decompress Correctness
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

using namespace cuda_zstd;

bool verify_data(const std::vector<uint8_t>& original, const std::vector<uint8_t>& decompressed) {
    if (original.size() != decompressed.size()) {
        std::cerr << "  ✗ FAILED: Size mismatch. Original=" << original.size()
                  << ", Decompressed=" << decompressed.size() << "\n";
        return false;
    }
    
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != decompressed[i]) {
            std::cerr << "  ✗ FAILED: Data mismatch at byte " << i << ". "
                      << "Expected=" << (int)original[i] 
                      << ", Got=" << (int)decompressed[i] << "\n";
            return false;
        }
    }
    
    std::cout << "  ✓ PASSED: Data verified.\n";
    return true;
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  Test: Round-Trip (Compress/Decompress)\n";
    std::cout << "========================================\n\n";

    const size_t data_size = 1024 * 1024; // 1 MB
    const int level = 5;

    // 1. Create test data
    std::vector<uint8_t> h_input(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        h_input[i] = static_cast<uint8_t>((i * 7) % 256);
    }
    
    // 2. Allocate GPU buffers
    void *d_input, *d_compressed, *d_decompressed, *d_temp_comp, *d_temp_decomp;
    if (cuda_zstd::safe_cuda_malloc(&d_input, data_size) != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaMalloc d_input\n";
        return 1;
    }
    if (cuda_zstd::safe_cuda_malloc(&d_compressed, data_size * 2) != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaMalloc d_compressed\n";
        cudaFree(d_input);
        return 1;
    }
    if (cuda_zstd::safe_cuda_malloc(&d_decompressed, data_size) != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaMalloc d_decompressed\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        return 1;
    }
    
    cuda_zstd::CompressionConfig config{.level = level};
    ZstdBatchManager manager(config);
    size_t comp_temp_size = manager.get_compress_temp_size(data_size);
    size_t decomp_temp_size = manager.get_decompress_temp_size(data_size * 2);
    size_t temp_size = std::max(comp_temp_size, decomp_temp_size);
    if (cuda_zstd::safe_cuda_malloc(&d_temp_comp, temp_size) != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaMalloc d_temp_comp\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        return 1;
    }
    d_temp_decomp = d_temp_comp; // Reuse workspace
    
    if (cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaMemcpy to d_input\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        cudaFree(d_temp_comp);
        return 1;
    }
    
    // 3. Compress
    std::cout << "Compressing 1MB with level " << level << "...\n";
    size_t compressed_size = data_size * 2;
    Status status = manager.compress(
        d_input, data_size,
        d_compressed, &compressed_size,
        d_temp_comp, temp_size,
        nullptr, 0
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaDeviceSynchronize after compress\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        cudaFree(d_temp_comp);
        return 1;
    }
    
    if (status != Status::SUCCESS) {
        std::cerr << "  ✗ FAILED: Compression returned " << status_to_string(status) << "\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        cudaFree(d_temp_comp);
        return 1;
    }
    std::cout << "  Compressed size: " << compressed_size << " bytes.\n";

    // 4. Decompress
    std::cout << "Decompressing...\n";
    size_t decompressed_size = data_size;
    status = manager.decompress(
        d_compressed, compressed_size,
        d_decompressed, &decompressed_size,
        d_temp_decomp, temp_size
    );
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "  ✗ FAILED: cudaDeviceSynchronize after decompress\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        cudaFree(d_temp_comp);
        return 1;
    }
    
    if (status != Status::SUCCESS) {
        std::cerr << "  ✗ FAILED: Decompression returned " << status_to_string(status) << "\n";
        cudaFree(d_input);
        cudaFree(d_compressed);
        cudaFree(d_decompressed);
        cudaFree(d_temp_comp);
        return 1;
    }
    std::cout << "  Decompressed size: " << decompressed_size << " bytes.\n";

    // 5. Verify
    std::vector<uint8_t> h_output(decompressed_size);
    cudaMemcpy(h_output.data(), d_decompressed, decompressed_size, cudaMemcpyDeviceToHost);
    
    bool success = verify_data(h_input, h_output);
    
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_temp_comp);
    
    std::cout << "\nTest complete. Result: " << (success ? "PASSED ✓" : "FAILED ✗") << "\n";
    return success ? 0 : 1;
}