// ============================================================================
// test_nvcomp_batch.cu - Verify NvcompV5BatchManager C++ API
// ============================================================================

#include "cuda_zstd_nvcomp.h" // Use the NVComp C++ API
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

using namespace cuda_zstd;
using namespace cuda_zstd::nvcomp_v5;

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  Test: NvcompV5BatchManager C++ API\n";
    std::cout << "========================================\n\n";
    
    const int batch_size = 8;
    const size_t chunk_size = 64 * 1024; // 64 KB

    // 1. Create NVComp options and manager
    NvcompV5Options opts;
    opts.level = 5;
    opts.chunk_size = chunk_size;
    
    NvcompV5BatchManager batch_manager(opts);
    std::cout << "Created NvcompV5BatchManager with level " << opts.level << "\n";

    // 2. Prepare host data
    std::vector<std::vector<byte_t>> h_inputs(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        h_inputs[i].resize(chunk_size);
        for (size_t j = 0; j < chunk_size; ++j) {
            h_inputs[i][j] = (byte_t)(i + j);
        }
    }
    
    // 3. Allocate device data as required by the NVComp API
    // (arrays of pointers)
    std::vector<void*> d_input_ptrs_vec(batch_size);
    std::vector<void*> d_output_ptrs_vec(batch_size);
    std::vector<size_t> h_input_sizes_vec(batch_size, chunk_size);
    
    size_t max_comp_size = batch_manager.get_max_compressed_chunk_size(chunk_size);
    
    void **d_input_ptrs, **d_output_ptrs;
    size_t *d_input_sizes, *d_output_sizes;

    cudaMalloc(&d_input_ptrs, batch_size * sizeof(void*));
    cudaMalloc(&d_output_ptrs, batch_size * sizeof(void*));
    cudaMalloc(&d_input_sizes, batch_size * sizeof(size_t));
    cudaMalloc(&d_output_sizes, batch_size * sizeof(size_t));

    for (int i = 0; i < batch_size; ++i) {
        cudaMalloc(&d_input_ptrs_vec[i], chunk_size);
        cudaMalloc(&d_output_ptrs_vec[i], max_comp_size);
        cudaMemcpy(d_input_ptrs_vec[i], h_inputs[i].data(), chunk_size, cudaMemcpyHostToDevice);
    }
    
    cudaMemcpy(d_input_ptrs, d_input_ptrs_vec.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_ptrs, d_output_ptrs_vec.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_sizes, h_input_sizes_vec.data(), batch_size * sizeof(size_t), cudaMemcpyHostToDevice);

    // 4. Get workspace
    size_t temp_size = batch_manager.get_compress_temp_size(d_input_sizes, batch_size);
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);
    std::cout << "Allocated " << temp_size / 1024 << " KB temp workspace.\n";
    
    // 5. Compress batch
    std::cout << "Compressing batch of " << batch_size << " items...\n";
    Status status = batch_manager.compress_async(
        (const void* const*)d_input_ptrs,
        d_input_sizes,
        batch_size,
        d_output_ptrs,
        d_output_sizes,
        d_temp,
        temp_size,
        0
    );
    cudaDeviceSynchronize();

    if (status != Status::SUCCESS) {
        std::cerr << "  ✗ FAILED: Batch compress returned " << status_to_string(status) << "\n";
        return 1;
    }
    
    std::cout << "  ✓ Batch compressed.\n";
    std::cout << "Stats: " << batch_manager.get_stats().bytes_produced << " total bytes.\n";

    // (A full test would also decompress and verify)

    // 6. Cleanup
    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_input_ptrs_vec[i]);
        cudaFree(d_output_ptrs_vec[i]);
    }
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_temp);
    
    std::cout << "\nTest complete. Result: PASSED ✓\n";
    return 0;
}