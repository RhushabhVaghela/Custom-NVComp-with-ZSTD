// ============================================================================
// test_c_api.c - Verify the C-API
// ============================================================================

#include "cuda_zstd_nvcomp.h" // C-API is in the nvcomp header
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    printf("\n========================================\n");
    printf("  Test: C-API Validation (NVCOMP v5)\n");
    printf("========================================\n\n");
    
    // 1. Create Manager
    int level = 3;
    nvcompZstdManagerHandle manager = nvcomp_zstd_create_manager_v5(level);
    if (!manager) {
        printf("  ✗ FAILED: nvcomp_zstd_create_manager_v5 returned NULL\n");
        return 1;
    }
    printf("  ✓ nvcomp_zstd_create_manager_v5 passed.\n");
    
    // 2. Prepare data
    size_t data_size = 128 * 1024;
    void* h_data = malloc(data_size); // Not strictly needed, but good practice
    void *d_input, *d_output, *d_temp;
    
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    
    size_t temp_size = nvcomp_zstd_get_compress_temp_size_v5(manager, data_size);
    cudaMalloc(&d_temp, temp_size);
    
    printf("  ✓ Workspace allocated: %zu KB\n", temp_size / 1024);
    
    // 3. Compress
    size_t compressed_size = 0;
    int err = nvcomp_zstd_compress_async_v5(
        manager,
        d_input,
        data_size,
        d_output,
        &compressed_size,
        d_temp,
        temp_size,
        0
    );
    cudaDeviceSynchronize();
    
    if (err != 0) {
        printf("  ✗ FAILED: nvcomp_zstd_compress_async_v5 returned error %d\n", err);
        return 1;
    }
    printf("  ✓ nvcomp_zstd_compress_async_v5 passed. Size: %zu\n", compressed_size);
    
    // 4. Destroy Manager
    nvcomp_zstd_destroy_manager_v5(manager);
    printf("  ✓ nvcomp_zstd_destroy_manager_v5 passed.\n");
    
    // Cleanup
    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    printf("\nTest complete. Result: PASSED ✓\n");
    return 0;
}