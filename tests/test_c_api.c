// ============================================================================
// test_c_api.c - Verify the C-API
// ============================================================================

#include "cuda_zstd_manager.h" // C-API is in the manager header
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    printf("\n========================================\n");
    printf("  Test: C-API Validation\n");
    printf("========================================\n\n");
    
    // 1. Create Manager
    int level = 3;
    cuda_zstd_manager_t* manager = cuda_zstd_create_manager(level);
    if (!manager) {
        printf("  ✗ FAILED: cuda_zstd_create_manager returned NULL\n");
        return 1;
    }
    printf("  ✓ cuda_zstd_create_manager passed.\n");
    
    // 2. Prepare data
    size_t data_size = 128 * 1024;
    void* h_data = malloc(data_size);
    void *d_input, *d_output, *d_temp;
    
    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size * 2);
    
    size_t temp_size = cuda_zstd_get_compress_workspace_size(manager, data_size);
    cudaMalloc(&d_temp, temp_size);
    
    printf("  ✓ Workspace allocated: %zu KB\n", temp_size / 1024);
    
    // 3. Compress
    size_t compressed_size = 0;
    int err = cuda_zstd_compress(
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
        printf("  ✗ FAILED: cuda_zstd_compress returned error %d (%s)\n", 
               err, cuda_zstd_get_error_string(err));
        return 1;
    }
    printf("  ✓ cuda_zstd_compress passed. Size: %zu\n", compressed_size);
    
    // 4. Destroy Manager
    cuda_zstd_destroy_manager(manager);
    printf("  ✓ cuda_zstd_destroy_manager passed.\n");
    
    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    printf("\nTest complete. Result: PASSED ✓\n");
    return 0;
}