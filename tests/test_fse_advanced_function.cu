#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include "cuda_error_checking.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <vector>

void test_bit_exact_fse_roundtrip() {
    const cuda_zstd::byte_t test_data[] = {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6};
    cuda_zstd::u32 data_size = sizeof(test_data);
    
    printf("Testing FSE roundtrip with %u bytes of data\n", data_size);
    
    // Allocate device memory
    cuda_zstd::byte_t* d_input = nullptr;
    cuda_zstd::byte_t* d_output = nullptr;
    cuda_zstd::u32* d_output_size = nullptr;
    
    if (!safe_cuda_malloc((void**)&d_input, data_size)) {
        printf("ERROR: CUDA malloc for d_input failed\n");
        assert(0);
    }
    
    if (!safe_cuda_malloc((void**)&d_output, data_size * 2)) {
        printf("ERROR: CUDA malloc for d_output failed\n");
        safe_cuda_free(d_input);
        assert(0);
    }
    
    if (!safe_cuda_malloc((void**)&d_output_size, sizeof(cuda_zstd::u32))) {
        printf("ERROR: CUDA malloc for d_output_size failed\n");
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        assert(0);
    }
    
    if (!safe_cuda_memcpy(d_input, test_data, data_size, cudaMemcpyHostToDevice)) {
        printf("ERROR: CUDA memcpy to d_input failed\n");
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        assert(0);
    }
    
    // Encode using the FSE API directly
    printf("Encoding with FSE...\n");
    cuda_zstd::Status status = cuda_zstd::fse::encode_fse_advanced(
        d_input, data_size,
        d_output, d_output_size,
        cuda_zstd::fse::TableType::LITERALS, 
        true,  // auto_table_log
        true,  // accurate_norm  
        false, // gpu_optimize
        0      // stream
    );
    
    if (status != cuda_zstd::Status::SUCCESS) {
        printf("ERROR: FSE encoding failed: %d\n", (int)status);
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        assert(0);
    }
    
    // Copy encoded size back
    cuda_zstd::u32 encoded_size = 0;
    if (!safe_cuda_memcpy(&encoded_size, d_output_size, sizeof(cuda_zstd::u32), cudaMemcpyDeviceToHost)) {
        printf("ERROR: CUDA memcpy for encoded_size failed\n");
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        assert(0);
    }
    
    printf("Encoded to %u bytes\n", encoded_size);
    
    // Allocate memory for decoded data
    cuda_zstd::byte_t* d_decoded = nullptr;
    if (!safe_cuda_malloc((void**)&d_decoded, data_size)) {
        printf("ERROR: CUDA malloc for d_decoded failed\n");
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        assert(0);
    }
    
    // Use the correct decode_fse signature: (input, size, output, output_size_ptr, stream)
    cuda_zstd::u32 h_decoded_size = 0;
    cuda_zstd::Status dec_status = cuda_zstd::fse::decode_fse(
        d_output, encoded_size,
        d_decoded, &h_decoded_size,
        0  // stream
    );
    
    if (dec_status != cuda_zstd::Status::SUCCESS) {
        printf("ERROR: FSE decoding failed: %d\n", (int)dec_status);
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        safe_cuda_free(d_decoded);
        assert(0);
    }
    
    printf("Decoded to %u bytes\n", h_decoded_size);
    
    // Verify
    assert(h_decoded_size == data_size);
    
    // Verify content
    std::vector<cuda_zstd::byte_t> decoded_data(data_size);
    if (!safe_cuda_memcpy(decoded_data.data(), d_decoded, data_size, cudaMemcpyDeviceToHost)) {
        printf("ERROR: CUDA memcpy for decoded data failed\n");
        safe_cuda_free(d_input);
        safe_cuda_free(d_output);
        safe_cuda_free(d_output_size);
        safe_cuda_free(d_decoded);
        assert(0);
    }
    
    for (cuda_zstd::u32 i = 0; i < data_size; i++) {
        assert(decoded_data[i] == test_data[i]);
    }
    
    printf("FSE roundtrip test passed!\n");
    
    // Cleanup
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_output_size);
    safe_cuda_free(d_decoded);
}

int main() {
    printf("=== FSE Advanced Function Test ===\n");
    test_bit_exact_fse_roundtrip();
    printf("All tests passed!\n");
    return 0;
}
