#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_utils.h"
#include "cuda_zstd_types.h"

using namespace cuda_zstd;
using namespace cuda_zstd::huffman;

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper to fill buffer with random data
void fill_random(std::vector<u8>& buffer, size_t size, u32 seed = 1234) {
    std::mt19937 gen(seed);
    // Use a skewed distribution to make Huffman meaningful
    std::exponential_distribution<> d(1.0); 
    
    for (size_t i = 0; i < size; ++i) {
        int val = static_cast<int>(d(gen) * 30.0);
        buffer[i] = static_cast<u8>(std::min(val, 255));
    }
}

void test_huffman_roundtrip(size_t input_size) {
    std::cout << "Testing Huffman Roundtrip with size: " << input_size << " bytes... ";
    
    // 1. Prepare Host Data
    std::vector<u8> h_input(input_size);
    fill_random(h_input, input_size);
    
    // 2. Allocate Device Memory
    u8 *d_input, *d_compressed, *d_decompressed;
    size_t max_compressed_size = (input_size * 2) + 65536; // Header + slack (64KB for safety)
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_compressed, max_compressed_size));
    CUDA_CHECK(cudaMalloc(&d_decompressed, input_size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    
    // 3. Prepare Huffman Table
    HuffmanTable table;
    CUDA_CHECK(cudaMalloc(&table.codes, 256 * sizeof(HuffmanCode)));
    
    // 4. Encode
    size_t compressed_size = 0;
    // Note: We are passing nullptr for workspace, so it should self-allocate
    Status status = encode_huffman(
        d_input,
        static_cast<u32>(input_size),
        table,
        d_compressed,
        &compressed_size,
        nullptr, // workspace
        0        // stream
    );
    
    if (status != Status::SUCCESS) {
        std::cerr << "FAILED: encode_huffman returned " << (int)status << std::endl;
        exit(1);
    }
    
    std::cout << "Compressed size: " << compressed_size << " bytes (" << (float)input_size/compressed_size << "x)" << std::endl;

    // 5. Decode
    size_t decompressed_size_out = 0;
    status = decode_huffman(
        d_compressed,
        compressed_size,
        table, // Unused by decode, but required by API signature
        d_decompressed,
        &decompressed_size_out,
        static_cast<u32>(input_size), // Expected size
        0 // stream
    );
    
    if (status != Status::SUCCESS) {
        std::cerr << "FAILED: decode_huffman returned " << (int)status << std::endl;
        exit(1);
    }
    
    // 6. Verify
    if (decompressed_size_out != input_size) {
        std::cerr << "FAILED: Size mismatch. Expected " << input_size << ", got " << decompressed_size_out << std::endl;
        exit(1);
    }
    
    std::vector<u8> h_output(input_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_decompressed, input_size, cudaMemcpyDeviceToHost));
    
    if (h_input != h_output) {
        std::cerr << "FAILED: Content mismatch!" << std::endl;
        // Find first mismatch
        for(size_t i=0; i<input_size; ++i) {
            if(h_input[i] != h_output[i]) {
                std::cerr << "Mismatch at index " << i << ": expected " << (int)h_input[i] << ", got " << (int)h_output[i] << std::endl;
                // Dump context
                size_t start = (i > 10) ? i - 10 : 0;
                size_t end = std::min(input_size, i + 10);
                std::cerr << "Context (Expected vs Got):" << std::endl;
                for(size_t j=start; j<end; ++j) {
                    std::cerr << "[" << j << "] " << (int)h_input[j] << " vs " << (int)h_output[j] << (i==j ? " <--" : "") << std::endl;
                }
                break;
            }
        }
        exit(1);
    }
    
    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_compressed));
    CUDA_CHECK(cudaFree(d_decompressed));
    CUDA_CHECK(cudaFree(table.codes));
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    try {
        test_huffman_roundtrip(1024);       // 1 KB
        test_huffman_roundtrip(65536);      // 64 KB
        test_huffman_roundtrip(1024 * 1024); // 1 MB
        test_huffman_roundtrip(16 * 1024 * 1024); // 16 MB
        
        std::cout << "\nAll Huffman tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
