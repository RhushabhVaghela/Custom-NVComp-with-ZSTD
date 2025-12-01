#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_utils.h"
#include "workspace_manager.h"

// Helper to measure time
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// Helper to generate Huffman-compressible data
void generate_huffman_data(std::vector<uint8_t>& data, size_t size) {
    std::mt19937 rng(42);
    // Skewed distribution (Zipf-like)
    std::discrete_distribution<> dist({
        1000, 500, 250, 125, 60, 30, 15, 8, 4, 2, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    }); 
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t)(dist(rng) % 256);
    }
}

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

void run_benchmark(size_t size, int iterations) {
    std::cout << "Benchmarking Size: " << size << " bytes" << std::endl;

    // 1. Prepare Data
    std::vector<uint8_t> h_input(size);
    generate_huffman_data(h_input, size);

    void *d_input, *d_compressed, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    // For Indexed Huffman: need space for data + header + chunk offsets
    // Worst case: full expansion (size*2) + 257B header + (size/4096)*4B offsets
    size_t max_offsets = ((size + 4095) / 4096) * 4;
    CUDA_CHECK(cudaMalloc(&d_compressed, size * 2 + 512 + max_offsets));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Clear any previous errors
    cudaGetLastError();
    cudaDeviceSynchronize();

    // Huffman Table
    cuda_zstd::huffman::HuffmanTable table;
    CUDA_CHECK(cudaMalloc(&table.codes, cuda_zstd::huffman::MAX_HUFFMAN_SYMBOLS * sizeof(cuda_zstd::huffman::HuffmanCode)));

    // 2. Benchmark Encode
    size_t compressed_size = 0;
    
    // Warmup
    cuda_zstd::huffman::encode_huffman(
        (const cuda_zstd::byte_t*)d_input, (uint32_t)size,
        table,
        (cuda_zstd::byte_t*)d_compressed, &compressed_size,
        nullptr, stream
    );
    
    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        cuda_zstd::huffman::encode_huffman(
            (const cuda_zstd::byte_t*)d_input, (uint32_t)size,
            table,
            (cuda_zstd::byte_t*)d_compressed, &compressed_size,
            nullptr, stream
        );
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double encode_ms = timer.elapsed_ms() / iterations;
    double encode_gbps = (size / 1e9) / (encode_ms / 1000.0);

    std::cout << "  Encode: " << encode_ms << " ms (" << encode_gbps << " GB/s)" << std::endl;
    std::cout << "  Compressed Size: " << compressed_size << " (Ratio: " << (double)size/compressed_size << "x)" << std::endl;

    // 3. Benchmark Decode
    size_t decompressed_size_out = 0;
    
    // Warmup
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, size, stream));
    cuda_zstd::huffman::decode_huffman(
        (const cuda_zstd::byte_t*)d_compressed, compressed_size,
        table,
        (cuda_zstd::byte_t*)d_output, &decompressed_size_out,
        (uint32_t)size, // Expected decompressed size
        stream
    );
    
    timer.reset();
    for (int i = 0; i < iterations; ++i) {
        cuda_zstd::huffman::decode_huffman(
            (const cuda_zstd::byte_t*)d_compressed, compressed_size,
            table,
            (cuda_zstd::byte_t*)d_output, &decompressed_size_out,
            (uint32_t)size,
            stream
        );
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double decode_ms = timer.elapsed_ms() / iterations;
    double decode_gbps = (size / 1e9) / (decode_ms / 1000.0);

    std::cout << "  Decode: " << decode_ms << " ms (" << decode_gbps << " GB/s)" << std::endl;

    // Verify
    std::vector<uint8_t> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));
    if (h_input != h_output) {
        std::cerr << "FAILED: Content mismatch!" << std::endl;
    } else {
        std::cout << "  Verification: PASSED" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

int main() {
    try {
        // Test various sizes
        std::vector<size_t> sizes = {
            4 * 1024,       // 4KB
            64 * 1024,      // 64KB
            1024 * 1024,    // 1MB
            16 * 1024 * 1024// 16MB
        };

        for (size_t size : sizes) {
            run_benchmark(size, 10);
            std::cout << "--------------------------------" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
