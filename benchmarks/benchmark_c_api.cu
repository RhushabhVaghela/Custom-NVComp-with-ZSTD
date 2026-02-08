// ==============================================================================
// benchmark_c_api.cu - C API Wrapper Performance Benchmark
// ==============================================================================
// Benchmarks the overhead of C API wrappers vs direct C++ API

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"

using namespace cuda_zstd;

void benchmark_cpp_vs_c_overhead() {
  std::cout << "\n=== C++ API Performance Benchmark ===" << std::endl;
  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;

  std::vector<size_t> sizes = {64 * 1024,        // 64KB
                               256 * 1024,       // 256KB
                               1024 * 1024,      // 1MB
                               4 * 1024 * 1024}; // 4MB

  const int warmup_runs = 2;
  const int benchmark_runs = 5;

  std::cout << std::setw(10) << "Size" << " | " << std::setw(12) << "Compress"
            << " | " << std::setw(12) << "Decompress" << std::endl;
  std::cout << std::setfill('-') << std::setw(40) << "-" << std::setfill(' ')
            << std::endl;

  for (size_t size : sizes) {
    // Generate test data
    std::vector<uint8_t> h_data(size);
    for (size_t i = 0; i < size; i++) {
      h_data[i] = static_cast<uint8_t>((i * 17 + i / 256) % 256);
    }

    // Allocate device memory
    void *d_input, *d_output, *d_decompressed, *d_temp;
    cuda_zstd::safe_cuda_malloc(&d_input, size);
    cuda_zstd::safe_cuda_malloc(&d_output, size * 2);
    cuda_zstd::safe_cuda_malloc(&d_decompressed, size);
    cudaMemcpy(d_input, h_data.data(), size, cudaMemcpyHostToDevice);

    // Create manager
    ZstdBatchManager manager(CompressionConfig{.level = 3});
    size_t temp_size = manager.get_compress_temp_size(size);
    cuda_zstd::safe_cuda_malloc(&d_temp, temp_size);

    // === Compression Benchmark ===
    size_t compressed_size = size * 2;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
      compressed_size = size * 2;
      manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                       temp_size, nullptr, 0);
      cudaDeviceSynchronize();
    }

    // Benchmark compression
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) {
      compressed_size = size * 2;
      manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                       temp_size, nullptr, 0);
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double compress_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchmark_runs;
    double compress_mbps = (size / (1024.0 * 1024.0)) / (compress_ms / 1000.0);

    // === Decompression Benchmark ===
    size_t decompressed_size = size;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
      decompressed_size = size;
      manager.decompress(d_output, compressed_size, d_decompressed,
                         &decompressed_size, d_temp, temp_size);
      cudaDeviceSynchronize();
    }

    // Benchmark decompression
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) {
      decompressed_size = size;
      manager.decompress(d_output, compressed_size, d_decompressed,
                         &decompressed_size, d_temp, temp_size);
      cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();

    double decompress_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchmark_runs;
    double decompress_mbps =
        (size / (1024.0 * 1024.0)) / (decompress_ms / 1000.0);

    std::cout << std::setw(6) << (size / 1024) << " KB | " << std::fixed
              << std::setprecision(1) << std::setw(8) << compress_mbps
              << " MB/s | " << std::setw(8) << decompress_mbps << " MB/s"
              << std::endl;

    cudaFree(d_temp);
    cudaFree(d_decompressed);
    cudaFree(d_output);
    cudaFree(d_input);
  }

  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;
}

void benchmark_manager_creation_overhead() {
  std::cout << "\n=== Manager Creation Overhead ===" << std::endl;

  const int iterations = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    ZstdBatchManager manager(CompressionConfig{.level = 3});
    // Manager destructor called automatically
  }
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  std::cout << "Manager creation/destruction: " << std::fixed
            << std::setprecision(1) << (elapsed_us / iterations) << " us/cycle"
            << std::endl;
}

int main() {
  std::cout << "C++ API Performance Benchmark" << std::endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device: " << prop.name << std::endl;

  benchmark_cpp_vs_c_overhead();
  benchmark_manager_creation_overhead();

  std::cout << "\n✓ Benchmark complete" << std::endl;
  return 0;
}
