// benchmark_fse_host_fallback.cu - Benchmark for FSE Host Fallback encoding
// Measures the actual latency of encode_sequences_with_predefined_fse

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Local CUDA error check macro (exits on failure, different from library
// version)
#define CUDA_CHECK_EXIT(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#include "cuda_zstd_manager.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_safe_alloc.h"

using namespace cuda_zstd;

// Timer helper
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

// Benchmark the full compress() path which uses
// encode_sequences_with_predefined_fse
void benchmark_compress_path(size_t input_size, int iterations = 5) {
  std::cout << "\n=== Benchmarking Full Compression Path ===" << std::endl;
  std::cout << "Input Size: " << input_size / 1024 << " KB" << std::endl;

  // Generate random compressible data
  std::vector<uint8_t> h_input(input_size);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < input_size; ++i) {
    h_input[i] =
        (uint8_t)(dist(rng) % 64); // Limit range for better compression
  }

  // Allocate device memory
  void *d_input, *d_compressed, *d_temp;
  size_t compressed_max = input_size * 2;
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_compressed, compressed_max));
  CUDA_CHECK_EXIT(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  // Create manager
  auto manager = cuda_zstd::create_manager();
  size_t temp_size = manager->get_compress_temp_size(input_size);
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_temp, temp_size));

  // Warmup
  size_t compressed_size = compressed_max;
  Status status =
      manager->compress(d_input, input_size, d_compressed, &compressed_size,
                        d_temp, temp_size, nullptr, 0, 0);
  cudaDeviceSynchronize();

  if (status != Status::SUCCESS) {
    std::cerr << "Warmup compression failed: " << (int)status << std::endl;
    return;
  }

  // Benchmark
  Timer timer;
  for (int i = 0; i < iterations; ++i) {
    compressed_size = compressed_max;
    manager->compress(d_input, input_size, d_compressed, &compressed_size,
                      d_temp, temp_size, nullptr, 0, 0);
  }
  cudaDeviceSynchronize();
  double avg_ms = timer.elapsed_ms() / iterations;

  double throughput_mbps = (input_size / 1e6) / (avg_ms / 1000.0);
  double ratio = (double)input_size / compressed_size;

  std::cout << "  Compressed Size: " << compressed_size << " bytes"
            << std::endl;
  std::cout << "  Compression Ratio: " << std::fixed << std::setprecision(2)
            << ratio << ":1" << std::endl;
  std::cout << "  Average Latency: " << std::fixed << std::setprecision(2)
            << avg_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
            << throughput_mbps << " MB/s" << std::endl;

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_temp);
}

// Benchmark decompression path
void benchmark_decompress_path(size_t input_size, int iterations = 5) {
  std::cout << "\n=== Benchmarking Full Decompression Path ===" << std::endl;
  std::cout << "Input Size: " << input_size / 1024 << " KB" << std::endl;

  // Generate and compress data first
  std::vector<uint8_t> h_input(input_size);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < input_size; ++i) {
    h_input[i] = (uint8_t)(dist(rng) % 64);
  }

  void *d_input, *d_compressed, *d_output, *d_temp;
  size_t compressed_max = input_size * 2;
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_compressed, compressed_max));
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_output, input_size));
  CUDA_CHECK_EXIT(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  auto manager = cuda_zstd::create_manager();
  size_t temp_size = manager->get_compress_temp_size(input_size);
  CUDA_CHECK_EXIT(cuda_zstd::safe_cuda_malloc(&d_temp, temp_size));

  // Compress
  size_t compressed_size = compressed_max;
  manager->compress(d_input, input_size, d_compressed, &compressed_size, d_temp,
                    temp_size, nullptr, 0, 0);
  cudaDeviceSynchronize();

  // Warmup decompression
  size_t decompressed_size = input_size;
  manager->decompress(d_compressed, compressed_size, d_output,
                      &decompressed_size, d_temp, temp_size);
  cudaDeviceSynchronize();

  // Benchmark decompression
  Timer timer;
  for (int i = 0; i < iterations; ++i) {
    decompressed_size = input_size;
    manager->decompress(d_compressed, compressed_size, d_output,
                        &decompressed_size, d_temp, temp_size);
  }
  cudaDeviceSynchronize();
  double avg_ms = timer.elapsed_ms() / iterations;

  double throughput_mbps = (input_size / 1e6) / (avg_ms / 1000.0);

  std::cout << "  Average Latency: " << std::fixed << std::setprecision(2)
            << avg_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
            << throughput_mbps << " MB/s" << std::endl;

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_output);
  cudaFree(d_temp);
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "  FSE Host Fallback Benchmarks" << std::endl;
  std::cout << "========================================" << std::endl;

  // Print GPU info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Memory: " << std::fixed << std::setprecision(1)
            << prop.totalGlobalMem / 1e9 << " GB" << std::endl;

  // Test various sizes
  std::vector<size_t> sizes = {
      64 * 1024,       // 64KB
      256 * 1024,      // 256KB
      1 * 1024 * 1024, // 1MB
      4 * 1024 * 1024, // 4MB
  };

  std::cout << "\n========== COMPRESSION BENCHMARKS ==========" << std::endl;
  for (size_t size : sizes) {
    benchmark_compress_path(size, 5);
  }

  std::cout << "\n========== DECOMPRESSION BENCHMARKS ==========" << std::endl;
  for (size_t size : sizes) {
    benchmark_decompress_path(size, 5);
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Benchmarks Complete" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
