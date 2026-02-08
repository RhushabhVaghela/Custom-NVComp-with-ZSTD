// ==============================================================================
// benchmark_adaptive.cu - Adaptive Level Selection Performance Benchmark
// ==============================================================================

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"

using namespace cuda_zstd;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void benchmark_adaptive_level_selection() {
  std::cout << "\n=== Adaptive Level Selection Benchmark ===" << std::endl;
  std::cout << std::setfill('=') << std::setw(55) << "=" << std::setfill(' ')
            << std::endl;

  std::vector<size_t> sizes = {64 * 1024,         // 64KB
                               256 * 1024,        // 256KB
                               1024 * 1024,       // 1MB
                               4 * 1024 * 1024,   // 4MB
                               16 * 1024 * 1024}; // 16MB

  std::vector<int> levels = {1, 3, 5, 9, 15, 19}; // Various compression levels

  const int warmup_runs = 2;
  const int benchmark_runs = 5;

  std::cout << std::setw(10) << "Size" << " | ";
  for (int level : levels) {
    std::cout << "Level " << std::setw(2) << level << " | ";
  }
  std::cout << std::endl;
  std::cout << std::setfill('-') << std::setw(55) << "-" << std::setfill(' ')
            << std::endl;

  for (size_t size : sizes) {
    std::cout << std::setw(6) << (size / 1024) << " KB | ";

    // Generate test data
    std::vector<uint8_t> h_data(size);
    for (size_t i = 0; i < size; i++) {
      h_data[i] = static_cast<uint8_t>((i * 17 + i / 256) % 256);
    }

    void *d_input, *d_output, *d_temp;
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_input, size));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, size * 2));
    cudaMemcpy(d_input, h_data.data(), size, cudaMemcpyHostToDevice);

    for (int level : levels) {
      ZstdBatchManager manager(CompressionConfig{.level = level});

      size_t temp_size = manager.get_compress_temp_size(size);
      CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_temp, temp_size));

      // Warmup
      size_t compressed_size = size * 2;
      for (int i = 0; i < warmup_runs; i++) {
        manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                         temp_size, nullptr, 0);
      }

      // Benchmark
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < benchmark_runs; i++) {
        compressed_size = size * 2;
        manager.compress(d_input, size, d_output, &compressed_size, d_temp,
                         temp_size, nullptr, 0);
        cudaDeviceSynchronize();
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = elapsed_ms / benchmark_runs;
      double throughput_mbps = (size / (1024.0 * 1024.0)) / (avg_ms / 1000.0);

      std::cout << std::fixed << std::setprecision(1) << std::setw(5)
                << throughput_mbps << " | ";

      cudaFree(d_temp);
    }
    std::cout << " MB/s" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
  }

  std::cout << std::setfill('=') << std::setw(55) << "=" << std::setfill(' ')
            << std::endl;
}

void benchmark_auto_level_decision() {
  std::cout << "\n=== Auto Level Decision Overhead ===" << std::endl;

  // Measure overhead of level selection algorithm
  const int iterations = 1000;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    // Simulate level selection based on data characteristics
    size_t test_size = rand() % (16 * 1024 * 1024);
    [[maybe_unused]] int selected_level;
    if (test_size < 64 * 1024) {
      selected_level = 1; // Fast for small data
    } else if (test_size < 1024 * 1024) {
      selected_level = 3; // Balanced
    } else {
      selected_level = 5; // Better ratio for large data
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  std::cout << "Level selection overhead: " << std::fixed
            << std::setprecision(3) << (elapsed_us / iterations) << " us/call"
            << std::endl;
}

int main() {
  std::cout << "Adaptive Level Selection Benchmark" << std::endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device: " << prop.name << std::endl;

  benchmark_adaptive_level_selection();
  benchmark_auto_level_decision();

  std::cout << "\n✓ Benchmark complete" << std::endl;
  return 0;
}
