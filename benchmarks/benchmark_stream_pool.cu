// ==============================================================================
// benchmark_stream_pool.cu - CUDA Stream Pool Performance Benchmark
// ==============================================================================

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "cuda_zstd_stream_pool.h"

using namespace cuda_zstd;

void benchmark_stream_acquisition() {
  std::cout << "\n=== Stream Pool Acquisition Benchmark ===" << std::endl;
  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;

  StreamPool pool(8);
  const int iterations = 10000;

  // Single-threaded acquisition using Guard pattern
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      auto guard = pool.acquire(); // RAII: automatically releases on scope exit
      cudaStream_t stream = guard.get_stream();
      (void)stream; // Use the stream
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Single-threaded acquire/release: " << std::fixed
              << std::setprecision(3) << (elapsed_us / iterations)
              << " us/cycle" << std::endl;
  }

  // Multi-threaded acquisition using Guard pattern
  {
    const int num_threads = 4;
    const int per_thread = iterations / num_threads;
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < num_threads; t++) {
      threads.emplace_back([&pool, per_thread]() {
        for (int i = 0; i < per_thread; i++) {
          auto guard = pool.acquire();
          cudaStream_t stream = guard.get_stream();
          (void)stream;
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Multi-threaded (" << num_threads
              << " threads) acquire/release: " << std::fixed
              << std::setprecision(3) << (elapsed_us / iterations)
              << " us/cycle" << std::endl;
  }
}

void benchmark_stream_vs_create() {
  std::cout << "\n=== Stream Pool vs cudaStreamCreate Comparison ==="
            << std::endl;

  StreamPool pool(8);
  const int iterations = 1000;

  // Stream pool with Guard pattern
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      auto guard = pool.acquire();
      cudaStream_t stream = guard.get_stream();
      // Simulate work
      cudaStreamSynchronize(stream);
      // Guard automatically releases stream on scope exit
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Using StreamPool:       " << std::fixed
              << std::setprecision(2) << (elapsed_us / iterations)
              << " us/iteration" << std::endl;
  }

  // Direct cudaStreamCreate/Destroy
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Using cudaStreamCreate: " << std::fixed
              << std::setprecision(2) << (elapsed_us / iterations)
              << " us/iteration" << std::endl;
  }

  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;
}

void benchmark_concurrent_work() {
  std::cout << "\n=== Concurrent Work with Stream Pool ===" << std::endl;

  StreamPool pool(8);
  const int num_streams = 8;
  const int work_iterations = 100;

  // Allocate test data
  void *d_data;
  const size_t data_size = 1024 * 1024; // 1MB
  cudaMalloc(&d_data, data_size);

  auto start = std::chrono::high_resolution_clock::now();

  // Use scoped guard pattern for concurrent work
  for (int iter = 0; iter < work_iterations; iter++) {
    std::vector<std::thread> threads;
    for (int i = 0; i < num_streams; i++) {
      threads.emplace_back([&pool, d_data, data_size, iter]() {
        auto guard = pool.acquire();
        cudaStream_t stream = guard.get_stream();
        cudaMemsetAsync(d_data, iter % 256, data_size, stream);
        cudaStreamSynchronize(stream);
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double total_data_gb =
      (data_size * num_streams * work_iterations) / (1024.0 * 1024.0 * 1024.0);
  double throughput_gbps = total_data_gb / (elapsed_ms / 1000.0);

  std::cout << "Concurrent work (" << num_streams << " streams): " << std::fixed
            << std::setprecision(2) << throughput_gbps << " GB/s" << std::endl;

  cudaFree(d_data);
}

int main() {
  std::cout << "CUDA Stream Pool Performance Benchmark" << std::endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device: " << prop.name << std::endl;

  benchmark_stream_acquisition();
  benchmark_stream_vs_create();
  benchmark_concurrent_work();

  std::cout << "\n✓ Benchmark complete" << std::endl;
  return 0;
}
