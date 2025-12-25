// ==============================================================================
// benchmark_xxhash.cu - XXHash Checksum Performance Benchmark
// ==============================================================================

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_zstd_types.h"
#include "cuda_zstd_xxhash.h"

using namespace cuda_zstd;

void benchmark_xxhash_throughput() {
  std::cout << "\n=== XXHash Checksum Performance Benchmark ===" << std::endl;
  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;

  // Test sizes from 1KB to 128MB
  std::vector<size_t> sizes = {
      1024,             // 1KB
      4096,             // 4KB
      16384,            // 16KB
      65536,            // 64KB
      262144,           // 256KB
      1024 * 1024,      // 1MB
      4 * 1024 * 1024,  // 4MB
      16 * 1024 * 1024, // 16MB
      64 * 1024 * 1024  // 64MB
  };

  const int warmup_runs = 3;
  const int benchmark_runs = 10;

  for (size_t size : sizes) {
    // Check if we have enough GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (size > free_mem * 0.8) {
      std::cout << "Skipping size " << (size / 1024 / 1024)
                << " MB - insufficient GPU memory" << std::endl;
      continue;
    }

    // Allocate and initialize device memory
    byte_t *d_data;
    u64 *d_hash_out;
    cudaError_t err = cudaMalloc(&d_data, size);
    if (err != cudaSuccess) {
      std::cout << "Failed to allocate " << (size / 1024 / 1024) << " MB"
                << std::endl;
      continue;
    }
    cudaMalloc(&d_hash_out, sizeof(u64));

    // Initialize with random data
    std::vector<byte_t> h_data(size);
    for (size_t i = 0; i < size; i++) {
      h_data[i] = static_cast<byte_t>(rand() % 256);
    }
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
      xxhash::compute_xxhash64(d_data, size, 0, d_hash_out, stream);
      cudaStreamSynchronize(stream);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) {
      xxhash::compute_xxhash64(d_data, size, 0, d_hash_out, stream);
      cudaStreamSynchronize(stream);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / benchmark_runs;
    double throughput_gbps =
        (size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << std::setw(8) << (size / 1024) << " KB: " << std::fixed
              << std::setprecision(2) << std::setw(8) << avg_ms << " ms | "
              << std::setw(8) << throughput_gbps << " GB/s" << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(d_hash_out);
    cudaFree(d_data);
  }

  std::cout << std::setfill('=') << std::setw(50) << "=" << std::setfill(' ')
            << std::endl;
}

int main() {
  std::cout << "XXHash Checksum Benchmark" << std::endl;

  // Check CUDA device
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device: " << prop.name << std::endl;

  benchmark_xxhash_throughput();

  std::cout << "\n✓ Benchmark complete" << std::endl;
  return 0;
}
