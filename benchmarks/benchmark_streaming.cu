// ==============================================================================
// benchmark_streaming.cu - Streaming Manager Performance Benchmark
// ==============================================================================

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_zstd_manager.h"

using namespace cuda_zstd;

void benchmark_chunked_compression() {
  std::cout << "\n=== Chunked Compression Benchmark ===" << std::endl;
  std::cout << std::setfill('=') << std::setw(55) << "=" << std::setfill(' ')
            << std::endl;

  // Test with various chunk sizes
  std::vector<std::pair<size_t, size_t>> configs = {
      {64 * 1024, 1024 * 1024},        // 64KB chunks, 1MB total -> 16 chunks
      {256 * 1024, 4 * 1024 * 1024},   // 256KB chunks, 4MB total -> 16 chunks
      {1024 * 1024, 16 * 1024 * 1024}, // 1MB chunks, 16MB total -> 16 chunks
  };

  const int warmup_runs = 1;
  const int benchmark_runs = 3;

  std::cout << std::setw(12) << "Chunk Size" << " | " << std::setw(12)
            << "Total Size" << " | " << std::setw(12) << "Throughput"
            << std::endl;
  std::cout << std::setfill('-') << std::setw(45) << "-" << std::setfill(' ')
            << std::endl;

  for (auto &config : configs) {
    size_t chunk_size = config.first;
    size_t total_size = config.second;
    size_t num_chunks = total_size / chunk_size;

    // Generate test data
    std::vector<uint8_t> h_data(total_size);
    for (size_t i = 0; i < total_size; i++) {
      h_data[i] = static_cast<uint8_t>((i * 17 + i / 256) % 256);
    }

    // Allocate device memory
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, total_size);
    cudaMalloc(&d_output, total_size * 2);
    cudaMemcpy(d_input, h_data.data(), total_size, cudaMemcpyHostToDevice);

    ZstdBatchManager manager(CompressionConfig{.level = 3});
    size_t temp_size = manager.get_compress_temp_size(chunk_size);
    cudaMalloc(&d_temp, temp_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup - compress all chunks
    for (int w = 0; w < warmup_runs; w++) {
      size_t output_offset = 0;
      for (size_t c = 0; c < num_chunks; c++) {
        size_t compressed_size = chunk_size * 2;
        manager.compress((uint8_t *)d_input + c * chunk_size, chunk_size,
                         (uint8_t *)d_output + output_offset, &compressed_size,
                         d_temp, temp_size, nullptr, 0, stream);
        output_offset += compressed_size;
      }
      cudaStreamSynchronize(stream);
    }

    // Benchmark - compress all chunks
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < benchmark_runs; r++) {
      size_t output_offset = 0;
      for (size_t c = 0; c < num_chunks; c++) {
        size_t compressed_size = chunk_size * 2;
        manager.compress((uint8_t *)d_input + c * chunk_size, chunk_size,
                         (uint8_t *)d_output + output_offset, &compressed_size,
                         d_temp, temp_size, nullptr, 0, stream);
        output_offset += compressed_size;
      }
      cudaStreamSynchronize(stream);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchmark_runs;
    double throughput_mbps =
        (total_size / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    std::cout << std::setw(8) << (chunk_size / 1024) << " KB | " << std::setw(8)
              << (total_size / 1024 / 1024) << " MB | " << std::fixed
              << std::setprecision(1) << std::setw(8) << throughput_mbps
              << " MB/s" << std::endl;

    cudaFree(d_temp);
    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_input);
  }

  std::cout << std::setfill('=') << std::setw(55) << "=" << std::setfill(' ')
            << std::endl;
}

void benchmark_single_vs_chunked() {
  std::cout << "\n=== Single vs Chunked Compression ===" << std::endl;

  const size_t total_size = 16 * 1024 * 1024; // 16MB
  const size_t chunk_size = 1024 * 1024;      // 1MB chunks
  const int benchmark_runs = 3;

  // Generate test data
  std::vector<uint8_t> h_data(total_size);
  for (size_t i = 0; i < total_size; i++) {
    h_data[i] = static_cast<uint8_t>((i * 17 + i / 256) % 256);
  }

  void *d_input, *d_output, *d_temp;
  cudaMalloc(&d_input, total_size);
  cudaMalloc(&d_output, total_size * 2);
  cudaMemcpy(d_input, h_data.data(), total_size, cudaMemcpyHostToDevice);

  // Single shot compression
  {
    ZstdBatchManager manager(CompressionConfig{.level = 3});
    size_t temp_size = manager.get_compress_temp_size(total_size);
    cudaMalloc(&d_temp, temp_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < benchmark_runs; r++) {
      size_t compressed_size = total_size * 2;
      manager.compress(d_input, total_size, d_output, &compressed_size, d_temp,
                       temp_size, nullptr, 0);
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchmark_runs;
    double throughput_mbps =
        (total_size / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    std::cout << "Single shot (16MB):     " << std::fixed
              << std::setprecision(1) << throughput_mbps << " MB/s"
              << std::endl;

    cudaFree(d_temp);
  }

  // Chunked compression (16 x 1MB)
  {
    ZstdBatchManager manager(CompressionConfig{.level = 3});
    size_t temp_size = manager.get_compress_temp_size(chunk_size);
    cudaMalloc(&d_temp, temp_size);

    size_t num_chunks = total_size / chunk_size;

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < benchmark_runs; r++) {
      size_t output_offset = 0;
      for (size_t c = 0; c < num_chunks; c++) {
        size_t compressed_size = chunk_size * 2;
        manager.compress((uint8_t *)d_input + c * chunk_size, chunk_size,
                         (uint8_t *)d_output + output_offset, &compressed_size,
                         d_temp, temp_size, nullptr, 0);
        output_offset += compressed_size;
      }
      cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchmark_runs;
    double throughput_mbps =
        (total_size / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    std::cout << "Chunked (16 x 1MB):     " << std::fixed
              << std::setprecision(1) << throughput_mbps << " MB/s"
              << std::endl;

    cudaFree(d_temp);
  }

  cudaFree(d_output);
  cudaFree(d_input);
}

int main() {
  std::cout << "Streaming/Chunked Compression Benchmark" << std::endl;

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device: " << prop.name << std::endl;

  benchmark_chunked_compression();
  benchmark_single_vs_chunked();

  std::cout << "\n✓ Benchmark complete" << std::endl;
  return 0;
}
