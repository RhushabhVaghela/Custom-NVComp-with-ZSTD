#include "../include/cuda_zstd_manager.h"
#include "../include/cuda_zstd_types.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      exit(1);                                                                 \
    }                                                                          \
  }

class Timer {
public:
  Timer() { reset(); }
  void reset() { start_ = std::chrono::high_resolution_clock::now(); }
  double elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
};

void generate_compressible_data(std::vector<byte_t> &data, size_t size) {
  std::mt19937 rng(42);
  // Generate pattern-rich compressible data
  for (size_t i = 0; i < size; ++i) {
    if (i % 16 == 0) {
      data[i] = rng() % 256;
    } else {
      data[i] = data[i - (i % 16)]; // Repeat pattern
    }
  }
}

void benchmark_decode(size_t input_size, int iterations) {
  std::cout << "\n=== Benchmark: " << (input_size / (1024 * 1024))
            << " MB ===" << std::endl;

  // Prepare input data
  std::vector<byte_t> h_input(input_size);
  generate_compressible_data(h_input, input_size);

  // Allocate device memory
  byte_t *d_input, *d_compressed, *d_output;
  CHECK(cudaMalloc(&d_input, input_size));
  CHECK(cudaMalloc(&d_compressed, input_size * 2));
  CHECK(cudaMalloc(&d_output, input_size));
  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  // Create manager
  auto manager = create_manager(3);

  // Allocate workspace for compression
  size_t compress_temp_size = manager->get_compress_temp_size(input_size);
  void *d_temp;
  CHECK(cudaMalloc(&d_temp, compress_temp_size));
  CHECK(cudaMemset(d_temp, 0, compress_temp_size));

  // First compress the data
  size_t compressed_size = input_size * 2;
  Status status =
      manager->compress(d_input, input_size, d_compressed, &compressed_size,
                        d_temp, compress_temp_size, nullptr, 0, 0);
  if (status != Status::SUCCESS) {
    std::cerr << "Compression failed: " << (int)status << std::endl;
    return;
  }
  CHECK(cudaDeviceSynchronize());

  double compression_ratio = (double)input_size / compressed_size;
  std::cout << "Compressed: " << compressed_size
            << " bytes (ratio: " << compression_ratio << "x)" << std::endl;

  // Reallocate workspace for decompression with proper size
  CHECK(cudaFree(d_temp));
  size_t decompress_temp_size =
      manager->get_decompress_temp_size(compressed_size);
  CHECK(cudaMalloc(&d_temp, decompress_temp_size));
  CHECK(cudaMemset(d_temp, 0, decompress_temp_size));

  // Warmup decompression
  size_t decompressed_size = input_size;
  status =
      manager->decompress(d_compressed, compressed_size, d_output,
                          &decompressed_size, d_temp, decompress_temp_size, 0);
  CHECK(cudaDeviceSynchronize());

  if (status != Status::SUCCESS) {
    std::cerr << "Decompression failed: " << (int)status << std::endl;
    return;
  }

  // Verify correctness (first run only)
  std::vector<byte_t> h_output(decompressed_size);
  CHECK(cudaMemcpy(h_output.data(), d_output, decompressed_size,
                   cudaMemcpyDeviceToHost));
  bool correct = (decompressed_size == input_size);
  if (correct) {
    for (size_t i = 0; i < input_size && correct; ++i) {
      if (h_output[i] != h_input[i]) {
        std::cerr << "Mismatch at byte " << i << std::endl;
        correct = false;
      }
    }
  }
  std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

  // Benchmark decompression
  Timer timer;
  for (int i = 0; i < iterations; ++i) {
    decompressed_size = input_size;
    manager->decompress(d_compressed, compressed_size, d_output,
                        &decompressed_size, d_temp, decompress_temp_size, 0);
  }
  CHECK(cudaDeviceSynchronize());

  double total_ms = timer.elapsed_ms();
  double avg_ms = total_ms / iterations;
  double throughput_gbps = (input_size / 1e9) / (avg_ms / 1000.0);

  std::cout << "Decompression:" << std::endl;
  std::cout << "  Average Time: " << avg_ms << " ms" << std::endl;
  std::cout << "  Throughput:   " << throughput_gbps << " GB/s" << std::endl;

  // Cleanup
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_compressed));
  CHECK(cudaFree(d_output));
  CHECK(cudaFree(d_temp));
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "FSE Decoder Throughput Benchmark" << std::endl;
  std::cout << "======================================" << std::endl;

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << std::endl;

  // Benchmark different sizes
  benchmark_decode(1 * 1024 * 1024, 10); // 1 MB
  benchmark_decode(16 * 1024 * 1024, 5); // 16 MB
  benchmark_decode(64 * 1024 * 1024, 3); // 64 MB

  std::cout << "\n======================================" << std::endl;
  std::cout << "Done!" << std::endl;

  return 0;
}
