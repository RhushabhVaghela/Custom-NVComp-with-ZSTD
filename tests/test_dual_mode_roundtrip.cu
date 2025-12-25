#include "../include/cuda_zstd_manager.h"
#include "../include/cuda_zstd_types.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;     \
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

void generate_test_data(std::vector<byte_t> &data, size_t size) {
  std::mt19937 rng(42);
  for (size_t i = 0; i < size; ++i) {
    if (i % 32 < 16) {
      data[i] = rng() % 64;
    } else {
      data[i] = rng() % 256;
    }
  }
}

bool test_dual_mode_roundtrip() {
  std::cout << "\n=== Dual Mode Roundtrip Test ===" << std::endl;

  const size_t size = 128 * 1024; // 128 KB
  std::vector<byte_t> h_input(size);
  generate_test_data(h_input, size);

  byte_t *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, size));
  CHECK_CUDA(cudaMalloc(&d_compressed, size * 2));
  CHECK_CUDA(cudaMalloc(&d_output, size));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

  auto manager = create_manager(3);

  // Get workspace
  size_t compress_temp = manager->get_compress_temp_size(size);
  void *d_temp;
  CHECK_CUDA(cudaMalloc(&d_temp, compress_temp));

  // Test NATIVE mode (default - current implementation)
  std::cout << "\nBitstreamKind::NATIVE mode:" << std::endl;
  {
    CHECK_CUDA(cudaMemset(d_temp, 0, compress_temp));
    size_t compressed_size = size * 2;

    Status status =
        manager->compress(d_input, size, d_compressed, &compressed_size, d_temp,
                          compress_temp, nullptr, 0, 0);
    std::cout << "  Compress: " << (status == Status::SUCCESS ? "OK" : "FAILED")
              << std::endl;
    std::cout << "  Compressed size: " << compressed_size << " bytes"
              << std::endl;

    if (status == Status::SUCCESS) {
      CHECK_CUDA(cudaFree(d_temp));
      size_t decompress_temp =
          manager->get_decompress_temp_size(compressed_size);
      CHECK_CUDA(cudaMalloc(&d_temp, decompress_temp));
      CHECK_CUDA(cudaMemset(d_temp, 0, decompress_temp));

      size_t decompressed_size = size;
      status =
          manager->decompress(d_compressed, compressed_size, d_output,
                              &decompressed_size, d_temp, decompress_temp, 0);
      std::cout << "  Decompress: "
                << (status == Status::SUCCESS ? "OK" : "FAILED") << std::endl;

      if (status == Status::SUCCESS) {
        std::vector<byte_t> h_output(decompressed_size);
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, decompressed_size,
                              cudaMemcpyDeviceToHost));

        bool match = (decompressed_size == size);
        for (size_t i = 0; match && i < size; ++i) {
          match = (h_output[i] == h_input[i]);
        }
        std::cout << "  Verify: " << (match ? "PASSED" : "FAILED") << std::endl;
      }
    }
  }

  std::cout << "\nNote: RAW mode (RFC 8878 compatible) is planned for future "
               "implementation."
            << std::endl;
  std::cout << "      Currently, only NATIVE mode is supported." << std::endl;

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_temp));

  return true;
}

void benchmark_decode(size_t size, int iterations) {
  std::cout << "\n=== Decode Benchmark: " << (size / (1024 * 1024))
            << " MB ===" << std::endl;

  std::vector<byte_t> h_input(size);
  generate_test_data(h_input, size);

  byte_t *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, size));
  CHECK_CUDA(cudaMalloc(&d_compressed, size * 2));
  CHECK_CUDA(cudaMalloc(&d_output, size));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

  auto manager = create_manager(3);

  // Compress first
  size_t compress_temp = manager->get_compress_temp_size(size);
  void *d_temp;
  CHECK_CUDA(cudaMalloc(&d_temp, compress_temp));
  CHECK_CUDA(cudaMemset(d_temp, 0, compress_temp));

  size_t compressed_size = size * 2;
  manager->compress(d_input, size, d_compressed, &compressed_size, d_temp,
                    compress_temp, nullptr, 0, 0);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(d_temp));
  size_t decompress_temp = manager->get_decompress_temp_size(compressed_size);
  CHECK_CUDA(cudaMalloc(&d_temp, decompress_temp));

  // Warmup
  size_t decompressed_size = size;
  manager->decompress(d_compressed, compressed_size, d_output,
                      &decompressed_size, d_temp, decompress_temp, 0);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark
  Timer timer;
  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaMemset(d_temp, 0, decompress_temp));
    decompressed_size = size;
    manager->decompress(d_compressed, compressed_size, d_output,
                        &decompressed_size, d_temp, decompress_temp, 0);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  double total_ms = timer.elapsed_ms();
  double avg_ms = total_ms / iterations;
  double throughput = (size / 1e9) / (avg_ms / 1000.0);

  std::cout << "  Time: " << avg_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_temp));
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "Dual Mode Roundtrip & Benchmark Test" << std::endl;
  std::cout << "======================================" << std::endl;

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << std::endl;

  test_dual_mode_roundtrip();

  std::cout << "\n--- Decode Benchmarks ---" << std::endl;
  benchmark_decode(1 * 1024 * 1024, 10); // 1 MB

  std::cout << "\n======================================" << std::endl;
  std::cout << "Tests Complete" << std::endl;
  std::cout << "======================================" << std::endl;

  return 0;
}
