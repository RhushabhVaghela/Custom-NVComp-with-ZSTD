#include "benchmark_results.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_utils.h"
#include "cuda_zstd_safe_alloc.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>


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

// Helper to generate FSE-compressible data
void generate_fse_data(std::vector<uint8_t> &data, size_t size) {
  std::mt19937 rng(42);
  // Use a skewed distribution to make it compressible
  std::discrete_distribution<> dist({100, 50, 25, 12, 6, 3, 1, 1});
  for (size_t i = 0; i < size; ++i) {
    data[i] = (uint8_t)dist(rng);
  }
}

#undef CUDA_CHECK // Override library version with benchmark-specific version
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void run_benchmark(size_t size, int iterations, const char *gpu_name) {
  std::cout << "Benchmarking Size: " << size << " bytes" << std::endl;

  // 1. Prepare Data
  std::vector<uint8_t> h_input(size);
  generate_fse_data(h_input, size);

  void *d_input, *d_compressed, *d_output, *d_temp;
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_input, size));
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_compressed, size * 2));
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_output, size));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

  auto manager = cuda_zstd::create_manager();
  size_t temp_size = manager->get_compress_temp_size(size);
  std::cout << "Temp size: " << temp_size << std::endl;
  CUDA_CHECK(cuda_zstd::safe_cuda_malloc(&d_temp, temp_size));

  // Clear any previous errors
  cudaGetLastError();
  cudaDeviceSynchronize();

  // 2. Compress (to prepare valid FSE input)
  // We use the manager to compress, which uses FSE internally for
  // literals/sequences But wait, the manager does full Zstd compression. To
  // benchmark *just* FSE decode, we should ideally call decode_fse directly.
  // But decode_fse expects a specific format (header + bitstream).
  // The easiest way is to compress a block and then decompress it.
  // The manager's decompress function calls decode_fse.

  size_t compressed_size = size * 2;
  cuda_zstd::Status status =
      manager->compress(d_input, size, d_compressed, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  if (status != cuda_zstd::Status::SUCCESS) {
    std::cerr << "Compression failed: " << (int)status << std::endl;
    return;
  }
  cudaDeviceSynchronize();

// 3. Benchmark CPU Path (Threshold = MAX)
// We set the threshold very high to force CPU execution
#ifdef _WIN32
  _putenv("CUDA_ZSTD_FSE_THRESHOLD=2147483647");
#else
  setenv("CUDA_ZSTD_FSE_THRESHOLD", "2147483647", 1);
#endif

  // Warmup
  size_t decompressed_size = size;
  manager->decompress(d_compressed, compressed_size, d_output,
                      &decompressed_size, d_temp, temp_size);

  Timer timer;
  for (int i = 0; i < iterations; ++i) {
    decompressed_size = size;
    manager->decompress(d_compressed, compressed_size, d_output,
                        &decompressed_size, d_temp, temp_size);
  }
  cudaDeviceSynchronize();
  double cpu_ms = timer.elapsed_ms() / iterations;
  double cpu_gbps = (size / 1e9) / (cpu_ms / 1000.0);

// 4. Benchmark GPU Path (Threshold = 0)
// We set the threshold to 0 to force GPU execution
#ifdef _WIN32
  _putenv("CUDA_ZSTD_FSE_THRESHOLD=0");
#else
  setenv("CUDA_ZSTD_FSE_THRESHOLD", "0", 1);
#endif

  // Warmup
  decompressed_size = size;
  manager->decompress(d_compressed, compressed_size, d_output,
                      &decompressed_size, d_temp, temp_size);

  timer.reset();
  for (int i = 0; i < iterations; ++i) {
    decompressed_size = size;
    manager->decompress(d_compressed, compressed_size, d_output,
                        &decompressed_size, d_temp, temp_size);
  }
  cudaDeviceSynchronize();
  double gpu_ms = timer.elapsed_ms() / iterations;
  double gpu_gbps = (size / 1e9) / (gpu_ms / 1000.0);

  std::cout << "  CPU Path: " << cpu_ms << " ms (" << cpu_gbps << " GB/s)"
            << std::endl;
  std::cout << "  GPU Path: " << gpu_ms << " ms (" << gpu_gbps << " GB/s)"
            << std::endl;

  log_benchmark_result("FSE_Benchmark_GPU", gpu_name, size, 1, size, gpu_ms,
                       gpu_gbps);
  std::cout << "  Speedup: " << cpu_ms / gpu_ms << "x" << std::endl;

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_output);
  cudaFree(d_temp);
}

int main() {
  try {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const char *gpu_name = prop.name;

    // Test various sizes
    std::vector<size_t> sizes = {
        4 * 1024,        // 4KB
        16 * 1024,       // 16KB
        64 * 1024,       // 64KB
        256 * 1024,      // 256KB
        1024 * 1024,     // 1MB
        4 * 1024 * 1024, // 4MB
        16 * 1024 * 1024 // 16MB
    };

    for (size_t size : sizes) {
      run_benchmark(size, 10, gpu_name);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
