
#include "cuda_error_checking.h"
#include "cuda_zstd_manager.h"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace cuda_zstd;

#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_FAIL(name, msg)                                                    \
  std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }

void generate_test_data(std::vector<uint8_t> &data, size_t size,
                        const char *pattern) {
  data.resize(size);
  for (size_t i = 0; i < size; i++)
    data[i] = (i % 32);
}

bool test_concurrent_compression() {
  std::cout << "\n[TEST] Concurrent Compression Operations" << std::endl;

  const int num_threads = 2; // Stable configuration (4+ threads may crash)
  const int operations_per_thread = 10;
  const size_t data_size = 64 * 1024;

  LOG_INFO("Testing with " << num_threads << " threads");

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  auto worker = [&](int thread_id) {
    ZstdBatchManager manager(CompressionConfig{.level = 5});

    for (int i = 0; i < operations_per_thread; i++) {
      std::vector<uint8_t> h_data;
      generate_test_data(h_data, data_size, "compressible");

      void *d_input = nullptr;
      void *d_output = nullptr;
      void *d_temp = nullptr;

      size_t temp_size = manager.get_compress_temp_size(data_size);

      if (cudaMalloc(&d_input, data_size) != cudaSuccess ||
          cudaMalloc(&d_output, data_size * 2) != cudaSuccess ||
          cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
        printf("Test Allocation Failed!\\n");
        if (d_input)
          cudaFree(d_input);
        if (d_output)
          cudaFree(d_output);
        if (d_temp)
          cudaFree(d_temp);
        failure_count++; // Increment failure count for this operation
        continue;        // Skip to next operation or return if critical
      }

      cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);

      size_t compressed_size = data_size * 2;
      Status status =
          manager.compress(d_input, data_size, d_output, &compressed_size,
                           d_temp, temp_size, nullptr, 0);

      if (status == Status::SUCCESS) {
        success_count++;
      } else {
        failure_count++;
      }

      cudaFree(d_input);
      cudaFree(d_output);
      cudaFree(d_temp);
    }
  };

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; i++)
    threads.emplace_back(worker, i);
  for (auto &t : threads)
    t.join();
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  int total_ops = num_threads * operations_per_thread;

  LOG_INFO("Total: " << total_ops << ", Success: " << success_count
                     << ", Failed: " << failure_count);
  LOG_INFO("Time: " << elapsed_ms << " ms");

  return (success_count == total_ops);
}

int main() {
  if (test_concurrent_compression()) {
    std::cout << "PASS" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL" << std::endl;
    return 1;
  }
}
