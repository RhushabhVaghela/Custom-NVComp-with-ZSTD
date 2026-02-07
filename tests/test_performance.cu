// ============================================================================
// test_performance.cu - Comprehensive Performance & Profiling Tests
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace cuda_zstd;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg)                                                    \
  std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_STATUS(status, msg)                                             \
  if ((status) != Status::SUCCESS) {                                           \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }

void print_separator() {
  std::cout << "========================================" << std::endl;
}

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
public:
  void start() { start_time_ = std::chrono::high_resolution_clock::now(); }

  double elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_time_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

class CudaTimer {
public:
  CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) { cudaEventRecord(start_, stream); }

  float elapsed_ms(cudaStream_t stream = 0) {
    cudaEventRecord(stop_, stream);
    cudaEventSynchronize(stop_);
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

private:
  cudaEvent_t start_, stop_;
};

// ============================================================================
// Data Generation
// ============================================================================

void generate_test_data(std::vector<uint8_t> &data, size_t size,
                        const char *pattern) {
  data.resize(size);
  if (strcmp(pattern, "compressible") == 0) {
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<uint8_t>(i % 64);
    }
  } else if (strcmp(pattern, "text") == 0) {
    const char *text =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    size_t len = strlen(text);
    for (size_t i = 0; i < size; i++) {
      data[i] = text[i % len];
    }
  } else {
    // random
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<uint8_t>((i * 1103515245 + 12345) & 0xFF);
    }
  }
}

// ============================================================================
// TEST SUITE 1: Component Timing
// ============================================================================

bool test_compression_timing_breakdown() {
  LOG_TEST("Compression Timing Breakdown");

  const size_t data_size = 1024 * 1024; // 1MB
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  // Allocate GPU memory with error checking
  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_compression_timing_breakdown",
             "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_compression_timing_breakdown",
             "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  // Copy input data to device
  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_compression_timing_breakdown",
             "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_compression_timing_breakdown", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_compression_timing_breakdown",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_compression_timing_breakdown",
             "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Enable profiling
  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  size_t compressed_size;
  CudaTimer timer;
  timer.start();

  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);

  float total_time = timer.elapsed_ms();

  ASSERT_STATUS(status, "Compression failed");

  // Get detailed metrics
  const auto &metrics = PerformanceProfiler::get_metrics();

  LOG_INFO("=== Timing Breakdown ===");
  LOG_INFO("Total time: " << std::fixed << std::setprecision(2) << total_time
                          << " ms");
  LOG_INFO("LZ77 time: " << metrics.lz77_time_ms << " ms");
  LOG_INFO("FSE encode time: " << metrics.fse_encode_time_ms << " ms");
  LOG_INFO("Huffman encode time: " << metrics.huffman_encode_time_ms << " ms");
  LOG_INFO("Sequence generation: " << metrics.sequence_generation_time_ms
                                   << " ms");

  double component_sum = metrics.lz77_time_ms + metrics.fse_encode_time_ms +
                         metrics.huffman_encode_time_ms +
                         metrics.sequence_generation_time_ms;

  LOG_INFO("Component sum: " << std::fixed << std::setprecision(2)
                             << component_sum << " ms");
  LOG_INFO("Overhead: " << (total_time - component_sum) << " ms");

  // Roundtrip verification: decompress and compare with original
  ASSERT_TRUE(compressed_size > 0, "Compressed size should be > 0");
  ASSERT_TRUE(compressed_size < data_size, "Compressible data should compress smaller than original");

  void *d_decompressed = nullptr;
  if (!safe_cuda_malloc(&d_decompressed, data_size)) {
    LOG_FAIL("test_compression_timing_breakdown", "CUDA malloc for d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    PerformanceProfiler::enable_profiling(false);
    return false;
  }

  size_t decompressed_size = data_size;
  status = manager->decompress(d_output, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size);
  ASSERT_STATUS(status, "Decompression failed in roundtrip verification");
  ASSERT_TRUE(decompressed_size == data_size, "Decompressed size should match original");

  std::vector<uint8_t> h_decompressed(data_size);
  if (!safe_cuda_memcpy(h_decompressed.data(), d_decompressed, data_size,
                        cudaMemcpyDeviceToHost)) {
    LOG_FAIL("test_compression_timing_breakdown", "CUDA memcpy from d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    safe_cuda_free(d_decompressed);
    PerformanceProfiler::enable_profiling(false);
    return false;
  }

  ASSERT_TRUE(memcmp(h_data.data(), h_decompressed.data(), data_size) == 0,
              "Decompressed data must match original input");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);
  safe_cuda_free(d_decompressed);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("Compression Timing Breakdown");
  return true;
}

bool test_decompression_timing() {
  LOG_TEST("Decompression Timing");

  const size_t data_size = 1024 * 1024;
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr,
       *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_decompression_timing", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_compressed, data_size * 2)) {
    LOG_FAIL("test_decompression_timing",
             "CUDA malloc for d_compressed failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size)) {
    LOG_FAIL("test_decompression_timing", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_decompression_timing", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_decompression_timing", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_decompression_timing",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_decompression_timing", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  // Compress first
  size_t compressed_size;
  Status status =
      manager->compress(d_input, data_size, d_compressed, &compressed_size,
                        d_temp, temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  LOG_INFO("Compressed: " << data_size << " -> " << compressed_size
                          << " bytes");

  // Time decompression
  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  CudaTimer timer;
  timer.start();

  size_t decompressed_size = data_size; // Must init to output buffer capacity
  status = manager->decompress(d_compressed, compressed_size, d_output,
                               &decompressed_size, d_temp, temp_size);

  float decomp_time = timer.elapsed_ms();

  ASSERT_STATUS(status, "Decompression failed");
  ASSERT_TRUE(decompressed_size == data_size, "Decompressed size mismatch");

  const auto &metrics = PerformanceProfiler::get_metrics();

  LOG_INFO("=== Decompression Timing ===");
  LOG_INFO("Total time: " << std::fixed << std::setprecision(2) << decomp_time
                          << " ms");
  LOG_INFO("Entropy decode time: " << metrics.entropy_decode_time_ms << " ms");
  LOG_INFO("Throughput: " << (data_size / (1024.0 * 1024.0)) /
                                 (decomp_time / 1000.0)
                          << " MB/s");

  // Roundtrip verification: copy decompressed data back and compare
  std::vector<uint8_t> h_decompressed(data_size);
  if (!safe_cuda_memcpy(h_decompressed.data(), d_output, data_size,
                        cudaMemcpyDeviceToHost)) {
    LOG_FAIL("test_decompression_timing", "CUDA memcpy from d_output failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    PerformanceProfiler::enable_profiling(false);
    return false;
  }

  ASSERT_TRUE(memcmp(h_data.data(), h_decompressed.data(), data_size) == 0,
              "Decompressed data must match original input");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_compressed);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("Decompression Timing");
  return true;
}

// ============================================================================
// TEST SUITE 2: Throughput Metrics
// ============================================================================

bool test_compression_throughput() {
  LOG_TEST("Compression Throughput Across Levels");

  const size_t data_size = 4 * 1024 * 1024; // 4MB
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_compression_throughput", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_compression_throughput", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_compression_throughput", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  std::cout
      << "\n  Level | Time (ms) | Throughput (MB/s) | Ratio | Size (KB)\n";
  std::cout << "  ------|-----------|-------------------|-------|----------\n";

  std::vector<int> levels = {1, 3, 5, 9, 15, 19};
  bool any_level_failed = false;

  for (int level : levels) {
    std::unique_ptr<ZstdManager> manager;
    try {
      manager = create_manager(level);
      if (!manager) {
        LOG_FAIL("test_compression_throughput",
                 "Failed to create manager for level " + std::to_string(level));
        any_level_failed = true;
        continue;
      }
    } catch (const std::exception &e) {
      LOG_FAIL("test_compression_throughput",
               "Manager creation failed for level " + std::to_string(level) +
                   ": " + e.what());
      any_level_failed = true;
      continue;
    }

    size_t temp_size = manager->get_compress_temp_size(data_size);
    void *d_temp = nullptr;
    if (!safe_cuda_malloc(&d_temp, temp_size)) {
      LOG_FAIL("test_compression_throughput", "CUDA malloc for d_temp failed");
      any_level_failed = true;
      continue;
    }

    CudaTimer timer;
    timer.start();

    size_t compressed_size;
    Status status =
        manager->compress(d_input, data_size, d_output, &compressed_size,
                          d_temp, temp_size, nullptr, 0, 0);

    float time_ms = timer.elapsed_ms();

    if (status == Status::SUCCESS) {
      double throughput = (data_size / (1024.0 * 1024.0)) / (time_ms / 1000.0);
      float ratio = get_compression_ratio(data_size, compressed_size);

      std::cout << "  " << std::setw(5) << level << " | " << std::setw(9)
                << std::fixed << std::setprecision(2) << time_ms << " | "
                << std::setw(17) << std::fixed << std::setprecision(1)
                << throughput << " | " << std::setw(5) << std::fixed
                << std::setprecision(2) << ratio << " | " << std::setw(8)
                << compressed_size / 1024 << "\n";

      // Roundtrip verification: decompress and verify data integrity
      void *d_decompressed = nullptr;
      if (!safe_cuda_malloc(&d_decompressed, data_size)) {
        LOG_FAIL("test_compression_throughput",
                 "CUDA malloc for d_decompressed failed at level " + std::to_string(level));
        any_level_failed = true;
        safe_cuda_free(d_temp);
        continue;
      }

      size_t decompressed_size = data_size;
      Status dec_status = manager->decompress(d_output, compressed_size, d_decompressed,
                                              &decompressed_size, d_temp, temp_size);
      if (dec_status != Status::SUCCESS) {
        LOG_FAIL("test_compression_throughput",
                 "Decompression failed at level " + std::to_string(level));
        any_level_failed = true;
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_temp);
        continue;
      }

      if (decompressed_size != data_size) {
        LOG_FAIL("test_compression_throughput",
                 "Decompressed size mismatch at level " + std::to_string(level));
        any_level_failed = true;
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_temp);
        continue;
      }

      std::vector<uint8_t> h_decompressed(data_size);
      if (!safe_cuda_memcpy(h_decompressed.data(), d_decompressed, data_size,
                            cudaMemcpyDeviceToHost)) {
        LOG_FAIL("test_compression_throughput",
                 "CUDA memcpy from d_decompressed failed at level " + std::to_string(level));
        any_level_failed = true;
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_temp);
        continue;
      }

      if (memcmp(h_data.data(), h_decompressed.data(), data_size) != 0) {
        LOG_FAIL("test_compression_throughput",
                 "Decompressed data mismatch at level " + std::to_string(level));
        any_level_failed = true;
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_temp);
        continue;
      }

      safe_cuda_free(d_decompressed);
    } else {
      std::cout << "  " << std::setw(5) << level << " | ERROR | - | - | -\n";
      any_level_failed = true;
    }

    safe_cuda_free(d_temp);
  }

  safe_cuda_free(d_input);
  safe_cuda_free(d_output);

  if (any_level_failed) {
    LOG_FAIL("test_compression_throughput", "One or more compression levels failed verification");
    return false;
  }

  LOG_PASS("Compression Throughput");
  return true;
}

bool test_decompression_throughput() {
  LOG_TEST("Decompression Throughput");

  const size_t data_size = 4 * 1024 * 1024;
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr,
       *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_decompression_throughput", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_compressed, data_size * 2)) {
    LOG_FAIL("test_decompression_throughput",
             "CUDA malloc for d_compressed failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size)) {
    LOG_FAIL("test_decompression_throughput",
             "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_decompression_throughput", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_decompression_throughput", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_decompression_throughput",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_decompression_throughput", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    return false;
  }

  // Compress
  size_t compressed_size;
  Status status =
      manager->compress(d_input, data_size, d_compressed, &compressed_size,
                        d_temp, temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  // Decompress multiple times for average
  const int num_iterations = 10;
  double total_time = 0;

  for (int i = 0; i < num_iterations; i++) {
    CudaTimer timer;
    timer.start();

    size_t decompressed_size = data_size; // Must init to output buffer capacity
    status = manager->decompress(d_compressed, compressed_size, d_output,
                                 &decompressed_size, d_temp, temp_size);

    if (status != Status::SUCCESS) {
      LOG_FAIL("test_decompression_throughput",
               "Decompression iteration " + std::to_string(i) + " failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      safe_cuda_free(d_temp);
      return false;
    }

    total_time += timer.elapsed_ms();
  }

  double avg_time = total_time / num_iterations;
  double throughput = (data_size / (1024.0 * 1024.0)) / (avg_time / 1000.0);

  LOG_INFO("Average decompression time: " << std::fixed << std::setprecision(2)
                                           << avg_time << " ms");
  LOG_INFO("Decompression throughput: " << std::fixed << std::setprecision(1)
                                         << throughput << " MB/s");

  // Verify final decompressed output matches original input
  size_t final_decompressed_size = data_size;
  status = manager->decompress(d_compressed, compressed_size, d_output,
                               &final_decompressed_size, d_temp, temp_size);
  ASSERT_STATUS(status, "Final verification decompression failed");
  ASSERT_TRUE(final_decompressed_size == data_size, "Decompressed size mismatch");

  std::vector<uint8_t> h_decompressed(data_size);
  if (!safe_cuda_memcpy(h_decompressed.data(), d_output, data_size,
                        cudaMemcpyDeviceToHost)) {
    LOG_FAIL("test_decompression_throughput", "CUDA memcpy from d_output failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return false;
  }

  ASSERT_TRUE(memcmp(h_data.data(), h_decompressed.data(), data_size) == 0,
              "Decompressed data must match original input");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_compressed);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  LOG_PASS("Decompression Throughput");
  return true;
}

bool test_memory_bandwidth() {
  LOG_TEST("Memory Bandwidth Measurement");

  const size_t data_size = 16 * 1024 * 1024; // 16MB
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_memory_bandwidth", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_memory_bandwidth",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  size_t compressed_size = data_size * 2;  // Initialize to output buffer capacity
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  const auto &metrics = PerformanceProfiler::get_metrics();

  LOG_INFO("=== Memory Bandwidth ===");
  LOG_INFO("Read bandwidth: " << std::fixed << std::setprecision(2)
                              << metrics.read_bandwidth_gbps << " GB/s");
  LOG_INFO("Write bandwidth: " << std::fixed << std::setprecision(2)
                               << metrics.write_bandwidth_gbps << " GB/s");
  LOG_INFO("Total bandwidth: " << std::fixed << std::setprecision(2)
                               << metrics.total_bandwidth_gbps << " GB/s");

  // Roundtrip verification: decompress and compare with original
  ASSERT_TRUE(compressed_size > 0, "Compressed size should be > 0");

  void *d_decompressed = nullptr;
  if (!safe_cuda_malloc(&d_decompressed, data_size)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA malloc for d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    PerformanceProfiler::enable_profiling(false);
    return false;
  }

  size_t decompressed_size = data_size;
  status = manager->decompress(d_output, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size);
  ASSERT_STATUS(status, "Decompression failed in roundtrip verification");
  ASSERT_TRUE(decompressed_size == data_size, "Decompressed size should match original");

  std::vector<uint8_t> h_decompressed(data_size);
  if (!safe_cuda_memcpy(h_decompressed.data(), d_decompressed, data_size,
                        cudaMemcpyDeviceToHost)) {
    LOG_FAIL("test_memory_bandwidth", "CUDA memcpy from d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    safe_cuda_free(d_decompressed);
    PerformanceProfiler::enable_profiling(false);
    return false;
  }

  ASSERT_TRUE(memcmp(h_data.data(), h_decompressed.data(), data_size) == 0,
              "Decompressed data must match original input");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);
  safe_cuda_free(d_decompressed);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("Memory Bandwidth");
  return true;
}

// ============================================================================
// TEST SUITE 3: Profiling API Tests
// ============================================================================

bool test_profiler_enable_disable() {
  LOG_TEST("Profiler Enable/Disable");

  // Initially disabled
  bool initially_enabled = PerformanceProfiler::is_profiling_enabled();
  LOG_INFO("Initially enabled: " << (initially_enabled ? "yes" : "no"));

  // Enable
  PerformanceProfiler::enable_profiling(true);
  ASSERT_TRUE(PerformanceProfiler::is_profiling_enabled(), "Should be enabled");
  LOG_INFO("✓ Profiling enabled");

  // Disable
  PerformanceProfiler::enable_profiling(false);
  ASSERT_TRUE(!PerformanceProfiler::is_profiling_enabled(),
              "Should be disabled");
  LOG_INFO("✓ Profiling disabled");

  LOG_PASS("Profiler Enable/Disable");
  return true;
}

bool test_named_timers() {
  LOG_TEST("Named Timer Accuracy");

  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  // Test named timers
  PerformanceProfiler::start_timer("test_operation");

  // Simulate work
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  PerformanceProfiler::stop_timer("test_operation");

  double elapsed = PerformanceProfiler::get_timer_ms("test_operation");

  LOG_INFO("Named timer 'test_operation': "
           << std::fixed << std::setprecision(2) << elapsed << " ms");

  ASSERT_TRUE(elapsed >= 90.0 && elapsed <= 150.0, "Timer should be ~100ms");
  LOG_INFO("✓ Named timer accurate");

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("Named Timer Accuracy");
  return true;
}

bool test_metrics_reset() {
  LOG_TEST("Metrics Reset");

  const size_t data_size = 1024 * 1024;
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_metrics_reset", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_metrics_reset", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_metrics_reset", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(3);
    if (!manager) {
      LOG_FAIL("test_metrics_reset", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_metrics_reset",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_metrics_reset", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  // Run compression
  size_t compressed_size;
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  auto metrics_before = PerformanceProfiler::get_metrics();
  LOG_INFO("Before reset - total time: " << metrics_before.total_time_ms
                                         << " ms");

  // Reset
  PerformanceProfiler::reset_metrics();

  auto metrics_after = PerformanceProfiler::get_metrics();
  LOG_INFO("After reset - total time: " << metrics_after.total_time_ms
                                        << " ms");

  ASSERT_TRUE(metrics_after.total_time_ms == 0.0, "Metrics should be reset");
  LOG_INFO("✓ Metrics reset successfully");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("Metrics Reset");
  return true;
}

bool test_csv_export() {
  LOG_TEST("CSV Export Functionality");

  const size_t data_size = 512 * 1024;
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_csv_export", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_csv_export", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_csv_export", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_csv_export", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_csv_export",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_csv_export", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  size_t compressed_size = data_size * 2;  // Initialize to output buffer capacity
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  // Export to CSV
  const char *csv_file = "performance_metrics.csv";
  PerformanceProfiler::export_metrics_csv(csv_file);

  // Check file exists and has content
  std::ifstream file(csv_file);
  ASSERT_TRUE(file.good(), "CSV file should be created");

  std::string line;
  int line_count = 0;
  while (std::getline(file, line)) {
    line_count++;
  }
  file.close();

  LOG_INFO("CSV file created with " << line_count << " lines");
  ASSERT_TRUE(line_count > 0, "CSV should have content");
  LOG_INFO("✓ CSV export successful");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("CSV Export");
  return true;
}

bool test_json_export() {
  LOG_TEST("JSON Export Functionality");

  const size_t data_size = 512 * 1024;
  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_json_export", "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_json_export", "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_json_export", "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_json_export", "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_json_export",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_json_export", "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  PerformanceProfiler::enable_profiling(true);
  PerformanceProfiler::reset_metrics();

  size_t compressed_size;
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Compression failed");

  // Export to JSON
  const char *json_file = "performance_metrics.json";
  PerformanceProfiler::export_metrics_json(json_file);

  // Check file exists
  std::ifstream file(json_file);
  ASSERT_TRUE(file.good(), "JSON file should be created");

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  file.close();

  ASSERT_TRUE(content.length() > 0, "JSON should have content");
  LOG_INFO("JSON file created (" << content.length() << " bytes)");
  LOG_INFO("✓ JSON export successful");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);

  PerformanceProfiler::enable_profiling(false);

  LOG_PASS("JSON Export");
  return true;
}

// ============================================================================
// TEST SUITE 4: Optimization Validation
// ============================================================================

bool test_memory_pool_performance_impact() {
  LOG_TEST("Memory Pool Performance Impact");

  const size_t data_size = 2 * 1024 * 1024;
  const int num_iterations = 20;

  std::vector<uint8_t> h_data;
  generate_test_data(h_data, data_size, "compressible");

  void *d_input = nullptr, *d_output = nullptr, *d_temp = nullptr;

  if (!safe_cuda_malloc(&d_input, data_size)) {
    LOG_FAIL("test_memory_pool_performance_impact",
             "CUDA malloc for d_input failed");
    return false;
  }

  if (!safe_cuda_malloc(&d_output, data_size * 2)) {
    LOG_FAIL("test_memory_pool_performance_impact",
             "CUDA malloc for d_output failed");
    safe_cuda_free(d_input);
    return false;
  }

  if (!safe_cuda_memcpy(d_input, h_data.data(), data_size,
                        cudaMemcpyHostToDevice)) {
    LOG_FAIL("test_memory_pool_performance_impact",
             "CUDA memcpy to d_input failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Create manager with safe error handling
  std::unique_ptr<ZstdManager> manager;
  try {
    manager = create_manager(5);
    if (!manager) {
      LOG_FAIL("test_memory_pool_performance_impact",
               "Failed to create manager");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      return false;
    }
  } catch (const std::exception &e) {
    LOG_FAIL("test_memory_pool_performance_impact",
             std::string("Manager creation failed: ") + e.what());
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  size_t temp_size = manager->get_compress_temp_size(data_size);
  if (!safe_cuda_malloc(&d_temp, temp_size)) {
    LOG_FAIL("test_memory_pool_performance_impact",
             "CUDA malloc for d_temp failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    return false;
  }

  // Warm up
  size_t compressed_size;
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size, d_temp,
                        temp_size, nullptr, 0, 0);
  ASSERT_STATUS(status, "Warm-up compression failed");

  // Measure with pool (multiple compressions)
  Timer timer;
  timer.start();

  for (int i = 0; i < num_iterations; i++) {
    status = manager->compress(d_input, data_size, d_output, &compressed_size,
                               d_temp, temp_size, nullptr, 0, 0);
    if (status != Status::SUCCESS) {
      LOG_FAIL("test_memory_pool_performance_impact",
               "Compression iteration " + std::to_string(i) + " failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_output);
      safe_cuda_free(d_temp);
      return false;
    }
  }
  cudaDeviceSynchronize();

  double total_time = timer.elapsed_ms();
  double avg_time = total_time / num_iterations;

  LOG_INFO("Average compression time: " << std::fixed << std::setprecision(2)
                                         << avg_time << " ms");
  LOG_INFO("Total time for " << num_iterations << " iterations: " << std::fixed
                              << std::setprecision(2) << total_time << " ms");

  // Roundtrip verification: decompress the last compressed output and verify
  ASSERT_TRUE(compressed_size > 0, "Compressed size should be > 0");

  void *d_decompressed = nullptr;
  if (!safe_cuda_malloc(&d_decompressed, data_size)) {
    LOG_FAIL("test_memory_pool_performance_impact", "CUDA malloc for d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    return false;
  }

  size_t decompressed_size = data_size;
  status = manager->decompress(d_output, compressed_size, d_decompressed,
                               &decompressed_size, d_temp, temp_size);
  ASSERT_STATUS(status, "Decompression failed in roundtrip verification");
  ASSERT_TRUE(decompressed_size == data_size, "Decompressed size should match original");

  std::vector<uint8_t> h_decompressed(data_size);
  if (!safe_cuda_memcpy(h_decompressed.data(), d_decompressed, data_size,
                        cudaMemcpyDeviceToHost)) {
    LOG_FAIL("test_memory_pool_performance_impact", "CUDA memcpy from d_decompressed failed");
    safe_cuda_free(d_input);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);
    safe_cuda_free(d_decompressed);
    return false;
  }

  ASSERT_TRUE(memcmp(h_data.data(), h_decompressed.data(), data_size) == 0,
              "Decompressed data must match original input");

  // Cleanup with safe free functions
  safe_cuda_free(d_input);
  safe_cuda_free(d_output);
  safe_cuda_free(d_temp);
  safe_cuda_free(d_decompressed);

  LOG_PASS("Memory Pool Performance Impact");
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << "\n";
  print_separator();
  std::cout << "CUDA ZSTD - Performance & Profiling Test Suite" << std::endl;
  print_separator();
  std::cout << "\n";

  int passed = 0;
  int total = 0;

  // Skip on CPU-only environments; otherwise print device info
  SKIP_IF_NO_CUDA_RET(0);
  check_cuda_device();

  // Component Timing
  print_separator();
  std::cout << "SUITE 1: Component Timing" << std::endl;
  print_separator();

  total++;
  if (test_compression_timing_breakdown())
    passed++;
  total++;
  if (test_decompression_timing())
    passed++;

  // Throughput Metrics
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 2: Throughput Metrics" << std::endl;
  print_separator();

  total++;
  if (test_compression_throughput())
    passed++;
  total++;
  if (test_decompression_throughput())
    passed++;
  total++;
  if (test_memory_bandwidth())
    passed++;

  // Profiling API
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 3: Profiling API Tests" << std::endl;
  print_separator();

  total++;
  if (test_profiler_enable_disable())
    passed++;
  total++;
  if (test_named_timers())
    passed++;
  total++;
  if (test_metrics_reset())
    passed++;
  total++;
  if (test_csv_export())
    passed++;
  total++;
  if (test_json_export())
    passed++;

  // Optimization Validation
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 4: Optimization Validation" << std::endl;
  print_separator();

  total++;
  if (test_memory_pool_performance_impact())
    passed++;

  // Summary
  std::cout << "\n";
  print_separator();
  std::cout << "TEST RESULTS" << std::endl;
  print_separator();
  std::cout << "Passed: " << passed << "/" << total << std::endl;
  std::cout << "Failed: " << (total - passed) << "/" << total << std::endl;

  if (passed == total) {
    std::cout << "\n✓ ALL TESTS PASSED" << std::endl;
  } else {
    std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
  }
  print_separator();
  std::cout << "\n";

  return (passed == total) ? 0 : 1;
}
