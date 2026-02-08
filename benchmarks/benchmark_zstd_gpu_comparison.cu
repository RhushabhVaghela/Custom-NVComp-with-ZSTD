// ============================================================================
// benchmark_zstd_gpu_comparison.cu
//
// Comprehensive benchmark comparing standard zstd library (CPU) vs GPU
// implementation (CUDA).
//
// Features:
// - Compression time comparison (CPU vs GPU)
// - Decompression time comparison (CPU vs GPU)
// - Compression ratio comparison
// - Throughput comparison (MB/s)
// - Resource utilization tracking (GPU VRAM, CPU memory)
// - Cross-compatibility verification (compress with one, decompress with other)
// - Multiple compression levels (1, 3, 6, 12, 19, 22)
// - Multiple file sizes (1MB, 10MB, 100MB, 256MB)
// - CSV export for further analysis
// - Speedup calculations (GPU vs CPU)
//
// Usage:
//   ./benchmark_zstd_gpu_comparison <input_file>
//   ./benchmark_zstd_gpu_comparison <input_file> --csv-output results.csv
//   ./benchmark_zstd_gpu_comparison --generate-data
//
// ============================================================================

#include "cuda_zstd_hash.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_utils.h"
#include "cuda_zstd_xxhash.h"
#include "cuda_zstd_safe_alloc.h"

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>


// Standard zstd library include
#include <zstd.h>

#ifdef __cplusplus
using namespace cuda_zstd;
#endif

// ============================================================================
// MACROS AND CONSTANTS
// ============================================================================

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_ZSTD(call)                                                       \
  do {                                                                         \
    size_t err = call;                                                         \
    if (ZSTD_isError(err)) {                                                   \
      std::cerr << "ZSTD Error: " << ZSTD_getErrorName(err) << " at "          \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Hardware-safe constants
#define MAX_OUTPUT_MULTIPLIER 1.5f
#define NUM_ITERATIONS 3 // Number of iterations for averaging
#define MAX_VRAM_PER_BENCHMARK (10ULL * 1024 * 1024 * 1024)
#define MAX_SAFE_DATA_SIZE (10ULL * 1024 * 1024 * 1024)

// Compression levels to test
const int COMPRESSION_LEVELS[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
const int NUM_COMPRESSION_LEVELS = 22;

// File sizes to test (in bytes)
const size_t FILE_SIZES[] = {
    1ULL * 1024 * 1024,   // 1 MB
    2ULL * 1024 * 1024,   // 2 MB
    4ULL * 1024 * 1024,   // 4 MB
    8ULL * 1024 * 1024,   // 8 MB
    16ULL * 1024 * 1024,  // 16 MB
    32ULL * 1024 * 1024,  // 32 MB
    64ULL * 1024 * 1024,  // 64 MB
    128ULL * 1024 * 1024, // 128 MB
    256ULL * 1024 * 1024, // 256 MB
    512ULL * 1024 * 1024, // 512 MB
    1ULL * 1024 * 1024 * 1024,  // 1 GB
    2ULL * 1024 * 1024 * 1024,  // 2 GB
    4ULL * 1024 * 1024 * 1024,  // 4 GB
    8ULL * 1024 * 1024 * 1024,  // 8 GB
    10ULL * 1024 * 1024 * 1024  // 10 GB
};
const int NUM_FILE_SIZES = 15;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Comprehensive benchmark result structure
struct ComparisonResult {
  // Test configuration
  size_t data_size;
  int compression_level;
  const char *data_pattern;

  // CPU (Standard zstd) Results
  double cpu_compress_time_ms;
  double cpu_decompress_time_ms;
  double cpu_compress_throughput_mbps;
  double cpu_decompress_throughput_mbps;
  double cpu_compression_ratio;
  size_t cpu_compressed_size;
  bool cpu_compress_success;
  bool cpu_decompress_success;

  // GPU (CUDA zstd) Results
  double gpu_compress_time_ms;
  double gpu_decompress_time_ms;
  double gpu_compress_throughput_mbps;
  double gpu_decompress_throughput_mbps;
  double gpu_compression_ratio;
  size_t gpu_compressed_size;
  bool gpu_compress_success;
  bool gpu_decompress_success;

  // Resource utilization
  size_t gpu_peak_vram_bytes;
  size_t cpu_peak_memory_bytes;
  double gpu_memory_alloc_time_ms;

  // Cross-compatibility tests
  bool cpu_compress_gpu_decompress_success;
  bool gpu_compress_cpu_decompress_success;
  double cpu_compress_gpu_decompress_time_ms;
  double gpu_compress_cpu_decompress_time_ms;

  // Comparison metrics
  double compress_speedup;       // GPU compression speedup vs CPU
  double decompress_speedup;     // GPU decompression speedup vs CPU
  double compression_ratio_diff; // Difference in compression ratio (GPU - CPU)
  double throughput_diff_mbps;   // Throughput difference (GPU - CPU)
};

struct GpuMemorySnapshot {
  size_t free_bytes;
  size_t total_bytes;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Get GPU memory snapshot
GpuMemorySnapshot get_gpu_memory_snapshot() {
  GpuMemorySnapshot snapshot;
  CHECK_CUDA(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
  return snapshot;
}

// Format bytes to human readable string
std::string format_bytes(size_t bytes) {
  if (bytes >= 1024ULL * 1024 * 1024) {
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
  } else if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + " KB";
  } else {
    return std::to_string(bytes) + " B";
  }
}

// Compute XXH64 checksum for data integrity verification
uint64_t compute_checksum(const void *data, size_t size) {
  return cuda_zstd::xxhash::xxhash_64_cpu((const unsigned char*)data, size, 0);
}

size_t compute_max_data_size() {
  // Use safety-buffer-aware VRAM query to prevent system instability
  size_t usable_vram = cuda_zstd::get_usable_vram();
  if (usable_vram == 0) {
    return MAX_SAFE_DATA_SIZE;
  }

  size_t vram_budget = std::min(usable_vram, (size_t)MAX_VRAM_PER_BENCHMARK);
  size_t per_test_budget = (size_t)(vram_budget * 0.6);
  size_t max_by_budget = per_test_budget / 3;

  return std::min(max_by_budget, (size_t)MAX_SAFE_DATA_SIZE);
}

// Load file into memory
bool load_file(const std::string &filename, std::vector<byte_t> &data,
               size_t max_size = 0) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t read_size = file_size;
  if (max_size > 0 && max_size < file_size) {
    read_size = max_size;
  }

  data.resize(read_size);
  if (!file.read(reinterpret_cast<char *>(data.data()), read_size)) {
    std::cerr << "Failed to read file: " << filename << std::endl;
    return false;
  }

  return true;
}

// Generate test data with specified pattern
void generate_test_data(byte_t *data, size_t size, const char *pattern_name) {
  std::string pattern(pattern_name);

  if (pattern == "random") {
    // Fully random data (hard to compress)
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<byte_t>(dist(rng));
    }
  } else if (pattern == "repetitive") {
    // All 'A's - highly compressible
    memset(data, 'A', size);
  } else if (pattern == "semi-random") {
    // Repeating 64-byte patterns with some randomness
    std::mt19937_64 rng(67890);
    std::uniform_int_distribution<int> dist(0, 255);
    const size_t block_size = 64;
    byte_t block[block_size];
    for (size_t i = 0; i < block_size; ++i) {
      block[i] = static_cast<byte_t>(dist(rng));
    }
    for (size_t i = 0; i < size; ++i) {
      data[i] = block[i % block_size];
    }
  } else if (pattern == "text-like") {
    // Text-like data with common patterns
    const char *patterns[] = {"the quick brown fox jumps over the lazy dog ",
                              "hello world hello world hello world ",
                              "aaaaaaaaabbbbbbbbbbbcccccccccdddddddddd ",
                              "0123456789012345678901234567890123456789 "};
    const int num_patterns = 4;
    size_t pattern_lens[num_patterns];
    for (int i = 0; i < num_patterns; i++) {
      pattern_lens[i] = strlen(patterns[i]);
    }

    size_t pos = 0;
    std::mt19937_64 rng(54321);
    std::uniform_int_distribution<int> dist(0, num_patterns - 1);

    while (pos < size) {
      int p = dist(rng);
      size_t len = pattern_lens[p];
      if (pos + len > size)
        len = size - pos;
      memcpy(data + pos, patterns[p], len);
      pos += len;
    }
  } else {
    // Default: random
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<byte_t>(dist(rng));
    }
  }
}

// ============================================================================
// CPU ZSTD BENCHMARKS
// ============================================================================

// CPU Compression benchmark
void benchmark_cpu_compress(const byte_t *input_data, size_t input_size,
                            int compression_level, ComparisonResult &result) {
  // Estimate maximum output size
  size_t max_output_size = ZSTD_compressBound(input_size);

  // Allocate output buffer
  std::vector<byte_t> compressed_data(max_output_size);

  // Warmup run
  ZSTD_compress(compressed_data.data(), max_output_size, input_data, input_size,
                compression_level);

  // Benchmark with multiple iterations
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t compressed_size =
        ZSTD_compress(compressed_data.data(), max_output_size, input_data,
                      input_size, compression_level);

    auto end = std::chrono::high_resolution_clock::now();

    if (!ZSTD_isError(compressed_size)) {
      double iter_time =
          std::chrono::duration<double, std::milli>(end - start).count();
      iteration_times.push_back(iter_time);
      result.cpu_compressed_size = compressed_size;
    }
  }

  if (!iteration_times.empty()) {
    // Calculate average
    double total_time = 0.0;
    for (double t : iteration_times)
      total_time += t;
    result.cpu_compress_time_ms = total_time / iteration_times.size();
    result.cpu_compress_throughput_mbps =
        (input_size / (1024.0 * 1024.0)) /
        (result.cpu_compress_time_ms / 1000.0);
    result.cpu_compression_ratio =
        static_cast<double>(result.cpu_compressed_size) /
        static_cast<double>(input_size);
    result.cpu_compress_success = true;
  } else {
    result.cpu_compress_time_ms = -1;
    result.cpu_compress_throughput_mbps = 0;
    result.cpu_compression_ratio = -1;
    result.cpu_compress_success = false;
  }
}

// CPU Decompression benchmark
void benchmark_cpu_decompress(const byte_t *compressed_data,
                              size_t compressed_size, size_t original_size,
                              ComparisonResult &result) {
  // Allocate output buffer
  std::vector<byte_t> decompressed_data(original_size);

  // Warmup run
  ZSTD_decompress(decompressed_data.data(), original_size, compressed_data,
                  compressed_size);

  // Benchmark with multiple iterations
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t decompressed_size =
        ZSTD_decompress(decompressed_data.data(), original_size,
                        compressed_data, compressed_size);

    auto end = std::chrono::high_resolution_clock::now();

    if (!ZSTD_isError(decompressed_size)) {
      double iter_time =
          std::chrono::duration<double, std::milli>(end - start).count();
      iteration_times.push_back(iter_time);
    }
  }

  if (!iteration_times.empty()) {
    double total_time = 0.0;
    for (double t : iteration_times)
      total_time += t;
    result.cpu_decompress_time_ms = total_time / iteration_times.size();
    result.cpu_decompress_throughput_mbps =
        (original_size / (1024.0 * 1024.0)) /
        (result.cpu_decompress_time_ms / 1000.0);
    result.cpu_decompress_success = true;
  } else {
    result.cpu_decompress_time_ms = -1;
    result.cpu_decompress_throughput_mbps = 0;
    result.cpu_decompress_success = false;
  }
}

// ============================================================================
// GPU ZSTD BENCHMARKS
// ============================================================================

// GPU Compression benchmark
void benchmark_gpu_compress(const byte_t *h_input, size_t input_size,
                            int compression_level, ComparisonResult &result) {
  GpuMemorySnapshot mem_before = get_gpu_memory_snapshot();

  // Allocate device memory
  void *d_input = nullptr;
  void *d_output = nullptr;
  size_t max_output_size =
      static_cast<size_t>(input_size * MAX_OUTPUT_MULTIPLIER);

  auto alloc_start = std::chrono::high_resolution_clock::now();
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, max_output_size));
  auto alloc_end = std::chrono::high_resolution_clock::now();
  result.gpu_memory_alloc_time_ms =
      std::chrono::duration<double, std::milli>(alloc_end - alloc_start)
          .count();

  // Copy input to device
  CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

  // Create manager and configure
  auto manager = create_batch_manager(compression_level);
  CompressionConfig config;
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(input_size, static_cast<size_t>(UINT32_MAX))),
      compression_level);
  manager->configure(config);

  // Get workspace size and allocate
  size_t workspace_size = manager->get_compress_temp_size(input_size);
  void *workspace = nullptr;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&workspace, workspace_size));

  // Track peak VRAM
  GpuMemorySnapshot mem_after_alloc = get_gpu_memory_snapshot();
  result.gpu_peak_vram_bytes =
      mem_before.free_bytes - mem_after_alloc.free_bytes;

  // Warmup
  size_t compressed_size = max_output_size;
  manager->compress(d_input, input_size, d_output, &compressed_size, workspace,
                    workspace_size, nullptr, 0, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark with CUDA events for accurate GPU timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    CHECK_CUDA(cudaEventRecord(start));

    size_t iter_compressed_size = max_output_size;
    Status status =
        manager->compress(d_input, input_size, d_output, &iter_compressed_size,
                          workspace, workspace_size, nullptr, 0, nullptr);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float iter_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));

    if (status == Status::SUCCESS) {
      iteration_times.push_back(iter_time_ms);
      compressed_size = iter_compressed_size;
    }
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  if (!iteration_times.empty()) {
    float total_time = 0.0f;
    for (float t : iteration_times)
      total_time += t;
    result.gpu_compress_time_ms = total_time / iteration_times.size();
    result.gpu_compress_throughput_mbps =
        (input_size / (1024.0 * 1024.0)) /
        (result.gpu_compress_time_ms / 1000.0);
    result.gpu_compressed_size = compressed_size;
    result.gpu_compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(input_size);
    result.gpu_compress_success = true;
  } else {
    result.gpu_compress_time_ms = -1;
    result.gpu_compress_throughput_mbps = 0;
    result.gpu_compress_success = false;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_input));
}

// GPU Decompression benchmark
void benchmark_gpu_decompress(const byte_t *h_compressed,
                              size_t compressed_size, size_t original_size,
                              int compression_level, ComparisonResult &result) {
  // Allocate device memory
  void *d_compressed = nullptr;
  void *d_output = nullptr;

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_compressed, compressed_size));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, original_size));

  // Copy compressed data to device
  CHECK_CUDA(cudaMemcpy(d_compressed, h_compressed, compressed_size,
                        cudaMemcpyHostToDevice));

  // Create decompression manager
  auto manager = create_batch_manager(compression_level);
  size_t workspace_size = manager->get_decompress_temp_size(compressed_size);
  void *workspace = nullptr;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&workspace, workspace_size));

  // Warmup
  size_t decompressed_size = original_size;
  manager->decompress(d_compressed, compressed_size, d_output,
                      &decompressed_size, workspace, workspace_size, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark with CUDA events
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    CHECK_CUDA(cudaEventRecord(start));

    size_t iter_decompressed_size = original_size;
    Status status = manager->decompress(d_compressed, compressed_size, d_output,
                                        &iter_decompressed_size, workspace,
                                        workspace_size, nullptr);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float iter_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&iter_time_ms, start, stop));

    if (status == Status::SUCCESS) {
      iteration_times.push_back(iter_time_ms);
    }
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  if (!iteration_times.empty()) {
    float total_time = 0.0f;
    for (float t : iteration_times)
      total_time += t;
    result.gpu_decompress_time_ms = total_time / iteration_times.size();
    result.gpu_decompress_throughput_mbps =
        (original_size / (1024.0 * 1024.0)) /
        (result.gpu_decompress_time_ms / 1000.0);
    result.gpu_decompress_success = true;
  } else {
    result.gpu_decompress_time_ms = -1;
    result.gpu_decompress_throughput_mbps = 0;
    result.gpu_decompress_success = false;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_compressed));
}

// ============================================================================
// CROSS-COMPATIBILITY TESTS
// ============================================================================

// Test: Compress with CPU, Decompress with GPU
bool test_cpu_compress_gpu_decompress(const byte_t *input_data,
                                      size_t input_size, int compression_level,
                                      double &time_ms) {
  // CPU Compression
  size_t max_output_size = ZSTD_compressBound(input_size);
  std::vector<byte_t> compressed_data(max_output_size);

  size_t compressed_size =
      ZSTD_compress(compressed_data.data(), max_output_size, input_data,
                    input_size, compression_level);

  if (ZSTD_isError(compressed_size)) {
    return false;
  }

  // GPU Decompression
  void *d_compressed = nullptr;
  void *d_output = nullptr;

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_compressed, compressed_size));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, input_size));
  CHECK_CUDA(cudaMemcpy(d_compressed, compressed_data.data(), compressed_size,
                        cudaMemcpyHostToDevice));

  auto manager = create_batch_manager(compression_level);
  size_t workspace_size = manager->get_decompress_temp_size(compressed_size);
  void *workspace = nullptr;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&workspace, workspace_size));

  auto start = std::chrono::high_resolution_clock::now();

  size_t decompressed_size = input_size;
  Status status = manager->decompress(d_compressed, compressed_size, d_output,
                                      &decompressed_size, workspace,
                                      workspace_size, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  // Verify integrity
  bool success = (status == Status::SUCCESS && decompressed_size == input_size);

  if (success) {
    // Copy back and verify
    std::vector<byte_t> decompressed_data(input_size);
    CHECK_CUDA(cudaMemcpy(decompressed_data.data(), d_output, input_size,
                          cudaMemcpyDeviceToHost));

    uint64_t original_checksum = compute_checksum(input_data, input_size);
    uint64_t decompressed_checksum =
        compute_checksum(decompressed_data.data(), input_size);
    success = (original_checksum == decompressed_checksum);
  }

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_compressed));

  return success;
}

// Test: Compress with GPU, Decompress with CPU
bool test_gpu_compress_cpu_decompress(const byte_t *input_data,
                                      size_t input_size, int compression_level,
                                      double &time_ms) {
  // GPU Compression
  void *d_input = nullptr;
  void *d_output = nullptr;
  size_t max_output_size =
      static_cast<size_t>(input_size * MAX_OUTPUT_MULTIPLIER);

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, max_output_size));
  CHECK_CUDA(
      cudaMemcpy(d_input, input_data, input_size, cudaMemcpyHostToDevice));

  auto manager = create_batch_manager(compression_level);
  CompressionConfig config;
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(input_size, static_cast<size_t>(UINT32_MAX))),
      compression_level);
  manager->configure(config);

  size_t workspace_size = manager->get_compress_temp_size(input_size);
  void *workspace = nullptr;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&workspace, workspace_size));

  size_t compressed_size = max_output_size;
  Status comp_status =
      manager->compress(d_input, input_size, d_output, &compressed_size,
                        workspace, workspace_size, nullptr, 0, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  if (comp_status != Status::SUCCESS) {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_input));
    return false;
  }

  // Copy compressed data back to host
  std::vector<byte_t> compressed_data(compressed_size);
  CHECK_CUDA(cudaMemcpy(compressed_data.data(), d_output, compressed_size,
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_input));

  // CPU Decompression
  std::vector<byte_t> decompressed_data(input_size);

  auto start = std::chrono::high_resolution_clock::now();

  size_t decompressed_size =
      ZSTD_decompress(decompressed_data.data(), input_size,
                      compressed_data.data(), compressed_size);

  auto end = std::chrono::high_resolution_clock::now();
  time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  if (ZSTD_isError(decompressed_size)) {
    return false;
  }

  // Verify integrity
  uint64_t original_checksum = compute_checksum(input_data, input_size);
  uint64_t decompressed_checksum =
      compute_checksum(decompressed_data.data(), input_size);

  return (original_checksum == decompressed_checksum);
}

// ============================================================================
// RESULT REPORTING
// ============================================================================

void compute_comparison_metrics(ComparisonResult &result) {
  if (result.cpu_compress_success && result.gpu_compress_success &&
      result.cpu_compress_time_ms > 0 && result.gpu_compress_time_ms > 0) {
    result.compress_speedup =
        result.cpu_compress_time_ms / result.gpu_compress_time_ms;
  } else {
    result.compress_speedup = 0;
  }

  if (result.cpu_decompress_success && result.gpu_decompress_success &&
      result.cpu_decompress_time_ms > 0 && result.gpu_decompress_time_ms > 0) {
    result.decompress_speedup =
        result.cpu_decompress_time_ms / result.gpu_decompress_time_ms;
  } else {
    result.decompress_speedup = 0;
  }

  if (result.cpu_compress_success && result.gpu_compress_success) {
    result.compression_ratio_diff =
        result.gpu_compression_ratio - result.cpu_compression_ratio;
    result.throughput_diff_mbps = result.gpu_compress_throughput_mbps -
                                  result.cpu_compress_throughput_mbps;
  } else {
    result.compression_ratio_diff = 0;
    result.throughput_diff_mbps = 0;
  }
}

void print_result_header() {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n+--------+--------+----------+-----------+-----------+-------"
               "----+-----------+-----------+--------+--------+--------+-------"
               "-+--------+--------+--------+\n";
  std::cout << "| Size   | Level  | Pattern  | CPU Comp  | GPU Comp  | CPU "
               "Decomp| GPU Decomp| CPU Thrpt | GPU Thrpt| CPU Ratio| GPU "
               "Ratio| C Speed | D Speed | C Delta|\n";
  std::cout << "|        |        |          | (ms)      | (ms)      | (ms)    "
               "  | (ms)      | (MB/s)    | (MB/s)   |          |          | "
               "up      | up      |        |\n";
  std::cout << "+--------+--------+----------+-----------+-----------+---------"
               "--+-----------+-----------+--------+--------+--------+--------+"
               "--------+--------+--------+\n";
}

void print_result_row(const ComparisonResult &r) {
  std::cout << "| " << std::setw(6) << (r.data_size / (1024 * 1024)) << "MB | ";
  std::cout << std::setw(6) << r.compression_level << " | ";
  std::cout << std::setw(8) << r.data_pattern << " | ";

  // CPU/GPU Compression times
  if (r.cpu_compress_success) {
    std::cout << std::setw(9) << r.cpu_compress_time_ms << " | ";
  } else {
    std::cout << std::setw(9) << "FAIL" << " | ";
  }

  if (r.gpu_compress_success) {
    std::cout << std::setw(9) << r.gpu_compress_time_ms << " | ";
  } else {
    std::cout << std::setw(9) << "FAIL" << " | ";
  }

  // CPU/GPU Decompression times
  if (r.cpu_decompress_success) {
    std::cout << std::setw(9) << r.cpu_decompress_time_ms << " | ";
  } else {
    std::cout << std::setw(9) << "FAIL" << " | ";
  }

  if (r.gpu_decompress_success) {
    std::cout << std::setw(9) << r.gpu_decompress_time_ms << " | ";
  } else {
    std::cout << std::setw(9) << "FAIL" << " | ";
  }

  // Throughputs
  if (r.cpu_compress_success) {
    std::cout << std::setw(9) << r.cpu_compress_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(9) << "N/A" << " | ";
  }

  if (r.gpu_compress_success) {
    std::cout << std::setw(8) << r.gpu_compress_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(8) << "N/A" << " | ";
  }

  // Ratios
  if (r.cpu_compress_success) {
    std::cout << std::setw(8) << r.cpu_compression_ratio << " | ";
  } else {
    std::cout << std::setw(8) << "N/A" << " | ";
  }

  if (r.gpu_compress_success) {
    std::cout << std::setw(8) << r.gpu_compression_ratio << " | ";
  } else {
    std::cout << std::setw(8) << "N/A" << " | ";
  }

  // Speedups
  if (r.compress_speedup > 0) {
    std::cout << std::setw(7) << r.compress_speedup << "x | ";
  } else {
    std::cout << std::setw(7) << "N/A" << " | ";
  }

  if (r.decompress_speedup > 0) {
    std::cout << std::setw(7) << r.decompress_speedup << "x | ";
  } else {
    std::cout << std::setw(7) << "N/A" << " | ";
  }

  // Compression ratio delta
  if (r.cpu_compress_success && r.gpu_compress_success) {
    std::cout << std::setw(6) << std::showpos << r.compression_ratio_diff
              << std::noshowpos << " |";
  } else {
    std::cout << std::setw(6) << "N/A" << " |";
  }

  std::cout << "\n";
}

void print_result_separator() {
  std::cout << "+--------+--------+----------+-----------+-----------+---------"
               "--+-----------+-----------+--------+--------+--------+--------+"
               "--------+--------+--------+\n";
}

void print_cross_compat_header() {
  std::cout
      << "\n+--------+--------+----------+------------------------+------------"
         "------------+------------------------+------------------------+\n";
  std::cout
      << "| Size   | Level  | Pattern  | CPU->GPU Decomp        | GPU->CPU "
         "Decomp        | CPU->GPU Time          | GPU->CPU Time          |\n";
  std::cout
      << "|        |        |          | Success                | Success      "
         "          | (ms)                   | (ms)                   |\n";
  std::cout
      << "+--------+--------+----------+------------------------+--------------"
         "----------+------------------------+------------------------+\n";
}

void print_cross_compat_row(const ComparisonResult &r) {
  std::cout << "| " << std::setw(6) << (r.data_size / (1024 * 1024)) << "MB | ";
  std::cout << std::setw(6) << r.compression_level << " | ";
  std::cout << std::setw(8) << r.data_pattern << " | ";

  std::cout << std::setw(22)
            << (r.cpu_compress_gpu_decompress_success ? "PASS" : "FAIL")
            << " | ";
  std::cout << std::setw(22)
            << (r.gpu_compress_cpu_decompress_success ? "PASS" : "FAIL")
            << " | ";

  if (r.cpu_compress_gpu_decompress_success) {
    std::cout << std::setw(22) << r.cpu_compress_gpu_decompress_time_ms
              << " | ";
  } else {
    std::cout << std::setw(22) << "N/A" << " | ";
  }

  if (r.gpu_compress_cpu_decompress_success) {
    std::cout << std::setw(22) << r.gpu_compress_cpu_decompress_time_ms << " |";
  } else {
    std::cout << std::setw(22) << "N/A" << " |";
  }

  std::cout << "\n";
}

void print_cross_compat_separator() {
  std::cout
      << "+--------+--------+----------+------------------------+--------------"
         "----------+------------------------+------------------------+\n";
}

void print_summary(const std::vector<ComparisonResult> &results) {
  int total_tests = results.size();
  int cpu_compress_success = 0;
  int gpu_compress_success = 0;
  int cpu_decompress_success = 0;
  int gpu_decompress_success = 0;
  int cross_compat_cpu_gpu = 0;
  int cross_compat_gpu_cpu = 0;

  double avg_compress_speedup = 0;
  double avg_decompress_speedup = 0;
  double max_compress_speedup = 0;
  double max_decompress_speedup = 0;
  double min_compress_speedup = 1e9;
  double min_decompress_speedup = 1e9;

  int valid_compress_comparisons = 0;
  int valid_decompress_comparisons = 0;

  for (const auto &r : results) {
    if (r.cpu_compress_success)
      cpu_compress_success++;
    if (r.gpu_compress_success)
      gpu_compress_success++;
    if (r.cpu_decompress_success)
      cpu_decompress_success++;
    if (r.gpu_decompress_success)
      gpu_decompress_success++;
    if (r.cpu_compress_gpu_decompress_success)
      cross_compat_cpu_gpu++;
    if (r.gpu_compress_cpu_decompress_success)
      cross_compat_gpu_cpu++;

    if (r.compress_speedup > 0) {
      avg_compress_speedup += r.compress_speedup;
      max_compress_speedup = std::max(max_compress_speedup, r.compress_speedup);
      min_compress_speedup = std::min(min_compress_speedup, r.compress_speedup);
      valid_compress_comparisons++;
    }

    if (r.decompress_speedup > 0) {
      avg_decompress_speedup += r.decompress_speedup;
      max_decompress_speedup =
          std::max(max_decompress_speedup, r.decompress_speedup);
      min_decompress_speedup =
          std::min(min_decompress_speedup, r.decompress_speedup);
      valid_decompress_comparisons++;
    }
  }

  if (valid_compress_comparisons > 0) {
    avg_compress_speedup /= valid_compress_comparisons;
  }
  if (valid_decompress_comparisons > 0) {
    avg_decompress_speedup /= valid_decompress_comparisons;
  }

  std::cout << "\n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "                           BENCHMARK SUMMARY                   "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "Total Test Configurations: " << total_tests << "\n\n";

  std::cout << "Compression Success Rates:\n";
  std::cout << "  CPU (Standard zstd): " << cpu_compress_success << "/"
            << total_tests << " (" << std::fixed << std::setprecision(1)
            << (100.0 * cpu_compress_success / total_tests) << "%)\n";
  std::cout << "  GPU (CUDA zstd):     " << gpu_compress_success << "/"
            << total_tests << " ("
            << (100.0 * gpu_compress_success / total_tests) << "%)\n\n";

  std::cout << "Decompression Success Rates:\n";
  std::cout << "  CPU (Standard zstd): " << cpu_decompress_success << "/"
            << total_tests << " ("
            << (100.0 * cpu_decompress_success / total_tests) << "%)\n";
  std::cout << "  GPU (CUDA zstd):     " << gpu_decompress_success << "/"
            << total_tests << " ("
            << (100.0 * gpu_decompress_success / total_tests) << "%)\n\n";

  std::cout << "Cross-Compatibility:\n";
  std::cout << "  CPU Compress -> GPU Decompress: " << cross_compat_cpu_gpu
            << "/" << total_tests << " ("
            << (100.0 * cross_compat_cpu_gpu / total_tests) << "%)\n";
  std::cout << "  GPU Compress -> CPU Decompress: " << cross_compat_gpu_cpu
            << "/" << total_tests << " ("
            << (100.0 * cross_compat_gpu_cpu / total_tests) << "%)\n\n";

  if (valid_compress_comparisons > 0) {
    std::cout << "Compression Speedup (GPU vs CPU):\n";
    std::cout << "  Average: " << std::setprecision(2) << avg_compress_speedup
              << "x\n";
    std::cout << "  Maximum: " << max_compress_speedup << "x\n";
    std::cout << "  Minimum: " << min_compress_speedup << "x\n\n";
  }

  if (valid_decompress_comparisons > 0) {
    std::cout << "Decompression Speedup (GPU vs CPU):\n";
    std::cout << "  Average: " << std::setprecision(2) << avg_decompress_speedup
              << "x\n";
    std::cout << "  Maximum: " << max_decompress_speedup << "x\n";
    std::cout << "  Minimum: " << min_decompress_speedup << "x\n\n";
  }

  std::cout << "==============================================================="
               "=================\n";
}

void export_results_csv(const std::vector<ComparisonResult> &results,
                        const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open CSV file for writing: " << filename
              << std::endl;
    return;
  }

  // Write header
  file << "data_size,compression_level,data_pattern,"
       << "cpu_compress_time_ms,cpu_decompress_time_ms,"
       << "cpu_compress_throughput_mbps,cpu_decompress_throughput_mbps,"
       << "cpu_compression_ratio,cpu_compressed_size,cpu_compress_success,cpu_"
          "decompress_success,"
       << "gpu_compress_time_ms,gpu_decompress_time_ms,"
       << "gpu_compress_throughput_mbps,gpu_decompress_throughput_mbps,"
       << "gpu_compression_ratio,gpu_compressed_size,gpu_compress_success,gpu_"
          "decompress_success,"
       << "gpu_peak_vram_bytes,gpu_memory_alloc_time_ms,"
       << "cpu_compress_gpu_decompress_success,gpu_compress_cpu_decompress_"
          "success,"
       << "cpu_compress_gpu_decompress_time_ms,gpu_compress_cpu_decompress_"
          "time_ms,"
       << "compress_speedup,decompress_speedup,compression_ratio_diff,"
          "throughput_diff_mbps\n";

  // Write data
  for (const auto &r : results) {
    file << r.data_size << "," << r.compression_level << "," << r.data_pattern
         << "," << r.cpu_compress_time_ms << "," << r.cpu_decompress_time_ms
         << "," << r.cpu_compress_throughput_mbps << ","
         << r.cpu_decompress_throughput_mbps << "," << r.cpu_compression_ratio
         << "," << r.cpu_compressed_size << ","
         << (r.cpu_compress_success ? 1 : 0) << ","
         << (r.cpu_decompress_success ? 1 : 0) << "," << r.gpu_compress_time_ms
         << "," << r.gpu_decompress_time_ms << ","
         << r.gpu_compress_throughput_mbps << ","
         << r.gpu_decompress_throughput_mbps << "," << r.gpu_compression_ratio
         << "," << r.gpu_compressed_size << ","
         << (r.gpu_compress_success ? 1 : 0) << ","
         << (r.gpu_decompress_success ? 1 : 0) << "," << r.gpu_peak_vram_bytes
         << "," << r.gpu_memory_alloc_time_ms << ","
         << (r.cpu_compress_gpu_decompress_success ? 1 : 0) << ","
         << (r.gpu_compress_cpu_decompress_success ? 1 : 0) << ","
         << r.cpu_compress_gpu_decompress_time_ms << ","
         << r.gpu_compress_cpu_decompress_time_ms << "," << r.compress_speedup
         << "," << r.decompress_speedup << "," << r.compression_ratio_diff
         << "," << r.throughput_diff_mbps << "\n";
  }

  file.close();
  std::cout << "\nResults exported to: " << filename << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [options] [input_file]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --generate-data          Generate synthetic test data "
               "instead of using input file\n";
  std::cout << "  --csv-output <filename>  Export results to CSV file\n";
  std::cout << "  --levels <list>          Comma-separated compression levels "
               "(default: 1,3,6,12,19,22)\n";
  std::cout << "  --sizes <list>           Comma-separated sizes in MB "
               "(default: 1,10,100,256)\n";
  std::cout << "  --help                   Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << program_name << " data.bin\n";
  std::cout << "  " << program_name
            << " --generate-data --csv-output results.csv\n";
  std::cout << "  " << program_name
            << " data.bin --levels 1,6,12 --sizes 10,100\n";
}

int main(int argc, char **argv) {
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "           ZSTD CPU vs GPU Comparison Benchmark                "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n\n";

  // Parse command line arguments
  std::string input_file;
  std::string csv_output = "benchmark_zstd_gpu_comparison_results.csv";
  bool generate_data = false;
  std::vector<int> levels_to_test;
  std::vector<size_t> sizes_to_test;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--generate-data") {
      generate_data = true;
    } else if (arg == "--csv-output" && i + 1 < argc) {
      csv_output = argv[++i];
    } else if (arg == "--levels" && i + 1 < argc) {
      std::string levels_str = argv[++i];
      size_t pos = 0;
      while ((pos = levels_str.find(',')) != std::string::npos) {
        levels_to_test.push_back(std::stoi(levels_str.substr(0, pos)));
        levels_str.erase(0, pos + 1);
      }
      levels_to_test.push_back(std::stoi(levels_str));
    } else if (arg == "--sizes" && i + 1 < argc) {
      std::string sizes_str = argv[++i];
      size_t pos = 0;
      while ((pos = sizes_str.find(',')) != std::string::npos) {
        sizes_to_test.push_back(std::stoull(sizes_str.substr(0, pos)) * 1024 *
                                1024);
        sizes_str.erase(0, pos + 1);
      }
      sizes_to_test.push_back(std::stoull(sizes_str) * 1024 * 1024);
    } else if (arg[0] != '-') {
      input_file = arg;
    }
  }

  // Use default levels and sizes if not specified
  if (levels_to_test.empty()) {
    for (int i = 0; i < NUM_COMPRESSION_LEVELS; i++) {
      levels_to_test.push_back(COMPRESSION_LEVELS[i]);
    }
  }

  if (sizes_to_test.empty()) {
    for (int i = 0; i < NUM_FILE_SIZES; i++) {
      sizes_to_test.push_back(FILE_SIZES[i]);
    }
  }

  size_t max_data_size = compute_max_data_size();
  std::vector<size_t> filtered_sizes;
  filtered_sizes.reserve(sizes_to_test.size());
  for (size_t size : sizes_to_test) {
    if (size <= max_data_size) {
      filtered_sizes.push_back(size);
    }
  }
  if (!filtered_sizes.empty()) {
    sizes_to_test.swap(filtered_sizes);
  }

  // Validate input
  if (!generate_data && input_file.empty()) {
    std::cerr << "Error: No input file specified. Use --generate-data to "
                 "generate synthetic data.\n";
    print_usage(argv[0]);
    return 1;
  }

  // Print configuration
  std::cout << "Configuration:\n";
  std::cout << "  Input: "
            << (generate_data ? "Generated synthetic data" : input_file)
            << "\n";
  std::cout << "  Compression levels: ";
  for (size_t i = 0; i < levels_to_test.size(); i++) {
    std::cout << levels_to_test[i];
    if (i < levels_to_test.size() - 1)
      std::cout << ", ";
  }
  std::cout << "\n";
  std::cout << "  Data sizes: ";
  for (size_t i = 0; i < sizes_to_test.size(); i++) {
    std::cout << (sizes_to_test[i] / (1024 * 1024)) << "MB";
    if (i < sizes_to_test.size() - 1)
      std::cout << ", ";
  }
  std::cout << "\n";
  std::cout << "  VRAM-limited max size: " << format_bytes(max_data_size) << "\n";
  std::cout << "  Iterations per test: " << NUM_ITERATIONS << "\n";
  std::cout << "  CSV output: " << csv_output << "\n\n";

  // Load or generate test data
  std::vector<byte_t> file_data;
  if (!generate_data) {
    // Load the largest size needed
    size_t max_size =
        *std::max_element(sizes_to_test.begin(), sizes_to_test.end());
    if (!load_file(input_file, file_data, max_size)) {
      return 1;
    }
    std::cout << "Loaded " << format_bytes(file_data.size()) << " from "
              << input_file << "\n\n";
  }

  // Data patterns to test
  const char *patterns[] = {"random", "repetitive", "semi-random", "text-like"};
  std::vector<ComparisonResult> all_results;

  // Run benchmarks
  for (size_t data_size : sizes_to_test) {
    std::cout << "\n--- Testing Data Size: " << format_bytes(data_size)
              << " ---\n";

    for (const char *pattern : patterns) {
      std::cout << "\n  Pattern: " << pattern << "\n";

      // Prepare test data
      std::vector<byte_t> test_data(data_size);
      if (generate_data) {
        generate_test_data(test_data.data(), data_size, pattern);
      } else {
        // Use file data, padded or truncated as needed
        size_t copy_size = std::min(data_size, file_data.size());
        memcpy(test_data.data(), file_data.data(), copy_size);
        if (copy_size < data_size) {
          // Pad with pattern
          generate_test_data(test_data.data() + copy_size,
                             data_size - copy_size, pattern);
        }
      }

      for (int level : levels_to_test) {
        std::cout << "    Level " << level << "... " << std::flush;

        ComparisonResult result = {};
        result.data_size = data_size;
        result.compression_level = level;
        result.data_pattern = pattern;

        // 1. CPU Compression Benchmark
        benchmark_cpu_compress(test_data.data(), data_size, level, result);

        // 2. CPU Decompression Benchmark (if compression succeeded)
        if (result.cpu_compress_success) {
          std::vector<byte_t> cpu_compressed(result.cpu_compressed_size);
          size_t max_output = ZSTD_compressBound(data_size);
          std::vector<byte_t> temp_compressed(max_output);
          size_t compressed_size =
              ZSTD_compress(temp_compressed.data(), max_output,
                            test_data.data(), data_size, level);
          memcpy(cpu_compressed.data(), temp_compressed.data(),
                 compressed_size);

          benchmark_cpu_decompress(cpu_compressed.data(), cpu_compressed.size(),
                                   data_size, result);
        }

        // 3. GPU Compression Benchmark
        benchmark_gpu_compress(test_data.data(), data_size, level, result);

        // 4. GPU Decompression Benchmark (if compression succeeded)
        if (result.gpu_compress_success) {
          // Need to get the compressed data from GPU
          void *d_input = nullptr;
          void *d_output = nullptr;
          size_t max_output_size =
              static_cast<size_t>(data_size * MAX_OUTPUT_MULTIPLIER);

          CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_input, data_size));
          CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_output, max_output_size));
          CHECK_CUDA(cudaMemcpy(d_input, test_data.data(), data_size,
                                cudaMemcpyHostToDevice));

          auto manager = create_batch_manager(level);
          CompressionConfig config;
          config.block_size = get_optimal_block_size(
              static_cast<u32>(
                  std::min(data_size, static_cast<size_t>(UINT32_MAX))),
              level);
          manager->configure(config);

          size_t workspace_size = manager->get_compress_temp_size(data_size);
          void *workspace = nullptr;
          CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&workspace, workspace_size));

          size_t gpu_compressed_size = max_output_size;
          manager->compress(d_input, data_size, d_output, &gpu_compressed_size,
                            workspace, workspace_size, nullptr, 0, nullptr);
          CHECK_CUDA(cudaDeviceSynchronize());

          std::vector<byte_t> gpu_compressed(gpu_compressed_size);
          CHECK_CUDA(cudaMemcpy(gpu_compressed.data(), d_output,
                                gpu_compressed_size, cudaMemcpyDeviceToHost));

          CHECK_CUDA(cudaFree(workspace));
          CHECK_CUDA(cudaFree(d_output));
          CHECK_CUDA(cudaFree(d_input));

          benchmark_gpu_decompress(gpu_compressed.data(), gpu_compressed.size(),
                                   data_size, level, result);
        }

        // 5. Cross-compatibility tests
        if (result.cpu_compress_success) {
          result.cpu_compress_gpu_decompress_success =
              test_cpu_compress_gpu_decompress(
                  test_data.data(), data_size, level,
                  result.cpu_compress_gpu_decompress_time_ms);
        }

        if (result.gpu_compress_success) {
          result.gpu_compress_cpu_decompress_success =
              test_gpu_compress_cpu_decompress(
                  test_data.data(), data_size, level,
                  result.gpu_compress_cpu_decompress_time_ms);
        }

        // Compute comparison metrics
        compute_comparison_metrics(result);

        all_results.push_back(result);

        std::cout << "Done (C Speedup: ";
        if (result.compress_speedup > 0) {
          std::cout << std::fixed << std::setprecision(2)
                    << result.compress_speedup << "x";
        } else {
          std::cout << "N/A";
        }
        std::cout << ")\n";

        // Allow GPU to cool
        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }
  }

  // Print detailed results
  std::cout << "\n\n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "                         DETAILED RESULTS                      "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n";

  print_result_header();
  for (const auto &r : all_results) {
    print_result_row(r);
  }
  print_result_separator();

  // Print cross-compatibility results
  std::cout << "\n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "                    CROSS-COMPATIBILITY RESULTS                "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n";

  print_cross_compat_header();
  for (const auto &r : all_results) {
    print_cross_compat_row(r);
  }
  print_cross_compat_separator();

  // Print summary
  print_summary(all_results);

  // Export to CSV
  export_results_csv(all_results, csv_output);

  std::cout << "\nBenchmark complete!\n";
  return 0;
}
