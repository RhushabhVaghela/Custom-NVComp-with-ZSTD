// ============================================================================
// benchmark_zstd_comparison.cu
//
// Comprehensive benchmark comparing standard zstd library compression
// against custom GPU-based cuda_zstd implementation.
//
// Metrics compared:
// - Encoding time (ms)
// - Throughput (MB/s)
// - Compression ratio (%)
// - Resource utilization (VRAM, execution time)
//
// Uses identical input data for fair comparison.
// ============================================================================

#include "cuda_zstd_hash.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_utils.h"
#include "cuda_zstd_xxhash.h"

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
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
#define MAX_VRAM_PER_BENCHMARK (8ULL * 1024 * 1024 * 1024) // 8GB max
#define MAX_OUTPUT_MULTIPLIER 1.5f

// ============================================================================
// DATA PATTERNS
// ============================================================================

enum class DataPattern {
  RANDOM,      // Fully random data (hard to compress)
  REPETITIVE,  // Highly repetitive (best compression)
  SEMI_RANDOM, // Semi-random with patterns
  INCREMENTAL  // Incrementing values
};

// ============================================================================
// BENCHMARK RESULT STRUCTURES
// ============================================================================

struct ZstdBenchmarkResult {
  // Input metadata
  size_t data_size;
  int compression_level;
  DataPattern pattern;

  // Standard zstd results
  double std_encode_time_ms;
  double std_throughput_mbps;
  double std_compression_ratio;
  size_t std_output_size;
  bool std_success;

  // CUDA zstd results
  double gpu_encode_time_ms;
  double gpu_throughput_mbps;
  double gpu_compression_ratio;
  size_t gpu_output_size;
  bool gpu_success;
  bool gpu_integrity_passed;

  // Resource utilization
  size_t gpu_peak_vram_bytes;
  double gpu_memory_alloc_time_ms;
  double gpu_kernel_launch_overhead_ms;

  // Comparison metrics
  double speedup_ratio;         // GPU speedup vs standard zstd
  double compression_delta;     // Difference in compression ratio
  double throughput_delta_mbps; // Difference in throughput
};

struct GpuMemorySnapshot {
  size_t free_bytes;
  size_t total_bytes;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Generate test data with specified pattern
void generate_test_data(void *ptr, size_t size, DataPattern pattern) {
  byte_t *data = static_cast<byte_t *>(ptr);

  switch (pattern) {
  case DataPattern::RANDOM: {
    std::mt19937_64 rng(12345 + static_cast<int>(pattern));
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<byte_t>(dist(rng));
    }
  } break;

  case DataPattern::REPETITIVE:
    // All 'A's - highly compressible
    memset(data, 'A', size);
    break;

  case DataPattern::SEMI_RANDOM: {
    // Repeating 64-byte patterns with some randomness
    std::mt19937_64 rng(67890 + static_cast<int>(pattern));
    std::uniform_int_distribution<int> dist(0, 255);
    const size_t block_size = 64;
    byte_t block[block_size];
    for (size_t i = 0; i < block_size; ++i) {
      block[i] = static_cast<byte_t>(dist(rng));
    }
    for (size_t i = 0; i < size; ++i) {
      data[i] = block[i % block_size];
    }
  } break;

  case DataPattern::INCREMENTAL:
    // Incrementing bytes (0-255 repeated)
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<byte_t>(i & 0xFF);
    }
    break;
  }
}

// Get GPU memory snapshot
GpuMemorySnapshot get_gpu_memory_snapshot() {
  GpuMemorySnapshot snapshot;
  CHECK_CUDA(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
  return snapshot;
}

// Compute XXH64 checksum
uint64_t compute_checksum(void *data, size_t size, cudaStream_t stream = 0) {
  u64 h_hash;
  cuda_zstd::xxhash::compute_xxhash64(data, size, 0, &h_hash, stream);
  cudaStreamSynchronize(stream ? stream : 0);
  return h_hash;
}

// ============================================================================
// STANDARD ZSTD BENCHMARK
// ============================================================================

ZstdBenchmarkResult benchmark_standard_zstd(const void *input_data,
                                            size_t input_size,
                                            int compression_level) {
  ZstdBenchmarkResult result = {};
  result.data_size = input_size;
  result.compression_level = compression_level;

  // Estimate maximum output size
  size_t max_output_size = ZSTD_compressBound(input_size);

  // Allocate output buffer
  std::vector<byte_t> compressed_data(max_output_size);

  // Warmup
  CHECK_ZSTD(ZSTD_compress(compressed_data.data(), max_output_size,
                           static_cast<const byte_t *>(input_data), input_size,
                           compression_level));

  // Benchmark compression
  auto start = std::chrono::high_resolution_clock::now();

  size_t compressed_size = ZSTD_compress(
      compressed_data.data(), max_output_size,
      static_cast<const byte_t *>(input_data), input_size, compression_level);

  auto end = std::chrono::high_resolution_clock::now();

  if (!ZSTD_isError(compressed_size)) {
    result.std_success = true;
    result.std_encode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.std_throughput_mbps =
        (input_size / (1024.0 * 1024.0)) / (result.std_encode_time_ms / 1000.0);
    result.std_compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(input_size);
    result.std_output_size = compressed_size;
  } else {
    result.std_success = false;
    result.std_encode_time_ms = -1;
    result.std_throughput_mbps = 0;
    result.std_compression_ratio = -1;
    result.std_output_size = 0;
  }

  return result;
}

// ============================================================================
// GPU ZSTD BENCHMARK
// ============================================================================

ZstdBenchmarkResult benchmark_cuda_zstd(const void *h_input, size_t input_size,
                                        int compression_level,
                                        DataPattern pattern) {
  ZstdBenchmarkResult result = {};
  result.data_size = input_size;
  result.compression_level = compression_level;
  result.pattern = pattern;

  // Track GPU memory
  GpuMemorySnapshot mem_before = get_gpu_memory_snapshot();
  size_t peak_vram = 0;

  // Allocate device memory
  void *d_input = nullptr;
  void *d_output = nullptr;
  size_t max_output_size =
      static_cast<size_t>(input_size * MAX_OUTPUT_MULTIPLIER);

  auto alloc_start = std::chrono::high_resolution_clock::now();
  CHECK_CUDA(cudaMalloc(&d_input, input_size));
  CHECK_CUDA(cudaMalloc(&d_output, max_output_size));
  auto alloc_end = std::chrono::high_resolution_clock::now();

  result.gpu_memory_alloc_time_ms =
      std::chrono::duration<double, std::milli>(alloc_end - alloc_start)
          .count();

  // Copy input to device
  CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

  // Compute original checksum
  uint64_t original_checksum = compute_checksum(d_input, input_size);

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
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

  // Track peak VRAM usage
  GpuMemorySnapshot mem_after_alloc = get_gpu_memory_snapshot();
  peak_vram = mem_before.free_bytes - mem_after_alloc.free_bytes;

  // Warmup
  size_t compressed_size = max_output_size;
  manager->compress(d_input, input_size, d_output, &compressed_size, workspace,
                    workspace_size, nullptr, 0, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark compression with kernel launch overhead measurement
  cudaEvent_t kernel_start, kernel_stop;
  CHECK_CUDA(cudaEventCreate(&kernel_start));
  CHECK_CUDA(cudaEventCreate(&kernel_stop));

  CHECK_CUDA(cudaEventRecord(kernel_start));

  Status status =
      manager->compress(d_input, input_size, d_output, &compressed_size,
                        workspace, workspace_size, nullptr, 0, nullptr);

  CHECK_CUDA(cudaEventRecord(kernel_stop));
  CHECK_CUDA(cudaEventSynchronize(kernel_stop));

  float kernel_time_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop));
  result.gpu_kernel_launch_overhead_ms = kernel_time_ms;

  CHECK_CUDA(cudaEventDestroy(kernel_start));
  CHECK_CUDA(cudaEventDestroy(kernel_stop));

  CHECK_CUDA(cudaDeviceSynchronize());

  // Track peak VRAM after compression
  GpuMemorySnapshot mem_after_compress = get_gpu_memory_snapshot();
  peak_vram = std::max(peak_vram,
                       mem_before.free_bytes - mem_after_compress.free_bytes);
  result.gpu_peak_vram_bytes = peak_vram;

  if (status == Status::SUCCESS) {
    result.gpu_success = true;
    result.gpu_encode_time_ms = kernel_time_ms;
    result.gpu_throughput_mbps =
        (input_size / (1024.0 * 1024.0)) / (result.gpu_encode_time_ms / 1000.0);
    result.gpu_compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(input_size);
    result.gpu_output_size = compressed_size;

    // Verify integrity by decompressing and checking checksum
    void *d_decompressed = nullptr;
    CHECK_CUDA(cudaMalloc(&d_decompressed, input_size));

    size_t decompressed_size = input_size;
    auto decomp_manager = create_batch_manager(compression_level);
    size_t decomp_workspace =
        decomp_manager->get_decompress_temp_size(compressed_size);
    void *decomp_workspace_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&decomp_workspace_ptr, decomp_workspace));

    Status dec_status = decomp_manager->decompress(
        d_output, compressed_size, d_decompressed, &decompressed_size,
        decomp_workspace_ptr, decomp_workspace, nullptr);

    if (dec_status == Status::SUCCESS) {
      uint64_t decompressed_checksum =
          compute_checksum(d_decompressed, input_size);
      result.gpu_integrity_passed =
          (original_checksum == decompressed_checksum);
    } else {
      result.gpu_integrity_passed = false;
    }

    CHECK_CUDA(cudaFree(d_decompressed));
    CHECK_CUDA(cudaFree(decomp_workspace_ptr));
  } else {
    result.gpu_success = false;
    result.gpu_encode_time_ms = -1;
    result.gpu_throughput_mbps = 0;
    result.gpu_compression_ratio = -1;
    result.gpu_output_size = 0;
    result.gpu_integrity_passed = false;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_input));

  return result;
}

// ============================================================================
// COMPARISON AND REPORTING
// ============================================================================

void compute_comparison_metrics(ZstdBenchmarkResult &result) {
  if (result.std_success && result.gpu_success) {
    // Speedup: how many times faster is GPU vs standard
    result.speedup_ratio =
        result.std_encode_time_ms / result.gpu_encode_time_ms;

    // Compression delta: positive means GPU compresses better
    result.compression_delta =
        result.std_compression_ratio - result.gpu_compression_ratio;

    // Throughput delta: positive means GPU has higher throughput
    result.throughput_delta_mbps =
        result.gpu_throughput_mbps - result.std_throughput_mbps;
  } else {
    result.speedup_ratio = 0;
    result.compression_delta = 0;
    result.throughput_delta_mbps = 0;
  }
}

void print_result_header() {
  std::cout << std::fixed << std::setprecision(2);
  std::cout
      << "+---------+--------+------+---------------+---------------+----------"
         "-----+---------------+---------------+----------+--------------+\n";
  std::cout
      << "| Size    | Level  | Pat  | Std Time(ms)  | GPU Time(ms)  | Std MB/s "
         "     | GPU MB/s      | Std Ratio     | GPU Ratio | Speedup    |\n";
  std::cout
      << "+---------+--------+------+---------------+---------------+----------"
         "-----+---------------+---------------+----------+--------------+\n";
}

void print_result_row(const ZstdBenchmarkResult &r) {
  std::cout << "| " << std::setw(7) << (r.data_size / (1024 * 1024))
            << " MB | ";
  std::cout << std::setw(6) << r.compression_level << " | ";
  std::cout << (char)('A' + static_cast<int>(r.pattern)) << "   | ";

  if (r.std_success) {
    std::cout << std::setw(13) << r.std_encode_time_ms << " | ";
  } else {
    std::cout << std::setw(13) << "FAILED" << " | ";
  }

  if (r.gpu_success) {
    std::cout << std::setw(13) << r.gpu_encode_time_ms << " | ";
  } else {
    std::cout << std::setw(13) << "FAILED" << " | ";
  }

  if (r.std_success) {
    std::cout << std::setw(13) << r.std_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(13) << "N/A" << " | ";
  }

  if (r.gpu_success) {
    std::cout << std::setw(13) << r.gpu_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(13) << "N/A" << " | ";
  }

  if (r.std_success) {
    std::cout << std::setw(11) << r.std_compression_ratio << " | ";
  } else {
    std::cout << std::setw(11) << "N/A" << " | ";
  }

  if (r.gpu_success) {
    std::cout << std::setw(8) << r.gpu_compression_ratio << " | ";
  } else {
    std::cout << std::setw(8) << "N/A" << " | ";
  }

  if (r.std_success && r.gpu_success) {
    std::cout << std::setw(10) << std::setprecision(2) << r.speedup_ratio
              << "x | ";
  } else {
    std::cout << std::setw(10) << "N/A" << " | ";
  }

  std::cout << "\n";
}

void print_resource_header() {
  std::cout << "\n+=========+==================+==================+============"
               "======+\n";
  std::cout << "| Size    | GPU Alloc(ms)    | Kernel Time(ms)  | Peak "
               "VRAM(MB)    |\n";
  std::cout << "+=========+==================+==================+=============="
               "====+\n";
}

void print_resource_row(const ZstdBenchmarkResult &r) {
  std::cout << "| " << std::setw(7) << (r.data_size / (1024 * 1024))
            << " MB | ";
  std::cout << std::setw(16) << std::setprecision(2)
            << r.gpu_memory_alloc_time_ms << " | ";
  std::cout << std::setw(16) << std::setprecision(2)
            << r.gpu_kernel_launch_overhead_ms << " | ";
  std::cout << std::setw(16) << std::setprecision(2)
            << (r.gpu_peak_vram_bytes / (1024.0 * 1024.0)) << " |\n";
}

void print_comparison_summary(const std::vector<ZstdBenchmarkResult> &results) {
  int total_tests = results.size();
  int std_success_count = 0;
  int gpu_success_count = 0;
  int gpu_integrity_count = 0;

  double total_speedup = 0;
  double max_speedup = 0;
  int max_speedup_idx = -1;
  double min_speedup = 1e9;
  int min_speedup_idx = -1;

  double avg_compression_delta = 0;
  double avg_throughput_delta = 0;

  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];

    if (r.std_success)
      std_success_count++;
    if (r.gpu_success)
      gpu_success_count++;
    if (r.gpu_integrity_passed)
      gpu_integrity_count++;

    if (r.std_success && r.gpu_success) {
      total_speedup += r.speedup_ratio;
      if (r.speedup_ratio > max_speedup) {
        max_speedup = r.speedup_ratio;
        max_speedup_idx = static_cast<int>(i);
      }
      if (r.speedup_ratio < min_speedup && r.speedup_ratio > 0) {
        min_speedup = r.speedup_ratio;
        min_speedup_idx = static_cast<int>(i);
      }
      avg_compression_delta += r.compression_delta;
      avg_throughput_delta += r.throughput_delta_mbps;
    }
  }

  int valid_comparisons = std_success_count;
  if (valid_comparisons > 0) {
    avg_compression_delta /= valid_comparisons;
    avg_throughput_delta /= valid_comparisons;
    total_speedup /= valid_comparisons;
  }

  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "       COMPARISON SUMMARY\n";
  std::cout << "========================================\n";
  std::cout << "Total Tests:           " << total_tests << "\n";
  std::cout << "Standard ZSTD Success: " << std_success_count << "\n";
  std::cout << "CUDA ZSTD Success:     " << gpu_success_count << "\n";
  std::cout << "GPU Integrity Passed:  " << gpu_integrity_count << "\n";
  std::cout << "\n";
  std::cout << "Performance Comparison:\n";
  std::cout << "  Average Speedup:     " << std::fixed << std::setprecision(2)
            << total_speedup << "x\n";
  if (max_speedup_idx >= 0) {
    std::cout << "  Max Speedup:         " << max_speedup << "x ("
              << (results[max_speedup_idx].data_size / (1024 * 1024))
              << " MB, level " << results[max_speedup_idx].compression_level
              << ")\n";
  }
  if (min_speedup_idx >= 0) {
    std::cout << "  Min Speedup:         " << min_speedup << "x ("
              << (results[min_speedup_idx].data_size / (1024 * 1024))
              << " MB, level " << results[min_speedup_idx].compression_level
              << ")\n";
  }
  std::cout << "\n";
  std::cout << "Compression Comparison:\n";
  std::cout << "  Avg Compression Delta: " << std::setprecision(4)
            << avg_compression_delta << " (positive = GPU better)\n";
  std::cout << "\n";
  std::cout << "Throughput Comparison:\n";
  std::cout << "  Avg Throughput Delta:  " << std::setprecision(2)
            << avg_throughput_delta << " MB/s (positive = GPU better)\n";
  std::cout << "========================================\n";
}

void export_results_csv(const std::vector<ZstdBenchmarkResult> &results,
                        const std::string &filename) {
  std::ofstream file(filename);
  file << "data_size,compression_level,pattern,"
       << "std_encode_time_ms,std_throughput_mbps,std_compression_ratio,std_"
          "output_size,std_success,"
       << "gpu_encode_time_ms,gpu_throughput_mbps,gpu_compression_ratio,gpu_"
          "output_size,gpu_success,gpu_integrity_passed,"
       << "gpu_peak_vram_bytes,gpu_memory_alloc_time_ms,gpu_kernel_launch_"
          "overhead_ms,"
       << "speedup_ratio,compression_delta,throughput_delta_mbps\n";

  for (const auto &r : results) {
    file << r.data_size << "," << r.compression_level << ","
         << static_cast<int>(r.pattern) << "," << r.std_encode_time_ms << ","
         << r.std_throughput_mbps << "," << r.std_compression_ratio << ","
         << r.std_output_size << "," << (r.std_success ? 1 : 0) << ","
         << r.gpu_encode_time_ms << "," << r.gpu_throughput_mbps << ","
         << r.gpu_compression_ratio << "," << r.gpu_output_size << ","
         << (r.gpu_success ? 1 : 0) << "," << (r.gpu_integrity_passed ? 1 : 0)
         << "," << r.gpu_peak_vram_bytes << "," << r.gpu_memory_alloc_time_ms
         << "," << r.gpu_kernel_launch_overhead_ms << "," << r.speedup_ratio
         << "," << r.compression_delta << "," << r.throughput_delta_mbps
         << "\n";
  }

  file.close();
  std::cout << "\nResults exported to: " << filename << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
  std::cout << "========================================\n";
  std::cout << "  ZSTD vs CUDA ZSTD Comparison Benchmark\n";
  std::cout << "========================================\n\n";

  // Configuration
  std::vector<size_t> data_sizes = {
      1ULL * 1024 * 1024,   // 1 MB
      10ULL * 1024 * 1024,  // 10 MB
      100ULL * 1024 * 1024, // 100 MB
  };

  std::vector<int> compression_levels = {1, 3, 5, 9, 12, 19, 22};
  std::vector<DataPattern> patterns = {
      DataPattern::RANDOM,
      DataPattern::REPETITIVE,
      DataPattern::SEMI_RANDOM,
  };

  // Parse command line arguments
  bool run_small = true;
  bool run_medium = true;
  bool run_large = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--small") {
      run_small = true;
      run_medium = false;
      run_large = false;
    } else if (arg == "--medium") {
      run_small = false;
      run_medium = true;
      run_large = false;
    } else if (arg == "--large") {
      run_small = false;
      run_medium = false;
      run_large = true;
    } else if (arg == "--all") {
      run_small = true;
      run_medium = true;
      run_large = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --small   Run only small data size (1MB)\n";
      std::cout << "  --medium  Run only medium data sizes (1-10MB)\n";
      std::cout << "  --large   Run large data sizes (up to 100MB)\n";
      std::cout << "  --all     Run all data sizes (default)\n";
      std::cout << "  --help    Show this help\n";
      return 0;
    }
  }

  // Adjust data sizes based on configuration
  std::vector<size_t> test_sizes;
  if (run_small)
    test_sizes.push_back(1ULL * 1024 * 1024);
  if (run_medium) {
    test_sizes.push_back(5ULL * 1024 * 1024);
    test_sizes.push_back(10ULL * 1024 * 1024);
  }
  if (run_large) {
    test_sizes.push_back(50ULL * 1024 * 1024);
    test_sizes.push_back(100ULL * 1024 * 1024);
  }

  std::cout << "Configuration:\n";
  std::cout << "  Data sizes: ";
  for (size_t s : test_sizes) {
    std::cout << (s / (1024 * 1024)) << "MB ";
  }
  std::cout << "\n";
  std::cout << "  Compression levels: ";
  for (int l : compression_levels) {
    std::cout << l << " ";
  }
  std::cout << "\n";
  std::cout << "  Patterns: RANDOM(A), REPETITIVE(B), SEMI_RANDOM(C)\n\n";

  std::vector<ZstdBenchmarkResult> all_results;

  // Run benchmarks
  for (size_t data_size : test_sizes) {
    std::cout << "\n--- Testing Data Size: " << (data_size / (1024 * 1024))
              << " MB ---\n";

    // Allocate host input buffer once
    void *h_input = malloc(data_size);

    for (DataPattern pattern : patterns) {
      std::cout << "\nPattern: ";
      switch (pattern) {
      case DataPattern::RANDOM:
        std::cout << "RANDOM";
        break;
      case DataPattern::REPETITIVE:
        std::cout << "REPETITIVE";
        break;
      case DataPattern::SEMI_RANDOM:
        std::cout << "SEMI_RANDOM";
        break;
      case DataPattern::INCREMENTAL:
        std::cout << "INCREMENTAL";
        break;
      }
      std::cout << "\n";

      // Generate test data (identical for both benchmarks)
      generate_test_data(h_input, data_size, pattern);

      for (int level : compression_levels) {
        std::cout << "  Level " << level << "... " << std::flush;

        // Benchmark standard zstd
        ZstdBenchmarkResult result =
            benchmark_standard_zstd(h_input, data_size, level);

        // Benchmark GPU zstd
        ZstdBenchmarkResult gpu_result =
            benchmark_cuda_zstd(h_input, data_size, level, pattern);

        // Merge results
        ZstdBenchmarkResult combined = result;
        combined.gpu_encode_time_ms = gpu_result.gpu_encode_time_ms;
        combined.gpu_throughput_mbps = gpu_result.gpu_throughput_mbps;
        combined.gpu_compression_ratio = gpu_result.gpu_compression_ratio;
        combined.gpu_output_size = gpu_result.gpu_output_size;
        combined.gpu_success = gpu_result.gpu_success;
        combined.gpu_integrity_passed = gpu_result.gpu_integrity_passed;
        combined.gpu_peak_vram_bytes = gpu_result.gpu_peak_vram_bytes;
        combined.gpu_memory_alloc_time_ms = gpu_result.gpu_memory_alloc_time_ms;
        combined.gpu_kernel_launch_overhead_ms =
            gpu_result.gpu_kernel_launch_overhead_ms;
        combined.pattern = pattern;

        // Compute comparison metrics
        compute_comparison_metrics(combined);

        all_results.push_back(combined);

        // Print progress indicator
        if (combined.gpu_success && combined.std_success) {
          std::cout << "GPU:" << combined.gpu_encode_time_ms << "ms "
                    << combined.gpu_throughput_mbps << "MB/s (" << std::showpos
                    << combined.speedup_ratio << "x" << std::noshowpos << ")\n";
        } else {
          std::cout << "FAILED\n";
        }

        // Allow GPU to cool between iterations
        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }

    free(h_input);
  }

  // Print results table
  std::cout << "\n\n";
  std::cout << "========================================\n";
  std::cout << "       DETAILED RESULTS\n";
  std::cout << "========================================\n\n";

  print_result_header();
  for (const auto &r : all_results) {
    print_result_row(r);
  }
  std::cout
      << "+---------+--------+------+---------------+---------------+----------"
         "-----+---------------+---------------+----------+--------------+\n";

  // Print resource utilization
  std::cout << "\n========================================\n";
  std::cout << "       RESOURCE UTILIZATION\n";
  std::cout << "========================================\n\n";

  print_resource_header();
  for (const auto &r : all_results) {
    print_resource_row(r);
  }
  std::cout << "+=========+==================+==================+=============="
               "====+\n";

  // Print summary
  print_comparison_summary(all_results);

  // Export to CSV
  export_results_csv(all_results, "zstd_comparison_results.csv");

  std::cout << "\nBenchmark complete!\n";
  return 0;
}
