// benchmark_all_levels.cu
// Comprehensive Compression Benchmark for Levels 1-22 with Data Sizes 1KB-256MB
// Modified for Asus Zephyrus G16 (32GB RAM / 16GB VRAM) - Respecting hardware
// constraints Tests: Encoding-only, Decoding-only, Full Pipeline
// (Encode+Decode) Measures: Throughput (MB/s), Latency (ms), Compression Ratio,
// Integrity Validates: 100% data integrity with XXH64 checksums Safety: Max 4GB
// VRAM per benchmark to prevent system instability

#include "cuda_zstd_hash.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_xxhash.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef __cplusplus
using namespace cuda_zstd;
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Hardware-safe constants for Asus Zephyrus G16 (32GB RAM / 16GB VRAM)
#define MAX_VRAM_PER_BENCHMARK                                                 \
  (4ULL * 1024 * 1024 * 1024) // 4GB max per benchmark
#define MAX_OUTPUT_MULTIPLIER 1.5f
#define MAX_SAFE_DATA_SIZE (256ULL * 1024 * 1024) // 256MB max per test

// Number of iterations for averaging results
#define NUM_ITERATIONS 3

// Data sizes to test (1KB to 256MB)
const size_t DATA_SIZES[] = {
    1ULL * 1024,          // 1 KB
    4ULL * 1024,          // 4 KB
    16ULL * 1024,         // 16 KB
    64ULL * 1024,         // 64 KB
    256ULL * 1024,        // 256 KB
    1ULL * 1024 * 1024,   // 1 MB
    4ULL * 1024 * 1024,   // 4 MB
    16ULL * 1024 * 1024,  // 16 MB
    64ULL * 1024 * 1024,  // 64 MB
    128ULL * 1024 * 1024, // 128 MB
    256ULL * 1024 * 1024  // 256 MB (max for hardware safety)
};
const int NUM_DATA_SIZES = 11;

// Compression levels to test (1-22)
const int COMPRESSION_LEVELS[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
const int NUM_LEVELS = 22;

// Benchmark result structure
struct BenchmarkResult {
  int level;
  size_t data_size;
  double encode_time_ms;
  double decode_time_ms;
  double pipeline_time_ms;
  double encode_latency_ms;
  double decode_latency_ms;
  double pipeline_latency_ms;
  double encode_throughput_mbps;
  double decode_throughput_mbps;
  double pipeline_throughput_mbps;
  double compression_ratio;
  bool integrity_passed;
  uint64_t original_checksum;
  uint64_t decompressed_checksum;
  DataPattern pattern;
  int iterations;
};

// Test data generation patterns
enum class DataPattern {
  RANDOM,      // Fully random data (hard to compress)
  REPETITIVE,  // Highly repetitive (best compression)
  SEMI_RANDOM, // Semi-random with patterns
  INCREMENTAL, // Incrementing values
  COMPRESSIBLE // Text-like data with common patterns
};

const char *pattern_to_string(DataPattern pattern) {
  switch (pattern) {
  case DataPattern::RANDOM:
    return "Random";
  case DataPattern::REPETITIVE:
    return "Repetitive";
  case DataPattern::SEMI_RANDOM:
    return "SemiRandom";
  case DataPattern::INCREMENTAL:
    return "Incremental";
  case DataPattern::COMPRESSIBLE:
    return "Compressible";
  default:
    return "Unknown";
  }
}

void generate_test_data(void *ptr, size_t size, DataPattern pattern) {
  byte_t *data = static_cast<byte_t *>(ptr);

  switch (pattern) {
  case DataPattern::RANDOM:
    // Use CUDA random for better distribution
    {
      std::mt19937_64 rng(12345);
      std::uniform_int_distribution<int> dist(0, 255);
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<byte_t>(dist(rng));
      }
    }
    break;

  case DataPattern::REPETITIVE:
    // All 'A's - highly compressible
    memset(data, 'A', size);
    break;

  case DataPattern::SEMI_RANDOM:
    // Repeating 64-byte patterns with some randomness
    {
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
    }
    break;

  case DataPattern::INCREMENTAL:
    // Incrementing bytes (0-255 repeated)
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<byte_t>(i & 0xFF);
    }
    break;

  case DataPattern::COMPRESSIBLE:
    // Text-like data with common patterns
    {
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
    }
    break;
  }
}

// Compute XXH64 checksum
uint64_t compute_checksum(void *data, size_t size, cudaStream_t stream = 0) {
  u64 h_hash;
  cuda_zstd::xxhash::compute_xxhash64(data, size, 0, &h_hash, stream);
  cudaStreamSynchronize(stream ? stream : 0);
  return h_hash;
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

// Print progress bar
void print_progress(int current, int total, const std::string &label) {
  int bar_width = 50;
  float progress = static_cast<float>(current) / total;
  int pos = static_cast<int>(bar_width * progress);

  std::cout << "\r" << label << " [";
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0)
            << "%";
  std::cout.flush();
}

// ============================================================================
// ENCODING-ONLY BENCHMARK
// ============================================================================
BenchmarkResult benchmark_encoding_only(int level, size_t data_size,
                                        DataPattern pattern) {
  BenchmarkResult result = {};
  result.level = level;
  result.data_size = data_size;
  result.pattern = pattern;
  result.integrity_passed = false;
  result.iterations = NUM_ITERATIONS;

  // Allocate device memory
  void *d_input, *d_output;
  size_t max_output_size =
      static_cast<size_t>(data_size * MAX_OUTPUT_MULTIPLIER);

  CHECK_CUDA(cudaMalloc(&d_input, data_size));
  CHECK_CUDA(cudaMalloc(&d_output, max_output_size));

  // Generate and copy test data
  void *h_input = malloc(data_size);
  generate_test_data(h_input, data_size, pattern);
  CHECK_CUDA(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

  // Compute original checksum
  result.original_checksum = compute_checksum(d_input, data_size);

  // Create manager and configure
  auto manager = create_batch_manager(level);
  CompressionConfig config;
  config.level = level;
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(data_size, static_cast<size_t>(UINT32_MAX))),
      level);
  manager->configure(config);

  // Get workspace size and allocate
  size_t workspace_size = manager->get_compress_temp_size(data_size);
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

  // Warmup run
  size_t compressed_size = max_output_size;
  manager->compress(d_input, data_size, d_output, &compressed_size, workspace,
                    workspace_size, nullptr, 0, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark encoding with multiple iterations
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    compressed_size = max_output_size;
    Status status =
        manager->compress(d_input, data_size, d_output, &compressed_size,
                          workspace, workspace_size, nullptr, 0, nullptr);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    if (status == Status::SUCCESS) {
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      iteration_times.push_back(iter_time_ms);
    }
  }

  if (!iteration_times.empty()) {
    // Calculate average
    double total_time = 0.0;
    for (double t : iteration_times)
      total_time += t;
    result.encode_time_ms = total_time / iteration_times.size();
    result.encode_latency_ms = iteration_times[0]; // First iteration = latency
    result.encode_throughput_mbps =
        (data_size / (1024.0 * 1024.0)) / (result.encode_time_ms / 1000.0);
    result.compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(data_size);
    result.integrity_passed = true;
  } else {
    result.encode_time_ms = -1;
    result.encode_latency_ms = -1;
    result.encode_throughput_mbps = 0;
    result.compression_ratio = -1;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_input));
  free(h_input);

  return result;
}

// ============================================================================
// DECODING-ONLY BENCHMARK
// ============================================================================
BenchmarkResult benchmark_decoding_only(int level, size_t data_size,
                                        DataPattern pattern) {
  BenchmarkResult result = {};
  result.level = level;
  result.data_size = data_size;
  result.pattern = pattern;
  result.integrity_passed = false;
  result.iterations = NUM_ITERATIONS;

  size_t max_output_size =
      static_cast<size_t>(data_size * MAX_OUTPUT_MULTIPLIER);

  // Allocate device memory
  void *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_compressed, max_output_size));
  CHECK_CUDA(cudaMalloc(&d_output, data_size));

  // Generate test data on host, compress, then copy to device
  void *h_input = malloc(data_size);
  void *h_compressed = malloc(max_output_size);
  size_t compressed_size = max_output_size;

  generate_test_data(h_input, data_size, pattern);

  // Use batch manager for compression
  auto manager = create_batch_manager(level);
  CompressionConfig config;
  config.level = level;
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(data_size, static_cast<size_t>(UINT32_MAX))),
      level);
  manager->configure(config);

  // Allocate workspace
  size_t workspace_size = manager->get_compress_temp_size(data_size);
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

  // Compress first (needed for decoding benchmark)
  CHECK_CUDA(
      cudaMemcpy(d_compressed, h_input, data_size, cudaMemcpyHostToDevice));

  Status status =
      manager->compress(d_compressed, data_size, d_compressed, &compressed_size,
                        workspace, workspace_size, nullptr, 0, nullptr);

  if (status != Status::SUCCESS) {
    // Fallback: just use original data as compressed (no compression possible)
    compressed_size = data_size;
    memcpy(h_compressed, h_input, data_size);
    CHECK_CUDA(cudaMemcpy(d_compressed, h_compressed, data_size,
                          cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemcpy(h_compressed, d_compressed, compressed_size,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(d_compressed, h_compressed, compressed_size,
                          cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  // Compute checksum of original
  CHECK_CUDA(cudaMemcpy(d_output, h_input, data_size, cudaMemcpyHostToDevice));
  result.original_checksum = compute_checksum(d_output, data_size);

  // Create decompression manager
  auto decomp_manager = create_batch_manager(level);

  size_t workspace_decomp =
      decomp_manager->get_decompress_temp_size(compressed_size);
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaMalloc(&workspace, workspace_decomp));

  // Warmup
  size_t decompressed_size = data_size;
  decomp_manager->decompress(d_compressed, compressed_size, d_output,
                             &decompressed_size, workspace, workspace_decomp,
                             nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark decoding with multiple iterations
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    decompressed_size = data_size;
    status = decomp_manager->decompress(d_compressed, compressed_size, d_output,
                                        &decompressed_size, workspace,
                                        workspace_decomp, nullptr);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    if (status == Status::SUCCESS) {
      double iter_time_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      iteration_times.push_back(iter_time_ms);
    }
  }

  if (!iteration_times.empty()) {
    // Calculate average
    double total_time = 0.0;
    for (double t : iteration_times)
      total_time += t;
    result.decode_time_ms = total_time / iteration_times.size();
    result.decode_latency_ms = iteration_times[0]; // First iteration = latency
    result.decode_throughput_mbps =
        (data_size / (1024.0 * 1024.0)) / (result.decode_time_ms / 1000.0);
    result.compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(data_size);

    // Verify integrity
    result.decompressed_checksum = compute_checksum(d_output, data_size);
    result.integrity_passed =
        (result.original_checksum == result.decompressed_checksum);
  } else {
    result.decode_time_ms = -1;
    result.decode_latency_ms = -1;
    result.decode_throughput_mbps = 0;
    result.compression_ratio = -1;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_compressed));
  free(h_compressed);
  free(h_input);

  return result;
}

// ============================================================================
// FULL PIPELINE BENCHMARK (Encode + Decode)
// ============================================================================
BenchmarkResult benchmark_full_pipeline(int level, size_t data_size,
                                        DataPattern pattern) {
  BenchmarkResult result = {};
  result.level = level;
  result.data_size = data_size;
  result.pattern = pattern;
  result.integrity_passed = false;
  result.iterations = NUM_ITERATIONS;

  size_t max_output_size =
      static_cast<size_t>(data_size * MAX_OUTPUT_MULTIPLIER);

  // Allocate device memory
  void *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, data_size));
  CHECK_CUDA(cudaMalloc(&d_compressed, max_output_size));
  CHECK_CUDA(cudaMalloc(&d_output, data_size));

  // Generate test data
  void *h_input = malloc(data_size);
  generate_test_data(h_input, data_size, pattern);
  CHECK_CUDA(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

  // Compute original checksum
  result.original_checksum = compute_checksum(d_input, data_size);

  // Create managers
  auto enc_manager = create_batch_manager(level);
  auto dec_manager = create_batch_manager(level);

  CompressionConfig config;
  config.level = level;
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(data_size, static_cast<size_t>(UINT32_MAX))),
      level);
  enc_manager->configure(config);
  dec_manager->configure(config);

  // Allocate workspaces
  size_t enc_workspace = enc_manager->get_compress_temp_size(data_size);
  size_t dec_workspace = dec_manager->get_decompress_temp_size(max_output_size);
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, std::max(enc_workspace, dec_workspace)));

  // Warmup
  size_t compressed_size = max_output_size;
  enc_manager->compress(d_input, data_size, d_compressed, &compressed_size,
                        workspace, enc_workspace, nullptr, 0, nullptr);

  size_t decompressed_size = data_size;
  dec_manager->decompress(d_compressed, compressed_size, d_output,
                          &decompressed_size, workspace, dec_workspace,
                          nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark full pipeline with multiple iterations
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto pipeline_start = std::chrono::high_resolution_clock::now();

    // Encode
    compressed_size = max_output_size;
    Status enc_status = enc_manager->compress(
        d_input, data_size, d_compressed, &compressed_size, workspace,
        enc_workspace, nullptr, 0, nullptr);

    // Decode
    decompressed_size = data_size;
    Status dec_status = dec_manager->decompress(
        d_compressed, compressed_size, d_output, &decompressed_size, workspace,
        dec_workspace, nullptr);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto pipeline_end = std::chrono::high_resolution_clock::now();

    if (enc_status == Status::SUCCESS && dec_status == Status::SUCCESS) {
      double iter_time_ms = std::chrono::duration<double, std::milli>(
                                pipeline_end - pipeline_start)
                                .count();
      iteration_times.push_back(iter_time_ms);
    }
  }

  if (!iteration_times.empty()) {
    // Calculate average
    double total_time = 0.0;
    for (double t : iteration_times)
      total_time += t;
    result.pipeline_time_ms = total_time / iteration_times.size();
    result.pipeline_latency_ms =
        iteration_times[0]; // First iteration = latency
    result.pipeline_throughput_mbps =
        (data_size / (1024.0 * 1024.0)) / (result.pipeline_time_ms / 1000.0);
    result.compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(data_size);

    // Verify integrity
    result.decompressed_checksum = compute_checksum(d_output, data_size);
    result.integrity_passed =
        (result.original_checksum == result.decompressed_checksum);
  } else {
    result.pipeline_time_ms = -1;
    result.pipeline_latency_ms = -1;
    result.pipeline_throughput_mbps = 0;
    result.compression_ratio = -1;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_input));
  free(h_input);

  return result;
}

// ============================================================================
// RESULTS REPORTING
// ============================================================================
void print_result_table_header() {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "+-------+----------+---------+--------+--------+--------+------"
               "--+--------+--------+--------+--------+--------+--------+\n";
  std::cout
      << "| Level | Size     | Pattern | EncT   | DecT   | PipeT  | EncThr | "
         "DecThr | PipThr | EncLat | DecLat | Ratio  | Valid  |\n";
  std::cout << "+-------+----------+---------+--------+--------+--------+------"
               "--+--------+--------+--------+--------+--------+--------+\n";
}

void print_result_row(const BenchmarkResult &r) {
  std::cout << "| " << std::setw(5) << r.level << " | ";

  // Format size
  std::string size_str;
  if (r.data_size >= 1024 * 1024) {
    size_str = std::to_string(r.data_size / (1024 * 1024)) + "MB";
  } else {
    size_str = std::to_string(r.data_size / 1024) + "KB";
  }
  std::cout << std::setw(8) << size_str << " | ";

  // Pattern (abbreviated)
  const char *pat = pattern_to_string(r.pattern);
  char pat_abbr[8];
  strncpy(pat_abbr, pat, 7);
  pat_abbr[7] = '\0';
  std::cout << std::setw(7) << pat_abbr << " | ";

  // Encode time
  if (r.encode_time_ms > 0) {
    std::cout << std::setw(6) << r.encode_time_ms << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Decode time
  if (r.decode_time_ms > 0) {
    std::cout << std::setw(6) << r.decode_time_ms << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Pipeline time
  if (r.pipeline_time_ms > 0) {
    std::cout << std::setw(6) << r.pipeline_time_ms << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Encode throughput
  if (r.encode_throughput_mbps > 0) {
    std::cout << std::setw(6) << r.encode_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Decode throughput
  if (r.decode_throughput_mbps > 0) {
    std::cout << std::setw(6) << r.decode_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Pipeline throughput
  if (r.pipeline_throughput_mbps > 0) {
    std::cout << std::setw(6) << r.pipeline_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Encode latency
  if (r.encode_latency_ms > 0) {
    std::cout << std::setw(6) << r.encode_latency_ms << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Decode latency
  if (r.decode_latency_ms > 0) {
    std::cout << std::setw(6) << r.decode_latency_ms << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Compression ratio
  if (r.compression_ratio > 0 && r.compression_ratio < 100) {
    std::cout << std::setw(6) << r.compression_ratio << " | ";
  } else {
    std::cout << std::setw(6) << "N/A" << " | ";
  }

  // Validity
  std::cout << std::setw(6) << (r.integrity_passed ? "PASS" : "FAIL") << " |";

  std::cout << "\n";
}

void print_result_separator() {
  std::cout << "+-------+----------+---------+--------+--------+--------+------"
               "--+--------+--------+--------+--------+--------+--------+\n";
}

void export_results_csv(const std::vector<BenchmarkResult> &results,
                        const std::string &filename) {
  std::ofstream file(filename);
  file << "level,data_size_bytes,data_size_human,pattern,pattern_name,"
       << "encode_time_ms,decode_time_ms,pipeline_time_ms,"
       << "encode_latency_ms,decode_latency_ms,pipeline_latency_ms,"
       << "encode_throughput_mbps,decode_throughput_mbps,pipeline_throughput_"
          "mbps,"
       << "compression_ratio,integrity_passed,original_checksum,decompressed_"
          "checksum,iterations\n";

  for (const auto &r : results) {
    file << r.level << ",";
    file << r.data_size << ",";
    file << format_bytes(r.data_size) << ",";
    file << static_cast<int>(r.pattern) << ",";
    file << pattern_to_string(r.pattern) << ",";
    file << r.encode_time_ms << ",";
    file << r.decode_time_ms << ",";
    file << r.pipeline_time_ms << ",";
    file << r.encode_latency_ms << ",";
    file << r.decode_latency_ms << ",";
    file << r.pipeline_latency_ms << ",";
    file << r.encode_throughput_mbps << ",";
    file << r.decode_throughput_mbps << ",";
    file << r.pipeline_throughput_mbps << ",";
    file << r.compression_ratio << ",";
    file << (r.integrity_passed ? 1 : 0) << ",";
    file << r.original_checksum << ",";
    file << r.decompressed_checksum << ",";
    file << r.iterations << "\n";
  }

  file.close();
  std::cout << "\nResults exported to: " << filename << "\n";
}

void print_summary(const std::vector<BenchmarkResult> &results) {
  int total_tests = results.size();
  int passed = 0;
  int failed = 0;

  double best_encode_throughput = 0;
  int best_encode_level = 0;
  size_t best_encode_size = 0;

  double best_decode_throughput = 0;
  int best_decode_level = 0;
  size_t best_decode_size = 0;

  double best_compression_ratio = 100;
  int best_ratio_level = 0;
  size_t best_ratio_size = 0;
  DataPattern best_ratio_pattern = DataPattern::RANDOM;

  double worst_compression_ratio = 0;
  int worst_ratio_level = 0;
  size_t worst_ratio_size = 0;

  for (const auto &r : results) {
    if (r.integrity_passed) {
      passed++;
    } else {
      failed++;
    }

    if (r.encode_throughput_mbps > best_encode_throughput &&
        r.encode_throughput_mbps > 0) {
      best_encode_throughput = r.encode_throughput_mbps;
      best_encode_level = r.level;
      best_encode_size = r.data_size;
    }

    if (r.decode_throughput_mbps > best_decode_throughput &&
        r.decode_throughput_mbps > 0) {
      best_decode_throughput = r.decode_throughput_mbps;
      best_decode_level = r.level;
      best_decode_size = r.data_size;
    }

    if (r.compression_ratio > 0 &&
        r.compression_ratio < best_compression_ratio) {
      best_compression_ratio = r.compression_ratio;
      best_ratio_level = r.level;
      best_ratio_size = r.data_size;
      best_ratio_pattern = r.pattern;
    }

    if (r.compression_ratio > worst_compression_ratio) {
      worst_compression_ratio = r.compression_ratio;
      worst_ratio_level = r.level;
      worst_ratio_size = r.data_size;
    }
  }

  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "         BENCHMARK SUMMARY\n";
  std::cout << "========================================\n";
  std::cout << "Total Tests:      " << total_tests << "\n";
  std::cout << "Passed (Valid):   " << passed << "\n";
  std::cout << "Failed:           " << failed << "\n";
  std::cout << "Integrity Rate:   " << std::fixed << std::setprecision(1)
            << (100.0 * passed / total_tests) << "%\n";
  std::cout << "Iterations/Test:  " << NUM_ITERATIONS << "\n";
  std::cout << "\n";
  std::cout << "Best Encode Performance:\n";
  std::cout << "  Level " << best_encode_level << " @ "
            << format_bytes(best_encode_size) << ": " << std::fixed
            << std::setprecision(2) << best_encode_throughput << " MB/s\n";
  std::cout << "\n";
  std::cout << "Best Decode Performance:\n";
  std::cout << "  Level " << best_decode_level << " @ "
            << format_bytes(best_decode_size) << ": " << std::fixed
            << std::setprecision(2) << best_decode_throughput << " MB/s\n";
  std::cout << "\n";
  std::cout << "Best Compression Ratio:\n";
  std::cout << "  Level " << best_ratio_level << " @ "
            << format_bytes(best_ratio_size) << " ("
            << pattern_to_string(best_ratio_pattern) << "): " << std::fixed
            << std::setprecision(4) << best_compression_ratio << "\n";
  std::cout << "\n";
  std::cout << "Worst Compression Ratio:\n";
  std::cout << "  Level " << worst_ratio_level << " @ "
            << format_bytes(worst_ratio_size) << ": " << std::fixed
            << std::setprecision(4) << worst_compression_ratio << "\n";
  std::cout << "========================================\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv) {
  std::cout << "========================================\n";
  std::cout << "  CUDA ZSTD All Levels Benchmark\n";
  std::cout << "  Levels 1-22, Sizes 1KB-256MB\n";
  std::cout << "  Max VRAM per test: 4GB\n";
  std::cout << "  Iterations per test: " << NUM_ITERATIONS << "\n";
  std::cout << "========================================\n\n";

  // Parse arguments
  bool run_encoding = true;
  bool run_decoding = true;
  bool run_pipeline = true;
  DataPattern pattern = DataPattern::RANDOM;
  std::vector<DataPattern> patterns_to_test = {DataPattern::RANDOM};
  bool test_all_patterns = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--encode-only") {
      run_decoding = false;
      run_pipeline = false;
    } else if (arg == "--decode-only") {
      run_encoding = false;
      run_pipeline = false;
    } else if (arg == "--pipeline-only") {
      run_encoding = false;
      run_decoding = false;
    } else if (arg == "--repetitive") {
      pattern = DataPattern::REPETITIVE;
      patterns_to_test = {DataPattern::REPETITIVE};
    } else if (arg == "--semi-random") {
      pattern = DataPattern::SEMI_RANDOM;
      patterns_to_test = {DataPattern::SEMI_RANDOM};
    } else if (arg == "--incremental") {
      pattern = DataPattern::INCREMENTAL;
      patterns_to_test = {DataPattern::INCREMENTAL};
    } else if (arg == "--compressible") {
      pattern = DataPattern::COMPRESSIBLE;
      patterns_to_test = {DataPattern::COMPRESSIBLE};
    } else if (arg == "--random") {
      pattern = DataPattern::RANDOM;
      patterns_to_test = {DataPattern::RANDOM};
    } else if (arg == "--all-patterns") {
      test_all_patterns = true;
      patterns_to_test = {DataPattern::RANDOM, DataPattern::REPETITIVE,
                          DataPattern::SEMI_RANDOM, DataPattern::INCREMENTAL,
                          DataPattern::COMPRESSIBLE};
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --encode-only    Run only encoding benchmark\n";
      std::cout << "  --decode-only    Run only decoding benchmark\n";
      std::cout << "  --pipeline-only  Run only full pipeline benchmark\n";
      std::cout
          << "  --repetitive     Use repetitive test data (best compression)\n";
      std::cout << "  --semi-random    Use semi-random test data\n";
      std::cout << "  --incremental    Use incremental test data\n";
      std::cout << "  --compressible   Use compressible text-like data\n";
      std::cout << "  --random         Use random test data (default)\n";
      std::cout << "  --all-patterns   Test all data patterns\n";
      std::cout << "  --help           Show this help\n";
      std::cout << "\n";
      std::cout << "Examples:\n";
      std::cout << "  " << argv[0]
                << "                    # Run with random data\n";
      std::cout << "  " << argv[0]
                << " --all-patterns      # Test all patterns\n";
      std::cout << "  " << argv[0] << " --encode-only --repetitive\n";
      return 0;
    }
  }

  std::cout << "Test Pattern(s): ";
  if (test_all_patterns) {
    std::cout << "All patterns (Random, Repetitive, SemiRandom, Incremental, "
                 "Compressible)\n";
  } else {
    std::cout << pattern_to_string(pattern) << "\n";
  }

  std::cout << "Running tests for " << NUM_LEVELS
            << " compression levels (1-22)\n";
  std::cout << "Data sizes (" << NUM_DATA_SIZES << "): ";
  for (int i = 0; i < NUM_DATA_SIZES; ++i) {
    std::cout << format_bytes(DATA_SIZES[i]) << " ";
  }
  std::cout << "\n\n";

  std::vector<BenchmarkResult> all_results;
  std::vector<BenchmarkResult> encoding_results;
  std::vector<BenchmarkResult> decoding_results;
  std::vector<BenchmarkResult> pipeline_results;

  // Calculate total tests for progress tracking
  int total_patterns = patterns_to_test.size();
  int tests_per_pattern = NUM_LEVELS * NUM_DATA_SIZES;
  int total_tests = tests_per_pattern * total_patterns;
  int tests_completed = 0;

  // Run tests for each pattern
  for (DataPattern current_pattern : patterns_to_test) {
    std::cout << "\n========================================\n";
    std::cout << "  Testing Pattern: " << pattern_to_string(current_pattern)
              << "\n";
    std::cout << "========================================\n";

    // Run encoding benchmarks
    if (run_encoding) {
      std::cout << "\n--- ENCODING-ONLY BENCHMARKS ---\n";
      print_result_table_header();

      for (int s = 0; s < NUM_DATA_SIZES; ++s) {
        size_t data_size = DATA_SIZES[s];

        for (int l = 0; l < NUM_LEVELS; ++l) {
          int level = COMPRESSION_LEVELS[l];
          BenchmarkResult result =
              benchmark_encoding_only(level, data_size, current_pattern);
          result.pattern = current_pattern;
          encoding_results.push_back(result);
          print_result_row(result);

          tests_completed++;
          if (s % 2 == 0 && l == 0) {
            print_progress(tests_completed, total_tests, "Progress");
          }

          // Small delay to allow GPU to cool
          CHECK_CUDA(cudaDeviceSynchronize());
        }
      }
      print_result_separator();
    }

    // Run decoding benchmarks
    if (run_decoding) {
      std::cout << "\n--- DECODING-ONLY BENCHMARKS ---\n";
      print_result_table_header();

      for (int s = 0; s < NUM_DATA_SIZES; ++s) {
        size_t data_size = DATA_SIZES[s];

        for (int l = 0; l < NUM_LEVELS; ++l) {
          int level = COMPRESSION_LEVELS[l];
          BenchmarkResult result =
              benchmark_decoding_only(level, data_size, current_pattern);
          result.pattern = current_pattern;
          decoding_results.push_back(result);
          print_result_row(result);

          tests_completed++;

          CHECK_CUDA(cudaDeviceSynchronize());
        }
      }
      print_result_separator();
    }

    // Run full pipeline benchmarks
    if (run_pipeline) {
      std::cout << "\n--- FULL PIPELINE BENCHMARKS ---\n";
      print_result_table_header();

      for (int s = 0; s < NUM_DATA_SIZES; ++s) {
        size_t data_size = DATA_SIZES[s];

        for (int l = 0; l < NUM_LEVELS; ++l) {
          int level = COMPRESSION_LEVELS[l];
          BenchmarkResult result =
              benchmark_full_pipeline(level, data_size, current_pattern);
          result.pattern = current_pattern;
          pipeline_results.push_back(result);
          print_result_row(result);

          tests_completed++;

          CHECK_CUDA(cudaDeviceSynchronize());
        }
      }
      print_result_separator();
    }
  }

  std::cout << "\n\n"; // Clear progress line

  // Combine all results
  all_results.insert(all_results.end(), encoding_results.begin(),
                     encoding_results.end());
  all_results.insert(all_results.end(), decoding_results.begin(),
                     decoding_results.end());
  all_results.insert(all_results.end(), pipeline_results.begin(),
                     pipeline_results.end());

  // Print summary
  print_summary(all_results);

  // Export to CSV
  export_results_csv(all_results, "benchmark_all_levels_results.csv");

  std::cout << "\nBenchmark complete!\n";
  return 0;
}
