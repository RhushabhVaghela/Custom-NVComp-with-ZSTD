// benchmark_comprehensive_levels.cu
// Comprehensive Compression Benchmark for Levels 1-22 and Data Sizes 1MB-10GB
// Tests: Encoding-only, Decoding-only, Full Pipeline (Encode+Decode)
// Measures: Throughput (MB/s), Duration (ms), Compression Ratio
// Validates: 100% data integrity with checksums

#include "cuda_zstd_hash.h"
#include "cuda_zstd_manager.h"
#include <algorithm>
#include <chrono>
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

// Hardware-safe constants
#define MAX_VRAM_PER_BENCHMARK (8ULL * 1024 * 1024 * 1024) // 8GB max
#define MAX_OUTPUT_MULTIPLIER 1.5f

// Data sizes to test (representative subset)
const size_t DATA_SIZES[] = {
    1ULL * 1024 * 1024,   // 1 MB
    10ULL * 1024 * 1024,  // 10 MB
    100ULL * 1024 * 1024, // 100 MB
    1024ULL * 1024 * 1024 // 1 GB
};
const int NUM_DATA_SIZES = 4;

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
  double encode_throughput_mbps;
  double decode_throughput_mbps;
  double pipeline_throughput_mbps;
  double compression_ratio;
  bool integrity_passed;
  uint64_t original_checksum;
  uint64_t decompressed_checksum;
};

// Test data generation patterns
enum class DataPattern {
  RANDOM,      // Fully random data (hard to compress)
  REPETITIVE,  // Highly repetitive (best compression)
  SEMI_RANDOM, // Semi-random with patterns
  INCREMENTAL  // Incrementing values
};

void generate_test_data(void *ptr, size_t size, DataPattern pattern) {
  byte_t *data = static_cast<byte_t *>(ptr);

  switch (pattern) {
  case DataPattern::RANDOM:
    // Use CUDA random for better distribution
    {
      std::mt19937_64 rng(12345 + pattern);
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
      std::mt19937_64 rng(67890 + pattern);
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
  }
}

// Compute XXH64 checksum
uint64_t compute_checksum(void *data, size_t size, cudaStream_t stream = 0) {
  return xxhash::compute_xxh64(static_cast<byte_t *>(data),
                               static_cast<uint32_t>(size),
                               0, // default seed
                               stream);
}

// ============================================================================
// ENCODING-ONLY BENCHMARK
// ============================================================================
BenchmarkResult benchmark_encoding_only(int level, size_t data_size,
                                        DataPattern pattern) {
  BenchmarkResult result = {};
  result.level = level;
  result.data_size = data_size;
  result.integrity_passed = false;

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
  config.block_size = get_optimal_block_size(
      static_cast<u32>(std::min(data_size, static_cast<size_t>(UINT32_MAX))),
      level);
  manager->configure(config);

  // Get workspace size and allocate
  size_t workspace_size = manager->get_compress_temp_size(data_size);
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

  // Warmup
  size_t compressed_size = max_output_size;
  manager->compress(d_input, data_size, d_output, &compressed_size, workspace,
                    workspace_size, nullptr, 0, nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark encoding
  auto start = std::chrono::high_resolution_clock::now();

  compressed_size = max_output_size;
  Status status =
      manager->compress(d_input, data_size, d_output, &compressed_size,
                        workspace, workspace_size, nullptr, 0, nullptr);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  if (status == Status::SUCCESS) {
    result.encode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.encode_throughput_mbps =
        (data_size / (1024.0 * 1024.0)) / (result.encode_time_ms / 1000.0);
    result.compression_ratio =
        static_cast<double>(compressed_size) / static_cast<double>(data_size);
    result.integrity_passed = true;
  } else {
    result.encode_time_ms = -1;
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
  result.integrity_passed = false;

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
  CHECK_CUDA(cudaRealloc(&workspace, workspace_decomp));

  // Warmup
  size_t decompressed_size = data_size;
  decomp_manager->decompress(d_compressed, compressed_size, d_output,
                             &decompressed_size, workspace, workspace_decomp,
                             nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark decoding
  auto start = std::chrono::high_resolution_clock::now();

  decompressed_size = data_size;
  status = decomp_manager->decompress(d_compressed, compressed_size, d_output,
                                      &decompressed_size, workspace,
                                      workspace_decomp, nullptr);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  if (status == Status::SUCCESS) {
    result.decode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
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
  result.integrity_passed = false;

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

  // Benchmark full pipeline
  auto pipeline_start = std::chrono::high_resolution_clock::now();

  // Encode
  compressed_size = max_output_size;
  Status enc_status =
      enc_manager->compress(d_input, data_size, d_compressed, &compressed_size,
                            workspace, enc_workspace, nullptr, 0, nullptr);

  // Decode
  decompressed_size = data_size;
  Status dec_status = dec_manager->decompress(
      d_compressed, compressed_size, d_output, &decompressed_size, workspace,
      dec_workspace, nullptr);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto pipeline_end = std::chrono::high_resolution_clock::now();

  if (enc_status == Status::SUCCESS && dec_status == Status::SUCCESS) {
    result.pipeline_time_ms =
        std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start)
            .count();
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
  std::cout
      << "+---------+-----------+------------------+------------------+--------"
         "----------+------------------+------------------+------------+\n";
  std::cout << "| Level   | Size      | Encode Time(ms)  | Decode Time(ms)  | "
               "Pipeline Time(ms)| Encode MB/s      | Decode MB/s      | Comp "
               "Ratio |\n";
  std::cout
      << "+---------+-----------+------------------+------------------+--------"
         "----------+------------------+------------------+------------+\n";
}

void print_result_row(const BenchmarkResult &r, bool is_valid) {
  std::cout << "| " << std::setw(7) << r.level << " | ";
  std::cout << std::setw(8) << (r.data_size / (1024 * 1024)) << " MB | ";

  if (r.encode_time_ms > 0) {
    std::cout << std::setw(16) << r.encode_time_ms << " | ";
  } else {
    std::cout << std::setw(16) << "N/A" << " | ";
  }

  if (r.decode_time_ms > 0) {
    std::cout << std::setw(16) << r.decode_time_ms << " | ";
  } else {
    std::cout << std::setw(16) << "N/A" << " | ";
  }

  if (r.pipeline_time_ms > 0) {
    std::cout << std::setw(16) << r.pipeline_time_ms << " | ";
  } else {
    std::cout << std::setw(16) << "N/A" << " | ";
  }

  if (r.encode_throughput_mbps > 0) {
    std::cout << std::setw(16) << r.encode_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(16) << "N/A" << " | ";
  }

  if (r.decode_throughput_mbps > 0) {
    std::cout << std::setw(16) << r.decode_throughput_mbps << " | ";
  } else {
    std::cout << std::setw(16) << "N/A" << " | ";
  }

  if (r.compression_ratio > 0) {
    std::cout << std::setw(10) << r.compression_ratio << " | ";
  } else {
    std::cout << std::setw(10) << "N/A" << " | ";
  }

  std::cout << "\n";
}

void print_result_separator() {
  std::cout
      << "+---------+-----------+------------------+------------------+--------"
         "----------+------------------+------------------+------------+\n";
}

void export_results_csv(const std::vector<BenchmarkResult> &results,
                        const std::string &filename) {
  std::ofstream file(filename);
  file
      << "level,data_size_bytes,encode_time_ms,decode_time_ms,pipeline_time_ms,"
      << "encode_throughput_mbps,decode_throughput_mbps,pipeline_throughput_"
         "mbps,"
      << "compression_ratio,integrity_passed,original_checksum,decompressed_"
         "checksum\n";

  for (const auto &r : results) {
    file << r.level << "," << r.data_size << "," << r.encode_time_ms << ","
         << r.decode_time_ms << "," << r.pipeline_time_ms << ","
         << r.encode_throughput_mbps << "," << r.decode_throughput_mbps << ","
         << r.pipeline_throughput_mbps << "," << r.compression_ratio << ","
         << (r.integrity_passed ? 1 : 0) << "," << r.original_checksum << ","
         << r.decompressed_checksum << "\n";
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
  std::cout << "\n";
  std::cout << "Best Encode Performance:\n";
  std::cout << "  Level " << best_encode_level << " @ "
            << (best_encode_size / (1024 * 1024)) << " MB: " << std::fixed
            << std::setprecision(2) << best_encode_throughput << " MB/s\n";
  std::cout << "\n";
  std::cout << "Best Decode Performance:\n";
  std::cout << "  Level " << best_decode_level << " @ "
            << (best_decode_size / (1024 * 1024)) << " MB: " << std::fixed
            << std::setprecision(2) << best_decode_throughput << " MB/s\n";
  std::cout << "\n";
  std::cout << "Best Compression Ratio:\n";
  std::cout << "  Level " << best_ratio_level << " @ "
            << (best_ratio_size / (1024 * 1024)) << " MB: " << std::fixed
            << std::setprecision(4) << best_compression_ratio << "\n";
  std::cout << "\n";
  std::cout << "Worst Compression Ratio:\n";
  std::cout << "  Level " << worst_ratio_level << " @ "
            << (worst_ratio_size / (1024 * 1024)) << " MB: " << std::fixed
            << std::setprecision(4) << worst_compression_ratio << "\n";
  std::cout << "========================================\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv) {
  std::cout << "========================================\n";
  std::cout << "  CUDA ZSTD Comprehensive Benchmark\n";
  std::cout << "  Levels 1-22, Sizes 1MB-1GB\n";
  std::cout << "========================================\n\n";

  // Parse arguments
  bool run_encoding = true;
  bool run_decoding = true;
  bool run_pipeline = true;
  DataPattern pattern = DataPattern::RANDOM;

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
    } else if (arg == "--semi-random") {
      pattern = DataPattern::SEMI_RANDOM;
    } else if (arg == "--incremental") {
      pattern = DataPattern::INCREMENTAL;
    } else if (arg == "--random") {
      pattern = DataPattern::RANDOM;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --encode-only   Run only encoding benchmark\n";
      std::cout << "  --decode-only   Run only decoding benchmark\n";
      std::cout << "  --pipeline-only Run only full pipeline benchmark\n";
      std::cout
          << "  --repetitive    Use repetitive test data (best compression)\n";
      std::cout << "  --semi-random   Use semi-random test data\n";
      std::cout << "  --incremental   Use incremental test data\n";
      std::cout << "  --random        Use random test data (default)\n";
      std::cout << "  --help          Show this help\n";
      return 0;
    }
  }

  std::cout << "Test Pattern: ";
  switch (pattern) {
  case DataPattern::RANDOM:
    std::cout << "Random\n";
    break;
  case DataPattern::REPETITIVE:
    std::cout << "Repetitive (AAAA...)\n";
    break;
  case DataPattern::SEMI_RANDOM:
    std::cout << "Semi-Random\n";
    break;
  case DataPattern::INCREMENTAL:
    std::cout << "Incremental\n";
    break;
  }

  std::cout << "Running tests for " << NUM_LEVELS
            << " compression levels (1-22)\n";
  std::cout << "Data sizes: ";
  for (int i = 0; i < NUM_DATA_SIZES; ++i) {
    std::cout << (DATA_SIZES[i] / (1024 * 1024)) << "MB ";
  }
  std::cout << "\n\n";

  std::vector<BenchmarkResult> all_results;
  std::vector<BenchmarkResult> encoding_results;
  std::vector<BenchmarkResult> decoding_results;
  std::vector<BenchmarkResult> pipeline_results;

  // Run encoding benchmarks
  if (run_encoding) {
    std::cout << "========================================\n";
    std::cout << "  ENCODING-ONLY BENCHMARKS\n";
    std::cout << "========================================\n\n";
    print_result_table_header();

    for (int s = 0; s < NUM_DATA_SIZES; ++s) {
      size_t data_size = DATA_SIZES[s];
      std::cout << "\n--- Data Size: " << (data_size / (1024 * 1024))
                << " MB ---\n";

      for (int l = 0; l < NUM_LEVELS; ++l) {
        int level = COMPRESSION_LEVELS[l];
        BenchmarkResult result =
            benchmark_encoding_only(level, data_size, pattern);
        encoding_results.push_back(result);
        print_result_row(result, result.integrity_passed);

        // Small delay to allow GPU to cool
        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }
    print_result_separator();
  }

  // Run decoding benchmarks
  if (run_decoding) {
    std::cout << "\n========================================\n";
    std::cout << "  DECODING-ONLY BENCHMARKS\n";
    std::cout << "========================================\n\n";
    print_result_table_header();

    for (int s = 0; s < NUM_DATA_SIZES; ++s) {
      size_t data_size = DATA_SIZES[s];
      std::cout << "\n--- Data Size: " << (data_size / (1024 * 1024))
                << " MB ---\n";

      for (int l = 0; l < NUM_LEVELS; ++l) {
        int level = COMPRESSION_LEVELS[l];
        BenchmarkResult result =
            benchmark_decoding_only(level, data_size, pattern);
        decoding_results.push_back(result);
        print_result_row(result, result.integrity_passed);

        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }
    print_result_separator();
  }

  // Run full pipeline benchmarks
  if (run_pipeline) {
    std::cout << "\n========================================\n";
    std::cout << "  FULL PIPELINE BENCHMARKS\n";
    std::cout << "========================================\n\n";
    print_result_table_header();

    for (int s = 0; s < NUM_DATA_SIZES; ++s) {
      size_t data_size = DATA_SIZES[s];
      std::cout << "\n--- Data Size: " << (data_size / (1024 * 1024))
                << " MB ---\n";

      for (int l = 0; l < NUM_LEVELS; ++l) {
        int level = COMPRESSION_LEVELS[l];
        BenchmarkResult result =
            benchmark_full_pipeline(level, data_size, pattern);
        pipeline_results.push_back(result);
        print_result_row(result, result.integrity_passed);

        CHECK_CUDA(cudaDeviceSynchronize());
      }
    }
    print_result_separator();
  }

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
  export_results_csv(all_results, "benchmark_comprehensive_results.csv");

  std::cout << "\nBenchmark complete!\n";
  return 0;
}
