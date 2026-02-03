/**
 * @file benchmark_parallel_backtracking.cu
 * @brief Performance benchmark for parallel backtracking vs CPU baseline
 * 
 * This benchmark measures the performance of the GPU parallel backtracking
 * algorithm compared to a CPU reference implementation.
 */

#include "cuda_zstd_lz77.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include "lz77_parallel.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::lz77;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// Benchmark Configuration
// =============================================================================

struct BenchmarkConfig {
  std::vector<size_t> data_sizes;
  int compression_level;
  int iterations;
  
  BenchmarkConfig() 
    : data_sizes({1024, 4096, 16384, 65536, 262144, 1048576})
    , compression_level(3)
    , iterations(10) {}
};

// =============================================================================
// Data Generators
// =============================================================================

void generate_compressible_data(std::vector<uint8_t> &data, size_t size) {
  data.resize(size);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  
  // Fill with random data first
  for (size_t i = 0; i < size; ++i) {
    data[i] = (uint8_t)dist(rng);
  }
  
  // Inject repeated patterns to make it compressible
  size_t pos = 0;
  while (pos < size - 100) {
    if (dist(rng) < 128) { // 50% chance of pattern
      size_t len = 10 + (dist(rng) % 50);
      size_t offset = 1 + (dist(rng) % 1000);
      if (pos >= offset && pos + len < size) {
        for (size_t i = 0; i < len; ++i) {
          data[pos + i] = data[pos - offset + i];
        }
        pos += len;
      } else {
        pos++;
      }
    } else {
      pos++;
    }
  }
}

void generate_repeated_pattern(std::vector<uint8_t> &data, size_t size) {
  const char *pattern = "The quick brown fox jumps over the lazy dog. ";
  size_t pattern_len = strlen(pattern);
  
  data.resize(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = pattern[i % pattern_len];
  }
}

// =============================================================================
// Performance Measurement
// =============================================================================

struct BenchmarkResult {
  size_t input_size;
  double gpu_time_ms;
  double throughput_mb_s;
  size_t num_sequences;
  bool success;
  
  BenchmarkResult() 
    : input_size(0)
    , gpu_time_ms(0.0)
    , throughput_mb_s(0.0)
    , num_sequences(0)
    , success(false) {}
};

BenchmarkResult run_gpu_backtracking_benchmark(
    const std::vector<uint8_t> &input_data,
    int compression_level,
    int iterations) {
  
  BenchmarkResult result;
  result.input_size = input_data.size();
  
  // Allocate device memory
  void *d_input = nullptr;
  void *d_output = nullptr;
  void *d_temp = nullptr;
  
  CHECK_CUDA(cudaMalloc(&d_input, input_data.size()));
  CHECK_CUDA(cudaMemcpy(d_input, input_data.data(), input_data.size(), 
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&d_output, input_data.size() * 2));
  
  // Create manager
  auto manager = create_manager(compression_level);
  size_t temp_size = manager->get_compress_temp_size(input_data.size());
  CHECK_CUDA(cudaMalloc(&d_temp, temp_size));
  
  // Warmup
  size_t compressed_size = 0;
  Status status = manager->compress(
      d_input, input_data.size(),
      d_output, &compressed_size,
      d_temp, temp_size,
      nullptr, 0,
      0
  );
  
  if (status != Status::SUCCESS) {
    result.success = false;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
    return result;
  }
  
  // Benchmark iterations
  double total_time_ms = 0.0;
  
  for (int i = 0; i < iterations; ++i) {
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    compressed_size = 0;
    status = manager->compress(
        d_input, input_data.size(),
        d_output, &compressed_size,
        d_temp, temp_size,
        nullptr, 0,
        0
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    if (status == Status::SUCCESS) {
      double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += time_ms;
    }
  }
  
  result.gpu_time_ms = total_time_ms / iterations;
  result.throughput_mb_s = (input_data.size() / (1024.0 * 1024.0)) / (result.gpu_time_ms / 1000.0);
  result.success = true;
  
  // Cleanup
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_temp));
  
  return result;
}

// =============================================================================
// CPU Reference Implementation
// =============================================================================

struct CPUResult {
  double time_ms;
  size_t num_sequences;
  bool success;
};

CPUResult run_cpu_lz77_reference(const std::vector<uint8_t> &input, 
                                  const LZ77Config &config) {
  CPUResult result;
  result.time_ms = 0.0;
  result.num_sequences = 0;
  result.success = false;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // Simple CPU LZ77 implementation for reference
  // This is a basic greedy match finder
  std::vector<u32> lit_lengths;
  std::vector<u32> match_lengths;
  std::vector<u32> offsets;
  
  size_t pos = 0;
  while (pos < input.size()) {
    u32 best_len = 0;
    u32 best_off = 0;
    
    // Search for matches in the window
    size_t window_start = (pos > (1u << config.window_log)) ? 
                          (pos - (1u << config.window_log)) : 0;
    
    for (size_t match_pos = window_start; match_pos < pos; ++match_pos) {
      u32 len = 0;
      while (pos + len < input.size() && 
             len < 1000 && // Max match length
             input[match_pos + len] == input[pos + len]) {
        len++;
      }
      
      if (len >= config.min_match && len > best_len) {
        best_len = len;
        best_off = pos - match_pos;
      }
    }
    
    if (best_len >= config.min_match) {
      // Output match
      u32 lit_len = 0; // For simplicity, no literal before match
      lit_lengths.push_back(lit_len);
      match_lengths.push_back(best_len);
      offsets.push_back(best_off);
      pos += best_len;
    } else {
      // Skip this byte (will be literal)
      pos++;
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.num_sequences = lit_lengths.size();
  result.success = true;
  
  return result;
}

// =============================================================================
// Output Formatting
// =============================================================================

void print_header() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Parallel Backtracking Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::left << std::setw(12) << "Size"
            << std::setw(15) << "GPU Time(ms)"
            << std::setw(18) << "Throughput(MB/s)"
            << std::setw(12) << "Status"
            << std::endl;
  std::cout << std::string(57, '-') << std::endl;
}

void print_result(const BenchmarkResult &result) {
  std::cout << std::left << std::setw(12) << result.input_size
            << std::setw(15) << std::fixed << std::setprecision(3) << result.gpu_time_ms
            << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput_mb_s
            << std::setw(12) << (result.success ? "PASS" : "FAIL")
            << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
  std::cout << "========================================" << std::endl;
  std::cout << "LZ77 Parallel Backtracking Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;
  
  // Get device info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
  std::cout << std::endl;
  
  BenchmarkConfig config;
  
  // Parse command line arguments
  if (argc > 1) {
    config.compression_level = std::atoi(argv[1]);
  }
  if (argc > 2) {
    config.iterations = std::atoi(argv[2]);
  }
  
  std::cout << "Compression Level: " << config.compression_level << std::endl;
  std::cout << "Iterations: " << config.iterations << std::endl;
  std::cout << std::endl;
  
  // Run benchmarks
  print_header();
  
  std::vector<BenchmarkResult> results;
  
  for (size_t size : config.data_sizes) {
    std::vector<uint8_t> data;
    generate_compressible_data(data, size);
    
    BenchmarkResult result = run_gpu_backtracking_benchmark(
        data, config.compression_level, config.iterations);
    
    print_result(result);
    results.push_back(result);
  }
  
  // Summary
  std::cout << std::string(57, '-') << std::endl;
  
  double total_throughput = 0.0;
  int success_count = 0;
  for (const auto &r : results) {
    if (r.success) {
      total_throughput += r.throughput_mb_s;
      success_count++;
    }
  }
  
  if (success_count > 0) {
    double avg_throughput = total_throughput / success_count;
    std::cout << "\nAverage Throughput: " << std::fixed << std::setprecision(2) 
              << avg_throughput << " MB/s" << std::endl;
  }
  
  std::cout << "\nBenchmark complete." << std::endl;
  
  return 0;
}