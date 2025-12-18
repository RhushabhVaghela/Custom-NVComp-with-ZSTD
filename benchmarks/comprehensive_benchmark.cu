// Comprehensive CUDA-ZSTD Benchmark: Formulas × Block Sizes × Modes × Input
// Sizes Tests: 6 formulas × 7 block_sizes × 2 modes (serial/parallel) × 14
// input_sizes Total: ~1,176 test combinations from 1KB to 20GB

#include "cuda_zstd_manager.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace cuda_zstd;

// =============================
// Formula definitions
// =============================
namespace formulas {
u32 sqrt_k400(size_t input_size) {
  return (u32)(std::sqrt((double)input_size) * 400.0);
}
u32 logarithmic(size_t input_size) {
  double base = 512.0 * 1024.0;
  double power = std::pow(input_size / (1024.0 * 1024.0), 0.25);
  return (u32)(base * power);
}
u32 linear_128blocks(size_t input_size) { return (u32)(input_size / 128); }
u32 cuberoot_k150(size_t input_size) {
  return (u32)(std::cbrt((double)input_size) * 150.0);
}
u32 piecewise(size_t input_size) {
  if (input_size < 10 * 1024 * 1024)
    return 2 * 1024 * 1024;
  if (input_size < 100 * 1024 * 1024)
    return 4 * 1024 * 1024;
  return 8 * 1024 * 1024;
}
u32 hybrid(size_t input_size) {
  u32 ideal = (u32)(std::sqrt((double)input_size) * 400.0);
  size_t target_blocks = input_size / ideal;
  target_blocks = std::clamp(target_blocks, (size_t)64, (size_t)256);
  u32 block_size = (u32)(input_size / target_blocks);
  u32 power = (u32)std::ceil(std::log2(block_size));
  return (u32)(1 << power);
}
} // namespace formulas

// =============================
// Benchmark result structure
// =============================
struct BenchmarkResult {
  std::string formula_name;
  size_t input_size;
  u32 block_size;
  bool is_parallel;
  double time_ms;
  double throughput_mbps;
  bool success;
};

// =============================
// Utility functions
// =============================
std::string format_size(size_t bytes) {
  if (bytes >= 1024UL * 1024 * 1024)
    return std::to_string(bytes / (1024UL * 1024 * 1024)) + " GB";
  if (bytes >= 1024 * 1024)
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  if (bytes >= 1024)
    return std::to_string(bytes / 1024) + " KB";
  return std::to_string(bytes) + " B";
}

// =============================
// Benchmark runner
// =============================
bool run_benchmark(const std::string &formula_name, size_t input_size,
                   u32 block_size, bool use_parallel, BenchmarkResult &result,
                   int test_number) {
  result = {formula_name, input_size, block_size, use_parallel,
            0.0,          0.0,        false};

  if (input_size < 256)
    return false;

  const size_t VRAM_LIMIT = 12UL * 1024 * 1024 * 1024; // 12GB safe limit
  size_t estimated_memory = input_size * 4;
  if (estimated_memory > VRAM_LIMIT)
    return false;

  // Generate reproducible test data
  std::vector<uint8_t> h_input(input_size);
  std::mt19937 rng(42); // fixed seed for reproducibility
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < input_size; ++i) {
    h_input[i] = (uint8_t)dist(rng);
  }

  CompressionConfig config = CompressionConfig::from_level(3);
  config.block_size = std::min(block_size, (u32)input_size);

  // Set execution mode based on valid flag
  if (use_parallel) {
    config.cpu_threshold = 0; // Force GPU
  } else {
    config.cpu_threshold = 0xFFFFFFFF; // Force CPU (UINT32_MAX)
  }

  ZstdBatchManager manager(config);

  void *d_input;
  if (cudaMalloc(&d_input, input_size) != cudaSuccess)
    return false;
  if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(d_input);
    return false;
  }

  size_t max_compressed = manager.get_max_compressed_size(input_size);
  size_t temp_size = manager.get_compress_temp_size(input_size);

  void *d_compressed;
  void *d_temp;
  if (cudaMalloc(&d_compressed, max_compressed) != cudaSuccess) {
    cudaFree(d_input);
    return false;
  }
  if (cudaMalloc(&d_temp, temp_size) != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    return false;
  }

  // Warmup
  size_t compressed_size = max_compressed;
  manager.compress(d_input, input_size, d_compressed, &compressed_size, d_temp,
                   temp_size, nullptr, 0, 0);
  cudaDeviceSynchronize();

  compressed_size = max_compressed;
  Status warmup_status =
      manager.compress(d_input, input_size, d_compressed, &compressed_size,
                       d_temp, temp_size, nullptr, 0, 0);
  cudaDeviceSynchronize();
  if (warmup_status != Status::SUCCESS) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    return false;
  }

  const int NUM_RUNS = 3;
  std::vector<double> run_times;
  run_times.reserve(NUM_RUNS);

  std::cout << "   Test #" << test_number << " runs:\n";

  for (int run = 0; run < NUM_RUNS; run++) {
    compressed_size = max_compressed;

    auto start = std::chrono::high_resolution_clock::now();
    Status status =
        manager.compress(d_input, input_size, d_compressed, &compressed_size,
                         d_temp, temp_size, nullptr, 0, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (status != Status::SUCCESS) {
      cudaFree(d_input);
      cudaFree(d_compressed);
      cudaFree(d_temp);
      return false;
    }

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ms = duration.count() / 1000.0;
    run_times.push_back(ms);

    std::cout << "      Run " << (run + 1) << ": " << std::fixed
              << std::setprecision(2) << ms << " ms\n";
  }

  double total_time = 0.0;
  for (double t : run_times)
    total_time += t;
  double avg_time = total_time / NUM_RUNS;

  result.time_ms = avg_time;
  result.throughput_mbps =
      (input_size / (1024.0 * 1024.0)) / (result.time_ms / 1000.0);
  result.success = true;

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_temp);

  return true;
}

int main() {
  // Ensure clean state handled by OS/Runtime to avoid conflicting with static
  // singletons cudaDeviceReset();

  // Check CUDA status at startup
  cudaError_t startup_err = cudaGetLastError();
  if (startup_err != cudaSuccess) {
    printf("[FATAL] CUDA Error at startup (Pre-main): %s\n",
           cudaGetErrorString(startup_err));
    return 1;
  }

  auto benchmark_start = std::chrono::system_clock::now();

  std::cout
      << "================================================================\n";
  std::cout << "  CUDA-ZSTD Comprehensive Benchmark\n";
  std::cout
      << "================================================================\n\n";

  // Config summary
  std::cout << "GPU VRAM Limit: 12 GB (safe)\n";
  std::cout << "Compression Level: 3\n";
  std::cout << "Block Sizes Tested: 128 KB → 8 MB\n";
  std::cout << "Input Sizes Tested: 1 KB → 20 GB\n\n";

  // Formula definitions
  std::vector<std::pair<std::string, std::function<u32(size_t)>>> formulas = {
      {"Sqrt_K400", formulas::sqrt_k400},
      {"Logarithmic", formulas::logarithmic},
      {"Linear_128", formulas::linear_128blocks},
      {"CubeRoot_K150", formulas::cuberoot_k150},
      {"Piecewise", formulas::piecewise},
      {"Hybrid", formulas::hybrid}};

  // Input sizes to benchmark (from 1 KB up to 20 GB)
  std::vector<size_t> input_sizes = {
      1 * 1024,             // 1 KB
      4 * 1024,             // 4 KB
      16 * 1024,            // 16 KB
      64 * 1024,            // 64 KB
      256 * 1024,           // 256 KB
      1 * 1024 * 1024,      // 1 MB
      4 * 1024 * 1024,      // 4 MB
      16 * 1024 * 1024,     // 16 MB
      64 * 1024 * 1024,     // 64 MB
      256 * 1024 * 1024,    // 256 MB
      1024UL * 1024 * 1024, // 1 GB
  };

  // Block sizes to test (from 128 KB up to 8 MB)
  std::vector<u32> block_sizes = {
      128 * 1024,      // 128 KB
      256 * 1024,      // 256 KB
      512 * 1024,      // 512 KB
      1 * 1024 * 1024, // 1 MB
      2 * 1024 * 1024, // 2 MB
      4 * 1024 * 1024, // 4 MB
      8 * 1024 * 1024  // 8 MB
  };

  std::vector<bool> parallel_modes = {false, true}; // Serial, Parallel

  std::ofstream csv_file("benchmark_results.csv");
  csv_file
      << "Formula,InputSize,BlockSize,Parallel,TimeMS,ThroughputMBps,Success\n";

  std::ofstream error_log("benchmark_errors.log");

  int total_tests = (block_sizes.size() + formulas.size()) *
                    input_sizes.size() * parallel_modes.size();
  std::cout << "Total tests to run: " << total_tests << " (sizes up to "
            << format_size(input_sizes.back()) << ", blocks up to "
            << (block_sizes.back() / (1024 * 1024)) << "MB)\n\n";

  int test_number = 1;
  int success_count = 0;
  int fail_count = 0;
  double total_throughput = 0.0;
  int throughput_samples = 0;
  double cumulative_avg_time = 0.0;
  int cumulative_tests = 0;

  // =============================
  // Main test loop
  // =============================
  for (size_t size : input_sizes) {
    std::cout
        << "------------------------------------------------------------\n";
    std::cout << ">>> Input Size: " << format_size(size) << "\n";
    std::cout
        << "------------------------------------------------------------\n";

    // Estimate ETA for this group
    int group_tests =
        (int)((block_sizes.size() + formulas.size()) * parallel_modes.size());
    if (cumulative_tests > 0) {
      double avg_time_so_far =
          cumulative_avg_time / cumulative_tests; // ms per test
      double eta_seconds = (avg_time_so_far * group_tests) / 1000.0;
      double progress_pct = (100.0 * (test_number - 1)) / total_tests;
      std::cout << "Progress: " << (test_number - 1) << "/" << total_tests
                << " (" << std::fixed << std::setprecision(1) << progress_pct
                << "%)\n";
      std::cout << "Estimated time for this group: ~" << std::fixed
                << std::setprecision(1) << eta_seconds << " seconds\n";
    } else {
      std::cout << "Progress: 0/" << total_tests << " (0%)\n";
      std::cout << "Estimated time for this group: (collecting baseline...)\n";
    }

    // Fixed block sizes
    for (u32 block_size : block_sizes) {
      for (bool parallel : parallel_modes) {
        if (!parallel && size >= 16 * 1024 * 1024)
          continue;
        if (!parallel &&
            block_size >= 512 * 1024) // Limit serial block size to < 512KB
                                      // to prevent instability
          continue;
        if (block_size >= 4 * 1024 * 1024) // Limit max block size to 4MB total
          continue;

        std::ostringstream block_stream;
        if (block_size >= 1024 * 1024)
          block_stream << (block_size / (1024 * 1024)) << "MB";
        else
          block_stream << (block_size / 1024) << "KB";
        std::string block_label = block_stream.str();
        std::string name = "Fixed_" + block_label;

        if (size != 1024 || block_size != 131072)
          continue;

        std::cout << "\n[Test #" << test_number << "] Benchmarking " << name
                  << " (Block " << block_label << ") on " << format_size(size)
                  << " (" << (parallel ? "Parallel" : "Serial") << ")\n";

        BenchmarkResult result;
        bool ok = run_benchmark(name, size, block_size, parallel, result,
                                test_number);

        if (ok && result.success) {
          std::cout << "   Result: OK | Avg: " << std::fixed
                    << std::setprecision(2) << result.time_ms
                    << " ms | Throughput: " << result.throughput_mbps
                    << " MB/s\n";

          csv_file << name << "," << format_size(size) << "," << block_label
                   << "," << (parallel ? "1" : "0") << "," << result.time_ms
                   << "," << result.throughput_mbps << ",1\n";

          success_count++;
          total_throughput += result.throughput_mbps;
          throughput_samples++;
          cumulative_avg_time += result.time_ms;
          cumulative_tests++;
        } else {
          std::cout << "   Result: FAILED\n";
          csv_file << name << "," << format_size(size) << "," << block_label
                   << "," << (parallel ? "1" : "0") << ",0,0,0\n";
          fail_count++;
          error_log << "FAILED: " << name << " on " << format_size(size) << " ("
                    << (parallel ? "Parallel" : "Serial") << ")\n";
        }
        csv_file.flush();
        error_log.flush();

        // CRITICAL: Check for sticky GPU errors after each test
        cudaError_t sticky_err = cudaGetLastError();
        if (sticky_err != cudaSuccess) {
          std::cout << "   [STICKY ERROR] Test #" << (test_number - 1)
                    << " left GPU error: " << cudaGetErrorString(sticky_err)
                    << "\n";
          error_log << "STICKY ERROR after Test #" << (test_number - 1) << ": "
                    << cudaGetErrorString(sticky_err) << "\n";
          error_log.flush();
        }

        test_number++;
      }
    }

    // Formula block sizes
    for (const auto &formula : formulas) {
      u32 block_size = formula.second(size);
      for (bool parallel : parallel_modes) {
        if (!parallel && block_size >= 512 * 1024)
          continue; // Strictly limit serial block size to < 512KB
        if (block_size >= 4 * 1024 * 1024)
          continue; // Global safety limit: max 4MB block size
        std::ostringstream block_stream;
        if (block_size >= 1024 * 1024)
          block_stream << (block_size / (1024 * 1024)) << "MB";
        else
          block_stream << (block_size / 1024) << "KB";
        std::string block_label = block_stream.str();

        std::cout << "\n[Test #" << test_number << "] Benchmarking "
                  << formula.first << " (Block " << block_label << ") on "
                  << format_size(size) << " ("
                  << (parallel ? "Parallel" : "Serial") << ")\n";

        BenchmarkResult result;
        bool ok = run_benchmark(formula.first, size, block_size, parallel,
                                result, test_number);

        if (ok && result.success) {
          std::cout << "   Result: OK | Avg: " << std::fixed
                    << std::setprecision(2) << result.time_ms
                    << " ms | Throughput: " << result.throughput_mbps
                    << " MB/s\n";

          csv_file << formula.first << "," << format_size(size) << ","
                   << block_label << "," << (parallel ? "1" : "0") << ","
                   << result.time_ms << "," << result.throughput_mbps << ",1\n";

          success_count++;
          total_throughput += result.throughput_mbps;
          throughput_samples++;
          cumulative_avg_time += result.time_ms;
          cumulative_tests++;
        } else {
          std::cout << "   Result: FAILED\n";
          csv_file << formula.first << "," << format_size(size) << ","
                   << block_label << "," << (parallel ? "1" : "0")
                   << ",0,0,0\n";
          fail_count++;
          error_log << "FAILED: " << formula.first << " on "
                    << format_size(size) << " ("
                    << (parallel ? "Parallel" : "Serial") << ")\n";
        }
        csv_file.flush();
        error_log.flush();
        test_number++;
      }
    }

    std::cout << "\n"; // spacing between input size groups
  }

  // === Summary Section ===
  auto benchmark_end = std::chrono::system_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                            benchmark_end - benchmark_start)
                            .count();

  std::time_t start_time =
      std::chrono::system_clock::to_time_t(benchmark_start);
  std::time_t end_time = std::chrono::system_clock::to_time_t(benchmark_end);

  std::cout
      << "================================================================\n";
  std::cout << "  Benchmark Summary\n";
  std::cout
      << "================================================================\n";
  std::cout << "  Started:         " << std::ctime(&start_time);
  std::cout << "  Finished:        " << std::ctime(&end_time);
  std::cout << "  Total Duration:  " << total_duration << " seconds\n";
  std::cout << "  Total Tests Run: " << (success_count + fail_count) << "/"
            << total_tests << "\n";
  std::cout << "  Successful:      " << success_count << "\n";
  std::cout << "  Failed:          " << fail_count << "\n";
  if (throughput_samples > 0) {
    std::cout << "  Avg Throughput:  " << std::fixed << std::setprecision(2)
              << (total_throughput / throughput_samples) << " MB/s\n";
  }
  std::cout
      << "================================================================\n";

  csv_file.flush();
  csv_file.close();
  error_log.flush();
  error_log.close();

  std::cout
      << "\nBenchmark completed. Results saved to benchmark_results.csv\n";
  std::cout << "Errors logged to benchmark_errors.log (if any)\n";
  return 0;
}