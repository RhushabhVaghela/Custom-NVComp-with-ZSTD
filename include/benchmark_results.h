#pragma once

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

inline void log_benchmark_result(const char *benchmark_name,
                                 const char *gpu_name, unsigned int block_size,
                                 unsigned int num_blocks, double total_bytes,
                                 double time_ms, double throughput_gbps) {
  const char *filename = "benchmark_results.csv";
  std::ofstream file;

  // Check if file exists to write header
  bool exists = false;
  {
    std::ifstream check(filename);
    exists = check.good();
  }

  file.open(filename, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << " for writing result.\n";
    return;
  }

  if (!exists) {
    file << "Timestamp,Benchmark,GPU,BlockSize,NumBlocks,TotalBytes,TimeMS,"
            "ThroughputGBps\n";
  }

  // Get current timestamp
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  struct tm *parts = std::localtime(&now_c);

  char time_str[64];
  std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", parts);

  file << time_str << ",";
  file << benchmark_name << ",";
  file << "\"" << gpu_name << "\",";
  file << block_size << ",";
  file << num_blocks << ",";
  file << std::fixed << std::setprecision(0) << total_bytes << ",";
  file << std::fixed << std::setprecision(2) << time_ms << ",";
  file << std::fixed << std::setprecision(4) << throughput_gbps << "\n";

  file.close();
  std::cout << "  [Result logged to " << filename << "]\n";
}
