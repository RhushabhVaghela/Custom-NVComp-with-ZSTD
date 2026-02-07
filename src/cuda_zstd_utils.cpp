// ============================================================================
// cuda_zstd_utils.cpp - Implementation of Shared Utility Functions
// ============================================================================

#include "cuda_zstd_utils.h"
#include "cuda_zstd_manager.h"
#include "performance_profiler.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace cuda_zstd {

// ============================================================================
// PerformanceProfiler Implementation
// ============================================================================

bool PerformanceProfiler::profiling_enabled_ = false;
DetailedPerformanceMetrics PerformanceProfiler::metrics_;

// Initialize pointers to nullptr to avoid static initialization order issues
std::unordered_map<std::string, double> *PerformanceProfiler::timers_ = nullptr;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point>
    *PerformanceProfiler::timer_start_ = nullptr;
std::unordered_map<std::string, cudaEvent_t>
    *PerformanceProfiler::cuda_timers_ = nullptr;
std::mutex *PerformanceProfiler::profiler_mutex_ = nullptr;

static std::once_flag profiler_init_flag;

void PerformanceProfiler::ensure_initialized() {
  std::call_once(profiler_init_flag, []() {
    profiler_mutex_ = new std::mutex();
    timers_ = new std::unordered_map<std::string, double>();
    timer_start_ = new std::unordered_map<
        std::string, std::chrono::high_resolution_clock::time_point>();
    cuda_timers_ = new std::unordered_map<std::string, cudaEvent_t>();
  });
}

void PerformanceProfiler::enable_profiling(bool enable) {
  ensure_initialized();
  std::lock_guard<std::mutex> lock(*profiler_mutex_);
  profiling_enabled_ = enable;
}

bool PerformanceProfiler::is_profiling_enabled() { return profiling_enabled_; }

const DetailedPerformanceMetrics &PerformanceProfiler::get_metrics() {
  return metrics_;
}

void PerformanceProfiler::reset_metrics() {
  ensure_initialized();
  std::lock_guard<std::mutex> lock(*profiler_mutex_);
  metrics_ = DetailedPerformanceMetrics();
  timers_->clear();
  timer_start_->clear();
  // Don't destroy cuda events, just clear map? Or reuse?
  // For simplicity, just clearing map (might leak events if not destroyed,
  // but this is a debug profiler).
  // Ideally we should destroy them.
  for (auto &pair : *cuda_timers_) {
    cudaEventDestroy(pair.second);
  }
  cuda_timers_->clear();
}

void PerformanceProfiler::print_metrics() {
  if (!profiling_enabled_)
    return;
  metrics_.print();
}

void PerformanceProfiler::start_timer(const char *name) {
  if (!profiling_enabled_)
    return;
  ensure_initialized();
  std::lock_guard<std::mutex> lock(*profiler_mutex_);
  (*timer_start_)[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceProfiler::stop_timer(const char *name) {
  if (!profiling_enabled_)
    return;
  ensure_initialized();
  std::lock_guard<std::mutex> lock(*profiler_mutex_);
  auto it = timer_start_->find(name);
  if (it != timer_start_->end()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - it->second;
    (*timers_)[name] += ms.count();
    timer_start_->erase(it);
  }
}

double PerformanceProfiler::get_timer_ms(const char *name) {
  ensure_initialized();
  std::lock_guard<std::mutex> lock(*profiler_mutex_);
  if (timers_->find(name) != timers_->end()) {
    return (*timers_)[name];
  }
  return 0.0;
}

void PerformanceProfiler::start_timer(const std::string &name) {
  start_timer(name.c_str());
}

void PerformanceProfiler::stop_timer(const std::string &name) {
  stop_timer(name.c_str());
}

double PerformanceProfiler::get_timer_ms(const std::string &name) {
  return get_timer_ms(name.c_str());
}

void PerformanceProfiler::record_lz77_time(double ms) {
  if (profiling_enabled_)
    metrics_.lz77_time_ms += ms;
}

void PerformanceProfiler::record_fse_time(double ms) {
  if (profiling_enabled_)
    metrics_.fse_encode_time_ms += ms;
}

void PerformanceProfiler::record_huffman_time(double ms) {
  if (profiling_enabled_)
    metrics_.huffman_encode_time_ms += ms;
}

void PerformanceProfiler::record_memory_usage(size_t bytes) {
  if (profiling_enabled_) {
    metrics_.current_memory_bytes = bytes;
    if (bytes > metrics_.peak_memory_bytes) {
      metrics_.peak_memory_bytes = bytes;
    }
  }
}

void PerformanceProfiler::record_kernel_launch() {
  if (profiling_enabled_)
    metrics_.kernel_launches++;
}

void PerformanceProfiler::export_metrics_csv(const char *filename) {
  metrics_.export_csv(filename);
}

void PerformanceProfiler::export_metrics_json(const char *filename) {
  metrics_.export_json(filename);
}

void PerformanceProfiler::export_metrics_csv(const std::string &filename) {
  metrics_.export_csv(filename);
}

void PerformanceProfiler::export_metrics_json(const std::string &filename) {
  metrics_.export_json(filename);
}

// ============================================================================
// DetailedPerformanceMetrics Implementation
// ============================================================================

void DetailedPerformanceMetrics::print() const {
  std::cout << "\n=== Performance Metrics ===\n";
  std::cout << "Total Time: " << total_time_ms << " ms\n";
  std::cout << "LZ77 Time: " << lz77_time_ms << " ms\n";
  std::cout << "FSE Encode Time: " << fse_encode_time_ms << " ms\n";
  std::cout << "Huffman Encode Time: " << huffman_encode_time_ms << " ms\n";
  std::cout << "Compression Throughput: " << compression_throughput_mbps
            << " MB/s\n";
  std::cout << "Peak Memory: " << peak_memory_bytes / (1024.0 * 1024.0)
            << " MB\n";
  std::cout << "Kernel Launches: " << kernel_launches << "\n";
  std::cout << "===========================\n";
}

void DetailedPerformanceMetrics::export_csv(const char *filename) const {
  std::ofstream f(filename);
  if (!f.is_open())
    return;

  f << "Metric,Value\n";
  f << "total_time_ms," << total_time_ms << "\n";
  f << "lz77_time_ms," << lz77_time_ms << "\n";
  f << "fse_encode_time_ms," << fse_encode_time_ms << "\n";
  f << "huffman_encode_time_ms," << huffman_encode_time_ms << "\n";
  f << "compression_throughput_mbps," << compression_throughput_mbps << "\n";
  f << "peak_memory_bytes," << peak_memory_bytes << "\n";
  f << "kernel_launches," << kernel_launches << "\n";
  f.close();
}

void DetailedPerformanceMetrics::export_json(const char *filename) const {
  std::ofstream f(filename);
  if (!f.is_open())
    return;

  f << "{\n";
  f << "  \"total_time_ms\": " << total_time_ms << ",\n";
  f << "  \"lz77_time_ms\": " << lz77_time_ms << ",\n";
  f << "  \"fse_encode_time_ms\": " << fse_encode_time_ms << ",\n";
  f << "  \"huffman_encode_time_ms\": " << huffman_encode_time_ms << ",\n";
  f << "  \"compression_throughput_mbps\": " << compression_throughput_mbps
    << ",\n";
  f << "  \"peak_memory_bytes\": " << peak_memory_bytes << ",\n";
  f << "  \"kernel_launches\": " << kernel_launches << "\n";
  f << "}\n";
  f.close();
}

} // namespace cuda_zstd
