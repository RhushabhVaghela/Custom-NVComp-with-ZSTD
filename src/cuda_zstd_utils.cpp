// ============================================================================
// cuda_zstd_utils.cpp - Implementation of Shared Utility Functions
// ============================================================================

#include "cuda_zstd_utils.h"
#include "cuda_zstd_manager.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <mutex>

namespace cuda_zstd {

// ============================================================================
// PerformanceProfiler Implementation
// ============================================================================

/*
bool PerformanceProfiler::profiling_enabled_ = false;
DetailedPerformanceMetrics PerformanceProfiler::metrics_;
std::unordered_map<std::string, double> PerformanceProfiler::timers_;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> PerformanceProfiler::timer_start_;
std::unordered_map<std::string, cudaEvent_t> PerformanceProfiler::cuda_timers_;
std::mutex PerformanceProfiler::profiler_mutex_;

void PerformanceProfiler::enable_profiling(bool enable) {}
bool PerformanceProfiler::is_profiling_enabled() { return false; }
const DetailedPerformanceMetrics& PerformanceProfiler::get_metrics() { static DetailedPerformanceMetrics m; return m; }
void PerformanceProfiler::reset_metrics() {}
void PerformanceProfiler::print_metrics() {}
void PerformanceProfiler::start_timer(const char* name) {}
void PerformanceProfiler::stop_timer(const char* name) {}
double PerformanceProfiler::get_timer_ms(const char* name) { return 0.0; }
void PerformanceProfiler::start_timer(const std::string& name) {}
void PerformanceProfiler::stop_timer(const std::string& name) {}
double PerformanceProfiler::get_timer_ms(const std::string& name) { return 0.0; }
void PerformanceProfiler::record_lz77_time(double ms) {}
void PerformanceProfiler::record_fse_time(double ms) {}
void PerformanceProfiler::record_huffman_time(double ms) {}
void PerformanceProfiler::record_memory_usage(size_t bytes) {}
void PerformanceProfiler::record_kernel_launch() {}
void PerformanceProfiler::export_metrics_csv(const char* filename) {}
void PerformanceProfiler::export_metrics_json(const char* filename) {}
void PerformanceProfiler::export_metrics_csv(const std::string& filename) {}
void PerformanceProfiler::export_metrics_json(const std::string& filename) {}
*/

// Stubs to satisfy linker - initialized to nullptr to avoid static init issues
bool PerformanceProfiler::profiling_enabled_ = false;
DetailedPerformanceMetrics PerformanceProfiler::metrics_;
std::unordered_map<std::string, double>* PerformanceProfiler::timers_ = nullptr;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point>* PerformanceProfiler::timer_start_ = nullptr;
std::unordered_map<std::string, cudaEvent_t>* PerformanceProfiler::cuda_timers_ = nullptr;
std::mutex* PerformanceProfiler::profiler_mutex_ = nullptr;

void PerformanceProfiler::enable_profiling(bool enable) {}
bool PerformanceProfiler::is_profiling_enabled() { return false; }
const DetailedPerformanceMetrics& PerformanceProfiler::get_metrics() { static DetailedPerformanceMetrics m; return m; }
void PerformanceProfiler::reset_metrics() {}
void PerformanceProfiler::print_metrics() {}
void PerformanceProfiler::start_timer(const char* name) {}
void PerformanceProfiler::stop_timer(const char* name) {}
double PerformanceProfiler::get_timer_ms(const char* name) { return 0.0; }
void PerformanceProfiler::start_timer(const std::string& name) {}
void PerformanceProfiler::stop_timer(const std::string& name) {}
double PerformanceProfiler::get_timer_ms(const std::string& name) { return 0.0; }
void PerformanceProfiler::record_lz77_time(double ms) {}
void PerformanceProfiler::record_fse_time(double ms) {}
void PerformanceProfiler::record_huffman_time(double ms) {}
void PerformanceProfiler::record_memory_usage(size_t bytes) {}
void PerformanceProfiler::record_kernel_launch() {}
void PerformanceProfiler::export_metrics_csv(const char* filename) {}
void PerformanceProfiler::export_metrics_json(const char* filename) {}
void PerformanceProfiler::export_metrics_csv(const std::string& filename) {}
void PerformanceProfiler::export_metrics_json(const std::string& filename) {}

// ============================================================================
// DetailedPerformanceMetrics Implementation
// ============================================================================

void DetailedPerformanceMetrics::print() const {
//     std::cout << "=== Performance Metrics ===" << std::endl;
//     std::cout << "LZ77 Time: " << lz77_time_ms << " ms" << std::endl;
//     std::cout << "FSE Encode Time: " << fse_encode_time_ms << " ms" << std::endl;
//     std::cout << "Huffman Encode Time: " << huffman_encode_time_ms << " ms" << std::endl;
//     std::cout << "Sequence Generation Time: " << sequence_generation_time_ms << " ms" << std::endl;
//     std::cout << "Total Time: " << total_time_ms << " ms" << std::endl;
//     std::cout << "Peak Memory: " << peak_memory_bytes << " bytes" << std::endl;
//     std::cout << "Kernel Launches: " << kernel_launches << std::endl;
}

void DetailedPerformanceMetrics::export_csv(const char* filename) const {
    PerformanceProfiler::export_metrics_csv(filename);
}

} // namespace cuda_zstd
