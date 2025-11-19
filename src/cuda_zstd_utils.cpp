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

bool PerformanceProfiler::profiling_enabled_ = false;
DetailedPerformanceMetrics PerformanceProfiler::metrics_;
std::unordered_map<std::string, double> PerformanceProfiler::timers_;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> PerformanceProfiler::timer_start_;
std::unordered_map<std::string, cudaEvent_t> PerformanceProfiler::cuda_timers_;
std::mutex PerformanceProfiler::profiler_mutex_;

void PerformanceProfiler::enable_profiling(bool enable) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    profiling_enabled_ = enable;
    if (!enable) {
        reset_metrics();
    }
}

bool PerformanceProfiler::is_profiling_enabled() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    return profiling_enabled_;
}

const DetailedPerformanceMetrics& PerformanceProfiler::get_metrics() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    return metrics_;
}

void PerformanceProfiler::reset_metrics() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_ = DetailedPerformanceMetrics();
    timers_.clear();
    timer_start_.clear();
    // Safe CUDA event destruction with error checking
    for (auto& pair : cuda_timers_) {
        cudaError_t err = cudaEventDestroy(pair.second);
        if (err != cudaSuccess) {
            std::cerr << "[Profiler] Warning: cudaEventDestroy failed for timer '" 
                     << pair.first << "': " << cudaGetErrorString(err) << std::endl;
        }
    }
    cuda_timers_.clear();
}

void PerformanceProfiler::print_metrics() {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.print();
}

void PerformanceProfiler::start_timer(const char* name) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    timer_start_[name] = std::chrono::high_resolution_clock::now();
    
    // Create CUDA event if it doesn't exist
    if (cuda_timers_.find(name) == cuda_timers_.end()) {
        cudaEventCreate(&cuda_timers_[name]);
    }
    cudaEventRecord(cuda_timers_[name], 0);
}

void PerformanceProfiler::stop_timer(const char* name) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    
    auto start_it = timer_start_.find(name);
    if (start_it != timer_start_.end()) {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start_it->second).count();
        timers_[name] = elapsed;
        timer_start_.erase(start_it);
    }
    
    // CUDA event timing
    auto cuda_it = cuda_timers_.find(name);
    if (cuda_it != cuda_timers_.end()) {
        cudaEventSynchronize(cuda_it->second);
    }
}

double PerformanceProfiler::get_timer_ms(const char* name) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    auto it = timers_.find(name);
    return (it != timers_.end()) ? it->second : 0.0;
}

void PerformanceProfiler::start_timer(const std::string& name) {
    start_timer(name.c_str());
}

void PerformanceProfiler::stop_timer(const std::string& name) {
    stop_timer(name.c_str());
}

double PerformanceProfiler::get_timer_ms(const std::string& name) {
    return get_timer_ms(name.c_str());
}

void PerformanceProfiler::record_lz77_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.lz77_time_ms += ms;
}

void PerformanceProfiler::record_fse_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.fse_encode_time_ms += ms;
}

void PerformanceProfiler::record_huffman_time(double ms) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.huffman_encode_time_ms += ms;
}

void PerformanceProfiler::record_memory_usage(size_t bytes) {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.current_memory_bytes = bytes;
    metrics_.peak_memory_bytes = std::max(metrics_.peak_memory_bytes, bytes);
}

void PerformanceProfiler::record_kernel_launch() {
    if (!profiling_enabled_) return;
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    metrics_.kernel_launches++;
}

void PerformanceProfiler::export_metrics_csv(const char* filename) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "Metric,Value,Unit\n";
    file << "LZ77 Time," << metrics_.lz77_time_ms << ",ms\n";
    file << "FSE Encode Time," << metrics_.fse_encode_time_ms << ",ms\n";
    file << "Huffman Encode Time," << metrics_.huffman_encode_time_ms << ",ms\n";
    file << "Sequence Generation Time," << metrics_.sequence_generation_time_ms << ",ms\n";
    file << "Total Time," << metrics_.total_time_ms << ",ms\n";
    file << "Peak Memory," << metrics_.peak_memory_bytes << ",bytes\n";
    file << "Kernel Launches," << metrics_.kernel_launches << ",count\n";
    file.close();
}

void PerformanceProfiler::export_metrics_json(const char* filename) {
    std::lock_guard<std::mutex> lock(profiler_mutex_);
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"lz77_time_ms\": " << metrics_.lz77_time_ms << ",\n";
    file << "  \"fse_encode_time_ms\": " << metrics_.fse_encode_time_ms << ",\n";
    file << "  \"huffman_encode_time_ms\": " << metrics_.huffman_encode_time_ms << ",\n";
    file << "  \"sequence_generation_time_ms\": " << metrics_.sequence_generation_time_ms << ",\n";
    file << "  \"total_time_ms\": " << metrics_.total_time_ms << ",\n";
    file << "  \"peak_memory_bytes\": " << metrics_.peak_memory_bytes << ",\n";
    file << "  \"kernel_launches\": " << metrics_.kernel_launches << "\n";
    file << "}\n";
    file.close();
}

void PerformanceProfiler::export_metrics_csv(const std::string& filename) {
    export_metrics_csv(filename.c_str());
}

void PerformanceProfiler::export_metrics_json(const std::string& filename) {
    export_metrics_json(filename.c_str());
}

// ============================================================================
// DetailedPerformanceMetrics Implementation
// ============================================================================

void DetailedPerformanceMetrics::print() const {
    std::cout << "=== Performance Metrics ===" << std::endl;
    std::cout << "LZ77 Time: " << lz77_time_ms << " ms" << std::endl;
    std::cout << "FSE Encode Time: " << fse_encode_time_ms << " ms" << std::endl;
    std::cout << "Huffman Encode Time: " << huffman_encode_time_ms << " ms" << std::endl;
    std::cout << "Sequence Generation Time: " << sequence_generation_time_ms << " ms" << std::endl;
    std::cout << "Total Time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Peak Memory: " << peak_memory_bytes << " bytes" << std::endl;
    std::cout << "Kernel Launches: " << kernel_launches << std::endl;
}

void DetailedPerformanceMetrics::export_csv(const char* filename) const {
    PerformanceProfiler::export_metrics_csv(filename);
}

} // namespace cuda_zstd
