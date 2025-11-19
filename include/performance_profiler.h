// ==============================================================================
// performance_profiler.h - Performance profiling and metrics
// ==============================================================================

#ifndef PERFORMANCE_PROFILER_H
#define PERFORMANCE_PROFILER_H

#include "common_types.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace compression {

struct DetailedPerformanceMetrics {
    double lz77_time_ms = 0.0;
    double hash_build_time_ms = 0.0;
    double match_finding_time_ms = 0.0;
    double optimal_parse_time_ms = 0.0;
    double sequence_generation_time_ms = 0.0;

    double entropy_encode_time_ms = 0.0;
    double fse_encode_time_ms = 0.0;
    double huffman_encode_time_ms = 0.0;

    double entropy_decode_time_ms = 0.0;
    double total_time_ms = 0.0;

    double compression_throughput_mbps = 0.0;
    double decompression_throughput_mbps = 0.0;

    size_t peak_memory_bytes = 0;
    size_t current_memory_bytes = 0;
    size_t workspace_size_bytes = 0;

    size_t input_bytes = 0;
    size_t output_bytes = 0;
    float compression_ratio = 0.0f;

    u32 kernel_launches = 0;
    double avg_kernel_time_ms = 0.0;

    double read_bandwidth_gbps = 0.0;
    double write_bandwidth_gbps = 0.0;
    double total_bandwidth_gbps = 0.0;

    float gpu_utilization_percent = 0.0f;
    float memory_utilization_percent = 0.0f;

    void print() const;
    void export_csv(const std::string& filename) const;
    void export_json(const std::string& filename) const;
};

class PerformanceProfiler {
public:
    static void enable_profiling(bool enable);
    static bool is_profiling_enabled();
    static const DetailedPerformanceMetrics& get_metrics();
    static void reset_metrics();
    static void print_metrics();

    static void start_timer(const char* name);
    static void stop_timer(const char* name);
    static double get_timer_ms(const char* name);

    static void start_timer(const std::string& name);
    static void stop_timer(const std::string& name);
    static double get_timer_ms(const std::string& name);

    static void record_lz77_time(double ms);
    static void record_fse_time(double ms);
    static void record_huffman_time(double ms);
    static void record_memory_usage(size_t bytes);
    static void record_kernel_launch();

    static void export_metrics_csv(const std::string& filename);
    static void export_metrics_json(const std::string& filename);

private:
    static bool profiling_enabled_;
    static DetailedPerformanceMetrics metrics_;
    static std::unordered_map<std::string, double> timers_;
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timer_start_;
    static std::mutex profiler_mutex_;
};

} // namespace compression

#endif // PERFORMANCE_PROFILER_H
