// throughput_display.h - Consistent throughput formatting across benchmarks
#pragma once

#include <cstdio>
#include <string>

namespace cuda_zstd {

/**
 * @brief Format throughput with automatic unit selection (GB/s or MB/s)
 * @param bytes_per_second Raw throughput in bytes/second
 * @return Formatted string like "1.23 GB/s (1234.56 MB/s)"
 */
inline std::string format_throughput(double bytes_per_second) {
  double mbps = bytes_per_second / (1024.0 * 1024.0);
  double gbps = mbps / 1024.0;

  char buf[64];
  if (gbps >= 1.0) {
    snprintf(buf, sizeof(buf), "%.2f GB/s (%.1f MB/s)", gbps, mbps);
  } else if (mbps >= 1.0) {
    snprintf(buf, sizeof(buf), "%.2f MB/s (%.4f GB/s)", mbps, gbps);
  } else {
    // Very slow: show KB/s
    double kbps = bytes_per_second / 1024.0;
    snprintf(buf, sizeof(buf), "%.2f KB/s (%.6f MB/s)", kbps, mbps);
  }
  return std::string(buf);
}

/**
 * @brief Calculate and format throughput from size and time
 * @param size_bytes Data size in bytes
 * @param time_ms Time taken in milliseconds
 * @return Formatted throughput string
 */
inline std::string format_throughput_from_time(size_t size_bytes,
                                               double time_ms) {
  if (time_ms <= 0)
    return "N/A";
  double bytes_per_second = (double)size_bytes / (time_ms / 1000.0);
  return format_throughput(bytes_per_second);
}

/**
 * @brief Get throughput in MB/s for calculations
 */
inline double throughput_mbps(size_t size_bytes, double time_ms) {
  if (time_ms <= 0)
    return 0.0;
  return (size_bytes / (1024.0 * 1024.0)) / (time_ms / 1000.0);
}

/**
 * @brief Get throughput in GB/s for calculations
 */
inline double throughput_gbps(size_t size_bytes, double time_ms) {
  return throughput_mbps(size_bytes, time_ms) / 1024.0;
}

} // namespace cuda_zstd
