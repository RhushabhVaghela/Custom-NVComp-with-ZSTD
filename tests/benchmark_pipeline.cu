#include "../src/pipeline_manager.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cuda_zstd;

// Generate a verifiable pattern (cyclic)
void fill_buffer_pattern(void *buf, size_t size) {
  uint8_t *ptr = (uint8_t *)buf;
// Simple fast pattern: count up
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = (uint8_t)(i & 0xFF);
  }
}

void run_benchmark(const char *name, size_t total_size_bytes) {
  std::cout
      << "\n------------------------------------------------------------\n";
  std::cout << "Benchmarking: " << name << " ("
            << total_size_bytes / (1024.0 * 1024.0 * 1024.0) << " GB)"
            << std::endl;
  std::cout << "------------------------------------------------------------\n";

  CompressionConfig config;
  config.block_size = 128 * 1024;

  // Batch size 64MB, 3 slots
  PipelinedBatchManager pipeline(config, 64 * 1024 * 1024, 3);

  // Pre-gen generic data chunk (64MB) to copy-paste for speed
  size_t chunk_size = 64 * 1024 * 1024;
  std::vector<uint8_t> source_data(chunk_size);
  fill_buffer_pattern(source_data.data(), chunk_size);

  size_t bytes_generated = 0;
  size_t bytes_compressed = 0;
  size_t chunks_processed = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  Status status = pipeline.compress_stream_pipeline(
      // Input Generator
      [&](void *h_in, size_t max_len, size_t *out_len) -> bool {
        if (bytes_generated >= total_size_bytes) {
          *out_len = 0;
          return false;
        }

        size_t remain = total_size_bytes - bytes_generated;
        size_t copy_size = std::min(max_len, remain);

        // Fast cyclic copy
        // For true 25GB benchmark, data generation must be faster than
        // compression memcpy is roughly 20GB/s+, should be fine.
        size_t src_offset = bytes_generated % chunk_size;
        size_t first_part = std::min(copy_size, chunk_size - src_offset);

        memcpy(h_in, source_data.data() + src_offset, first_part);
        if (first_part < copy_size) {
          memcpy((uint8_t *)h_in + first_part, source_data.data(),
                 copy_size - first_part);
        }

        bytes_generated += copy_size;
        *out_len = copy_size;
        return true;
      },
      // Output Consumer
      [&](const void *h_out, size_t size) {
        bytes_compressed += size;
        chunks_processed++;
        // Optional: Verify integrity or write to /dev/null
      });

  auto end_time = std::chrono::high_resolution_clock::now();

  if (status != Status::SUCCESS) {
    std::cerr << "Pipeline FAILED with status " << (int)status << std::endl;
    return;
  }

  std::chrono::duration<double> diff = end_time - start_time;
  double seconds = diff.count();
  double throughput_gbps =
      (total_size_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;

  std::cout << "Status: SUCCESS" << std::endl;
  std::cout << "Time:   " << std::fixed << std::setprecision(3) << seconds
            << " s" << std::endl;
  std::cout << "Size:   " << bytes_generated << " bytes -> " << bytes_compressed
            << " bytes" << std::endl;
  std::cout << "Ratio:  " << (double)bytes_generated / bytes_compressed
            << std::endl;
  std::cout << "Speed:  " << throughput_gbps << " GB/s" << std::endl;
}

int main(int argc, char **argv) {
  // 512MB
  run_benchmark("512 MB Dataset", 512ULL * 1024 * 1024);

  // 1GB
  run_benchmark("1 GB Dataset", 1ULL * 1024 * 1024 * 1024);

  // 2GB
  run_benchmark("2 GB Dataset", 2ULL * 1024 * 1024 * 1024);

  return 0;
}
