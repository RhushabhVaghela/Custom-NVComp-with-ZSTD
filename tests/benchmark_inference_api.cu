// ============================================================================
// benchmark_inference_api.cu - Performance Benchmarks for Inference-Ready API
// ============================================================================
// Measures throughput and latency of zero-malloc decompression for LLM
// inference. Simulates realistic inference patterns with layer-wise streaming.
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include "throughput_display.h"
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace cuda_zstd;

// ============================================================================
// Benchmark Utilities
// ============================================================================

void print_separator() {
  std::cout << "========================================" << std::endl;
}

void print_header(const char *name) {
  std::cout << "\n";
  print_separator();
  std::cout << "BENCHMARK: " << name << std::endl;
  print_separator();
}

void generate_model_weight_data(std::vector<uint8_t> &data, size_t size) {
  data.resize(size);
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<uint8_t>((i * 1103515245 + 12345) % 256);
    if (i % 8 < 4) {
      data[i] = data[i] & 0xF0; // Simulate quantization pattern
    }
  }
}

struct BenchmarkResult {
  double throughput_gbps;
  double throughput_mbps; // Added for dual display
  double avg_latency_ms;
  double min_latency_ms;
  double max_latency_ms;
  size_t total_bytes;
  int iterations;
};

void print_result(const BenchmarkResult &result, const char *name) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n  Results for " << name << ":" << std::endl;
  std::cout << "    Throughput:   " << result.throughput_mbps << " MB/s ("
            << result.throughput_gbps << " GB/s)" << std::endl;
  std::cout << "    Avg Latency:  " << result.avg_latency_ms << " ms"
            << std::endl;
  std::cout << "    Min Latency:  " << result.min_latency_ms << " ms"
            << std::endl;
  std::cout << "    Max Latency:  " << result.max_latency_ms << " ms"
            << std::endl;
  std::cout << "    Total Data:   " << (result.total_bytes / (1024.0 * 1024.0))
            << " MB" << std::endl;
  std::cout << "    Iterations:   " << result.iterations << std::endl;
}

// ============================================================================
// BENCHMARK 1: decompress_to_preallocated Throughput
// ============================================================================

BenchmarkResult benchmark_decompress_to_preallocated(size_t data_size,
                                                     int iterations) {
  std::vector<uint8_t> h_input;
  generate_model_weight_data(h_input, data_size);

  void *d_input, *d_compressed, *d_output, *d_temp;
  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_compressed, data_size * 2);
  cudaMalloc(&d_output, data_size);

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  size_t temp_size = manager.get_compress_temp_size(data_size);
  cudaMalloc(&d_temp, temp_size);

  // Compress once
  size_t compressed_size = data_size * 2;
  manager.compress(d_input, data_size, d_compressed, &compressed_size, d_temp,
                   temp_size, nullptr, 0);

  // Warmup
  for (int i = 0; i < 3; i++) {
    size_t actual_size;
    manager.decompress_to_preallocated(d_compressed, compressed_size, d_output,
                                       data_size, &actual_size, d_temp,
                                       temp_size, 0);
  }
  cudaDeviceSynchronize();

  // Benchmark
  std::vector<double> latencies;
  auto overall_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t actual_size;
    manager.decompress_to_preallocated(d_compressed, compressed_size, d_output,
                                       data_size, &actual_size, d_temp,
                                       temp_size, 0);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();
    latencies.push_back(latency);
  }

  auto overall_end = std::chrono::high_resolution_clock::now();
  double total_time_sec =
      std::chrono::duration<double>(overall_end - overall_start).count();

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_output);
  cudaFree(d_temp);

  BenchmarkResult result;
  result.total_bytes = data_size * iterations;
  result.iterations = iterations;
  result.throughput_gbps =
      (result.total_bytes / (1024.0 * 1024.0 * 1024.0)) / total_time_sec;
  result.throughput_mbps = result.throughput_gbps * 1024.0;
  result.avg_latency_ms =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) /
      latencies.size();
  result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
  result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());

  return result;
}

void run_benchmark_throughput_sweep() {
  print_header("Throughput Sweep (decompress_to_preallocated)");

  std::vector<size_t> sizes = {
      64 * 1024,        // 64 KB
      256 * 1024,       // 256 KB
      1 * 1024 * 1024,  // 1 MB
      4 * 1024 * 1024,  // 4 MB
      16 * 1024 * 1024, // 16 MB
      64 * 1024 * 1024  // 64 MB
  };

  std::cout << "\nSize (KB)\tMB/s\t\tGB/s\t\tAvg Latency (ms)" << std::endl;
  std::cout << "--------\t----\t\t----\t\t-----------------" << std::endl;

  for (size_t size : sizes) {
    int iterations = (size < 1024 * 1024) ? 100 : 20;
    BenchmarkResult result =
        benchmark_decompress_to_preallocated(size, iterations);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << (size / 1024) << "\t\t" << result.throughput_mbps << "\t\t"
              << result.throughput_gbps << "\t\t" << result.avg_latency_ms
              << std::endl;
  }
}

// ============================================================================
// BENCHMARK 2: Zipper Pattern (Buffer Reuse) Latency
// ============================================================================

void run_benchmark_zipper_pattern() {
  print_header("Zipper Pattern (Buffer Reuse Latency)");

  const size_t layer_size =
      4 * 1024 * 1024;       // 4 MB per layer (simulates cold weights)
  const int num_layers = 20; // Simulate 20 layers
  const int iterations = 5;

  std::cout << "\nSimulating " << num_layers << " layers of "
            << (layer_size / (1024 * 1024)) << " MB each" << std::endl;

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  // Pre-allocate workspace and Zipper Buffer
  void *workspace;
  size_t workspace_size;
  manager.allocate_inference_workspace(layer_size, layer_size * 2, &workspace,
                                       &workspace_size);

  void *zipper_buffer;
  cudaMalloc(&zipper_buffer, layer_size);

  // Pre-compress all layers
  std::vector<void *> compressed_buffers(num_layers);
  std::vector<size_t> compressed_sizes(num_layers);

  void *d_input;
  cudaMalloc(&d_input, layer_size);

  size_t total_compressed = 0;
  for (int i = 0; i < num_layers; i++) {
    std::vector<uint8_t> h_data;
    generate_model_weight_data(h_data, layer_size);

    cudaMalloc(&compressed_buffers[i], layer_size * 2);
    cudaMemcpy(d_input, h_data.data(), layer_size, cudaMemcpyHostToDevice);

    compressed_sizes[i] = layer_size * 2;
    manager.compress(d_input, layer_size, compressed_buffers[i],
                     &compressed_sizes[i], workspace, workspace_size, nullptr,
                     0);
    total_compressed += compressed_sizes[i];
  }

  double compression_ratio =
      (num_layers * layer_size) / (double)total_compressed;
  std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
            << compression_ratio << ":1" << std::endl;

  // Warmup
  for (int i = 0; i < num_layers; i++) {
    size_t actual_size;
    manager.decompress_to_preallocated(
        compressed_buffers[i], compressed_sizes[i], zipper_buffer, layer_size,
        &actual_size, workspace, workspace_size, 0);
  }
  cudaDeviceSynchronize();

  // Benchmark multiple full passes
  std::vector<double> pass_times;
  for (int iter = 0; iter < iterations; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_layers; i++) {
      size_t actual_size;
      manager.decompress_to_preallocated(
          compressed_buffers[i], compressed_sizes[i], zipper_buffer, layer_size,
          &actual_size, workspace, workspace_size, 0);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();
    pass_times.push_back(elapsed);
  }

  double avg_pass_time =
      std::accumulate(pass_times.begin(), pass_times.end(), 0.0) /
      pass_times.size();
  double avg_layer_time = avg_pass_time / num_layers;
  double total_data = num_layers * layer_size;
  double throughput_gbps =
      (total_data / (1024.0 * 1024.0 * 1024.0)) / (avg_pass_time / 1000.0);

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Avg pass time:  " << std::fixed << std::setprecision(2)
            << avg_pass_time << " ms" << std::endl;
  std::cout << "  Avg layer time: " << std::fixed << std::setprecision(2)
            << avg_layer_time << " ms" << std::endl;
  std::cout << "  Throughput:     " << std::fixed << std::setprecision(2)
            << throughput_gbps << " GB/s" << std::endl;

  // Estimate tokens/sec for inference
  // Assumption: 70B model, ~80 layers, ~1.1 GB compressed cold data per token
  double estimated_cold_data_per_token = 1.1 * 1024 * 1024 * 1024; // 1.1 GB
  double token_time_sec =
      estimated_cold_data_per_token / (throughput_gbps * 1024 * 1024 * 1024);
  double tokens_per_sec = 1.0 / token_time_sec;

  std::cout << "\n  Estimated 70B Inference:" << std::endl;
  std::cout << "    Cold data per token: 1.1 GB" << std::endl;
  std::cout << "    Est. decompression latency: " << std::fixed
            << std::setprecision(0) << (token_time_sec * 1000) << " ms"
            << std::endl;
  std::cout << "    Est. tokens/sec (decomp only): " << std::fixed
            << std::setprecision(1) << tokens_per_sec << std::endl;

  // Cleanup
  manager.free_inference_workspace(workspace);
  cudaFree(zipper_buffer);
  cudaFree(d_input);
  for (int i = 0; i < num_layers; i++) {
    cudaFree(compressed_buffers[i]);
  }
}

// ============================================================================
// BENCHMARK 3: Comparison with Standard decompress_batch
// ============================================================================

void run_benchmark_preallocated_vs_standard() {
  print_header("Preallocated vs Standard decompress_batch");

  const int num_items = 10;
  const size_t item_size = 1 * 1024 * 1024; // 1 MB each
  const int iterations = 20;

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  // Pre-compress all items
  std::vector<void *> compressed_buffers(num_items);
  std::vector<void *> output_buffers(num_items);
  std::vector<size_t> compressed_sizes(num_items);

  void *d_temp;
  size_t temp_size = manager.get_compress_temp_size(item_size) * num_items;
  cudaMalloc(&d_temp, temp_size);

  void *d_input;
  cudaMalloc(&d_input, item_size);

  for (int i = 0; i < num_items; i++) {
    std::vector<uint8_t> h_data;
    generate_model_weight_data(h_data, item_size);

    cudaMalloc(&compressed_buffers[i], item_size * 2);
    cudaMalloc(&output_buffers[i],
               item_size); // Pre-allocate for preallocated test

    cudaMemcpy(d_input, h_data.data(), item_size, cudaMemcpyHostToDevice);

    compressed_sizes[i] = item_size * 2;
    manager.compress(d_input, item_size, compressed_buffers[i],
                     &compressed_sizes[i], d_temp, temp_size / num_items,
                     nullptr, 0);
  }

  // Prepare batch items for preallocated test
  std::vector<BatchItem> items_preallocated(num_items);
  for (int i = 0; i < num_items; i++) {
    items_preallocated[i].input_ptr = compressed_buffers[i];
    items_preallocated[i].input_size = compressed_sizes[i];
    items_preallocated[i].output_ptr = output_buffers[i];
    items_preallocated[i].output_size = item_size;
  }

  // Warmup
  for (int warmup = 0; warmup < 3; warmup++) {
    manager.decompress_batch_preallocated(items_preallocated, d_temp, temp_size,
                                          0);
  }
  cudaDeviceSynchronize();

  // Benchmark preallocated
  std::vector<double> times_preallocated;
  for (int iter = 0; iter < iterations; iter++) {
    // Reset output sizes
    for (int i = 0; i < num_items; i++) {
      items_preallocated[i].output_size = item_size;
    }

    auto start = std::chrono::high_resolution_clock::now();
    manager.decompress_batch_preallocated(items_preallocated, d_temp, temp_size,
                                          0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    times_preallocated.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  double avg_preallocated = std::accumulate(times_preallocated.begin(),
                                            times_preallocated.end(), 0.0) /
                            iterations;
  double throughput_preallocated =
      (num_items * item_size / (1024.0 * 1024.0 * 1024.0)) /
      (avg_preallocated / 1000.0);

  std::cout << "\nResults:" << std::endl;
  std::cout << "  decompress_batch_preallocated:" << std::endl;
  std::cout << "    Avg time:    " << std::fixed << std::setprecision(2)
            << avg_preallocated << " ms" << std::endl;
  std::cout << "    Throughput:  " << std::fixed << std::setprecision(2)
            << throughput_preallocated << " GB/s" << std::endl;

  // Cleanup
  cudaFree(d_temp);
  cudaFree(d_input);
  for (int i = 0; i < num_items; i++) {
    cudaFree(compressed_buffers[i]);
    cudaFree(output_buffers[i]);
  }
}

// ============================================================================
// BENCHMARK 4: Double-Buffer Pipelining Simulation
// ============================================================================

void run_benchmark_double_buffer_pipeline() {
  print_header("Double-Buffer Pipelining Simulation");

  const size_t layer_size = 2 * 1024 * 1024; // 2 MB per layer
  const int num_layers = 40;

  std::cout << "\nSimulating double-buffered pipeline:" << std::endl;
  std::cout << "  " << num_layers << " layers x "
            << (layer_size / (1024 * 1024)) << " MB" << std::endl;

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  // Allocate double buffer
  void *buffer_a, *buffer_b;
  cudaMalloc(&buffer_a, layer_size);
  cudaMalloc(&buffer_b, layer_size);

  void *workspace;
  size_t workspace_size;
  manager.allocate_inference_workspace(layer_size, layer_size * 2, &workspace,
                                       &workspace_size);

  // Create two streams for pipelining
  cudaStream_t stream_decompress, stream_compute;
  cudaStreamCreate(&stream_decompress);
  cudaStreamCreate(&stream_compute);

  // Pre-compress all layers
  std::vector<void *> compressed_buffers(num_layers);
  std::vector<size_t> compressed_sizes(num_layers);

  void *d_input;
  cudaMalloc(&d_input, layer_size);

  for (int i = 0; i < num_layers; i++) {
    std::vector<uint8_t> h_data;
    generate_model_weight_data(h_data, layer_size);

    cudaMalloc(&compressed_buffers[i], layer_size * 2);
    cudaMemcpy(d_input, h_data.data(), layer_size, cudaMemcpyHostToDevice);

    compressed_sizes[i] = layer_size * 2;
    manager.compress(d_input, layer_size, compressed_buffers[i],
                     &compressed_sizes[i], workspace, workspace_size, nullptr,
                     0);
  }

  // Simulate pipelined inference
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_layers; i++) {
    void *current_buffer = (i % 2 == 0) ? buffer_a : buffer_b;

    // Decompress current layer
    size_t actual_size;
    manager.decompress_to_preallocated(
        compressed_buffers[i], compressed_sizes[i], current_buffer, layer_size,
        &actual_size, workspace, workspace_size, 0);

    // In real inference, compute would overlap with next layer's decompress
    // For benchmark, we just measure sequential time
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  double total_data = num_layers * layer_size;
  double throughput_gbps =
      (total_data / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
  double avg_layer_ms = elapsed_ms / num_layers;

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Total time:     " << std::fixed << std::setprecision(2)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "  Avg per layer:  " << std::fixed << std::setprecision(2)
            << avg_layer_ms << " ms" << std::endl;
  std::cout << "  Throughput:     " << std::fixed << std::setprecision(2)
            << throughput_gbps << " GB/s" << std::endl;

  // Cleanup
  cudaStreamDestroy(stream_decompress);
  cudaStreamDestroy(stream_compute);
  manager.free_inference_workspace(workspace);
  cudaFree(buffer_a);
  cudaFree(buffer_b);
  cudaFree(d_input);
  for (int i = 0; i < num_layers; i++) {
    cudaFree(compressed_buffers[i]);
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "\n";
  print_separator();
  std::cout << "CUDA ZSTD - Inference API Benchmarks" << std::endl;
  print_separator();

  // Check CUDA device
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return 1;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::cout << "Device: " << props.name << std::endl;
  std::cout << "Compute Capability: " << props.major << "." << props.minor
            << std::endl;

  // Run benchmarks
  run_benchmark_throughput_sweep();
  run_benchmark_zipper_pattern();
  run_benchmark_preallocated_vs_standard();
  run_benchmark_double_buffer_pipeline();

  std::cout << "\n";
  print_separator();
  std::cout << "Benchmarks Complete" << std::endl;
  print_separator();

  return 0;
}
