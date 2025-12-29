#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define BENCH_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct BenchmarkResult {
  size_t size;
  size_t batch_count;
  double compress_mbps;
  double decompress_mbps;
};

BenchmarkResult run_batch_benchmark(ZstdBatchManager &manager, size_t size,
                                    int batch_count, cudaStream_t stream) {
  // 1. Prepare data
  size_t total_input_size = size * batch_count;
  size_t total_output_size = total_input_size * 2; // Safe upper bound

  std::vector<uint8_t> h_input(size);
  // Fill with compressible data (repeating pattern)
  for (size_t i = 0; i < size; i++)
    h_input[i] = (i % 256);

  // Replicate to full batch buffer on host for convenience, or just map
  // pointers We will allocate ONE large device buffer and point into it
  void *d_input_base;
  void *d_output_base;
  void *d_decompressed_base;

  BENCH_CHECK(cudaMalloc(&d_input_base, total_input_size));
  BENCH_CHECK(cudaMalloc(&d_output_base, total_output_size));
  BENCH_CHECK(cudaMalloc(&d_decompressed_base, total_input_size));

  // Fill device input
  // Copy the single item pattern 'batch_count' times
  for (int i = 0; i < batch_count; i++) {
    BENCH_CHECK(cudaMemcpyAsync((uint8_t *)d_input_base + i * size,
                                h_input.data(), size, cudaMemcpyHostToDevice,
                                stream));
  }
  BENCH_CHECK(cudaStreamSynchronize(stream));

  // 2. Prepare BatchItems
  std::vector<BatchItem> items(batch_count);
  std::vector<size_t> input_sizes(batch_count, size);
  std::vector<size_t> compressed_sizes_vec(batch_count);

  for (int i = 0; i < batch_count; i++) {
    items[i].input_ptr = (uint8_t *)d_input_base + i * size;
    items[i].input_size = size;
    items[i].output_ptr =
        (uint8_t *)d_output_base + i * size * 2; // Stride 2x to be safe
    items[i].output_size = size * 2;
  }

  // 3. Workspace
  size_t temp_size = manager.get_batch_compress_temp_size(input_sizes);
  void *d_temp;
  BENCH_CHECK(cudaMalloc(&d_temp, temp_size));

  // 4. Warmup
  manager.compress_batch(items, d_temp, temp_size, stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  // 5. Compress Benchmark
  auto start_c = std::chrono::high_resolution_clock::now();

  // Run multiple iterations if batch is small
  int iterations = (total_input_size < 100 * 1024 * 1024) ? 10 : 1;

  for (int k = 0; k < iterations; k++) {
    // Reset output sizes just in case
    for (int i = 0; i < batch_count; i++)
      items[i].output_size = size * 2;
    manager.compress_batch(items, d_temp, temp_size, stream);
  }
  BENCH_CHECK(cudaStreamSynchronize(stream));
  auto end_c = std::chrono::high_resolution_clock::now();

  double ms_c =
      std::chrono::duration<double, std::milli>(end_c - start_c).count();
  double total_bytes = (double)total_input_size * iterations;
  double c_mbps = (total_bytes / (1024.0 * 1024.0)) / (ms_c / 1000.0);

  // Update compressed sizes for decompression
  for (int i = 0; i < batch_count; i++) {
    if (items[i].status != Status::SUCCESS) {
      std::cerr << "Compression failed for item " << i
                << " Status=" << (int)items[i].status << std::endl;
      exit(1);
    }
    compressed_sizes_vec[i] = items[i].output_size;
    if (i == 0)
      std::cout << "Item 0 compressed size: " << items[i].output_size
                << std::endl;
  }

  cudaFree(d_temp);

  // 6. Decompress Benchmark
  // Update items for decompression
  // Swap input/output for decompression semantics?
  // BatchItem struct usage in decompress_batch:
  // input_ptr = compressed data
  // output_ptr = decompressed buffer

  // We reuse the items vector but need to ensure pointers are correct
  std::vector<BatchItem> d_items(batch_count);
  for (int i = 0; i < batch_count; i++) {
    d_items[i].input_ptr = items[i].output_ptr;   // compressed data
    d_items[i].input_size = items[i].output_size; // actual compressed size
    d_items[i].output_ptr = (uint8_t *)d_decompressed_base + i * size;
    d_items[i].output_size = size;
  }

  size_t d_temp_size =
      manager.get_batch_decompress_temp_size(compressed_sizes_vec);
  BENCH_CHECK(cudaMalloc(&d_temp, d_temp_size));

  // Warmup
  manager.decompress_batch(d_items, d_temp, d_temp_size, stream);
  BENCH_CHECK(cudaStreamSynchronize(stream));

  auto start_d = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < iterations; k++) {
    // Reset sizes
    for (int i = 0; i < batch_count; i++)
      d_items[i].output_size = size;
    manager.decompress_batch(d_items, d_temp, d_temp_size, stream);
  }
  BENCH_CHECK(cudaStreamSynchronize(stream));
  auto end_d = std::chrono::high_resolution_clock::now();

  double ms_d =
      std::chrono::duration<double, std::milli>(end_d - start_d).count();
  double d_mbps = (total_bytes / (1024.0 * 1024.0)) / (ms_d / 1000.0);

  cudaFree(d_temp);
  cudaFree(d_input_base);
  cudaFree(d_output_base);
  cudaFree(d_decompressed_base);

  return {size, (size_t)batch_count, c_mbps, d_mbps};
}

BenchmarkResult run_graph_benchmark(ZstdBatchManager &manager, size_t size,
                                    int batch_count, cudaStream_t stream) {
  // 1. Prepare data (Same as normal)
  size_t total_input_size = size * batch_count;
  size_t total_output_size = total_input_size * 2;

  void *d_input_base;
  void *d_output_base;
  void *d_decompressed_base;

  BENCH_CHECK(cudaMalloc(&d_input_base, total_input_size));
  BENCH_CHECK(cudaMalloc(&d_output_base, total_output_size));
  BENCH_CHECK(cudaMalloc(&d_decompressed_base, total_input_size));

  std::vector<uint8_t> h_input(size);
  for (size_t i = 0; i < size; i++)
    h_input[i] = (i % 256);
  for (int i = 0; i < batch_count; i++) {
    BENCH_CHECK(cudaMemcpyAsync((uint8_t *)d_input_base + i * size,
                                h_input.data(), size, cudaMemcpyHostToDevice,
                                stream));
  }
  BENCH_CHECK(cudaStreamSynchronize(stream));

  // 2. BatchItems
  std::vector<BatchItem> items(batch_count);
  std::vector<size_t> input_sizes(batch_count, size);
  std::vector<size_t> compressed_sizes_vec(batch_count);

  for (int i = 0; i < batch_count; i++) {
    items[i].input_ptr = (uint8_t *)d_input_base + i * size;
    items[i].input_size = size;
    items[i].output_ptr = (uint8_t *)d_output_base + i * size * 2;
    items[i].output_size = size * 2;
  }

  // 3. Workspace
  size_t per_item_temp =
      manager.get_max_compressed_size(size) + 4000000; // rough estimate
  // Better: get actual temp size for one item
  std::vector<size_t> single_size = {size};
  size_t one_temp_size = manager.get_batch_compress_temp_size(single_size);

  void *d_temp_all;
  // Reuse single workspace for serialized graph
  BENCH_CHECK(cudaMalloc(&d_temp_all, one_temp_size));

  // 4. Capture Graph
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < batch_count; i++) {
    manager.compress(items[i].input_ptr, items[i].input_size,
                     items[i].output_ptr, &items[i].output_size, d_temp_all,
                     one_temp_size, nullptr, 0, stream);
  }
  cudaStreamEndCapture(stream, &graph);
  BENCH_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // 5. Benchmark
  auto start_c = std::chrono::high_resolution_clock::now();
  int iterations = (batch_count < 1000) ? 10 : 1;
  for (int k = 0; k < iterations; k++) {
    BENCH_CHECK(cudaGraphLaunch(graphExec, stream));
  }
  BENCH_CHECK(cudaStreamSynchronize(stream));
  auto end_c = std::chrono::high_resolution_clock::now();

  double ms_c =
      std::chrono::duration<double, std::milli>(end_c - start_c).count();
  double total_bytes = (double)total_input_size * iterations;
  double c_mbps = (total_bytes / (1024.0 * 1024.0)) / (ms_c / 1000.0);

  BENCH_CHECK(cudaGraphExecDestroy(graphExec));
  BENCH_CHECK(cudaGraphDestroy(graph));
  cudaFree(d_temp_all);
  cudaFree(d_input_base);
  cudaFree(d_output_base);
  cudaFree(d_decompressed_base);

  return {size, (size_t)batch_count, c_mbps, 0.0};
}

int main() {
  // Use Level 1 (Greedy) to avoid CPU-based Backtracking (Level 3/Optimal)
  auto manager = create_batch_manager(1);
  cudaStream_t stream;
  BENCH_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // --- Batch Performance Section ---
  std::cout << "=== ZSTD GPU Batch Performance ===" << std::endl;
  std::cout << "Target: >10GB/s (Batched)" << std::endl;
  std::cout << std::setw(10) << "Size" << " | " << std::setw(8) << "Batch"
            << " | " << std::setw(15) << "Compress (MB/s)" << " | "
            << std::setw(15) << "Decompress (MB/s)" << std::endl;
  std::cout << "-------------------------------------------------------------"
            << std::endl;

  // Test cases: {size, batch_count}
  std::cout << "Skipping Standard Batch (Sequential) - Running Parallel "
               "Multi-Manager Batch..."
            << std::endl;
  std::cout << "Using OpenMP Max Threads: " << omp_get_max_threads()
            << std::endl;

  std::vector<std::pair<size_t, int>> batch_cases = {
      {4096, 2000},
      {16384, 1500},
      {64 * 1024, 1000},
      {256 * 1024, 500},
  };

  for (auto &c : batch_cases) {
    size_t chunk_size = c.first;
    int batch_cnt = c.second;

    std::cout << "Running Parallel Batch: Size=" << chunk_size
              << ", Count=" << batch_cnt << "..." << std::endl;

    // Allocate host buffer with pattern
    std::vector<uint8_t> h_pattern(chunk_size);
    for (size_t j = 0; j < chunk_size; ++j)
      h_pattern[j] = (j % 256);

    // device pointers vectors
    std::vector<void *> d_inputs(batch_cnt);
    std::vector<void *> d_outputs(batch_cnt);
    std::vector<size_t> d_out_sizes_host(batch_cnt,
                                         chunk_size * 2); // init with capacity

    // Allocate all device buffers (Sequential alloc, parallel execute)
    // Minimizing alloc overhead via one big malloc? To keep it simple, discrete
    // mallocs (Simulate real world fragmentation).
    for (int i = 0; i < batch_cnt; ++i) {
      BENCH_CHECK(cudaMalloc(&d_inputs[i], chunk_size));
      BENCH_CHECK(cudaMalloc(&d_outputs[i], chunk_size * 2));
      BENCH_CHECK(cudaMemcpy(d_inputs[i], h_pattern.data(), chunk_size,
                             cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
      // Each thread creates its own Manager (Level 1 Greedy)
      auto mgr = create_batch_manager(1);
      cudaStream_t t_stream;
      cudaStreamCreate(&t_stream);

#pragma omp for schedule(dynamic)
      for (int i = 0; i < batch_cnt; ++i) {
        size_t out_size_temp = chunk_size * 2;
        mgr->compress(d_inputs[i], chunk_size, d_outputs[i], &out_size_temp,
                      nullptr, 0, nullptr, 0, t_stream);
        // We can optionally sync stream here if we want to measure "Latency per
        // item" but for throughput we let them queue. However, since we destroy
        // stream at end of parallel region, we must sync at end of region.
      }

      cudaStreamSynchronize(t_stream);
      cudaStreamDestroy(t_stream);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gb_s = (double)(batch_cnt * chunk_size) / (ms * 1e6);

    std::cout << "Parallel Batch " << chunk_size << " | " << batch_cnt << " | "
              << std::fixed << std::setprecision(2) << ms << " ms | " << gb_s
              << " GB/s" << std::endl;

    // Cleanup
    for (int i = 0; i < batch_cnt; ++i) {
      cudaFree(d_inputs[i]);
      cudaFree(d_outputs[i]);
    }
  }

  cudaStreamDestroy(stream);
  return 0;
}
