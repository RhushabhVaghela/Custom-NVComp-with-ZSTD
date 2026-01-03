// benchmark_pipeline_streaming.cu
// Benchmarks the Double-Buffered / Async Pipeline performance of FSE Encoding
// Simulates 50GB file compression by looping over a 1GB buffer.

#include "cuda_zstd_manager.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace cuda_zstd;

// --- Test Data Generation ---
void generate_random_data(void *ptr, size_t size) {
  // FILL WITH REPEATING PATTERN FOR HIGH COMPRESSIBILITY
  memset(ptr, 'A', size);
}

int main(int argc, char **argv) {
  size_t CHUNK_SIZE = 16 * 1024 * 1024;          // 16 MB chunks
  size_t TOTAL_SIZE = 5ULL * 1024 * 1024 * 1024; // 5 GB Total (Simulated)
  // Note: Using 5GB to keep runtime reasonable but large enough for stability.

  if (argc > 1)
    CHUNK_SIZE = std::stoull(argv[1]);
  if (argc > 2)
    TOTAL_SIZE = std::stoull(argv[2]) * 1024 * 1024;

  std::cout << "Pipeline Benchmark (FSE Encoding)" << std::endl;
  std::cout << "Chunk Size: " << (CHUNK_SIZE / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Total Size: " << (TOTAL_SIZE / 1024 / 1024) << " MB"
            << std::endl;

  // 1. Setup Manager
  auto manager = create_streaming_manager(3);
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  manager->init_compression(
      stream, CHUNK_SIZE); // Preallocates Tables + Workspace for 16MB

  // 2. Allocate Host Buffers (Pinned for Async Copy)
  void *h_input, *h_output;
  CHECK_CUDA(cudaMallocHost(&h_input, CHUNK_SIZE));
  CHECK_CUDA(cudaMallocHost(&h_output, CHUNK_SIZE * 2)); // Conservative

  generate_random_data(h_input, CHUNK_SIZE);

  // 3. Warmup
  size_t out_size = CHUNK_SIZE * 2;
  manager->compress_chunk(h_input, CHUNK_SIZE, h_output, &out_size, false,
                          stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 4. Run Benchmark
  size_t iterations = TOTAL_SIZE / CHUNK_SIZE;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    size_t compressed_size = CHUNK_SIZE * 2;
    // Note: Reusing same input/output buffer for simulation.
    // In real pipeline, h_input would be evolving.
    // The Manager handles H2D copy internally in `compress_chunk`.
    // It relies on Async Copy if h_input is Pinned.

    Status s =
        manager->compress_chunk(h_input, CHUNK_SIZE, h_output, &compressed_size,
                                (i == iterations - 1), stream);
    if (s != Status::SUCCESS) {
      std::cerr << "Compression Failed: " << (int)s << std::endl;
      break;
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end_time = std::chrono::high_resolution_clock::now();

  double duration_sec =
      std::chrono::duration<double>(end_time - start_time).count();
  double throughput_gbps =
      (double)TOTAL_SIZE / duration_sec / (1024.0 * 1024.0 * 1024.0);

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Duration: " << duration_sec << " s" << std::endl;
  std::cout << "Throughput: " << throughput_gbps << " GB/s" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  // Cleanup
  CHECK_CUDA(cudaFreeHost(h_input));
  CHECK_CUDA(cudaFreeHost(h_output));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return 0;
}
