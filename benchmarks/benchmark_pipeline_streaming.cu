// benchmark_pipeline_streaming.cu
// Benchmarks the Double-Buffered / Async Pipeline performance of FSE Encoding
// MODIFIED for RTX 5080 (16GB VRAM) - Reduced memory usage

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

// Hardware-safe constants for RTX 5080 (16GB VRAM)
#define MAX_VRAM_PER_BENCHMARK                                                 \
  (8ULL * 1024 * 1024 * 1024)                // Max 8GB VRAM per benchmark
#define SAFE_CHUNK_SIZE (4ULL * 1024 * 1024) // 4 MB chunks (reduced from 16MB)
#define MAX_TOTAL_SIZE                                                         \
  (2ULL * 1024 * 1024 * 1024) // 2 GB total (reduced from 5GB)

using namespace cuda_zstd;

// --- Test Data Generation ---
void generate_random_data(void *ptr, size_t size) {
  // FILL WITH REPEATING PATTERN FOR HIGH COMPRESSIBILITY
  memset(ptr, 'A', size);
}

int main(int argc, char **argv) {
  size_t CHUNK_SIZE = SAFE_CHUNK_SIZE; // 4 MB chunks
  size_t TOTAL_SIZE = MAX_TOTAL_SIZE;  // 2 GB Total

  if (argc > 1)
    CHUNK_SIZE = std::stoull(argv[1]);
  if (argc > 2)
    TOTAL_SIZE = std::stoull(argv[2]) * 1024 * 1024;

  // Memory safety check
  if (CHUNK_SIZE > 8 * 1024 * 1024) {
    std::cerr << "WARNING: Chunk size too large for safety, capping at 8MB\n";
    CHUNK_SIZE = 8 * 1024 * 1024;
  }
  if (TOTAL_SIZE > 4ULL * 1024 * 1024 * 1024) {
    std::cerr << "WARNING: Total size too large for safety, capping at 4GB\n";
    TOTAL_SIZE = 4ULL * 1024 * 1024 * 1024;
  }

  std::cout << "Pipeline Benchmark (FSE Encoding) - RTX 5080 Safe Mode"
            << std::endl;
  std::cout << "Chunk Size: " << (CHUNK_SIZE / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Total Size: " << (TOTAL_SIZE / 1024 / 1024) << " MB"
            << std::endl;
  size_t est_vram = (CHUNK_SIZE * 3) / (1024 * 1024);
  std::cout << "Estimated VRAM Usage: ~" << est_vram << " MB" << std::endl;

  // 1. Setup Manager
  auto manager = create_streaming_manager(3);
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  manager->init_compression(
      stream, CHUNK_SIZE); // Preallocates Tables + Workspace for 4MB

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
