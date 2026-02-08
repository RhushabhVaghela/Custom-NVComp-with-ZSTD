// ============================================================================
// benchmark_streaming.cu - Throughput Benchmark for Streaming API
// MODIFIED for RTX 5080 (16GB VRAM) - Reduced memory usage
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Hardware-safe constants
#define MAX_VRAM_CAP (8ULL * 1024 * 1024 * 1024)         // 8GB upper cap
#define MAX_CHUNK_SIZE (4ULL * 1024 * 1024)                 // Max 4MB chunk size
#define MAX_TOTAL_DATA (256ULL * 1024 * 1024)               // Max 256MB total data

// Dynamic VRAM limit: min(hardcoded cap, actual usable VRAM)
static size_t get_max_vram_budget() {
    size_t usable = cuda_zstd::get_usable_vram();
    if (usable == 0) return MAX_VRAM_CAP;
    return std::min((size_t)MAX_VRAM_CAP, usable);
}

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

#define BENCH_CHECK_STATUS(call)                                               \
  do {                                                                         \
    Status s = call;                                                           \
    if (s != Status::SUCCESS) {                                                \
      std::cerr << "Manager Error: " << (int)s << " at " << __FILE__ << ":"    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void print_header() {
  std::cout << std::left << std::setw(15) << "Type" << std::setw(15)
            << "Chunk Size" << std::setw(15) << "Data Size" << std::setw(15)
            << "Time (ms)" << std::setw(15) << "Throughput" << std::endl;
  std::cout << std::string(75, '-') << std::endl;
}

struct ChunkDesc {
  size_t offset;
  size_t size;
};

void run_streaming_benchmark(size_t chunk_size, int num_chunks) {
  size_t total_input_size = chunk_size * num_chunks;
  
  // Memory safety check
  if (chunk_size > MAX_CHUNK_SIZE) {
    std::cout << "Skipping - chunk size " << (chunk_size/1024/1024) << "MB exceeds safety limit\n";
    return;
  }
  if (total_input_size > MAX_TOTAL_DATA) {
    std::cout << "Skipping - total size " << (total_input_size/1024/1024) << "MB exceeds safety limit\n";
    return;
  }

  // 1. Prepare Input Data (Repeated pattern to ensure compressibility)
  std::vector<uint8_t> h_input_chunk(chunk_size);
  for (size_t i = 0; i < chunk_size; i++) {
    h_input_chunk[i] = (i % 256);
  }

  void *d_input_chunk;

  // Output buffer must be large enough to hold the ENTIRE stream for
  // decompression test
  size_t max_output_size = total_input_size * 1.5; // Safety margin
  void *d_stream_output;
  void *d_decomp_output_chunk;

  BENCH_CHECK(cuda_zstd::safe_cuda_malloc(&d_input_chunk, chunk_size));
  BENCH_CHECK(cuda_zstd::safe_cuda_malloc(&d_stream_output, max_output_size));
  BENCH_CHECK(cuda_zstd::safe_cuda_malloc(&d_decomp_output_chunk, chunk_size));

  BENCH_CHECK(cudaMemcpy(d_input_chunk, h_input_chunk.data(), chunk_size,
                         cudaMemcpyHostToDevice));

  // Metadata to track the stream
  std::vector<ChunkDesc> stream_chunks;
  stream_chunks.reserve(num_chunks);

  // Optimize: Force GPU execution (cpu_threshold = 0) to avoid host fallback
  // for small chunks
  CompressionConfig config;
  config.level = 1;
  config.cpu_threshold = 0; // FORCE GPU

  ZstdStreamingManager manager(config);

  // =================================================================================
  // COMPRESSION BENCHMARK
  // =================================================================================
  manager.init_compression();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  size_t current_output_offset = 0;

  for (int i = 0; i < num_chunks; i++) {
    bool is_last = (i == num_chunks - 1);

    // Calculate pointer to current output position
    byte_t *d_curr_out = (byte_t *)d_stream_output + current_output_offset;

    size_t produced = 0;

    // Use the SAME input chunk repeatedly, but compressor treats it as a
    // continuous stream because we don't reset_init.
    BENCH_CHECK_STATUS(manager.compress_chunk(d_input_chunk, chunk_size,
                                              d_curr_out, &produced, is_last));

    // Store metadata for decompression
    stream_chunks.push_back({current_output_offset, produced});
    current_output_offset += produced;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms_comp = 0;
  cudaEventElapsedTime(&ms_comp, start, stop);

  double mb_total = (double)total_input_size / (1024.0 * 1024.0);
  double mb_s_comp = mb_total / (ms_comp / 1000.0);

  std::cout << std::left << std::setw(15) << "Compress" << std::setw(15)
            << chunk_size << std::setw(15) << (int)mb_total << " MB"
            << std::setw(15) << std::fixed << std::setprecision(2) << ms_comp
            << std::setw(15) << std::setprecision(2) << mb_s_comp << " MB/s"
            << std::endl;

  // =================================================================================
  // DECOMPRESSION BENCHMARK
  // =================================================================================
  // Now we have a valid compressed stream in d_stream_output and metadata in
  // stream_chunks.

  manager.init_decompression();

  cudaEventRecord(start);

  for (int i = 0; i < num_chunks; i++) {
    bool is_last = (i == num_chunks - 1);
    ChunkDesc &desc = stream_chunks[i];

    byte_t *d_curr_in = (byte_t *)d_stream_output + desc.offset;

    // We reuse the single output chunk buffer because we don't need to save the
    // result
    size_t decomp_produced = chunk_size;

    BENCH_CHECK_STATUS(manager.decompress_chunk(d_curr_in, desc.size,
                                                d_decomp_output_chunk,
                                                &decomp_produced, &is_last));
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms_decomp = 0;
  cudaEventElapsedTime(&ms_decomp, start, stop);

  double mb_s_decomp = mb_total / (ms_decomp / 1000.0);

  std::cout << std::left << std::setw(15) << "Decompress" << std::setw(15)
            << chunk_size << std::setw(15) << (int)mb_total << " MB"
            << std::setw(15) << std::fixed << std::setprecision(2) << ms_decomp
            << std::setw(15) << std::setprecision(2) << mb_s_decomp << " MB/s"
            << std::endl;

  // Cleanup
  cudaFree(d_input_chunk);
  cudaFree(d_stream_output);
  cudaFree(d_decomp_output_chunk);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  size_t vram_budget = get_max_vram_budget();
  std::cout << "Streaming Benchmark" << std::endl;
  std::cout << "Max Chunk Size: " << (MAX_CHUNK_SIZE/1024/1024) << " MB" << std::endl;
  std::cout << "Max Total Data: " << (MAX_TOTAL_DATA/1024/1024) << " MB" << std::endl;
  std::cout << "Usable VRAM:    " << (vram_budget/1024/1024) << " MB" << std::endl << std::endl;
  
  print_header();

  // Test various chunk sizes (reduced for safety)
  // Small Chunks (Latency dominated)
  run_streaming_benchmark(4096, 5000); // 4KB chunks, 20MB total

  // Medium Chunks
  run_streaming_benchmark(65536, 2000); // 64KB chunks, 130MB total

  // Large Chunks (Throughput dominated) - reduced from 200 to 100 chunks
  run_streaming_benchmark(1048576, 100); // 1MB chunks, 100MB total

  // Very Large - removed 4MB chunks test as it's close to safety limit
  // run_streaming_benchmark(4 * 1048576, 50); // 4MB chunks, 200MB total

  return 0;
}
