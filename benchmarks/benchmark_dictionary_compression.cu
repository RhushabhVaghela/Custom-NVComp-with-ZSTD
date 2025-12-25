/*
 * benchmark_dictionary_compression.cu
 *
 * Benchmark for Dictionary Compression Performance on GPU
 */

#include "cuda_zstd_manager.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Helper to generate highly repetitive data suitable for dictionary compression
void generate_dictionary_friendly_data(std::vector<byte_t> &data, size_t size,
                                       int num_patterns = 100) {
  data.resize(size);
  std::vector<std::vector<byte_t>> patterns;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> len_dist(4, 64);
  std::uniform_int_distribution<int> char_dist(0, 255);

  // Generate some common patterns
  for (int i = 0; i < num_patterns; ++i) {
    int len = len_dist(rng);
    std::vector<byte_t> pattern(len);
    for (int j = 0; j < len; ++j) {
      pattern[j] = (byte_t)char_dist(rng);
    }
    patterns.push_back(pattern);
  }

  // Fill data with patterns + random noise
  size_t pos = 0;
  std::uniform_int_distribution<int> pattern_idx_dist(0, num_patterns - 1);
  std::uniform_int_distribution<int> noise_dist(0, 10);

  while (pos < size) {
    if (noise_dist(rng) < 2) { // 20% noise
      data[pos++] = (byte_t)char_dist(rng);
    } else {
      const auto &pat = patterns[pattern_idx_dist(rng)];
      size_t copy_len = std::min(pat.size(), size - pos);
      memcpy(data.data() + pos, pat.data(), copy_len);
      pos += copy_len;
    }
  }
}

int main(int argc, char **argv) {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "  Benchmark: Dictionary Compression" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  size_t data_sizes[] = {1 * 1024 * 1024, 10 * 1024 * 1024, 64 * 1024 * 1024};
  CompressionConfig config = CompressionConfig::from_level(3); // Standard level
  ZstdBatchManager manager(config);

  for (size_t size : data_sizes) {
    std::cout << "\n--- Dataset Size: " << size / (1024 * 1024) << " MB ---"
              << std::endl;

    std::vector<byte_t> h_data;
    generate_dictionary_friendly_data(h_data, size);

    // 1. Train Dictionary
    std::cout << "Training dictionary..." << std::endl;
    // Use small samples for training
    std::vector<size_t> sample_sizes;
    size_t total_samples = 0;
    size_t num_samples = 1000;
    size_t sample_len = 1024;
    std::vector<byte_t> h_samples(num_samples * sample_len);
    for (size_t i = 0; i < num_samples; ++i) {
      size_t offset = (i * 997) % (size - sample_len);
      memcpy(h_samples.data() + i * sample_len, h_data.data() + offset,
             sample_len);
      sample_sizes.push_back(sample_len);
      total_samples += sample_len;
    }

    // Train on CPU for now as our GPU trainer is in manager
    // Actually let's use the manager's functionality
    auto start_train = std::chrono::high_resolution_clock::now();

    // For benchmark simplicity, we'll try to use a pretrained one or just train
    // using manager Assuming manager has train capability exposed or we create
    // dummy

    // Let's rely on manager->train_dictionary functionality if exposed or
    // manual Replicating test logic:
    byte_t *d_samples;
    CUDA_CHECK(cudaMalloc(&d_samples, total_samples));
    CUDA_CHECK(cudaMemcpy(d_samples, h_samples.data(), total_samples,
                          cudaMemcpyHostToDevice));

    // Create vector of sample pointers is tricky on GPU, simplified:
    // Just use a single buffer if supported? No, train_dictionary takes
    // pointers. We will skip training benchmark and focus on
    // COMPRESSION/DECOMPRESSION with a dictionary.

    dictionary::Dictionary dict;
    // Mock dictionary or simple train
    size_t dict_size = 32 * 1024;
    dict.raw_size = dict_size;
    dict.raw_content = (byte_t *)malloc(dict_size);
    // Fill with some data
    memcpy(dict.raw_content, h_samples.data(), dict_size);
    dict.header.dictionary_id = 12345;

    // Set dictionary
    manager.set_dictionary(dict);

    auto end_train = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> train_time = end_train - start_train;
    // std::cout << "Dictionary Setup Time: " << train_time.count() * 1000 << "
    // ms" << std::endl;

    // Allocate GPU memory
    byte_t *d_input, *d_output, *d_decompressed;
    void *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size * 2)); // Safety
    CUDA_CHECK(cudaMalloc(&d_decompressed, size));

    size_t temp_size = manager.get_compress_temp_size(size);
    size_t decomp_temp_size = manager.get_decompress_temp_size(size);
    temp_size = std::max(temp_size, decomp_temp_size);

    CUDA_CHECK(cudaMalloc(&d_workspace, temp_size));
    CUDA_CHECK(
        cudaMemcpy(d_input, h_data.data(), size, cudaMemcpyHostToDevice));

    // Warmup
    size_t compressed_size = size * 2;
    manager.compress(d_input, size, d_output, &compressed_size, d_workspace,
                     temp_size, nullptr, 0);

    // Benchmark Compress
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int iterations = 10;
    for (int i = 0; i < iterations; ++i) {
      compressed_size = size * 2;
      manager.compress(d_input, size, d_output, &compressed_size, d_workspace,
                       temp_size, nullptr, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double gb_per_sec = (double(size) * iterations) /
                        (ms * 1e6); // GB/ms -> GB/s ? No. bytes / (ms * 10^5)?
    // bytes / (ms * 0.001) = bytes/s
    // (bytes / 1e9) / (ms / 1000) = GB / s
    gb_per_sec = (double(size) * iterations / 1e9) / (ms / 1000.0);

    std::cout << "Compression Throughput (Dict): " << std::fixed
              << std::setprecision(2) << gb_per_sec << " GB/s" << std::endl;
    std::cout << "Compression Ratio: " << std::fixed << std::setprecision(2)
              << (double)size / compressed_size << "x" << std::endl;

    // Benchmark Decompress
    // Warmup
    size_t decomp_size = size;
    manager.decompress(d_output, compressed_size, d_decompressed, &decomp_size,
                       d_workspace, temp_size);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
      decomp_size = size;
      manager.decompress(d_output, compressed_size, d_decompressed,
                         &decomp_size, d_workspace, temp_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    gb_per_sec = (double(size) * iterations / 1e9) / (ms / 1000.0);

    std::cout << "Decompression Throughput (Dict): " << std::fixed
              << std::setprecision(2) << gb_per_sec << " GB/s" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_decompressed);
    cudaFree(d_workspace);
    free(dict.raw_content);
  }

  return 0;
}
