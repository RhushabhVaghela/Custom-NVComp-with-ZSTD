// Extended Validation Tests for Parallel Compression
#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

bool test_compression(const char *test_name, size_t input_size,
                      bool use_random = false) {
  std::cout << "\n=== " << test_name << " ===" << std::endl;
  std::cout << "Input Size: " << input_size / (1024.0 * 1024.0) << " MB"
            << std::endl;

  // Generate input data
  std::vector<uint8_t> h_input(input_size);
  if (use_random) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < input_size; ++i) {
      h_input[i] = dist(rng);
    }
  } else {
    for (size_t i = 0; i < input_size; ++i) {
      h_input[i] = (uint8_t)(i % 256);
    }
  }

  // Initialize manager
  CompressionConfig config = CompressionConfig::from_level(3);
  ZstdBatchManager manager(config);

  // Allocate GPU buffers
  void *d_input;
  if (cuda_zstd::safe_cuda_malloc(&d_input, input_size) != cudaSuccess) {
    std::cerr << "Failed to allocate d_input (" << input_size << " bytes)" << std::endl;
    return false;
  }
  if (cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cerr << "Failed to copy input to device" << std::endl;
    cudaFree(d_input);
    return false;
  }

  size_t max_compressed_size = manager.get_max_compressed_size(input_size);
  void *d_compressed;
  if (cuda_zstd::safe_cuda_malloc(&d_compressed, max_compressed_size) != cudaSuccess) {
    std::cerr << "Failed to allocate d_compressed (" << max_compressed_size << " bytes)" << std::endl;
    cudaFree(d_input);
    return false;
  }

  size_t temp_size = manager.get_compress_temp_size(input_size);
  void *d_temp;
  if (cuda_zstd::safe_cuda_malloc(&d_temp, temp_size) != cudaSuccess) {
    std::cerr << "Failed to allocate d_temp (" << temp_size << " bytes)" << std::endl;
    cudaFree(d_input);
    cudaFree(d_compressed);
    return false;
  }

  // Compress
  auto start = std::chrono::high_resolution_clock::now();

  size_t compressed_size = max_compressed_size;
  Status status =
      manager.compress(d_input, input_size, d_compressed, &compressed_size,
                       d_temp, temp_size, nullptr, 0, 0);

  auto compress_end = std::chrono::high_resolution_clock::now();

  if (status != Status::SUCCESS) {
    std::cerr << "Compression Failed: " << (int)status << std::endl;
    std::cerr.flush();
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    return false;
  }

  double ratio = (double)input_size / compressed_size;
  std::cout << "Compressed Size: " << compressed_size / (1024.0 * 1024.0)
            << " MB" << std::endl;
  std::cout << "Compression Ratio: " << ratio << std::endl;

  // Decompress
  void *d_decompressed;
  if (cuda_zstd::safe_cuda_malloc(&d_decompressed, input_size) != cudaSuccess) {
    std::cerr << "Failed to allocate d_decompressed (" << input_size << " bytes)" << std::endl;
    std::cerr.flush();
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    return false;
  }

  size_t decompressed_size = input_size;
  status = manager.decompress(d_compressed, compressed_size, d_decompressed,
                              &decompressed_size, d_temp, temp_size, 0);

  auto decompress_end = std::chrono::high_resolution_clock::now();

  if (status != Status::SUCCESS) {
    std::cerr << "Decompression Failed: " << (int)status << std::endl;
    std::cerr.flush();
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    cudaFree(d_decompressed);
    return false;
  }

  // Verify
  std::vector<uint8_t> h_output(input_size);
  if (cudaMemcpy(h_output.data(), d_decompressed, input_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    cudaFree(d_temp);
    cudaFree(d_decompressed);
    return false;
  }

  bool match = (h_input == h_output);

  // Timing
  auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         compress_end - start)
                         .count();
  auto decompress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           decompress_end - compress_end)
                           .count();

  std::cout << "Compress Time: " << compress_ms << " ms" << std::endl;
  std::cout << "Decompress Time: " << decompress_ms << " ms" << std::endl;
  std::cout << "Compress Throughput: "
            << (input_size / (1024.0 * 1024.0)) / (compress_ms / 1000.0)
            << " MB/s" << std::endl;

  if (match) {
    std::cout << "✅ Verification PASSED" << std::endl;
  } else {
    std::cerr << "❌ Verification FAILED" << std::endl;
    for (size_t i = 0; i < input_size; ++i) {
      if (h_input[i] != h_output[i]) {
        std::cerr << "First mismatch at index " << i
                  << ": input=" << (int)h_input[i]
                  << ", output=" << (int)h_output[i] << std::endl;
        break;
      }
    }
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_temp);
  cudaFree(d_decompressed);

  return match;
}

int main() {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "   CUDA-ZSTD Extended Validation Test Suite" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  int passed = 0;
  int total = 0;

  // Test 1: 2MB (small, below original test size)
  total++;
  if (test_compression("Test 1: 2MB Sequential Pattern", 2 * 1024 * 1024,
                       false)) {
    passed++;
  }

  // Test 2: 2MB (Hardware limit safe)
  total++;
  if (test_compression("Test 2: 2MB Sequential Pattern (Safe Limit)",
                       2 * 1024 * 1024, false)) {
    passed++;
  }

  // Test 3: 1MB (Variation)
  total++;
  if (test_compression("Test 3: 1MB Sequential Pattern", 1 * 1024 * 1024,
                       false)) {
    passed++;
  }

  // Test 4: 2MB Random (incompressible)
  total++;
  if (test_compression("Test 4: 2MB Random Data", 2 * 1024 * 1024, true)) {
    passed++;
  }

  // Test 5: Large Sequential (Memory Verification)
  // Dynamically pick the largest size whose workspace fits in usable VRAM.
  // get_compress_temp_size() scales O(N_blocks^2) so 128MB needs ~13 GB workspace.
  // Try sizes from 128MB down to 4MB; skip entirely if even 4MB doesn't fit.
  {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    const size_t candidates[] = {128ULL*1024*1024, 64ULL*1024*1024,
                                 32ULL*1024*1024, 16ULL*1024*1024,
                                 8ULL*1024*1024, 4ULL*1024*1024};
    size_t chosen_size = 0;

    CompressionConfig probe_config = CompressionConfig::from_level(3);
    ZstdBatchManager probe_mgr(probe_config);

    for (size_t sz : candidates) {
      size_t temp_need = probe_mgr.get_compress_temp_size(sz);
      size_t input_need = sz;
      size_t output_need = probe_mgr.get_max_compressed_size(sz);
      size_t decomp_need = sz; // decompressed buffer
      size_t total_need = temp_need + input_need + output_need + decomp_need
                          + VRAM_SAFETY_BUFFER_BYTES;
      if (free_mem >= total_need) {
        chosen_size = sz;
        break;
      }
    }

    if (chosen_size > 0) {
      total++;
      std::string label = "Test 5: " + std::to_string(chosen_size / (1024*1024))
                          + "MB Sequential (Memory Verification)";
      if (test_compression(label.c_str(), chosen_size, false)) {
        passed++;
      }
    } else {
      std::cout << "\n=== Test 5: Large Sequential (Memory Verification) ===" << std::endl;
      std::cout << "[SKIP] Insufficient VRAM for any candidate size (free: "
                << free_mem / (1024*1024) << " MB)" << std::endl;
    }
  }

  // Summary
  std::cout << "\n=================================================="
            << std::endl;
  std::cout << "Test Results: " << passed << "/" << total << " passed"
            << std::endl;
  std::cout << "=================================================="
            << std::endl;

  return (passed == total) ? 0 : 1;
}
