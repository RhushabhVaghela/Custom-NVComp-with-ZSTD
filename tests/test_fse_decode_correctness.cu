#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __LINE__ << ": "                        \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  }

// Generate reproducible test data
void generate_test_data(std::vector<byte_t> &data, size_t size, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);

  // Fill with random data
  // For larger sizes, we can use a faster generation method or just fill
  if (size > 1024 * 1024) {
    // Fast fill for large data to avoid slow test startup
    for (size_t i = 0; i < size; ++i) {
      data[i] = (byte_t)(i % 256);
    }
    // scramble a bit
    for (size_t i = 0; i < size; i += 1024) {
      data[i] = dist(rng);
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      if (i % 32 < 16) {
        data[i] = dist(rng) % 64; // Low entropy region
      } else {
        data[i] = dist(rng); // High entropy
      }
    }
  }
}

bool test_roundtrip(size_t size, int seed, float *enc_gbps, float *dec_gbps) {
  std::cout << "  Testing " << size << " bytes ("
            << (float)size / (1024.0f * 1024.0f) << " MB)..." << std::endl;

  // Use pinned memory for faster host-device transfers on larger sizes
  byte_t *h_input;
  CHECK_CUDA(cudaMallocHost(&h_input, size));

  // Fill input
  std::vector<byte_t> temp_input(size);
  generate_test_data(temp_input, size, seed);
  memcpy(h_input, temp_input.data(), size);

  byte_t *d_input, *d_compressed, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, size));
  size_t max_compressed_size =
      size + 512 + (size / 128); // Add slight overhead buffer
  CHECK_CUDA(cudaMalloc(&d_compressed, max_compressed_size));
  CHECK_CUDA(cudaMalloc(&d_output, size));
  CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

  cudaStream_t stream = 0;

  u32 *d_comp_size;
  CHECK_CUDA(cudaMalloc(&d_comp_size, sizeof(u32)));
  CHECK_CUDA(cudaMemset(d_comp_size, 0, sizeof(u32)));

  // Setup Timing Events
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // --- Compression ---
  std::cout << "    Compressing..." << std::endl;
  u64 *d_offsets = nullptr;
  CHECK_CUDA(cudaEventRecord(start, stream));
  Status status =
      fse::encode_fse_advanced(d_input, (u32)size, d_compressed, d_comp_size,
                               true, stream, nullptr, &d_offsets);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (compress: " << (int)status << ")" << std::endl;
    return false;
  }

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  if (enc_gbps)
    *enc_gbps = (size / 1e9f) / (ms / 1000.0f);
  // std::cout << "    Enc Time: " << ms << " ms (" << *enc_gbps << " GB/s)" <<
  // std::endl;

  u32 h_comp_size = 0;
  CHECK_CUDA(cudaMemcpy(&h_comp_size, d_comp_size, sizeof(u32),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_comp_size));

  // std::cout << "    Compressed size: " << h_comp_size << " bytes (" <<
  // (float)h_comp_size/size * 100.0f << "%)" << std::endl;

  // --- Decompression ---
  std::cout << "    Decompressing..." << std::endl;
  CHECK_CUDA(cudaGetLastError()); // Clear error

  u32 decompressed_size = 0;
  CHECK_CUDA(cudaEventRecord(start, stream));
  status = fse::decode_fse(d_compressed, h_comp_size, d_output,
                           &decompressed_size, d_offsets, stream);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());

  if (status != Status::SUCCESS) {
    std::cerr << " FAILED (decompress: " << (int)status << ")" << std::endl;
    return false;
  }

  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  if (dec_gbps)
    *dec_gbps = (size / 1e9f) / (ms / 1000.0f);
  // std::cout << "    Dec Time: " << ms << " ms (" << *dec_gbps << " GB/s)" <<
  // std::endl;

  std::cout << "    Verifying..." << std::endl;

  // Verify on host
  byte_t *h_output;
  CHECK_CUDA(cudaMallocHost(&h_output, size));
  CHECK_CUDA(cudaMemcpy(h_output, d_output, decompressed_size,
                        cudaMemcpyDeviceToHost));

  bool match = (decompressed_size == size);
  if (!match) {
    std::cerr << "SIZE MISMATCH: Expected " << size << ", Got "
              << decompressed_size << std::endl;
  } else {
    // Check content
    int errors = 0;
    // For large validations, check samples or use simple loops
    for (size_t i = 0; i < size; ++i) {
      if (h_output[i] != h_input[i]) {
        if (errors < 10) {
          std::cerr << " Mismatch at " << i << ": In=" << (int)h_input[i]
                    << " Out=" << (int)h_output[i] << std::endl;
        }
        match = false;
        errors++;
        if (errors > 100)
          break;
      }
    }
    if (errors > 0)
      std::cout << "    Total errors: " << errors << std::endl;
  }

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_compressed));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFreeHost(h_input));
  CHECK_CUDA(cudaFreeHost(h_output));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  if (d_offsets)
    CHECK_CUDA(cudaFree(d_offsets));

  return match;
}

int main() {
  setenv("CUDA_ZSTD_FSE_THRESHOLD", "0", 1); // Enable for GPU-only debugging

  std::cout << "======================================" << std::endl;
  std::cout << "FSE Scalability & Correctness Tests" << std::endl;
  std::cout << "======================================" << std::endl;

  int passed = 0;
  int failed = 0;

  // Progressive sizes: 1KB, 1MB, 10MB, 100MB, 512MB, 1GB
  // Note: 1GB Input + 1GB Output + 1GB Comp + Workspace ~= 3GB VRAM.
  // RTX 5080 (16GB) should handle 1GB easily.
  std::vector<size_t> sizes = {1024,
                               1024 * 1024,
                               10 * 1024 * 1024,
                               100 * 1024 * 1024,
                               512 * 1024 * 1024,
                               1024 * 1024 * 1024};

  // Warmup
  float d_dummy;
  std::cout << "Warmup Run (1MB)..." << std::endl;
  test_roundtrip(1024 * 1024, 123, &d_dummy, &d_dummy);
  std::cout << "Warmup Complete.\n" << std::endl;

  printf("%-15s | %-15s | %-15s | %-10s\n", "Size", "Encode GB/s",
         "Decode GB/s", "Status");
  printf("----------------------------------------------------------------\n");

  for (size_t size : sizes) {
    float enc_perf = 0;
    float dec_perf = 0;
    if (test_roundtrip(size, 42, &enc_perf, &dec_perf)) {
      passed++;
      // Check for reasonably valid values
      if (enc_perf < 0.001)
        enc_perf = 0;
      if (dec_perf < 0.001)
        dec_perf = 0;

      // Formatting size string
      char size_str[32];
      if (size >= 1024 * 1024 * 1024)
        snprintf(size_str, 32, "%.2f GB", size / 1e9f);
      else if (size >= 1024 * 1024)
        snprintf(size_str, 32, "%.1f MB", size / 1e6f);
      else
        snprintf(size_str, 32, "%zu B", size);

      // We print clean table row HERE, replacing the verbose output from inside
      // function? Actually, function prints progress. Let's just print Summary
      // row.
      printf("[RESULT] %-8s | %6.2f GB/s    | %6.2f GB/s    | PASSED\n",
             size_str, enc_perf, dec_perf);
    } else {
      failed++;
      char size_str[32];
      snprintf(size_str, 32, "%zu", size);
      printf("[RESULT] %-8s |      -         |      -         | FAILED\n",
             size_str);
    }
    printf(
        "----------------------------------------------------------------\n");
  }

  std::cout << "\n======================================" << std::endl;
  std::cout << "Results: " << passed << " PASSED, " << failed << " FAILED"
            << std::endl;
  std::cout << "======================================" << std::endl;

  return failed > 0 ? 1 : 0;
}
