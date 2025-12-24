/**
 * benchmark_fse_gpu.cu - GPU-to-GPU FSE Benchmark
 *
 * Measures PURE GPU performance (no host transfers)
 * Data is pre-allocated on device.
 */

#include "cuda_zstd_fse.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

void generate_test_data(std::vector<uint8_t> &data) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);

  // Generate mixed entropy data
  for (size_t i = 0; i < data.size(); ++i) {
    if (i % 2 == 0)
      data[i] = dist(rng) & 0x0F; // Low entropy
    else
      data[i] = dist(rng); // High entropy
  }
}

int main() {
  std::cout << "\n======================================================"
            << std::endl;
  std::cout << "  GPU-to-GPU FSE Benchmark (Pure Kernel Performance)"
            << std::endl;
  std::cout << "======================================================\n"
            << std::endl;

  std::vector<size_t> data_sizes = {
      64ULL * 1024 * 1024,   // 64 MB
      128ULL * 1024 * 1024,  // 128 MB
      256ULL * 1024 * 1024,  // 256 MB
      512ULL * 1024 * 1024,  // 512 MB
      1024ULL * 1024 * 1024, // 1 GB
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << std::left << std::setw(12) << "Data Size" << " | "
            << std::setw(12) << "Encode Time" << " | " << std::setw(12)
            << "Encode GB/s" << " | " << std::setw(12) << "Decode Time" << " | "
            << std::setw(12) << "Decode GB/s" << " | " << std::setw(8)
            << "Ratio" << " | Status" << std::endl;
  std::cout << std::string(90, '-') << std::endl;

  // Persistent contexts for reuse across iterations
  FSEContext encode_ctx = {};
  FSEDecodeContext decode_ctx = {};

  for (size_t data_size : data_sizes) {
    // Generate test data on host
    std::vector<uint8_t> h_data(data_size);
    generate_test_data(h_data);

    // Allocate device memory
    byte_t *d_input, *d_output, *d_decompressed;
    u32 *d_comp_size;
    size_t max_comp_size = data_size + (data_size >> 7) + 4096;

    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, max_comp_size);
    cudaMalloc(&d_decompressed, data_size);
    cudaMalloc(&d_comp_size, sizeof(u32));

    // Copy data to device (one-time, not timed)
    cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);

    // Warmup
    encode_fse_advanced(d_input, (u32)data_size, d_output, d_comp_size, true,
                        stream, &encode_ctx, nullptr);
    cudaStreamSynchronize(stream);

    // =================== ENCODE BENCHMARK ===================
    const int NUM_RUNS = 5;
    double total_encode_time = 0;
    u32 comp_size = 0;
    Status status = Status::SUCCESS;

    for (int r = 0; r < NUM_RUNS; ++r) {
      auto start = std::chrono::high_resolution_clock::now();

      status =
          encode_fse_advanced(d_input, (u32)data_size, d_output, d_comp_size,
                              true, stream, &encode_ctx, nullptr);
      cudaStreamSynchronize(stream);

      auto end = std::chrono::high_resolution_clock::now();
      total_encode_time += std::chrono::duration<double>(end - start).count();

      if (status != Status::SUCCESS) {
        std::cerr << "Encode failed!" << std::endl;
        break;
      }

      // Read back size for next step
      cudaMemcpy(&comp_size, d_comp_size, sizeof(u32), cudaMemcpyDeviceToHost);
    }

    double avg_encode_time = total_encode_time / NUM_RUNS;
    double encode_gbps = (data_size / 1e9) / avg_encode_time;

    // =================== DECODE BENCHMARK ===================
    // Get offsets for decode (needed for parallel decoding)
    u64 *d_offsets = nullptr;
    // Single run to populate offsets
    encode_fse_advanced(d_input, (u32)data_size, d_output, d_comp_size, true,
                        stream, &encode_ctx, &d_offsets);
    cudaStreamSynchronize(stream);

    u32 actual_dec_size = 0;
    double total_decode_time = 0;

    // Warmup Decode
    decode_fse(d_output, comp_size, d_decompressed, &actual_dec_size, d_offsets,
               stream, &decode_ctx);

    for (int r = 0; r < NUM_RUNS; ++r) {
      actual_dec_size = (u32)data_size;
      auto start = std::chrono::high_resolution_clock::now();

      // Pass &decode_ctx to reuse memory
      status = decode_fse(d_output, comp_size, d_decompressed, &actual_dec_size,
                          d_offsets, stream, &decode_ctx);
      cudaStreamSynchronize(stream);

      auto end = std::chrono::high_resolution_clock::now();
      total_decode_time += std::chrono::duration<double>(end - start).count();

      if (status != Status::SUCCESS) {
        std::cerr << "Decode failed!" << std::endl;
        break;
      }
    }

    double avg_decode_time = total_decode_time / NUM_RUNS;
    double decode_gbps = (data_size / 1e9) / avg_decode_time;

    // Verify Integrity (Full Check)
    std::vector<uint8_t> h_decompressed(data_size);
    cudaMemcpy(h_decompressed.data(), d_decompressed, data_size,
               cudaMemcpyDeviceToHost);

    // Check size
    bool passed = (actual_dec_size == data_size);
    if (passed) {
      // Full memcmp
      if (std::memcmp(h_decompressed.data(), h_data.data(), data_size) != 0) {
        passed = false;
      }
    }

    double ratio = (double)data_size / comp_size;

    // Print results
    char size_str[32];
    if (data_size >= 1024 * 1024 * 1024)
      snprintf(size_str, 32, "%zu GB", data_size >> 30);
    else
      snprintf(size_str, 32, "%zu MB", data_size >> 20);

    std::cout << std::left << std::setw(12) << size_str << " | " << std::fixed
              << std::setprecision(3) << std::setw(10) << avg_encode_time * 1000
              << "ms | " << std::setw(10) << encode_gbps << "  | "
              << std::setw(10) << avg_decode_time * 1000 << "ms | "
              << std::setw(10) << decode_gbps << "  | " << std::setw(6)
              << std::setprecision(2) << ratio << "x | "
              << (passed ? "PASS" : "FAIL") << std::endl;

    // Cleanup Loop Resources
    if (d_offsets)
      cudaFree(d_offsets);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_decompressed);
    cudaFree(d_comp_size);
  }

  // Free Persistent Contexts
  if (encode_ctx.d_dev_symbol_table)
    cudaFree(encode_ctx.d_dev_symbol_table);
  if (encode_ctx.d_dev_next_state)
    cudaFree(encode_ctx.d_dev_next_state);
  if (encode_ctx.d_bitstreams)
    cudaFree(encode_ctx.d_bitstreams);
  if (encode_ctx.d_chunk_bit_counts)
    cudaFree(encode_ctx.d_chunk_bit_counts);
  if (encode_ctx.d_chunk_start_states)
    cudaFree(encode_ctx.d_chunk_start_states);
  if (encode_ctx.d_chunk_offsets)
    cudaFree(encode_ctx.d_chunk_offsets);

  // FSEDecodeContext is freed automatically by destructor on stack exit?
  // Let's check definition... yes, ~FSEDecodeContext calls cudaFree.
  // BUT we should verify if the struct is copyable/movable.
  // It's a POD-like struct with a destructor, better rely on manual cleanup if
  // possible or ensure scope is correct. It is stack allocated here, so it will
  // be destroyed at end of main.

  cudaStreamDestroy(stream);
  std::cout << "\n" << std::endl;

  return 0;
}
