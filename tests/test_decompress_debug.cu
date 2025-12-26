// test_decompress_debug.cu - Debug test to find decompression corruption source
#include "cuda_zstd_manager.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>


using namespace cuda_zstd;

void dump_hex(const char *label, const void *data, size_t size,
              size_t max_bytes = 64) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  std::cout << label << " (first " << std::min(size, max_bytes) << " of "
            << size << " bytes):\n  ";
  for (size_t i = 0; i < std::min(size, max_bytes); ++i) {
    printf("%02X ", bytes[i]);
    if ((i + 1) % 16 == 0)
      printf("\n  ");
  }
  printf("\n");
}

int main() {
  std::cout << "=== Decompression Debug Test ===\n\n";

  // Create simple test data
  const size_t data_size = 64 * 1024; // 64KB
  std::vector<uint8_t> h_input(data_size);

  // Pattern: repeating 0-255 (compressible)
  for (size_t i = 0; i < data_size; ++i) {
    h_input[i] = static_cast<uint8_t>(i % 256);
  }

  std::cout << "[1] Input data created: " << data_size << " bytes\n";
  dump_hex("Input", h_input.data(), data_size);

  // Allocate GPU buffers
  void *d_input, *d_compressed, *d_decompressed, *d_temp;
  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_compressed, data_size * 2);
  cudaMalloc(&d_decompressed, data_size);

  CompressionConfig config{.level = 3};
  ZstdBatchManager manager(config);
  size_t temp_size = std::max(manager.get_compress_temp_size(data_size),
                              manager.get_decompress_temp_size(data_size * 2));
  cudaMalloc(&d_temp, temp_size);

  // Copy input to device
  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  // === COMPRESS ===
  std::cout << "\n[2] Compressing...\n";
  size_t compressed_size = data_size * 2;
  Status status =
      manager.compress(d_input, data_size, d_compressed, &compressed_size,
                       d_temp, temp_size, nullptr, 0, nullptr);
  cudaDeviceSynchronize();

  if (status != Status::SUCCESS) {
    std::cerr << "Compression failed: " << status_to_string(status) << "\n";
    return 1;
  }

  std::cout << "  Compressed size: " << compressed_size << " bytes\n";
  std::cout << "  Compression ratio: " << (double)data_size / compressed_size
            << ":1\n";

  // Copy compressed data to host for inspection
  std::vector<uint8_t> h_compressed(compressed_size);
  cudaMemcpy(h_compressed.data(), d_compressed, compressed_size,
             cudaMemcpyDeviceToHost);
  dump_hex("Compressed data", h_compressed.data(), compressed_size);

  // Check magic number
  uint32_t magic = *reinterpret_cast<uint32_t *>(h_compressed.data());
  printf("  Magic number: 0x%08X (expected 0xFD2FB528 for ZSTD)\n", magic);

  if (magic == 0xFD2FB528) {
    std::cout << "  ✓ Valid ZSTD magic number\n";
  } else if ((magic & 0xFFFFFFF0) == 0x184D2A50) {
    std::cout << "  ⚠ Skippable frame magic (custom metadata present)\n";
  } else {
    std::cout << "  ✗ Invalid magic number!\n";
  }

  // === DECOMPRESS ===
  std::cout << "\n[3] Decompressing...\n";

  // Initialize output buffer to recognizable pattern (0xAA) to see what gets
  // written
  cudaMemset(d_decompressed, 0xAA, data_size);

  size_t decompressed_size = data_size;
  status = manager.decompress(d_compressed, compressed_size, d_decompressed,
                              &decompressed_size, d_temp, temp_size, nullptr);
  cudaDeviceSynchronize();

  if (status != Status::SUCCESS) {
    std::cerr << "Decompression failed: " << status_to_string(status) << "\n";
    return 1;
  }

  std::cout << "  Decompressed size: " << decompressed_size << " bytes\n";

  // Copy decompressed data to host
  std::vector<uint8_t> h_output(decompressed_size);
  cudaMemcpy(h_output.data(), d_decompressed, decompressed_size,
             cudaMemcpyDeviceToHost);
  dump_hex("Decompressed data", h_output.data(), decompressed_size);

  // === VERIFY ===
  std::cout << "\n[4] Verification:\n";

  // Check if output is all 0xAA (unchanged)
  int unchanged_count = 0;
  int ff_count = 0;
  for (size_t i = 0; i < std::min(decompressed_size, (size_t)256); ++i) {
    if (h_output[i] == 0xAA)
      unchanged_count++;
    if (h_output[i] == 0xFF)
      ff_count++;
  }

  if (unchanged_count > 200) {
    std::cout
        << "  ⚠ OUTPUT UNCHANGED! Decompression wrote nothing to buffer.\n";
    std::cout << "  → Likely issue: execute_sequences not writing output\n";
  } else if (ff_count > 200) {
    std::cout << "  ⚠ OUTPUT IS ALL 0xFF! Memory may be uninitialized or "
                 "overwritten.\n";
  }

  // Compare with original
  int mismatches = 0;
  int first_mismatch = -1;
  for (size_t i = 0; i < std::min(h_input.size(), h_output.size()); ++i) {
    if (h_input[i] != h_output[i]) {
      if (first_mismatch < 0)
        first_mismatch = i;
      mismatches++;
    }
  }

  if (mismatches == 0 && h_input.size() == h_output.size()) {
    std::cout << "  ✓ PERFECT MATCH! All " << data_size << " bytes verified.\n";
  } else {
    std::cout << "  ✗ MISMATCH: " << mismatches << " bytes differ\n";
    if (first_mismatch >= 0) {
      std::cout << "  First mismatch at byte " << first_mismatch
                << ": expected " << (int)h_input[first_mismatch] << ", got "
                << (int)h_output[first_mismatch] << "\n";
    }
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  cudaFree(d_temp);

  return (mismatches == 0) ? 0 : 1;
}
