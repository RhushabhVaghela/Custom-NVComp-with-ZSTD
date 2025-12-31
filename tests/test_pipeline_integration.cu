/**
 * test_pipeline_integration.cu - Comprehensive Pipeline Integration Test
 *
 * Validates:
 * 1. Pipelined Compression correctness across various sizes (Small, Medium,
 * Large, Odd).
 * 2. Data Integrity: Output must be valid ZSTD frames (verified via system
 * `zstd -t`).
 */

#include "../src/pipeline_manager.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Ensure we link/include necessary components
using namespace cuda_zstd;

// --- Helper: Generate Data ---
void generate_data(std::vector<uint8_t> &data) {
  std::mt19937 rng(12345); // Deterministic seed
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : data)
    b = dist(rng);
}

// --- Helper: Verify via System ZSTD ---
bool verify_zstd_integrity(const std::string &filename) {
  std::string cmd = "zstd -q -t " + filename;
  int ret = std::system(cmd.c_str());
  return (ret == 0);
}

bool run_test_case(size_t data_size, size_t batch_size,
                   const std::string &name) {
  std::cout << "[TEST] " << name << " (Data: " << (data_size / 1024.0 / 1024.0)
            << " MB, Batch: " << (batch_size / 1024.0 / 1024.0) << " MB)... ";

  std::vector<uint8_t> h_data(data_size);
  generate_data(h_data);

  // Setup Pipeline
  CompressionConfig config;
  config.level = 1; // Fast level for testing

  PipelinedBatchManager manager(config, batch_size, 3);

  // Output File
  std::string out_filename =
      "test_temp_" + std::to_string(std::rand()) + ".zst";
  std::ofstream outfile(out_filename, std::ios::binary);

  // Callbacks
  size_t read_offset = 0;
  auto input_cb = [&](void *h_input, size_t max_len, size_t *out_len) -> bool {
    size_t rem = data_size - read_offset;
    if (rem == 0) {
      *out_len = 0;
      return false;
    }
    size_t copy_sz = std::min(rem, max_len);
    std::memcpy(h_input, h_data.data() + read_offset, copy_sz);
    read_offset += copy_sz;
    *out_len = copy_sz;
    return (read_offset < data_size);
  };

  auto output_cb = [&](const void *h_out, size_t size) {
    outfile.write(reinterpret_cast<const char *>(h_out), size);
  };

  Status s = manager.compress_stream_pipeline(input_cb, output_cb);
  outfile.close();

  if (s != Status::SUCCESS) {
    std::cout << "FAIL (Pipeline Error: " << (int)s << ")" << std::endl;
    std::remove(out_filename.c_str());
    return false;
  }

  // Verify Integrity
  bool valid = verify_zstd_integrity(out_filename);
  std::remove(out_filename.c_str());

  if (valid) {
    std::cout << "PASS" << std::endl;
    return true;
  } else {
    std::cout << "FAIL (ZSTD Integrity Check Failed)" << std::endl;
    return false;
  }
}

// --- Helper: Check if zstd CLI is available ---
bool is_zstd_available() {
  int ret = std::system("which zstd > /dev/null 2>&1");
  return (ret == 0);
}

int main() {
  std::cout << "Running Pipeline Integration Tests..." << std::endl;

  // Check if zstd CLI is available - skip gracefully if not installed
  if (!is_zstd_available()) {
    std::cout << "\n[SKIP] zstd CLI not found. Install zstd to run these tests."
              << std::endl;
    std::cout << "Tests skipped (no zstd CLI available)." << std::endl;
    return 0; // Return success - test is skipped, not failed
  }

  bool all_passed = true;

  // 1. Small Data (< Batch)
  all_passed &= run_test_case(16 * 1024 * 1024, 64 * 1024 * 1024, "Small Data");

  // 2. Exact Batch Size
  all_passed &=
      run_test_case(64 * 1024 * 1024, 64 * 1024 * 1024, "Exact Batch");

  // 3. Multi-Batch (Aligned)
  all_passed &=
      run_test_case(256 * 1024 * 1024, 64 * 1024 * 1024, "Multi-Batch Aligned");

  // 4. Multi-Batch (Unaligned / Odd)
  all_passed &=
      run_test_case(300 * 1024 * 1024, 64 * 1024 * 1024, "Multi-Batch Odd");

  // 5. Large Data (1 GB)
  all_passed &=
      run_test_case(1024ULL * 1024 * 1024, 128 * 1024 * 1024, "Large 1GB");

  if (all_passed) {
    std::cout << "\nAll Integration Tests PASSED." << std::endl;
    return 0;
  } else {
    std::cerr << "\nSome Tests FAILED." << std::endl;
    return 1;
  }
}
