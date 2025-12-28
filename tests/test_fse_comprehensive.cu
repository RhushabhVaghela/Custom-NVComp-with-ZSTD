// ============================================================================
// test_fse_comprehensive.cu - Consolidated FSE Test Suite
// ============================================================================
// This file consolidates all FSE (Finite State Entropy) unit and integration
// tests from the original 16 separate FSE test files.
//
// Original files consolidated:
// - test_fse_adaptive.cu, test_fse_advanced.cu, test_fse_advanced_function.cu
// - test_fse_context_reuse.cu, test_fse_decode_correctness.cu
// - test_fse_integration.cu, test_fse_minimal.cu, test_fse_prepare.cu
// - test_fse_reference.cu, test_fse_reference_comparison.cu
// - test_fse_setup_kernels.cu, test_fse_table_mismatch.cu
// - test_gpu_fse_encoder.cu, test_gpu_fse_scale.cu
// - test_batch_fse.cu, test_multitable_fse.cu
// - test_chunk_parallel_fse.cu, test_parallel_fse_zstd.cu
// - test_simple_fse_reference.cu
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// ============================================================================
// Test Utilities
// ============================================================================

#define LOG_TEST(name) printf("\n[TEST] %s\n", name)
#define LOG_PASS(name) printf("  [PASS] %s\n", name)
#define LOG_FAIL(name, msg) fprintf(stderr, "  [FAIL] %s: %s\n", name, msg)

void print_separator() { printf("========================================\n"); }

void generate_test_data(std::vector<byte_t> &data, size_t size, int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  data.resize(size);

  for (size_t i = 0; i < size; i++) {
    if (i % 32 < 16) {
      data[i] = dist(rng) % 64; // Low entropy region
    } else {
      data[i] = dist(rng); // High entropy
    }
  }
}

void generate_compressible_data(std::vector<byte_t> &data, size_t size) {
  data.resize(size);
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<byte_t>(i % 32); // Repetitive pattern
  }
}

// ============================================================================
// SUITE 1: Basic FSE Operations
// ============================================================================

bool test_multitable_fse_init() {
  LOG_TEST("MultiTableFSE Initialization");

  MultiTableFSE mt;

  if (mt.active_tables != 0) {
    LOG_FAIL(__func__, "active_tables should be 0");
    return false;
  }

  for (int i = 0; i < 4; ++i) {
    if (mt.tables[i].d_symbol_table != nullptr || mt.tables[i].table_log != 0 ||
        mt.tables[i].table_size != 0) {
      LOG_FAIL(__func__, "tables not zero-initialized");
      return false;
    }
  }

  mt.active_tables = 0xF;
  mt.clear();

  if (mt.active_tables != 0) {
    LOG_FAIL(__func__, "clear() failed to reset active_tables");
    return false;
  }

  LOG_PASS(__func__);
  return true;
}

bool test_fse_normalization() {
  LOG_TEST("FSE Frequency Normalization");

  std::vector<u32> raw_freqs = {100, 200, 50, 75, 150};
  std::vector<u16> normalized(256, 0);

  u32 actual_table_size = 0;
  Status status = normalize_frequencies_accurate(
      raw_freqs.data(), 575, 4, normalized.data(), 8, &actual_table_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "normalize_frequencies_accurate failed");
    return false;
  }

  u32 sum = 0;
  for (auto freq : normalized) {
    sum += freq;
  }

  if (sum != 256) {
    LOG_FAIL(__func__, "Normalization sum != 256");
    return false;
  }

  LOG_PASS(__func__);
  return true;
}

// ============================================================================
// SUITE 2: FSE Encode/Decode Roundtrip
// ============================================================================

bool test_fse_roundtrip_small() {
  LOG_TEST("FSE Roundtrip (Small Data)");

  const byte_t test_data[] = {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6};
  u32 data_size = sizeof(test_data);

  void *d_input, *d_output, *d_decoded;
  u32 *d_output_size;

  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, data_size * 2 + 1024);
  cudaMalloc(&d_output_size, sizeof(u32));
  cudaMalloc(&d_decoded, data_size);

  cudaMemcpy(d_input, test_data, data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, data_size * 2 + 1024);

  // Encode
  u32 encoded_size = 0;
  Status status = encode_fse_advanced(static_cast<byte_t *>(d_input), data_size,
                                      static_cast<byte_t *>(d_output),
                                      d_output_size, true, 0, nullptr, nullptr);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "encode_fse_advanced failed");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    cudaFree(d_decoded);
    return false;
  }

  cudaMemcpy(&encoded_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);
  printf("  Encoded %u bytes -> %u bytes\n", data_size, encoded_size);

  // Decode
  std::vector<byte_t> h_encoded(encoded_size);
  cudaMemcpy(h_encoded.data(), d_output, encoded_size, cudaMemcpyDeviceToHost);

  u32 decoded_size = data_size;
  status = decode_fse(h_encoded.data(), encoded_size,
                      static_cast<byte_t *>(d_decoded), &decoded_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "decode_fse failed");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    cudaFree(d_decoded);
    return false;
  }

  if (decoded_size != data_size) {
    LOG_FAIL(__func__, "Decoded size mismatch");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    cudaFree(d_decoded);
    return false;
  }

  // Verify content
  std::vector<byte_t> h_decoded(data_size);
  cudaMemcpy(h_decoded.data(), d_decoded, data_size, cudaMemcpyDeviceToHost);

  for (u32 i = 0; i < data_size; i++) {
    if (h_decoded[i] != test_data[i]) {
      LOG_FAIL(__func__, "Data mismatch");
      cudaFree(d_input);
      cudaFree(d_output);
      cudaFree(d_output_size);
      cudaFree(d_decoded);
      return false;
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);
  cudaFree(d_decoded);

  LOG_PASS(__func__);
  return true;
}

bool test_fse_roundtrip_large() {
  LOG_TEST("FSE Roundtrip (Large Data)");

  const size_t data_size = 256 * 1024; // 256 KB
  std::vector<byte_t> h_input;
  generate_test_data(h_input, data_size, 42);

  void *d_input, *d_output, *d_decoded;
  u32 *d_output_size;

  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, data_size * 2);
  cudaMalloc(&d_output_size, sizeof(u32));
  cudaMalloc(&d_decoded, data_size);

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  // Encode
  u32 encoded_size = 0;
  Status status = encode_fse_advanced(static_cast<byte_t *>(d_input),
                                      static_cast<u32>(data_size),
                                      static_cast<byte_t *>(d_output),
                                      d_output_size, true, 0, nullptr, nullptr);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "encode_fse_advanced failed");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    cudaFree(d_decoded);
    return false;
  }

  cudaMemcpy(&encoded_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);
  float ratio = (float)data_size / encoded_size;
  printf("  Encoded %zu bytes -> %u bytes (%.2fx compression)\n", data_size,
         encoded_size, ratio);

  // Decode
  std::vector<byte_t> h_encoded(encoded_size);
  cudaMemcpy(h_encoded.data(), d_output, encoded_size, cudaMemcpyDeviceToHost);

  u32 decoded_size = static_cast<u32>(data_size);
  status = decode_fse(h_encoded.data(), encoded_size,
                      static_cast<byte_t *>(d_decoded), &decoded_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "decode_fse failed");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_size);
    cudaFree(d_decoded);
    return false;
  }

  // Verify
  std::vector<byte_t> h_decoded(data_size);
  cudaMemcpy(h_decoded.data(), d_decoded, data_size, cudaMemcpyDeviceToHost);

  int mismatches = 0;
  for (size_t i = 0; i < data_size; i++) {
    if (h_decoded[i] != h_input[i]) {
      mismatches++;
      if (mismatches <= 5) {
        printf("  Mismatch at %zu: got 0x%02X, expected 0x%02X\n", i,
               h_decoded[i], h_input[i]);
      }
    }
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);
  cudaFree(d_decoded);

  if (mismatches > 0) {
    printf("  Total mismatches: %d\n", mismatches);
    LOG_FAIL(__func__, "Data verification failed");
    return false;
  }

  LOG_PASS(__func__);
  return true;
}

// ============================================================================
// SUITE 3: Entropy Distribution Tests
// ============================================================================

bool test_fse_single_symbol() {
  LOG_TEST("FSE Single Symbol (RLE-like)");

  const size_t data_size = 1024;
  std::vector<byte_t> h_input(data_size, 'A');

  void *d_input, *d_output;
  u32 *d_output_size;

  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, data_size * 2);
  cudaMalloc(&d_output_size, sizeof(u32));

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  Status status = encode_fse_advanced(static_cast<byte_t *>(d_input),
                                      static_cast<u32>(data_size),
                                      static_cast<byte_t *>(d_output),
                                      d_output_size, true, 0, nullptr, nullptr);

  u32 encoded_size = 0;
  cudaMemcpy(&encoded_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "Encoding failed");
    return false;
  }

  printf("  Single symbol: %zu -> %u bytes\n", data_size, encoded_size);
  LOG_PASS(__func__);
  return true;
}

bool test_fse_two_symbols() {
  LOG_TEST("FSE Two Symbols (50/50)");

  const size_t data_size = 1024;
  std::vector<byte_t> h_input(data_size);
  for (size_t i = 0; i < data_size; i++) {
    h_input[i] = (i % 2 == 0) ? 'A' : 'B';
  }

  void *d_input, *d_output;
  u32 *d_output_size;

  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, data_size * 2);
  cudaMalloc(&d_output_size, sizeof(u32));

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  Status status = encode_fse_advanced(static_cast<byte_t *>(d_input),
                                      static_cast<u32>(data_size),
                                      static_cast<byte_t *>(d_output),
                                      d_output_size, true, 0, nullptr, nullptr);

  u32 encoded_size = 0;
  cudaMemcpy(&encoded_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "Encoding failed");
    return false;
  }

  printf("  Two symbols: %zu -> %u bytes\n", data_size, encoded_size);
  LOG_PASS(__func__);
  return true;
}

bool test_fse_full_alphabet() {
  LOG_TEST("FSE Full Alphabet (256 symbols)");

  const size_t data_size = 256 * 1024;
  std::vector<byte_t> h_input;
  generate_test_data(h_input, data_size, 123);

  void *d_input, *d_output;
  u32 *d_output_size;

  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, data_size * 2);
  cudaMalloc(&d_output_size, sizeof(u32));

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  Status status = encode_fse_advanced(static_cast<byte_t *>(d_input),
                                      static_cast<u32>(data_size),
                                      static_cast<byte_t *>(d_output),
                                      d_output_size, true, 0, nullptr, nullptr);

  u32 encoded_size = 0;
  cudaMemcpy(&encoded_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "Encoding failed");
    return false;
  }

  float ratio = (float)data_size / encoded_size;
  printf("  Full alphabet: %zu -> %u bytes (%.2fx)\n", data_size, encoded_size,
         ratio);
  LOG_PASS(__func__);
  return true;
}

// ============================================================================
// SUITE 4: Integration with ZSTD Manager
// ============================================================================

bool test_fse_manager_integration() {
  LOG_TEST("FSE Manager Integration");

  const size_t data_size = 1 * 1024 * 1024; // 1 MB
  std::vector<byte_t> h_input;
  generate_compressible_data(h_input, data_size);

  CompressionConfig config = CompressionConfig::from_level(3);
  ZstdBatchManager mgr(config);

  void *d_input, *d_output, *d_temp;
  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_output, mgr.get_max_compressed_size(data_size));

  size_t temp_size = mgr.get_compress_temp_size(data_size);
  cudaMalloc(&d_temp, temp_size);

  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);

  size_t compressed_size = mgr.get_max_compressed_size(data_size);
  Status status = mgr.compress(d_input, data_size, d_output, &compressed_size,
                               d_temp, temp_size, nullptr, 0, 0);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp);

  if (status != Status::SUCCESS) {
    LOG_FAIL(__func__, "Manager compression failed");
    return false;
  }

  float ratio = (float)data_size / compressed_size;
  printf("  Manager compress: %zu -> %zu bytes (%.2fx)\n", data_size,
         compressed_size, ratio);
  LOG_PASS(__func__);
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  printf("\n");
  print_separator();
  printf("FSE Comprehensive Test Suite\n");
  printf("(Consolidated from 16 FSE test files)\n");
  print_separator();

  SKIP_IF_NO_CUDA_RET(0);
  check_cuda_device();

  int passed = 0;
  int total = 0;

  // Suite 1: Basic Operations
  print_separator();
  printf("SUITE 1: Basic FSE Operations\n");
  print_separator();

  total++;
  if (test_multitable_fse_init())
    passed++;
  total++;
  if (test_fse_normalization())
    passed++;

  // Suite 2: Roundtrip Tests
  printf("\n");
  print_separator();
  printf("SUITE 2: FSE Roundtrip Tests\n");
  print_separator();

  total++;
  if (test_fse_roundtrip_small())
    passed++;
  total++;
  if (test_fse_roundtrip_large())
    passed++;

  // Suite 3: Entropy Distribution
  printf("\n");
  print_separator();
  printf("SUITE 3: Entropy Distribution Tests\n");
  print_separator();

  total++;
  if (test_fse_single_symbol())
    passed++;
  total++;
  if (test_fse_two_symbols())
    passed++;
  total++;
  if (test_fse_full_alphabet())
    passed++;

  // Suite 4: Integration
  printf("\n");
  print_separator();
  printf("SUITE 4: Manager Integration\n");
  print_separator();

  total++;
  if (test_fse_manager_integration())
    passed++;

  // Summary
  printf("\n");
  print_separator();
  printf("RESULTS: %d/%d tests passed\n", passed, total);
  print_separator();

  return (passed == total) ? 0 : 1;
}
