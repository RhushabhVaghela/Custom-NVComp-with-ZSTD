#include "cuda_zstd_fse.h" // Our GPU FSE implementation
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// Reference Zstd headers
extern "C" {
#define FSE_STATIC_LINKING_ONLY
#define ZSTD_STATIC_LINKING_ONLY
#include "common/fse.h"
#include "zstd.h"
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

/**
 * Simple FSE Decoder Implementation
 * Based on Zstd FSE specification
 */
struct SimpleFSEDecoder {
  // Decode table: indexed by state
  struct DecoderTableEntry {
    uint8_t symbol;        // Symbol to output
    uint8_t nbBits;        // Number of bits to read for next state
    uint16_t newStateBase; // Base value for next state
  };

  std::vector<DecoderTableEntry> table;
  uint32_t table_log;
  uint32_t table_size;

  // Build decode table from normalized frequency counts
  bool buildTable(const int16_t *norm_freqs, size_t max_symbol,
                  unsigned table_log_param) {
    table_log = table_log_param;
    table_size = 1U << table_log;
    table.resize(table_size);

    // Step 1: Spread symbols across table using "spread" algorithm
    // This matches Zstd's FSE_buildDTable_internal
    uint32_t position = 0;
    for (size_t symbol = 0; symbol <= max_symbol; symbol++) {
      int freq = norm_freqs[symbol];
      if (freq == 0)
        continue;
      if (freq == -1) { // Special: 1 occurrence (less than 1 in normalized)
        freq = 1;
      }

      for (int i = 0; i < freq; i++) {
        table[position].symbol = symbol;
        // Spread pattern: incrementby odd number to distribute evenly
        position = (position + (table_size >> 1) + (table_size >> 3) + 3) &
                   (table_size - 1);
      }
    }

    // Step 2: Fill in nbBits and newStateBase for each table entry
    for (uint32_t state = 0; state < table_size; state++) {
      uint8_t symbol = table[state].symbol;
      int freq = (norm_freqs[symbol] == -1) ? 1 : norm_freqs[symbol];

      // Calculate nbBits: log2(tableSize / freq)
      uint32_t max_bits_out = table_log - 1;
      uint32_t min_state_plus = freq << max_bits_out;

      while (min_state_plus > table_size) {
        max_bits_out--;
        min_state_plus = freq << max_bits_out;
      }

      table[state].nbBits = max_bits_out;
      table[state].newStateBase = (freq << max_bits_out) - table_size;
    }

    return true;
  }

  // Decode compressed data
  bool decode(const uint8_t *compressed, size_t compressed_size,
              uint8_t *output, size_t output_size) {
    if (compressed_size < 1)
      return false;

    // Parse FSE header to get normalized frequencies
    // For now, we'll use the reference decoder
    // TODO: Implement FSE header parsing

    printf("⏭️  Custom FSE decoder needs header parsing implementation\\n");
    return false;
  }
};

/**
 * Test 1: Single Symbol (Simplest Case)
 * Input: All bytes are the same symbol (e.g., all 'A')
 * Expected: FSE should encode this very efficiently (RLE-like)
 */
bool test_single_symbol() {
  printf("\n=== Test 1: Single Symbol ===\n");

  const size_t input_size = 1024;
  const uint8_t symbol = 'A';

  // Create input: all same symbol
  std::vector<uint8_t> input(input_size, symbol);

  // Step 1: Compress with reference Zstd (which uses FSE internally)
  size_t max_compressed_size = ZSTD_compressBound(input_size);
  std::vector<uint8_t> compressed(max_compressed_size);

  size_t compressed_size =
      ZSTD_compress(compressed.data(), compressed.capacity(), input.data(),
                    input.size(), 1); // Level 1 for minimal complexity

  if (ZSTD_isError(compressed_size)) {
    printf("❌ Reference Zstd compression failed: %s\n",
           ZSTD_getErrorName(compressed_size));
    return false;
  }

  printf("Input size: %zu bytes\n", input_size);
  printf("Compressed size: %zu bytes (%.1f%% ratio)\n", compressed_size,
         100.0 * compressed_size / input_size);

  // Step 2: Decompress with reference Zstd (sanity check)
  std::vector<uint8_t> decompressed(input_size);
  size_t decompressed_size =
      ZSTD_decompress(decompressed.data(), decompressed.capacity(),
                      compressed.data(), compressed_size);

  if (ZSTD_isError(decompressed_size)) {
    printf("❌ Reference Zstd decompression failed: %s\n",
           ZSTD_getErrorName(decompressed_size));
    return false;
  }

  // Verify reference encode/decode roundtrip works
  if (decompressed_size != input_size) {
    printf("❌ Size mismatch: expected %zu, got %zu\n", input_size,
           decompressed_size);
    return false;
  }

  if (memcmp(input.data(), decompressed.data(), input_size) != 0) {
    printf("❌ Content mismatch in reference roundtrip\n");
    return false;
  }

  printf("✅ Reference Zstd FSE roundtrip: PASS\n");

  // Step 3: Analyze the compressed format
  printf("\nCompressed data (first 32 bytes):\n");
  for (size_t i = 0; i < std::min(compressed_size, size_t(32)); i++) {
    printf("%02X ", compressed[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n");

  // TODO: Implement our clean decoder and compare
  printf("⏭️  Custom decoder not yet implemented\n");

  return true;
}

/**
 * Test 2: Two Symbols (50/50 distribution)
 */
bool test_two_symbols() {
  printf("\n=== Test 2: Two Symbols (50/50) ===\n");

  const size_t input_size = 1024;
  std::vector<uint8_t> input(input_size);

  // Alternate between 'A' and 'B'
  for (size_t i = 0; i < input_size; i++) {
    input[i] = (i % 2 == 0) ? 'A' : 'B';
  }

  // Encode with reference ZSTD
  size_t max_compressed_size = ZSTD_compressBound(input_size);
  std::vector<uint8_t> compressed(max_compressed_size);

  size_t compressed_size = ZSTD_compress(
      compressed.data(), compressed.capacity(), input.data(), input.size(), 1);

  if (ZSTD_isError(compressed_size)) {
    printf("❌ Compression failed: %s\n", ZSTD_getErrorName(compressed_size));
    return false;
  }

  printf("Compressed: %zu → %zu bytes (%.1f%%)\n", input_size, compressed_size,
         100.0 * compressed_size / input_size);

  // Decode and verify
  std::vector<uint8_t> decompressed(input_size);
  size_t decompressed_size =
      ZSTD_decompress(decompressed.data(), decompressed.capacity(),
                      compressed.data(), compressed_size);

  if (ZSTD_isError(decompressed_size) || decompressed_size != input_size ||
      memcmp(input.data(), decompressed.data(), input_size) != 0) {
    printf("❌ Reference roundtrip failed\n");
    return false;
  }

  printf("✅ Reference roundtrip: PASS\n");
  return true;
}

/**
 * Test 3: Small Alphabet (10 symbols, varied distribution)
 */
bool test_small_alphabet() {
  printf("\n=== Test 3: Small Alphabet (10 symbols) ===\n");

  const size_t input_size = 4096;
  std::vector<uint8_t> input(input_size);

  // Create distribution: symbol i appears (i+1) times per 55 bytes
  // Frequencies: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (sum=55)
  size_t pos = 0;
  while (pos < input_size) {
    for (int symbol = 0; symbol < 10 && pos < input_size; symbol++) {
      for (int count = 0; count < symbol + 1 && pos < input_size; count++) {
        input[pos++] = symbol;
      }
    }
  }

  // Encode
  size_t max_compressed_size = ZSTD_compressBound(input_size);
  std::vector<uint8_t> compressed(max_compressed_size);

  size_t compressed_size = ZSTD_compress(
      compressed.data(), compressed.capacity(), input.data(), input.size(), 1);

  if (ZSTD_isError(compressed_size)) {
    printf("❌ Compression failed\n");
    return false;
  }

  printf("Compressed: %zu → %zu bytes (%.1f%%)\n", input_size, compressed_size,
         100.0 * compressed_size / input_size);

  // Verify
  std::vector<uint8_t> decompressed(input_size);
  size_t decompressed_size =
      ZSTD_decompress(decompressed.data(), decompressed.capacity(),
                      compressed.data(), compressed_size);

  if (ZSTD_isError(decompressed_size) || decompressed_size != input_size ||
      memcmp(input.data(), decompressed.data(), input_size) != 0) {
    printf("❌ Reference roundtrip failed\n");
    return false;
  }

  printf("✅ Reference roundtrip: PASS\n");
  return true;
}

/**
 * Test 4: Full Alphabet (256 symbols, realistic distribution)
 */
bool test_full_alphabet() {
  printf("\n=== Test 4: Full Alphabet (256 symbols) ===\n");

  const size_t input_size = 256 * 1024; // 256KB like our main test
  std::vector<uint8_t> input(input_size);

  // Generate pseudo-random but reproducible data
  unsigned int seed = 42;
  for (size_t i = 0; i < input_size; i++) {
    seed = seed * 1103515245 + 12345;
    input[i] = (seed >> 16) & 0xFF;
  }

  // Encode
  size_t max_compressed_size = ZSTD_compressBound(input_size);
  std::vector<uint8_t> compressed(max_compressed_size);

  size_t compressed_size = ZSTD_compress(
      compressed.data(), compressed.capacity(), input.data(), input.size(), 1);

  if (ZSTD_isError(compressed_size)) {
    printf("❌ Compression failed\n");
    return false;
  }

  printf("Compressed: %zu → %zu bytes (%.1f%%)\n", input_size, compressed_size,
         100.0 * compressed_size / input_size);

  // Verify
  std::vector<uint8_t> decompressed(input_size);
  size_t decompressed_size =
      ZSTD_decompress(decompressed.data(), decompressed.capacity(),
                      compressed.data(), compressed_size);

  if (ZSTD_isError(decompressed_size) || decompressed_size != input_size ||
      memcmp(input.data(), decompressed.data(), input_size) != 0) {
    printf("❌ Reference roundtrip failed\n");
    return false;
  }

  printf("✅ Reference roundtrip: PASS\n");
  return true;
}

int main() {
  printf("╔══════════════════════════════════════════════════════════╗\n");
  printf("║  FSE Reference Implementation Testing                   ║\n");
  printf("║  Goal: Identify divergence from Zstd specification      ║\n");
  printf("╚══════════════════════════════════════════════════════════╝\n");

  int passed = 0;
  int total = 5;

  if (test_single_symbol())
    passed++;
  if (test_two_symbols())
    passed++;
  if (test_small_alphabet())
    passed++;
  if (test_full_alphabet())
    passed++;
  // if (test_gpu_fse_comparison()) passed++;

  printf("\n╔══════════════════════════════════════════════════════════╗\n");
  printf("║  Results: %d/%d tests passed                             \n",
         passed, total);
  printf("╚══════════════════════════════════════════════════════════╝\n");

  return (passed == total) ? 0 : 1;
}
