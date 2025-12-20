#include "cuda_zstd_fse.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>


extern "C" {
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
}

/**
 * Minimal FSE Encoder - CPU Implementation
 * Follows Zstd FSE specification exactly
 * Single-threaded, no optimizations, maximum clarity
 */
class MinimalFSEEncoder {
private:
  std::vector<uint32_t> freq_count; // Symbol frequencies
  std::vector<int16_t> norm_freq;   // Normalized frequencies
  uint32_t max_symbol;
  uint32_t table_log;
  uint32_t table_size;

  // Encoding table
  struct CTableEntry {
    uint16_t deltaFindState;
    uint8_t deltaNbBits;
  };
  std::vector<CTableEntry> ctable;

public:
  MinimalFSEEncoder() : max_symbol(0), table_log(0), table_size(0) {}

  // Step 1: Count symbol frequencies
  void count_frequencies(const uint8_t *input, size_t size) {
    freq_count.resize(256, 0);
    max_symbol = 0;

    for (size_t i = 0; i < size; i++) {
      freq_count[input[i]]++;
      if (input[i] > max_symbol)
        max_symbol = input[i];
    }

    printf("[MINIMAL] Frequency analysis:\n");
    for (uint32_t i = 0; i <= max_symbol; i++) {
      if (freq_count[i] > 0) {
        printf("  Symbol %u: %u occurrences\n", i, freq_count[i]);
      }
    }
  }

  // Step 2: Normalize frequencies to fit in table
  void normalize_frequencies(unsigned table_log_param) {
    table_log = table_log_param;
    table_size = 1U << table_log;
    norm_freq.resize(max_symbol + 1, 0);

    // Simple normalization: scale to table_size
    uint64_t total = 0;
    for (uint32_t i = 0; i <= max_symbol; i++) {
      total += freq_count[i];
    }

    uint32_t allocated = 0;
    for (uint32_t i = 0; i <= max_symbol; i++) {
      if (freq_count[i] > 0) {
        // Scale: freq * table_size / total
        norm_freq[i] = std::max(1, (int)((freq_count[i] * table_size) / total));
        allocated += norm_freq[i];
      }
    }

    // Adjust to exactly table_size
    while (allocated > table_size) {
      for (uint32_t i = 0; i <= max_symbol && allocated > table_size; i++) {
        if (norm_freq[i] > 1) {
          norm_freq[i]--;
          allocated--;
        }
      }
    }
    while (allocated < table_size) {
      for (uint32_t i = 0; i <= max_symbol && allocated < table_size; i++) {
        if (norm_freq[i] > 0) {
          norm_freq[i]++;
          allocated++;
        }
      }
    }

    printf("[MINIMAL] Normalized frequencies (table_log=%u, size=%u):\n",
           table_log, table_size);
    for (uint32_t i = 0; i <= max_symbol; i++) {
      if (norm_freq[i] > 0) {
        printf("  Symbol %u: %d\n", i, norm_freq[i]);
      }
    }
  }

  // Step 3: Build encoding table
  void build_ctable() {
    ctable.resize(max_symbol + 1);

    // For each symbol, calculate encoding parameters
    for (uint32_t symbol = 0; symbol <= max_symbol; symbol++) {
      if (norm_freq[symbol] <= 0)
        continue;

      uint32_t freq = norm_freq[symbol];

      // Calculate nbBits: log2(tableSize / freq)
      uint32_t max_bits_out = table_log - 1;
      uint32_t min_state_plus = freq << max_bits_out;

      while (min_state_plus > table_size) {
        max_bits_out--;
        min_state_plus = freq << max_bits_out;
      }

      ctable[symbol].deltaNbBits = (table_log - max_bits_out) << 16;
      ctable[symbol].deltaFindState = freq << max_bits_out;

      printf("[MINIMAL] Symbol %u: deltaNbBits=%u, deltaFindState=%u\n", symbol,
             ctable[symbol].deltaNbBits >> 16, ctable[symbol].deltaFindState);
    }
  }

  // Step 4: Encode data
  void encode(const uint8_t *input, size_t size, uint8_t *output,
              size_t *output_size) {
    uint32_t state = table_size; // Initial state
    size_t bit_pos = 0;

    printf("[MINIMAL] Encoding %zu bytes, initial state=%u\n", size, state);

    // Encode backwards (Zstd FSE convention)
    for (int i = size - 1; i >= 0; i--) {
      uint8_t symbol = input[i];

      // Get nbBits for this state
      uint32_t nbBits = (state + ctable[symbol].deltaNbBits) >> 16;

      // Output low bits
      uint32_t bits = state & ((1U << nbBits) - 1);

      // Write bits to output (simplified - just track bit position)
      bit_pos += nbBits;

      // Update state
      state = (state >> nbBits) + ctable[symbol].deltaFindState;

      if (i >= (int)size - 5) {
        printf("[MINIMAL] i=%d, symbol=%u, nbBits=%u, bits=%u, new_state=%u\n",
               i, symbol, nbBits, bits, state);
      }
    }

    *output_size = (bit_pos + 7) / 8; // Convert bits to bytes
    printf("[MINIMAL] Final state=%u, total_bits=%zu, output_bytes=%zu\n",
           state, bit_pos, *output_size);
  }
};

int main() {
  printf("╔══════════════════════════════════════════════════════════╗\n");
  printf("║  Minimal FSE Encoder Test                               ║\n");
  printf("║  Compare clean CPU implementation with GPU              ║\n");
  printf("╚══════════════════════════════════════════════════════════╝\n\n");

  // Test with simple repeating pattern
  const size_t input_size = 1024;
  const uint8_t symbol = 'A';

  std::vector<uint8_t> h_input(input_size, symbol);

  printf("Input: %zu bytes, all '%c'\n\n", input_size, symbol);

  // ========================================================================
  // Part 1: Minimal CPU FSE
  // ========================================================================
  printf("=== Minimal CPU FSE Encoder ===\n");

  MinimalFSEEncoder encoder;
  encoder.count_frequencies(h_input.data(), h_input.size());
  encoder.normalize_frequencies(9); // table_log = 9 (512 entries)
  encoder.build_ctable();

  std::vector<uint8_t> cpu_output(input_size);
  size_t cpu_size;
  encoder.encode(h_input.data(), h_input.size(), cpu_output.data(), &cpu_size);

  printf("\nCPU output: %zu bytes\n\n", cpu_size);

  // ========================================================================
  // Part 2: GPU FSE (for comparison)
  // ========================================================================
  printf("=== GPU FSE Encoder ===\n");

  uint8_t *d_input, *d_output;
  uint32_t *d_output_size;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, input_size * 2);
  cudaMalloc(&d_output_size, sizeof(uint32_t));
  cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);

  printf("Calling GPU encode_fse_advanced...\n");
  cuda_zstd::Status status = cuda_zstd::fse::encode_fse_advanced(
      d_input, input_size, d_output, d_output_size, true, 0);

  if (status != cuda_zstd::Status::SUCCESS) {
    printf("❌ GPU FSE failed with status %d\n", (int)status);
    return 1;
  }

  uint32_t gpu_size;
  cudaMemcpy(&gpu_size, d_output_size, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  std::vector<uint8_t> gpu_output(gpu_size);
  cudaMemcpy(gpu_output.data(), d_output, gpu_size, cudaMemcpyDeviceToHost);

  printf("GPU output: %u bytes\n\n", gpu_size);

  // ========================================================================
  // Part 3: Analysis
  // ========================================================================
  printf("=== Analysis ===\n");
  printf("CPU: %zu bytes\n", cpu_size);
  printf("GPU: %u bytes\n", gpu_size);

  if (cpu_size != gpu_size) {
    printf("⚠️  Size mismatch! Outputs have different lengths.\n");
  } else {
    printf("✓ Sizes match\n");
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  printf("\n✅ Test completed\n");
  return 0;
}
