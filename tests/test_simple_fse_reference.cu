#include "cuda_error_checking.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>


using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Simple bitstream writer
struct SimpleBitstream {
  std::vector<byte_t> buffer;
  u32 bit_pos;
  u64 bit_buffer;
  u32 bits_in_buffer;

  SimpleBitstream()
      : buffer(1024), bit_pos(0), bit_buffer(0), bits_in_buffer(0) {}

  void addBits(u32 value, u32 numBits) {
    bit_buffer |= (u64)value << bits_in_buffer;
    bits_in_buffer += numBits;

    while (bits_in_buffer >= 8) {
      buffer[bit_pos++] = (byte_t)(bit_buffer & 0xFF);
      bit_buffer >>= 8;
      bits_in_buffer -= 8;
    }
  }

  void flush() {
    while (bits_in_buffer > 0) {
      buffer[bit_pos++] = (byte_t)(bit_buffer & 0xFF);
      bit_buffer >>= 8;
      bits_in_buffer = (bits_in_buffer >= 8) ? bits_in_buffer - 8 : 0;
    }
  }

  void print_bytes(const char *label) {
    printf("%s (%u bytes):", label, bit_pos);
    for (u32 i = 0; i < bit_pos; i++) {
      if (i % 16 == 0)
        printf("\n  ");
      printf("%02X ", buffer[i]);
    }
    printf("\n");
  }
};

int main() {
  printf("=== Simple FSE Reference Test ===\n");

  // Test 4 data: {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6}
  byte_t test_data[] = {1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6};
  u32 data_size = 11;

  printf("Input: ");
  for (u32 i = 0; i < data_size; i++)
    printf("%u ", test_data[i]);
  printf("\n");

  // Build FSE table
  FSEStats stats;
  stats.total_count = data_size;
  memset(stats.frequencies, 0, sizeof(stats.frequencies));

  for (u32 i = 0; i < data_size; i++) {
    stats.frequencies[test_data[i]]++;
  }

  stats.max_symbol = 0;
  stats.unique_symbols = 0;
  for (int i = 0; i < 256; i++) {
    if (stats.frequencies[i] > 0) {
      stats.max_symbol = i;
      stats.unique_symbols++;
    }
  }

  printf("Max symbol: %u, Unique symbols: %u\n", stats.max_symbol,
         stats.unique_symbols);
  printf("Frequencies: ");
  for (u32 i = 0; i <= stats.max_symbol; i++) {
    if (stats.frequencies[i] > 0)
      printf("%u:%u ", i, stats.frequencies[i]);
  }
  printf("\n");

  u32 table_log =
      select_optimal_table_log(stats.frequencies, stats.total_count,
                               stats.max_symbol, stats.unique_symbols);
  u32 table_size = 1u << table_log;

  printf("Table log: %u, Table size: %u\n", table_log, table_size);

  std::vector<u16> h_normalized(stats.max_symbol + 1);
  normalize_frequencies_accurate(stats.frequencies, stats.total_count,
                                 table_size, h_normalized.data(),
                                 stats.max_symbol, nullptr);

  printf("Normalized freqs: ");
  for (u32 i = 0; i <= stats.max_symbol; i++) {
    if (h_normalized[i] > 0)
      printf("%u:%u ", i, h_normalized[i]);
  }
  printf("\n");

  FSEEncodeTable h_table;
  FSE_buildCTable_Host(h_normalized.data(), stats.max_symbol, table_log,
                       &h_table);

  // Encode using same logic as GPU
  SimpleBitstream bs;
  u32 state = table_size;

  printf("\n=== ENCODING (Forward 0->N) ===\n");
  for (u32 i = 0; i < data_size; i++) {
    u8 symbol = test_data[i];
    const FSEEncodeTable::FSEEncodeSymbol &symInfo =
        h_table.d_symbol_table[symbol];

    u32 old_state = state;
    u32 nbBitsOut = (state + symInfo.deltaNbBits) >> 16;
    u32 val = state & ((1u << nbBitsOut) - 1);

    bs.addBits(val, nbBitsOut);

    u32 nextIdx = (state >> nbBitsOut) + symInfo.deltaFindState;
    state = h_table.d_next_state[nextIdx];

    printf(
        "[%u] sym=%u: state=%u -> val=%u (%u bits) -> nextIdx=%u -> state=%u\n",
        i, symbol, old_state, val, nbBitsOut, nextIdx, state);
  }

  printf("\nFinal state: %u\n", state);
  printf("Writing final state (%u bits)\n", table_log);
  bs.addBits(state, table_log);

  bs.flush();

  bs.print_bytes("Reference Output");

  // Now run GPU encoder and compare
  byte_t *d_input = nullptr;
  byte_t *d_output = nullptr;
  u32 encoded_size = 0;

  CUDA_CHECK(cudaMalloc(&d_input, data_size));
  CUDA_CHECK(cudaMalloc(&d_output, data_size * 2 + 1024));
  CUDA_CHECK(cudaMemcpy(d_input, test_data, data_size, cudaMemcpyHostToDevice));

  Status status = encode_fse_advanced_debug(d_input, data_size, d_output,
                                            &encoded_size, true, 0);

  if (status != Status::SUCCESS) {
    printf("ERROR: GPU encoding failed: %d\n", (int)status);
    return 1;
  }

  printf("\nGPU encoded size: %u bytes\n", encoded_size);

  std::vector<byte_t> gpu_output(encoded_size);
  CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, encoded_size,
                        cudaMemcpyDeviceToHost));

  printf("GPU Output:");
  for (u32 i = 0; i < encoded_size; i++) {
    if (i % 16 == 0)
      printf("\n  ");
    printf("%02X ", gpu_output[i]);
  }
  printf("\n");

  // Skip header
  u32 header_size = 12 + (stats.max_symbol + 1) * 2;
  printf("\nHeader size: %u bytes\n", header_size);
  printf("GPU Bitstream (after header):");
  for (u32 i = header_size; i < encoded_size; i++) {
    if ((i - header_size) % 16 == 0)
      printf("\n  ");
    printf("%02X ", gpu_output[i]);
  }
  printf("\n");

  // Compare
  printf("\n=== COMPARISON ===\n");
  bool match = true;
  u32 bitstream_size = encoded_size - header_size;
  for (u32 i = 0; i < std::min(bs.bit_pos, bitstream_size); i++) {
    byte_t ref_byte = bs.buffer[i];
    byte_t gpu_byte = gpu_output[header_size + i];
    if (ref_byte != gpu_byte) {
      printf("MISMATCH at byte %u: Reference=0x%02X, GPU=0x%02X\n", i, ref_byte,
             gpu_byte);
      match = false;
    }
  }

  if (match && bs.bit_pos == bitstream_size) {
    printf("✅ BITSTREAMS MATCH!\n");
  } else {
    printf("❌ BITSTREAMS DIFFER\n");
    printf("Reference size: %u bytes, GPU size: %u bytes\n", bs.bit_pos,
           bitstream_size);
  }

  // Cleanup
  delete[] h_table.d_symbol_table;
  delete[] h_table.d_next_state;
  delete[] h_table.d_state_to_symbol;

  cudaFree(d_input);
  cudaFree(d_output);

  return match ? 0 : 1;
}
