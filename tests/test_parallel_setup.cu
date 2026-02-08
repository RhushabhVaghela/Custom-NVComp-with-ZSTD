
#include "../include/cuda_zstd_types.h"
#include "../src/cuda_zstd_fse_chunk_kernel.cuh" // Access internal kernels directly
#include "cuda_zstd_safe_alloc.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Helper to check CUDA errors
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      exit(1);                                                                 \
    }                                                                          \
  }

void generate_random_data(std::vector<byte_t> &data, size_t size) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < size; ++i) {
    data[i] = (byte_t)dist(rng);
  }
}

// Dummy FSE Table Generator (We just need VALID tables, doesn't matter if they
// match data perfectly for this test) Actually, for the setup kernel to run
// without crashing, it needs valid tables that don't produce out-of-bounds. We
// can use a simple trivial table: 1 state, 1 symbol? No, let's use a small
// valid table. Or just copy the logic from benchmark_phase3? Ideally we rely on
// the implementation details: it uses `symbolTT` and `stateTable`. We can
// manually construct a minimal valid table.

void build_dummy_table(u16 *h_stateTable,
                       cuda_zstd::fse::GPU_FSE_SymbolTransform *h_symbolTT,
                       u16 tableLog) {
  u32 tableSize = 1 << tableLog;
  // Fill state table with valid next states
  for (u32 i = 0; i < tableSize; ++i) {
    h_stateTable[i] = (u16)((i + 1) % tableSize); // Simple cycle
    if (h_stateTable[i] == 0)
      h_stateTable[i] = 1; // Avoid state 0 if problematic
  }

  // Fill symbolTT
  for (int s = 0; s < 256; ++s) {
    h_symbolTT[s].deltaNbBits = 1 << 16; // 1 bit?
    h_symbolTT[s].deltaFindState = 0;
  }
}

int main() {
  std::cout << "Running test_parallel_setup..." << std::endl;

  const u32 chunk_size = 32; // Small chunk for testing
  const u32 num_chunks = 10;
  const u32 input_size = chunk_size * num_chunks;
  const u16 tableLog = 10;

  // 1. Prepare Data
  std::vector<byte_t> h_input(input_size);
  generate_random_data(h_input, input_size);

  byte_t *d_input;
  CHECK(cuda_zstd::safe_cuda_malloc(&d_input, input_size));
  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));

  // 2. Prepare Tables (Dummy)
  std::vector<u16> h_stateTable(1 << tableLog);
  std::vector<cuda_zstd::fse::GPU_FSE_SymbolTransform> h_symbolTT(256);
  build_dummy_table(h_stateTable.data(), h_symbolTT.data(), tableLog);

  u16 *d_stateTable;
  cuda_zstd::fse::GPU_FSE_SymbolTransform *d_symbolTT;
  CHECK(cuda_zstd::safe_cuda_malloc(&d_stateTable, h_stateTable.size() * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_symbolTT,
                   h_symbolTT.size() *
                       sizeof(cuda_zstd::fse::GPU_FSE_SymbolTransform)));
  CHECK(cudaMemcpy(d_stateTable, h_stateTable.data(),
                   h_stateTable.size() * sizeof(u16), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_symbolTT, h_symbolTT.data(),
                   h_symbolTT.size() *
                       sizeof(cuda_zstd::fse::GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));

  // 3. Allocate Outputs
  u16 *d_seq_states;
  u32 *d_seq_bits;
  u16 *d_par_states;
  u32 *d_par_bits;

  CHECK(cuda_zstd::safe_cuda_malloc(&d_seq_states, num_chunks * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_seq_bits, num_chunks * sizeof(u32)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_par_states, num_chunks * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_par_bits, num_chunks * sizeof(u32)));

  CHECK(cudaMemset(d_seq_states, 0, num_chunks * sizeof(u16)));
  CHECK(cudaMemset(d_par_states, 0, num_chunks * sizeof(u16)));

  // 4. Run Parallel (Candidate)
  std::cout << "Launching Parallel..." << std::endl;
  // Launch 1 block with num_chunks threads (or enough blocks)
  // Kernel expects grid * block threads >= num_chunks
  cuda_zstd::fse::fse_compute_states_kernel_parallel<<<1, num_chunks>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_par_states, d_par_bits);
  CHECK(cudaDeviceSynchronize());

  // 5. Verify (Host Calculation)
  std::vector<u16> h_par_states(num_chunks);
  std::vector<u32> h_par_bits(num_chunks);
  CHECK(cudaMemcpy(h_par_states.data(), d_par_states, num_chunks * sizeof(u16),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_par_bits.data(), d_par_bits, num_chunks * sizeof(u32),
                   cudaMemcpyDeviceToHost));

  bool pass = true;
  for (int i = 0; i < num_chunks; ++i) {
    // Calculate Expected State for Independent Chunk
    u32 start_idx = i * chunk_size;
    u32 end_idx = std::min((u32)((i + 1) * chunk_size), input_size);
    if (end_idx == 0)
      continue;

    byte_t last_symbol = h_input[end_idx - 1];

    // Manual Init State Calculation using Dummy Table Logic
    // In dummy table:
    // deltaNbBits = 0 => nbBitsOut = (0 + (1<<15)) >> 16 = 0?
    // Wait, review build_dummy_table logic below.
    // We need check what we programmed.

    // Replicating Kernel Logic:
    // nbBitsOut = (deltaNbBits + (1<<15)) >> 16
    // tempValue = (nbBitsOut << 16) - deltaNbBits
    // tableIndex = (tempValue >> nbBitsOut) + deltaFindState

    // For dummy table in main (see below):
    // deltaNbBits = 1 << 16 (65536) -> nbBitsOut = (65536 + 32768) >> 16 = 1.
    // tempValue = (1 << 16) - 65536 = 0.
    // deltaFindState = 0.
    // tableIndex = (0 >> 1) + 0 = 0.

    // StateTable[0] -> 1 (from build_dummy_table)
    // So expected state should be 1.

    u16 expected_state = 1;

    if (h_par_states[i] != expected_state) {
      std::cout << "MISMATCH at chunk " << i << ": Expected=" << expected_state
                << " Par=" << h_par_states[i] << std::endl;
      pass = false;
    }

    // Check Bit Count?
    // Loop runs for (end_idx - 1) - start_idx times.
    // Each step adds nbBits.
    // State 1 -> Loop Index = (1 >> 0) + 0 = 1.
    // StateTable[1] = 2.
    // Next State 2 -> Index 2...
    // For Dummy Table:
    // deltaNbBits = 1<<16 implies nbBits = (state + 65536) >> 16.
    // Since state < 1024 (tableLog 10), state+65536 is ~65536..66560.
    // Result >> 16 is 1.
    // So always 1 bit per symbol.
    // Init symbol gives 1 bit.
    // Loop gives (end_idx - 1 - start_idx) bits.
    // Flush gives tableLog = 10 bits.
    // Total = 1 + (size - 1) + 10 = size + 10.

    u32 chunk_len = end_idx - start_idx;
    u32 expected_bits = chunk_len + 10;

    if (h_par_bits[i] != expected_bits) {
      std::cout << "BIT COUNT MISMATCH at chunk " << i
                << ": Expected=" << expected_bits << " Par=" << h_par_bits[i]
                << std::endl;
      pass = false;
    }
  }

  if (pass) {
    std::cout << "TEST PASSED" << std::endl;
  } else {
    std::cout << "TEST FAILED" << std::endl;
    return 1;
  }

  return 0;
}
