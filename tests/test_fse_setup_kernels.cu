#include "../include/cuda_zstd_types.h"
#include "../src/cuda_zstd_fse_chunk_kernel.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      exit(1);                                                                 \
    }                                                                          \
  }

// Helper to build a simple FSE table for testing
void build_test_table(u16 *h_stateTable, GPU_FSE_SymbolTransform *h_symbolTT,
                      u16 tableLog) {
  u32 tableSize = 1 << tableLog;
  for (u32 i = 0; i < tableSize; ++i) {
    h_stateTable[i] = (u16)((i + 1) % tableSize);
    if (h_stateTable[i] == 0)
      h_stateTable[i] = 1;
  }
  for (int s = 0; s < 256; ++s) {
    h_symbolTT[s].deltaNbBits = 1 << 16; // 1 bit per symbol
    h_symbolTT[s].deltaFindState = 0;
  }
}

// Test: Parallel kernel produces expected bit counts for independent chunks
bool test_parallel_bit_counts() {
  std::cout << "[TEST] Parallel Kernel Bit Count Correctness..." << std::endl;

  const u32 input_size = 64 * 1024; // 64KB
  const u32 chunk_size = 4 * 1024;  // 4KB chunks
  const u16 tableLog = 9;
  const u32 tableSize = 1 << tableLog;
  const u32 num_chunks = (input_size + chunk_size - 1) / chunk_size;

  // Host Data
  std::vector<byte_t> h_input(input_size);
  std::mt19937 rng(42);
  for (u32 i = 0; i < input_size; ++i) {
    h_input[i] = rng() % 256;
  }

  std::vector<u16> h_stateTable(tableSize);
  std::vector<GPU_FSE_SymbolTransform> h_symbolTT(256);
  build_test_table(h_stateTable.data(), h_symbolTT.data(), tableLog);

  // Device Allocations
  byte_t *d_input;
  u16 *d_stateTable, *d_par_states;
  GPU_FSE_SymbolTransform *d_symbolTT;
  u32 *d_par_bits;

  CHECK(cudaMalloc(&d_input, input_size));
  CHECK(cudaMalloc(&d_stateTable, tableSize * sizeof(u16)));
  CHECK(cudaMalloc(&d_symbolTT, 256 * sizeof(GPU_FSE_SymbolTransform)));
  CHECK(cudaMalloc(&d_par_states, num_chunks * sizeof(u16)));
  CHECK(cudaMalloc(&d_par_bits, num_chunks * sizeof(u32)));

  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_stateTable, h_stateTable.data(), tableSize * sizeof(u16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_symbolTT, h_symbolTT.data(),
                   256 * sizeof(GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Run Parallel
  const u32 blockSize = 256;
  const u32 gridSize = (num_chunks + blockSize - 1) / blockSize;
  fse_compute_states_kernel_parallel<<<gridSize, blockSize, 0, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_par_states, d_par_bits);
  CHECK(cudaStreamSynchronize(stream));

  // Verify: Each chunk should have chunk_size + tableLog bits (1 bit per symbol
  // + tableLog)
  std::vector<u32> h_par_bits(num_chunks);
  CHECK(cudaMemcpy(h_par_bits.data(), d_par_bits, num_chunks * sizeof(u32),
                   cudaMemcpyDeviceToHost));

  bool pass = true;
  for (u32 i = 0; i < num_chunks; ++i) {
    u32 expected = chunk_size + tableLog; // 4096 + 9 = 4105
    if (h_par_bits[i] != expected) {
      std::cerr << "  Mismatch at chunk " << i << ": expected=" << expected
                << " got=" << h_par_bits[i] << std::endl;
      pass = false;
    }
  }

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_stateTable));
  CHECK(cudaFree(d_symbolTT));
  CHECK(cudaFree(d_par_states));
  CHECK(cudaFree(d_par_bits));
  CHECK(cudaStreamDestroy(stream));

  std::cout << (pass ? "  PASSED" : "  FAILED") << std::endl;
  return pass;
}

// Test: Shared Memory kernel produces same results as Global Memory kernel
bool test_shmem_vs_global() {
  std::cout << "[TEST] SharedMem vs GlobalMem Consistency..." << std::endl;

  const u32 input_size = 64 * 1024;
  const u32 chunk_size = 4 * 1024;
  const u16 tableLog = 9;
  const u32 tableSize = 1 << tableLog;
  const u32 num_chunks = (input_size + chunk_size - 1) / chunk_size;

  std::vector<byte_t> h_input(input_size);
  std::mt19937 rng(123);
  for (u32 i = 0; i < input_size; ++i)
    h_input[i] = rng() % 256;

  std::vector<u16> h_stateTable(tableSize);
  std::vector<GPU_FSE_SymbolTransform> h_symbolTT(256);
  build_test_table(h_stateTable.data(), h_symbolTT.data(), tableLog);

  byte_t *d_input;
  u16 *d_stateTable, *d_global_states, *d_shmem_states;
  GPU_FSE_SymbolTransform *d_symbolTT;
  u32 *d_global_bits, *d_shmem_bits;

  CHECK(cudaMalloc(&d_input, input_size));
  CHECK(cudaMalloc(&d_stateTable, tableSize * sizeof(u16)));
  CHECK(cudaMalloc(&d_symbolTT, 256 * sizeof(GPU_FSE_SymbolTransform)));
  CHECK(cudaMalloc(&d_global_states, num_chunks * sizeof(u16)));
  CHECK(cudaMalloc(&d_shmem_states, num_chunks * sizeof(u16)));
  CHECK(cudaMalloc(&d_global_bits, num_chunks * sizeof(u32)));
  CHECK(cudaMalloc(&d_shmem_bits, num_chunks * sizeof(u32)));

  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_stateTable, h_stateTable.data(), tableSize * sizeof(u16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_symbolTT, h_symbolTT.data(),
                   256 * sizeof(GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  const u32 blockSize = 256;
  const u32 gridSize = (num_chunks + blockSize - 1) / blockSize;

  // Global Memory Kernel
  fse_compute_states_kernel_parallel<<<gridSize, blockSize, 0, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_global_states, d_global_bits);
  CHECK(cudaStreamSynchronize(stream));

  // Shared Memory Kernel
  size_t sharedMemSize = tableSize * sizeof(u16);
  sharedMemSize = (sharedMemSize + 7) & ~7;
  sharedMemSize += 256 * sizeof(GPU_FSE_SymbolTransform);

  fse_compute_states_kernel_parallel_shmem<<<gridSize, blockSize, sharedMemSize,
                                             stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_shmem_states, d_shmem_bits);
  CHECK(cudaStreamSynchronize(stream));

  // Compare
  std::vector<u32> h_global_bits(num_chunks), h_shmem_bits(num_chunks);
  CHECK(cudaMemcpy(h_global_bits.data(), d_global_bits,
                   num_chunks * sizeof(u32), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_shmem_bits.data(), d_shmem_bits, num_chunks * sizeof(u32),
                   cudaMemcpyDeviceToHost));

  bool pass = true;
  for (u32 i = 0; i < num_chunks; ++i) {
    if (h_global_bits[i] != h_shmem_bits[i]) {
      std::cerr << "  Mismatch at chunk " << i
                << ": global=" << h_global_bits[i]
                << " shmem=" << h_shmem_bits[i] << std::endl;
      pass = false;
    }
  }

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_stateTable));
  CHECK(cudaFree(d_symbolTT));
  CHECK(cudaFree(d_global_states));
  CHECK(cudaFree(d_shmem_states));
  CHECK(cudaFree(d_global_bits));
  CHECK(cudaFree(d_shmem_bits));
  CHECK(cudaStreamDestroy(stream));

  std::cout << (pass ? "  PASSED" : "  FAILED") << std::endl;
  return pass;
}

// Test: Edge case - Single chunk (input_size < chunk_size)
bool test_single_chunk() {
  std::cout << "[TEST] Single Chunk Edge Case..." << std::endl;

  const u32 input_size = 1024;      // 1KB
  const u32 chunk_size = 64 * 1024; // 64KB (larger than input)
  const u16 tableLog = 9;
  const u32 tableSize = 1 << tableLog;
  const u32 num_chunks = 1;

  std::vector<byte_t> h_input(input_size, 0xAB);
  std::vector<u16> h_stateTable(tableSize);
  std::vector<GPU_FSE_SymbolTransform> h_symbolTT(256);
  build_test_table(h_stateTable.data(), h_symbolTT.data(), tableLog);

  byte_t *d_input;
  u16 *d_stateTable, *d_states;
  GPU_FSE_SymbolTransform *d_symbolTT;
  u32 *d_bits;

  CHECK(cudaMalloc(&d_input, input_size));
  CHECK(cudaMalloc(&d_stateTable, tableSize * sizeof(u16)));
  CHECK(cudaMalloc(&d_symbolTT, 256 * sizeof(GPU_FSE_SymbolTransform)));
  CHECK(cudaMalloc(&d_states, num_chunks * sizeof(u16)));
  CHECK(cudaMalloc(&d_bits, num_chunks * sizeof(u32)));

  CHECK(
      cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_stateTable, h_stateTable.data(), tableSize * sizeof(u16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_symbolTT, h_symbolTT.data(),
                   256 * sizeof(GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  fse_compute_states_kernel_parallel<<<1, 1, 0, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_states, d_bits);
  CHECK(cudaStreamSynchronize(stream));

  u32 h_bits;
  CHECK(cudaMemcpy(&h_bits, d_bits, sizeof(u32), cudaMemcpyDeviceToHost));

  // For 1024 symbols at 1 bit each + tableLog = 1024 + 9 = 1033
  bool pass = (h_bits == input_size + tableLog);
  if (!pass) {
    std::cerr << "  Expected " << (input_size + tableLog) << " bits, got "
              << h_bits << std::endl;
  }

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_stateTable));
  CHECK(cudaFree(d_symbolTT));
  CHECK(cudaFree(d_states));
  CHECK(cudaFree(d_bits));
  CHECK(cudaStreamDestroy(stream));

  std::cout << (pass ? "  PASSED" : "  FAILED") << std::endl;
  return pass;
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "Running FSE Setup Kernel Unit Tests" << std::endl;
  std::cout << "======================================" << std::endl;

  int passed = 0, failed = 0;

  if (test_parallel_bit_counts())
    passed++;
  else
    failed++;
  if (test_shmem_vs_global())
    passed++;
  else
    failed++;
  if (test_single_chunk())
    passed++;
  else
    failed++;

  std::cout << "======================================" << std::endl;
  std::cout << "Results: " << passed << " PASSED, " << failed << " FAILED"
            << std::endl;

  return (failed == 0) ? 0 : 1;
}
