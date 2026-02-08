#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "../include/cuda_zstd_types.h"
#include "../src/cuda_zstd_fse_chunk_kernel.cuh"
#include "cuda_zstd_safe_alloc.h"

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

// Dummy table builder (same as unit test)
void build_dummy_table(u16 *h_stateTable, GPU_FSE_SymbolTransform *h_symbolTT,
                       u16 tableLog) {
  u32 tableSize = 1 << tableLog;
  for (u32 i = 0; i < tableSize; ++i) {
    h_stateTable[i] = (u16)((i + 1) % tableSize);
    if (h_stateTable[i] == 0)
      h_stateTable[i] = 1;
  }
  for (int s = 0; s < 256; ++s) {
    h_symbolTT[s].deltaNbBits = 1 << 16;
    h_symbolTT[s].deltaFindState = 0;
  }
}

int main() {
  cudaDeviceReset(); // Clean start
  std::cout << "Running benchmark_setup_throughput..." << std::endl;

  const u32 chunk_size = 64 * 1024;
  const size_t input_size = 64 * 1024 * 1024; // 64 MB
  const u32 num_chunks = (input_size + chunk_size - 1) / chunk_size;
  const u16 tableLog = 11;

  std::cout << "Input Size: " << (input_size / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Chunks: " << num_chunks << std::endl;

  // Allocations
  byte_t *d_input;
  u16 *d_stateTable;
  GPU_FSE_SymbolTransform *d_symbolTT;
  u16 *d_seq_states, *d_par_states;
  u32 *d_seq_bits, *d_par_bits;

  CHECK(
      cuda_zstd::safe_cuda_malloc(&d_input, input_size)); // No data needed, just pointer access
  // But we should fill it with random data to avoid 0s (which might end loop
  // early?) Actually current loop logic depends on symbols. Random is better.
  // Fill on device?
  // Or just Memset? Memset 0 might trigger `current_idx == 0` check? No.
  // Memset is fast.
  CHECK(cudaMemset(d_input, 1, input_size));

  u32 tableSize = 1 << tableLog;
  std::vector<u16> h_stateTable(tableSize);
  std::vector<GPU_FSE_SymbolTransform> h_symbolTT(256);
  build_dummy_table(h_stateTable.data(), h_symbolTT.data(), tableLog);

  CHECK(cuda_zstd::safe_cuda_malloc(&d_stateTable, tableSize * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_symbolTT, 256 * sizeof(GPU_FSE_SymbolTransform)));
  CHECK(cudaMemcpy(d_stateTable, h_stateTable.data(), tableSize * sizeof(u16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_symbolTT, h_symbolTT.data(),
                   256 * sizeof(GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));

  CHECK(cuda_zstd::safe_cuda_malloc(&d_seq_states, num_chunks * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_seq_bits, num_chunks * sizeof(u32)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_par_states, num_chunks * sizeof(u16)));
  CHECK(cuda_zstd::safe_cuda_malloc(&d_par_bits, num_chunks * sizeof(u32)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Warmup Sequential
  fse_compute_states_kernel_sequential<<<1, 1, 0, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_seq_states, d_seq_bits);
  CHECK(cudaStreamSynchronize(stream));

  // Benchmark Sequential
  auto start_seq = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5; ++i) {
    fse_compute_states_kernel_sequential<<<1, 1, 0, stream>>>(
        d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
        d_seq_states, d_seq_bits);
  }
  CHECK(cudaStreamSynchronize(stream));
  auto end_seq = std::chrono::high_resolution_clock::now();
  double time_seq =
      std::chrono::duration<double>(end_seq - start_seq).count() / 5.0;
  double gbps_seq = (input_size / 1e9) / time_seq;
  std::cout << "Sequential Throughput: " << gbps_seq << " GB/s ("
            << (time_seq * 1000.0) << " ms)" << std::endl;

  // Warmup Parallel
  int blockSize = 256;
  int gridSize = (num_chunks + blockSize - 1) / blockSize;
  fse_compute_states_kernel_parallel<<<gridSize, blockSize, 0, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_par_states, d_par_bits);
  CHECK(cudaStreamSynchronize(stream));

  // Benchmark Parallel (Global Mem)
  auto start_par = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) { // More iterations for fast kernel
    fse_compute_states_kernel_parallel<<<gridSize, blockSize, 0, stream>>>(
        d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
        d_par_states, d_par_bits);
  }
  CHECK(cudaStreamSynchronize(stream));
  auto end_par = std::chrono::high_resolution_clock::now();
  double time_par =
      std::chrono::duration<double>(end_par - start_par).count() / 100.0;
  double gbps_par = (input_size / 1e9) / time_par;
  std::cout << "Parallel (GlobalMem) Throughput: " << gbps_par << " GB/s ("
            << (time_par * 1000.0) << " ms)" << std::endl;

  // Benchmark Parallel (Shared Mem)
  // Reuse tableSize from earlier (line 70)

  size_t sharedMemSize = tableSize * sizeof(u16);
  sharedMemSize = (sharedMemSize + 7) & ~7; // Align
  sharedMemSize += 256 * sizeof(GPU_FSE_SymbolTransform);

  // Warmup
  fse_compute_states_kernel_parallel_shmem<<<gridSize, blockSize, sharedMemSize,
                                             stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_par_states, d_par_bits);
  CHECK(cudaStreamSynchronize(stream));

  auto start_shmem = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    fse_compute_states_kernel_parallel_shmem<<<gridSize, blockSize,
                                               sharedMemSize, stream>>>(
        d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
        d_par_states, d_par_bits);
  }
  CHECK(cudaStreamSynchronize(stream));
  auto end_shmem = std::chrono::high_resolution_clock::now();
  double time_shmem =
      std::chrono::duration<double>(end_shmem - start_shmem).count() / 100.0;
  double gbps_shmem = (input_size / 1e9) / time_shmem;

  std::cout << "Parallel (SharedMem) Throughput: " << gbps_shmem << " GB/s ("
            << (time_shmem * 1000.0) << " ms)" << std::endl;

  // Benchmark Parallel (Bufffered Int4)
  fse_compute_states_kernel_parallel_shmem_int4<<<gridSize, blockSize,
                                                  sharedMemSize, stream>>>(
      d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
      d_par_states, d_par_bits);
  CHECK(cudaStreamSynchronize(stream));

  auto start_buf = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    fse_compute_states_kernel_parallel_shmem_int4<<<gridSize, blockSize,
                                                    sharedMemSize, stream>>>(
        d_input, input_size, d_stateTable, d_symbolTT, tableLog, chunk_size,
        d_par_states, d_par_bits);
  }
  CHECK(cudaStreamSynchronize(stream));
  auto end_buf = std::chrono::high_resolution_clock::now();
  double time_buf =
      std::chrono::duration<double>(end_buf - start_buf).count() / 100.0;
  double gbps_buf = (input_size / 1e9) / time_buf;

  std::cout << "Parallel (Buffered) Throughput:  " << gbps_buf << " GB/s ("
            << (time_buf * 1000.0) << " ms)" << std::endl;

  std::cout << "Speedup (Seq -> ParGlobal): " << (time_seq / time_par) << "x"
            << std::endl;
  std::cout << "Speedup (Seq -> ParShmem): " << (time_seq / time_shmem) << "x"
            << std::endl;
  std::cout << "Speedup (Seq -> ParBuf):   " << (time_seq / time_buf) << "x"
            << std::endl;
  std::cout << "Speedup (Shmem -> Buf):    " << (time_shmem / time_buf) << "x"
            << std::endl;

  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_stateTable));
  CHECK(cudaFree(d_symbolTT));
  CHECK(cudaFree(d_seq_states));
  return 0;
}
