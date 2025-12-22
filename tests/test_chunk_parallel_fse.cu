#include "../src/cuda_zstd_fse_chunk_kernel.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Helper to check CUDA errors
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Test Configuration
const int NUM_SYMBOLS = 100;
const int CHUNK_SIZE = 50;

void setup_manual_table(u16 **d_stateTable,
                        GPU_FSE_SymbolTransform **d_symbolTT) {
  // Trivial Table: Log=1. States 2,3.
  // Symbol 0: nbBits=1, find=1 -> index= (state>>1)+1 = (1)+1=2.
  // Symbol 1: nbBits=1, find=1 -> index= (state>>1)+1 = 2.

  // stateTable[2] = 2; stateTable[3] = 3;
  // index 2 -> State 2.
  // index 3 -> State 3.
  // (We use indices 2,3 to store states 2,3).

  std::vector<u16> h_stateTable(4);
  h_stateTable[2] = 2;
  h_stateTable[3] = 3;

  std::vector<GPU_FSE_SymbolTransform> h_symbolTT(2); // Symbols 0, 1
  // nbBits = 1. delta >= 65536 - state. Max state 3.
  // delta = 65536. (3+65536)>>16 = 1.
  // deltaNbBits = 65536.
  // deltaFindState = 1.
  h_symbolTT[0] = {1, 65536};
  h_symbolTT[1] = {1, 65536};

  CHECK(cudaMalloc(d_stateTable, 4 * sizeof(u16)));
  CHECK(cudaMalloc(d_symbolTT, 2 * sizeof(GPU_FSE_SymbolTransform)));

  CHECK(cudaMemcpy(*d_stateTable, h_stateTable.data(), 4 * sizeof(u16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(*d_symbolTT, h_symbolTT.data(),
                   2 * sizeof(GPU_FSE_SymbolTransform),
                   cudaMemcpyHostToDevice));
}

void verification_test() {
  std::cout << "Running Chunk Parallel Kernels Verification..." << std::endl;

  // 1. Setup Input
  std::vector<byte_t> h_symbols(NUM_SYMBOLS);
  for (int i = 0; i < NUM_SYMBOLS; i++)
    h_symbols[i] = i % 2; // Pattern 0, 1, 0, 1...

  byte_t *d_symbols;
  CHECK(cudaMalloc(&d_symbols, NUM_SYMBOLS));
  CHECK(cudaMemcpy(d_symbols, h_symbols.data(), NUM_SYMBOLS,
                   cudaMemcpyHostToDevice));

  // 2. Setup Table (Manual)
  u16 *d_stateTable;
  GPU_FSE_SymbolTransform *d_symbolTT;
  setup_manual_table(&d_stateTable, &d_symbolTT);

  // 3. Compute States (Pre-pass)
  // Expect 2 chunks (50 each).
  // Logic: Chunk 0 ends at 50, needs start state from end of Chunk 0
  // transition? Start of Chunk 1 matches End of Chunk 0. We scan Chunk 1 and
  // Chunk 0. Output: Start state for Chunk 0, Chunk 1? We only need Start State
  // for Chunk 0 if it wasn't the last chunk? No, Chunk 1 is last (self-init).
  // Chunk 0 needs start state. fse_compute_states_kernel outputs `out_states`
  // for chunks 0, 1...

  u32 num_chunks = (NUM_SYMBOLS + CHUNK_SIZE - 1) / CHUNK_SIZE;
  u16 *d_out_states;
  u32 *d_bit_counts;
  CHECK(cudaMalloc(&d_out_states, num_chunks * sizeof(u16)));
  CHECK(cudaMalloc(&d_bit_counts, num_chunks * sizeof(u32)));

  fse_compute_states_kernel<<<1, 1>>>(d_symbols, NUM_SYMBOLS, d_stateTable,
                                      d_symbolTT, 1, CHUNK_SIZE, d_out_states,
                                      d_bit_counts);
  CHECK(cudaDeviceSynchronize());

  // 4. Encode Chunks
  byte_t *d_output_buffer;
  u32 *d_chunk_offsets;
  u32 stride = 1024;
  CHECK(cudaMalloc(&d_output_buffer, num_chunks * stride));
  CHECK(cudaMalloc(&d_chunk_offsets, num_chunks * sizeof(u32)));

  fse_encode_chunk_kernel<<<1, num_chunks>>>(
      d_symbols, NUM_SYMBOLS, d_stateTable, d_symbolTT, 1, d_out_states,
      CHUNK_SIZE, d_output_buffer, d_chunk_offsets, stride);
  CHECK(cudaDeviceSynchronize());

  // 5. Verify Bit Counts & Sizes
  std::vector<u32> h_chunk_sizes(num_chunks);
  CHECK(cudaMemcpy(h_chunk_sizes.data(), d_chunk_offsets,
                   num_chunks * sizeof(u32), cudaMemcpyDeviceToHost));

  std::cout << "Chunk Sizes: " << h_chunk_sizes[0] << ", " << h_chunk_sizes[1]
            << std::endl;
  // Logic: 50 symbols. 1 bit each. 50 bits = 7 bytes (6.25).
  // Plus flush?
  // Chunk 0 flushes state (Log 1 = 1 bit?) Or Log bits?
  // Chunk 1 flushes bits.
  // Expect > 0.
  if (h_chunk_sizes[0] == 0 || h_chunk_sizes[1] == 0) {
    std::cerr << "FAILED: Chunk size is 0." << std::endl;
    exit(1);
  }

  std::cout << "Verification PASSED!" << std::endl;

  CHECK(cudaFree(d_symbols));
  CHECK(cudaFree(d_stateTable));
  CHECK(cudaFree(d_symbolTT));
  CHECK(cudaFree(d_out_states));
  CHECK(cudaFree(d_bit_counts));
  CHECK(cudaFree(d_output_buffer));
  CHECK(cudaFree(d_chunk_offsets));
}

int main() {
  verification_test();
  return 0;
}
