/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 */

#include "../src/cuda_zstd_fse_reference.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_utils.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// Helper types
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

// Helper to build a proper GPU CTable using FSE_buildCTable_Device().
// This allocates all required device arrays and uses the k_build_ctable kernel
// to correctly populate them, following the pattern from
// encode_sequences_with_predefined_fse() in cuda_zstd_manager.cu.
void build_gpu_table(const std::vector<short> &counts, unsigned table_log,
                     cuda_zstd::fse::FSEEncodeTable &gpu_table,
                     cuda_zstd::fse::FSEEncodeTable *d_table_slot) {
  using namespace cuda_zstd::fse;

  u32 max_symbol = (u32)(counts.size() - 1);
  u32 table_size = 1u << table_log;

  // 1. Fill host-side descriptor
  FSEEncodeTable h_desc;
  memset(&h_desc, 0, sizeof(h_desc));
  h_desc.table_log = table_log;
  h_desc.table_size = table_size;
  h_desc.max_symbol = max_symbol;

  // 2. Allocate ALL device arrays the kernel and encoder need
  cudaMalloc(&h_desc.d_symbol_table,
             (max_symbol + 1) * sizeof(FSEEncodeTable::FSEEncodeSymbol));
  cudaMalloc(&h_desc.d_next_state, table_size * sizeof(u16));
  cudaMalloc(&h_desc.d_state_to_symbol, table_size * sizeof(u8));
  cudaMalloc(&h_desc.d_symbol_first_state, (max_symbol + 1) * sizeof(u16));
  h_desc.d_nbBits_table = nullptr;     // Not used by k_encode_fse_interleaved
  h_desc.d_next_state_vals = nullptr;   // Not used by k_encode_fse_interleaved

  // 3. Zero-initialize to avoid garbage reads
  cudaMemset(h_desc.d_symbol_table, 0,
             (max_symbol + 1) * sizeof(FSEEncodeTable::FSEEncodeSymbol));
  cudaMemset(h_desc.d_symbol_first_state, 0, (max_symbol + 1) * sizeof(u16));

  // 4. Copy descriptor to the device table slot
  cudaMemcpy(d_table_slot, &h_desc, sizeof(FSEEncodeTable),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // 5. Upload normalized counts as u32 array
  std::vector<u32> h_norm_u32(max_symbol + 1);
  for (u32 i = 0; i <= max_symbol; ++i)
    h_norm_u32[i] = (u32)counts[i];

  u32 *d_norm;
  cudaMalloc(&d_norm, (max_symbol + 1) * sizeof(u32));
  cudaMemcpy(d_norm, h_norm_u32.data(), (max_symbol + 1) * sizeof(u32),
             cudaMemcpyHostToDevice);

  // 6. Build the CTable via the GPU kernel
  FSE_buildCTable_Device(d_norm, max_symbol, table_log, d_table_slot, nullptr,
                         0, 0);
  cudaDeviceSynchronize();

  // 7. Copy the populated descriptor back to host for reference
  cudaMemcpy(&gpu_table, d_table_slot, sizeof(FSEEncodeTable),
             cudaMemcpyDeviceToHost);

  cudaFree(d_norm);
}

void test_GPU_Encoding() {
  std::cout << "[TEST] GPU Encoding Verification" << std::endl;

  // 1. Setup Table (Uniform log 2)
  std::vector<short> counts = {1, 1, 1, 1};
  unsigned table_log = 2;
  std::vector<cuda_zstd::fse::FSE_CTable_Entry> ref_table;
  cuda_zstd::fse::build_fse_ctable_reference(ref_table, counts, table_log);

  // 2. Reference Encoding: Interleaved 3 Streams
  std::vector<u8> ref_bitstream;
  {
    // LL: C, B, A
    // OF: A, A, A
    // ML: A, A, A
    u32 symC = 2;
    u32 symB = 1;
    u32 symA = 0;

    // Init States (From Last Symbol, Index 2)
    // LL Init from C. OF Init from A. ML Init from A.
    u32 stateLL = ref_table[symC].nextState;
    u32 stateOF = ref_table[symA].nextState;
    u32 stateML = ref_table[symA].nextState;

    u64 bitContainer = 0;
    u32 bitCount = 0;

    // Loop N-2 down to 0 (Indices 1, 0)
    // Order: OF, ML, LL.

    // Iteration 1 (LL: C->B, OF: A->A, ML: A->A)
    {
      // OF (A->A)
      cuda_zstd::fse::fse_encode_step(stateOF, symA, symA, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
      // ML (A->A)
      cuda_zstd::fse::fse_encode_step(stateML, symA, symA, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
      // LL (C->B)
      cuda_zstd::fse::fse_encode_step(stateLL, symC, symB, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
    }

    // Iteration 2 (LL: B->A, OF: A->A, ML: A->A)
    {
      // OF (A->A)
      cuda_zstd::fse::fse_encode_step(stateOF, symA, symA, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
      // ML (A->A)
      cuda_zstd::fse::fse_encode_step(stateML, symA, symA, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
      // LL (B->A)
      cuda_zstd::fse::fse_encode_step(stateLL, symB, symA, ref_table,
                                      ref_bitstream, bitContainer, bitCount);
    }

    // Final Flush (Seq 0)
    // Order: ML, OF, LL

    u32 mask = (1 << table_log) - 1;
    u32 finalML = stateML & mask;
    u32 finalOF = stateOF & mask;
    u32 finalLL = stateLL & mask;

    bitContainer |= ((u64)finalML << bitCount);
    bitCount += table_log;
    bitContainer |= ((u64)finalOF << bitCount);
    bitCount += table_log;
    bitContainer |= ((u64)finalLL << bitCount);
    bitCount += table_log;

    // Sentinel
    bitContainer |= (1ULL << bitCount);
    bitCount++;

    while (bitCount > 0) {
      ref_bitstream.push_back((u8)bitContainer);
      bitContainer >>= 8;
      bitCount = (bitCount >= 8) ? bitCount - 8 : 0;
    }
  }

  std::cout << "  Ref Bitstream: ";
  for (auto b : ref_bitstream)
    printf("%02X ", b);
  std::cout << std::endl;

  // 3. GPU Run
  // Allocate array of 3 tables on device (LL, OF, ML — all same for this test)
  cuda_zstd::fse::FSEEncodeTable *d_gpu_tables;
  cudaMalloc(&d_gpu_tables, 3 * sizeof(cuda_zstd::fse::FSEEncodeTable));

  // Build all 3 tables using the proper CTable builder kernel
  cuda_zstd::fse::FSEEncodeTable h_gpu_table_struct;
  build_gpu_table(counts, table_log, h_gpu_table_struct, &d_gpu_tables[0]);

  // Copy slot 0 to slots 1 and 2 (same table for all 3 streams in this test)
  cudaMemcpy(&d_gpu_tables[1], &d_gpu_tables[0],
             sizeof(cuda_zstd::fse::FSEEncodeTable), cudaMemcpyDeviceToDevice);
  cudaMemcpy(&d_gpu_tables[2], &d_gpu_tables[0],
             sizeof(cuda_zstd::fse::FSEEncodeTable), cudaMemcpyDeviceToDevice);

  u8 *d_ll_codes;
  cudaMalloc(&d_ll_codes, 3);
  u8 h_codes_input[] = {0, 1, 2}; // A, B, C. Kernel uses [2] as Init.
  cudaMemcpy(d_ll_codes, h_codes_input, 3, cudaMemcpyHostToDevice);

  // Dummy pointers for others/extras
  u32 *d_extras;
  cudaMalloc(&d_extras, 3 * sizeof(u32));
  cudaMemset(d_extras, 0, 3 * sizeof(u32));
  u8 *d_bits;
  cudaMalloc(&d_bits, 3);
  cudaMemset(d_bits, 0, 3);
  u8 *d_codes_dummy;
  cudaMalloc(&d_codes_dummy, 3);
  cudaMemset(d_codes_dummy, 0, 3); // 0 = A

  // Output
  u8 *d_bitstream;
  cudaMalloc(&d_bitstream, 128);
  size_t *d_out_pos;
  cudaMalloc(&d_out_pos, sizeof(size_t));

  cuda_zstd::fse::launch_fse_encoding_kernel(
      d_ll_codes, d_extras, d_bits,    // LL
      d_codes_dummy, d_extras, d_bits, // OF (dummy)
      d_codes_dummy, d_extras, d_bits, // ML (dummy)
      3,                               // Num Symbols
      d_bitstream, d_out_pos, 128, d_gpu_tables, 0);

  // Read Output
  size_t out_pos;
  cudaMemcpy(&out_pos, d_out_pos, sizeof(size_t), cudaMemcpyDeviceToHost);
  std::vector<u8> gpu_bitstream(out_pos);
  cudaMemcpy(gpu_bitstream.data(), d_bitstream, out_pos,
             cudaMemcpyDeviceToHost);

  std::cout << "  GPU Bitstream: ";
  for (auto b : gpu_bitstream)
    printf("%02X ", b);
  std::cout << std::endl;

  // Compare
  bool match = (gpu_bitstream.size() == ref_bitstream.size());
  if (match) {
    for (size_t i = 0; i < gpu_bitstream.size(); ++i)
      if (gpu_bitstream[i] != ref_bitstream[i])
        match = false;
  }

  if (match)
    std::cout << "[PASS] GPU Matches Ref" << std::endl;
  else {
    std::cout << "[FAIL] Mismatch" << std::endl;
    std::cout << "Ref: ";
    for (auto b : ref_bitstream)
      printf("%02X ", b);
    printf("\n");
    exit(1);
  }
}

int main() {
  test_GPU_Encoding();
  return 0;
}
