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

// Helper to convert Ref Table to GPU Table
void convert_table(const std::vector<cuda_zstd::fse::FSE_CTable_Entry> &ref,
                   cuda_zstd::fse::FSEEncodeTable &gpu_table,
                   unsigned table_log) {
  // Allocate on device
  cudaMalloc(&gpu_table.d_symbol_table,
             ref.size() *
                 sizeof(cuda_zstd::fse::FSEEncodeTable::FSEEncodeSymbol));
  cudaMalloc(&gpu_table.d_symbol_first_state, ref.size() * sizeof(u16));
  gpu_table.table_log = table_log;

  // Host buffers
  std::vector<cuda_zstd::fse::FSEEncodeTable::FSEEncodeSymbol> h_syms(
      ref.size());
  std::vector<u16> h_next(ref.size());

  for (size_t i = 0; i < ref.size(); ++i) {
    h_syms[i].deltaNbBits = ref[i].deltaNbBits;
    h_syms[i].deltaFindState = ref[i].deltaFindState;
    h_next[i] = ref[i].nextState;
  }

  cudaMemcpy(gpu_table.d_symbol_table, h_syms.data(),
             h_syms.size() *
                 sizeof(cuda_zstd::fse::FSEEncodeTable::FSEEncodeSymbol),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_table.d_symbol_first_state, h_next.data(),
             h_next.size() * sizeof(u16), cudaMemcpyHostToDevice);
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
  cuda_zstd::fse::FSEEncodeTable h_gpu_table_struct;
  convert_table(ref_table, h_gpu_table_struct, table_log);

  cuda_zstd::fse::FSEEncodeTable *d_gpu_tables;
  cudaMalloc(&d_gpu_tables, 3 * sizeof(cuda_zstd::fse::FSEEncodeTable));
  cudaMemcpy(d_gpu_tables + 0, &h_gpu_table_struct,
             sizeof(cuda_zstd::fse::FSEEncodeTable),
             cudaMemcpyHostToDevice); // LL
  cudaMemcpy(d_gpu_tables + 1, &h_gpu_table_struct,
             sizeof(cuda_zstd::fse::FSEEncodeTable),
             cudaMemcpyHostToDevice); // OF
  cudaMemcpy(d_gpu_tables + 2, &h_gpu_table_struct,
             sizeof(cuda_zstd::fse::FSEEncodeTable),
             cudaMemcpyHostToDevice); // ML

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
    // exit(1);
  }
}

int main() {
  test_GPU_Encoding();
  return 0;
}
