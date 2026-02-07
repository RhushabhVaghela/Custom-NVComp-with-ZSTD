/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Test: Verify GPU FSE encoding kernel produces correct output by comparing
 * against a CPU-side reference that uses the SAME algorithm (ZSTD standard
 * FSE_encodeSymbol with d_next_state table lookup).
 *
 * The GPU kernel k_encode_fse_interleaved implements:
 *   nbBitsOut = (state + symbolTT.deltaNbBits) >> 16;
 *   write_bits(state, nbBitsOut);
 *   state = stateTable[(state >> nbBitsOut) + symbolTT.deltaFindState];
 * which matches zstd/lib/compress/fse_compress.c FSE_encodeSymbol exactly.
 */

#include "cuda_zstd_fse.h"
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_utils.h"
#include <cassert>
#include <cstring>
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

// Host-side structures to hold GPU table data read back for CPU reference encoding
struct HostFSETables {
  std::vector<cuda_zstd::fse::FSEEncodeTable::FSEEncodeSymbol> symbol_table;
  std::vector<u16> next_state;
  std::vector<u16> symbol_first_state;
  u32 table_log;
  u32 table_size;
  u32 max_symbol;
};

// Read GPU table data back to host for CPU-side reference encoding
void read_gpu_table_to_host(const cuda_zstd::fse::FSEEncodeTable &gpu_desc,
                            HostFSETables &h) {
  h.table_log = gpu_desc.table_log;
  h.table_size = gpu_desc.table_size;
  h.max_symbol = gpu_desc.max_symbol;

  h.symbol_table.resize(h.max_symbol + 1);
  cudaMemcpy(h.symbol_table.data(), gpu_desc.d_symbol_table,
             (h.max_symbol + 1) * sizeof(h.symbol_table[0]),
             cudaMemcpyDeviceToHost);

  h.next_state.resize(h.table_size);
  cudaMemcpy(h.next_state.data(), gpu_desc.d_next_state,
             h.table_size * sizeof(u16), cudaMemcpyDeviceToHost);

  h.symbol_first_state.resize(h.max_symbol + 1);
  cudaMemcpy(h.symbol_first_state.data(), gpu_desc.d_symbol_first_state,
             (h.max_symbol + 1) * sizeof(u16), cudaMemcpyDeviceToHost);
}

// CPU-side FSE_encodeSymbol matching ZSTD standard exactly:
//   nbBitsOut = (state + symbolTT.deltaNbBits) >> 16;
//   BIT_addBits(state, nbBitsOut);
//   state = stateTable[(state >> nbBitsOut) + symbolTT.deltaFindState];
void cpu_fse_encode_symbol(u32 &state, u8 code, const HostFSETables &tbl,
                           u64 &bitContainer, u32 &bitCount,
                           std::vector<u8> &bitstream) {
  auto &sym = tbl.symbol_table[code];
  u32 nbBitsOut = (state + sym.deltaNbBits) >> 16;
  u64 val = state & ((1ULL << nbBitsOut) - 1);
  bitContainer |= (val << bitCount);
  bitCount += nbBitsOut;
  while (bitCount >= 8) {
    bitstream.push_back((u8)(bitContainer & 0xFF));
    bitContainer >>= 8;
    bitCount -= 8;
  }
  state = tbl.next_state[(state >> nbBitsOut) + sym.deltaFindState];
}

void test_GPU_Encoding() {
  std::cout << "[TEST] GPU Encoding Verification" << std::endl;

  // 1. Setup Table (Uniform log 2)
  std::vector<short> counts = {1, 1, 1, 1};
  unsigned table_log = 2;

  // 2. Build GPU tables
  cuda_zstd::fse::FSEEncodeTable *d_gpu_tables;
  cudaMalloc(&d_gpu_tables, 3 * sizeof(cuda_zstd::fse::FSEEncodeTable));

  cuda_zstd::fse::FSEEncodeTable h_gpu_table_struct;
  build_gpu_table(counts, table_log, h_gpu_table_struct, &d_gpu_tables[0]);

  // Copy slot 0 to slots 1 and 2 (same table for all 3 streams in this test)
  cudaMemcpy(&d_gpu_tables[1], &d_gpu_tables[0],
             sizeof(cuda_zstd::fse::FSEEncodeTable), cudaMemcpyDeviceToDevice);
  cudaMemcpy(&d_gpu_tables[2], &d_gpu_tables[0],
             sizeof(cuda_zstd::fse::FSEEncodeTable), cudaMemcpyDeviceToDevice);

  // 3. Read GPU table back to host for CPU reference encoding
  HostFSETables h_tbl;
  read_gpu_table_to_host(h_gpu_table_struct, h_tbl);

  // 4. CPU Reference Encoding (mirrors k_encode_fse_interleaved exactly)
  //    LL: [A, B, C]   (codes: 0, 1, 2)
  //    OF: [A, A, A]   (codes: 0, 0, 0)
  //    ML: [A, A, A]   (codes: 0, 0, 0)
  //    extras/bits: all zero
  u8 ll_codes[] = {0, 1, 2};
  u8 of_codes[] = {0, 0, 0};
  u8 ml_codes[] = {0, 0, 0};
  u32 num_symbols = 3;
  u32 last = num_symbols - 1;

  std::vector<u8> ref_bitstream;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  // Step 1: Init states from last sequence
  u32 stateLL = h_tbl.symbol_first_state[ll_codes[last]];
  u32 stateOF = h_tbl.symbol_first_state[of_codes[last]];
  u32 stateML = h_tbl.symbol_first_state[ml_codes[last]];

  // Step 2: Write extras for last seq (all zero → nothing)

  // Step 3: Loop from N-2 down to 0, order: OF, ML, LL transitions then extras
  for (u32 i = num_symbols - 2; ; i--) {
    cpu_fse_encode_symbol(stateOF, of_codes[i], h_tbl, bitContainer, bitCount, ref_bitstream);
    cpu_fse_encode_symbol(stateML, ml_codes[i], h_tbl, bitContainer, bitCount, ref_bitstream);
    cpu_fse_encode_symbol(stateLL, ll_codes[i], h_tbl, bitContainer, bitCount, ref_bitstream);
    // extras all zero → nothing to write
    if (i == 0) break;
  }

  // Step 4: Flush final states (ML, OF, LL)
  auto write_bits = [&](u64 val, u32 nbBits) {
    if (nbBits == 0) return;
    val &= ((1ULL << nbBits) - 1);
    bitContainer |= (val << bitCount);
    bitCount += nbBits;
    while (bitCount >= 8) {
      ref_bitstream.push_back((u8)(bitContainer & 0xFF));
      bitContainer >>= 8;
      bitCount -= 8;
    }
  };
  write_bits(stateML, h_tbl.table_log);
  write_bits(stateOF, h_tbl.table_log);
  write_bits(stateLL, h_tbl.table_log);

  // Step 5: Sentinel bit
  write_bits(1, 1);

  // Step 6: Final flush
  if (bitCount > 0) {
    ref_bitstream.push_back((u8)(bitContainer & 0xFF));
  }

  std::cout << "  Ref Bitstream: ";
  for (auto b : ref_bitstream)
    printf("%02X ", b);
  std::cout << std::endl;

  // 5. GPU Run
  u8 *d_ll_codes;
  cudaMalloc(&d_ll_codes, 3);
  cudaMemcpy(d_ll_codes, ll_codes, 3, cudaMemcpyHostToDevice);

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
  cudaDeviceSynchronize();

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

  // 6. Compare
  bool match = (gpu_bitstream.size() == ref_bitstream.size());
  if (match) {
    for (size_t i = 0; i < gpu_bitstream.size(); ++i)
      if (gpu_bitstream[i] != ref_bitstream[i])
        match = false;
  }

  if (match) {
    std::cout << "[PASS] GPU Matches Ref" << std::endl;
  } else {
    std::cout << "[FAIL] Mismatch" << std::endl;
    std::cout << "  Ref (" << ref_bitstream.size() << " bytes): ";
    for (auto b : ref_bitstream)
      printf("%02X ", b);
    printf("\n");
    std::cout << "  GPU (" << gpu_bitstream.size() << " bytes): ";
    for (auto b : gpu_bitstream)
      printf("%02X ", b);
    printf("\n");
    exit(1);
  }

  // Cleanup
  cudaFree(d_ll_codes);
  cudaFree(d_extras);
  cudaFree(d_bits);
  cudaFree(d_codes_dummy);
  cudaFree(d_bitstream);
  cudaFree(d_out_pos);
  // Note: d_gpu_tables and its sub-arrays are leaked for simplicity in test
}

int main() {
  test_GPU_Encoding();
  return 0;
}
