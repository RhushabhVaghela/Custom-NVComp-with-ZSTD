/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 */

#include "../src/cuda_zstd_fse_reference.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_utils.h"
#include <cassert>
#include <iostream>
#include <vector>

// Handle types if not in global
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

void test_PredefinedTableStructure() {
  std::cout << "[TEST] Predefined Table Structure" << std::endl;

  std::vector<short> counts = {1, 1, 1, 1};
  unsigned table_log = 2;
  std::vector<cuda_zstd::fse::FSE_CTable_Entry> table;

  cuda_zstd::fse::build_fse_ctable_reference(table, counts, table_log);

  if (table.size() != 4) {
    std::cerr << "[FAIL] Table size " << table.size() << " != 4" << std::endl;
    exit(1);
  }

  unsigned expected_dNb = (2 << 16) - 4; // 131068

  for (size_t i = 0; i < table.size(); ++i) {
    std::cout << "  Sym " << i << ": dNbBits=" << table[i].deltaNbBits
              << ", dFindState=" << table[i].deltaFindState
              << ", nextState=" << table[i].nextState << std::endl;

    if (table[i].deltaNbBits != expected_dNb) {
      std::cerr << "[FAIL] Sym " << i << " dNbBits " << table[i].deltaNbBits
                << " != " << expected_dNb << std::endl;
      exit(1);
    }
  }

  std::cout << "[PASS] Predefined Table Structure" << std::endl;
}

void test_SingleStreamEncoding() {
  std::cout << "[TEST] Single Stream Encoding" << std::endl;
  // Setup simple table (log 2, 4 symbols uniform)
  std::vector<short> counts = {1, 1, 1, 1};
  unsigned table_log = 2;
  std::vector<cuda_zstd::fse::FSE_CTable_Entry> table;
  cuda_zstd::fse::build_fse_ctable_reference(table, counts, table_log);

  // Symbols to encode: 0, 1, 2, 3
  u32 symD = 3;
  u32 symC = 2;
  u32 symB = 1;
  u32 symA = 0;

  // 1. Initialize State from D
  u32 state = table[symD].nextState;

  std::vector<u8> bitstream;
  u64 bitContainer = 0;
  u32 bitCount = 0;

  // 2. Encode C, B, A
  cuda_zstd::fse::fse_encode_step(state, symC, symB, table, bitstream,
                                  bitContainer, bitCount);
  cuda_zstd::fse::fse_encode_step(state, symB, symA, table, bitstream,
                                  bitContainer, bitCount);
  cuda_zstd::fse::fse_encode_step(state, symA, symA, table, bitstream,
                                  bitContainer, bitCount);

  // Flush
  bitContainer |= (1ULL << bitCount);
  bitCount++;
  while (bitCount > 0) {
    bitstream.push_back((u8)bitContainer);
    bitContainer >>= 8;
    if (bitCount >= 8)
      bitCount -= 8;
    else
      bitCount = 0;
  }

  std::cout << "  Bitstream size: " << bitstream.size() << std::endl;
  for (auto b : bitstream)
    printf("  Byte: 0x%02X\n", b);

  if (bitstream.size() != 1) {
    std::cerr << "[FAIL] Expected 1 byte, got " << bitstream.size()
              << std::endl;
    // exit(1);
  }

  std::cout << "  Final State: " << state << std::endl;
  std::cout << "[PASS] Single Stream Encoding (Basic Run)" << std::endl;
}

int main() {
  test_PredefinedTableStructure();
  test_SingleStreamEncoding();
  return 0;
}
