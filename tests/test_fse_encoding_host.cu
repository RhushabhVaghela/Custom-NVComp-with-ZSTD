
#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <iostream>
#include <vector>

// Test
void test_build_ctable_simple() {
  using namespace cuda_zstd;
  using namespace cuda_zstd::fse;

  // Normalized counts for a simple distribution
  // Symbols: 0(1), 1(1). Total 2. TableLog 1 => table_size 2.
  u16 norm[] = {1, 1};
  u32 max_sym = 1;
  u32 table_log = 1; // log2(table_size) = log2(2) = 1

  FSEEncodeTable table;
  memset(&table, 0, sizeof(table));

  // FSE_buildCTable_Host allocates its own arrays via new[]
  Status s = FSE_buildCTable_Host(norm, max_sym, table_log, table);
  if (s != Status::SUCCESS) {
    std::cerr << "Build Failed" << std::endl;
    exit(1);
  }

  // Verify the CTable was populated (basic sanity checks)
  // For a {1,1} distribution with log=1, both symbols should have entries.
  assert(table.d_symbol_table != nullptr && "Symbol table must be allocated");
  assert(table.d_next_state != nullptr && "Next state table must be allocated");

  // Read back and verify symbol entries are initialized (not all zero)
  // FSE_buildCTable_Host uses new[], so d_symbol_table is host memory
  FSEEncodeTable::FSEEncodeSymbol *h_syms = table.d_symbol_table;
  // For prob=1, deltaNbBits should be non-zero (encoding requires bits)
  bool sym0_ok = (h_syms[0].deltaNbBits != 0 || h_syms[0].deltaFindState != 0);
  bool sym1_ok = (h_syms[1].deltaNbBits != 0 || h_syms[1].deltaFindState != 0);
  if (!sym0_ok || !sym1_ok) {
    std::cerr << "FAIL: CTable symbol entries appear uninitialized" << std::endl;
    std::cerr << "  sym0: deltaNbBits=" << h_syms[0].deltaNbBits
              << " deltaFindState=" << h_syms[0].deltaFindState << std::endl;
    std::cerr << "  sym1: deltaNbBits=" << h_syms[1].deltaNbBits
              << " deltaFindState=" << h_syms[1].deltaFindState << std::endl;
    delete[] table.d_symbol_table;
    delete[] table.d_next_state;
    delete[] table.d_state_to_symbol;
    delete[] table.d_nbBits_table;
    delete[] table.d_next_state_vals;
    exit(1);
  }

  delete[] table.d_symbol_table;
  delete[] table.d_next_state;
  delete[] table.d_state_to_symbol;
  delete[] table.d_nbBits_table;
  delete[] table.d_next_state_vals;
}

int main() {
  test_build_ctable_simple();
  std::cout << "Test Host Encoding: PASS" << std::endl;
  return 0;
}
