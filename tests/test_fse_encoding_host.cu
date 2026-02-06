
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
  // Symbols: 0(1), 1(1). Total 2. TableLog 1.
  u16 norm[] = {1, 1};
  u32 max_sym = 1;
  u32 table_size = 2; // log 1

  FSEEncodeTable table;
  // Mock device pointers with host pointers for testing (assuming BuildCTable
  // supports it or we mock copy) Wait, FSE_buildCTable_Host calls cudaMemcpy.
  // This test uses Unified Memory and requires a GPU.

  // Allocate Unified Memory or Device Memory
  cudaMallocManaged(&table.d_symbol_table,
                    (max_sym + 1) * sizeof(FSEEncodeTable::FSEEncodeSymbol));
  cudaMallocManaged(&table.d_next_state, table_size * sizeof(u16));

  Status s = FSE_buildCTable_Host(norm, max_sym, table_size, table);
  if (s != Status::SUCCESS) {
    std::cerr << "Build Failed" << std::endl;
    exit(1);
  }

  cudaDeviceSynchronize();

  // Verify the CTable was populated (basic sanity checks)
  // For a {1,1} distribution with log=1, both symbols should have entries.
  // d_symbol_table should be non-null and d_next_state should be non-null
  // (we already verified build succeeded above).
  // Verify that symbol_table entries have sensible values:
  // Each symbol with prob=1 in a tableLog=1 table should get exactly 1 state.
  assert(table.d_symbol_table != nullptr && "Symbol table must be allocated");
  assert(table.d_next_state != nullptr && "Next state table must be allocated");

  // Read back and verify symbol entries are initialized (not all zero)
  FSEEncodeTable::FSEEncodeSymbol h_syms[2];
  cudaMemcpy(h_syms, table.d_symbol_table, 2 * sizeof(h_syms[0]),
             cudaMemcpyDeviceToHost);
  // For prob=1, deltaNbBits should be non-zero (encoding requires bits)
  bool sym0_ok = (h_syms[0].deltaNbBits != 0 || h_syms[0].deltaFindState != 0);
  bool sym1_ok = (h_syms[1].deltaNbBits != 0 || h_syms[1].deltaFindState != 0);
  if (!sym0_ok || !sym1_ok) {
    std::cerr << "FAIL: CTable symbol entries appear uninitialized" << std::endl;
    cudaFree(table.d_symbol_table);
    cudaFree(table.d_next_state);
    exit(1);
  }

  cudaFree(table.d_symbol_table);
  cudaFree(table.d_next_state);
}

int main() {
  test_build_ctable_simple();
  std::cout << "Test Host Encoding: PASS" << std::endl;
  return 0;
}
