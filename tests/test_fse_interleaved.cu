
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Helper to print bits
void print_bits(const byte_t *data, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (int b = 7; b >= 0; b--) {
      std::cout << ((data[i] >> b) & 1);
    }
    std::cout << " ";
  }
  std::cout << std::endl;
}

void test_fse_interleaved_simple() {
  std::cout << "Testing k_encode_fse_interleaved..." << std::endl;

  // 1. Setup Table
  u32 max_symbol = 1;
  u32 table_log = 5;
  u32 table_size = 1 << table_log; 

  // Counts: Sym0=16, Sym1=16. Prob 50% each.
  // Normalized: 16, 16.
  std::vector<u32> h_counts = {16, 16};

  u32 *d_counts;
  cudaMalloc(&d_counts, h_counts.size() * sizeof(u32));
  cudaMemcpy(d_counts, h_counts.data(), h_counts.size() * sizeof(u32),
             cudaMemcpyHostToDevice);

  FSEEncodeTable h_table_desc;
  h_table_desc.table_log = table_log;
  h_table_desc.max_symbol = max_symbol;
  h_table_desc.table_size = table_size;

  // Allocate device buffers for the table
  cudaMalloc(&h_table_desc.d_symbol_table,
             (max_symbol + 1) * sizeof(FSEEncodeTable::FSEEncodeSymbol));
  cudaMalloc(&h_table_desc.d_next_state, table_size * sizeof(u16));
  cudaMalloc(&h_table_desc.d_nbBits_table, table_size * sizeof(u8));
  cudaMalloc(&h_table_desc.d_symbol_first_state,
             (max_symbol + 1) * sizeof(u16));
  cudaMalloc(&h_table_desc.d_state_to_symbol, table_size * sizeof(u8));
  cudaMalloc(&h_table_desc.d_next_state_vals, table_size * sizeof(u16));
  cudaMalloc(&h_table_desc.d_next_state_vals, table_size * sizeof(u16));

  // We need 3 tables: LL, OF, ML. For simplicity, use same table for all.
  FSEEncodeTable *d_table_ptr;
  cudaMalloc(&d_table_ptr, 3 * sizeof(FSEEncodeTable));

  // Build ONE table
  cudaStream_t stream = 0;
  FSEEncodeTable *d_single_table;
  cudaMalloc(&d_single_table, sizeof(FSEEncodeTable));
  cudaMemcpy(d_single_table, &h_table_desc, sizeof(FSEEncodeTable),
             cudaMemcpyHostToDevice); // Copy desc

  // We fixed the launcher to take `d_table` (pointer to struct on device).
  // Wait, `FSE_buildCTable_Device` takes `FSEEncodeTable* d_table`.
  // And `k_build` writes to `d_table->d_symbol_table`.
  // So `d_table` must be accessible on GPU.
  // And its members (d_symbol_table) must be valid GPU pointers.
  // Yes.

  Status s = FSE_buildCTable_Device(d_counts, max_symbol, table_log,
                                    d_single_table, nullptr, 0, stream);
  assert(s == Status::SUCCESS);

  // Copy the same table descriptor to all 3 slots [LL, OF, ML]
  // Actually, `k_encode` takes `const FSEEncodeTable* table`. It expects an
  // ARRAY of 3 tables. So we need to copy `h_table_desc` into 3 slots on host,
  // then copy to device.

  FSEEncodeTable h_tables[3];
  h_tables[0] = h_table_desc; // Share same buffers
  h_tables[1] = h_table_desc;
  h_tables[2] = h_table_desc;

  cudaMemcpy(d_table_ptr, h_tables, 3 * sizeof(FSEEncodeTable),
             cudaMemcpyHostToDevice);

  // 2. Setup Sequences
  // 1 Sequence: LL=0, OF=0, ML=0.
  // Sym 0 has prob 50%. bit cost 1.
  // State transition should be simple.

  u32 num_seq = 1;
  u8 *d_ll, *d_of, *d_ml;
  u32 *d_ll_extras, *d_of_extras, *d_ml_extras;
  u8 *d_ll_bits, *d_of_bits, *d_ml_bits;

  cudaMalloc(&d_ll, sizeof(u8));
  cudaMalloc(&d_of, sizeof(u8));
  cudaMalloc(&d_ml, sizeof(u8));

  cudaMalloc(&d_ll_extras, sizeof(u32));
  cudaMalloc(&d_of_extras, sizeof(u32));
  cudaMalloc(&d_ml_extras, sizeof(u32));
  cudaMemset(d_ll_extras, 0, sizeof(u32));
  cudaMemset(d_of_extras, 0, sizeof(u32));
  cudaMemset(d_ml_extras, 0, sizeof(u32));

  cudaMalloc(&d_ll_bits, sizeof(u8));
  cudaMalloc(&d_of_bits, sizeof(u8));
  cudaMalloc(&d_ml_bits, sizeof(u8));
  cudaMemset(d_ll_bits, 0, sizeof(u8));
  cudaMemset(d_of_bits, 0, sizeof(u8));
  cudaMemset(d_ml_bits, 0, sizeof(u8));

  u8 zero = 0;
  cudaMemcpy(d_ll, &zero, sizeof(u8), cudaMemcpyHostToDevice);
  cudaMemcpy(d_of, &zero, sizeof(u8), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ml, &zero, sizeof(u8), cudaMemcpyHostToDevice);

  // 3. Setup Bitstream
  size_t capacity = 32;
  byte_t *d_bitstream;
  cudaMalloc(&d_bitstream, capacity);
  cudaMemset(d_bitstream, 0, capacity);

  size_t *d_pos;
  cudaMalloc(&d_pos, sizeof(size_t));

  // 4. Launch Encoding
  // 4. Launch Encoding
  Status launchStatus = launch_fse_encoding_kernel(
      d_ll, d_ll_extras, d_ll_bits, d_of, d_of_extras, d_of_bits, d_ml,
      d_ml_extras, d_ml_bits, num_seq, d_bitstream, d_pos, capacity,
      d_table_ptr, stream);
  assert(launchStatus == Status::SUCCESS);

  cudaDeviceSynchronize();

  // 5. Verify Output
  size_t h_pos;
  cudaMemcpy(&h_pos, d_pos, sizeof(size_t), cudaMemcpyDeviceToHost);

  std::cout << "Output Size: " << h_pos << std::endl;

  std::vector<byte_t> h_out(capacity);
  cudaMemcpy(h_out.data(), d_bitstream, capacity, cudaMemcpyDeviceToHost);

  print_bits(h_out.data(), h_pos + 1); // Print a bit more

  // Expected:
  // 1 Seq.
  // Init States (LL, OF, ML) using Last Sym (0,0,0).
  // Sym 0 -> Prob 1/2.
  // TableLog=1.
  // State Range for 0: [2, 3]? No.
  // Freq=1. Norm=1.
  // deltaNbBits = (1<<16) - 2.
  // deltaFindState = -1.
  //
  // Init State = cumFreq[0] = 0?
  // Wait. In `k_build_ctable`:
  // normalized_counters = {1, 1}.
  // s=0: bits=1. minStatePlus=2. deltaNbBits = (1<<16)-2.
  // s=1: bits=1. minStatePlus=2. ...
  // Prefix sum:
  // [0]=0, [1]=1, [2]=2.
  // s=0: deltaFindState = cum[0] - 1 = -1.
  // s=1: deltaFindState = cum[1] - 1 = 0.
  //
  // So State Init:
  // LL=0 -> stateLL = deltaFindState[0] + 1 = 0.
  // OF=0 -> stateOF = 0.
  // ML=0 -> stateML = 0.
  //
  // Loop: 1 seq. idx=0 used for Init. loop num_seq-2 (1-2 = -1). Loop doesn't
  // run. Correct.
  //
  // Final Flush:
  // Write ML, OF, LL states.
  // State=0. TableLog=1.
  // Write 1 bit '0' for ML.
  // Write 1 bit '0' for OF.
  // Write 1 bit '0' for LL.
  // Total 3 bits "000".
  // Sentinel: '1'.
  // Total bits: "0001".
  // 4 bits.
  // Should be 1 byte.
  // Value: 0b00001000? Or 0b0001?
  // Logic: `bitContainer |= (val << bitCount)`.
  // bitCount 0 -> ML(0) -> Cont ...0
  // bitCount 1 -> OF(0) -> Cont ...00
  // bitCount 2 -> LL(0) -> Cont ...000
  // bitCount 3 -> Sentinel(1) -> Cont ...1000 (0x8)
  //
  // Byte: 0x08.
  // Output Size: 1.

  if (h_pos == 1 && h_out[0] == 0x08) {
    std::cout << "SUCCESS: Output matches expectation (0x08)." << std::endl;
  } else {
    std::cout << "FAILURE: Got size " << h_pos << " byte 0x" << std::hex
              << (int)h_out[0] << std::dec << std::endl;
    exit(1);
  }

  // Cleanup
  cudaFree(d_counts);
  cudaFree(h_table_desc.d_symbol_table);
  cudaFree(d_table_ptr);
  cudaFree(d_single_table);
  cudaFree(d_ll);
  cudaFree(d_of);
  cudaFree(d_ml);
  cudaFree(d_ll_extras);
  cudaFree(d_of_extras);
  cudaFree(d_ml_extras);
  cudaFree(d_ll_bits);
  cudaFree(d_of_bits);
  cudaFree(d_ml_bits);
  cudaFree(d_bitstream);
  cudaFree(d_pos);
}

int main() {
  test_fse_interleaved_simple();
  return 0;
}
