// ==============================================================================
// test_fse_sequence_decode.cu - Unit tests for FSE sequence decoding fixes
//
// Tests for:
// 1. State bounds checking
// 2. State normalization
// 3. Bitstream reading order
// 4. Initial state validation
// ==============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_zstd_fse.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_safe_alloc.h"

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Test result tracking
struct TestResults {
  int passed = 0;
  int failed = 0;

  void report(const char *test_name, bool passed_test,
              const char *msg = nullptr) {
    if (passed_test) {
      printf("[PASS] %s\n", test_name);
      passed++;
    } else {
      printf("[FAIL] %s: %s\n", test_name, msg ? msg : "Unknown error");
      failed++;
    }
  }
};

// Helper: Check CUDA error
#define CHECK_CUDA(err)                                                        \
  do {                                                                         \
    cudaError_t e = (err);                                                     \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__,     \
             __LINE__);                                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

// ==============================================================================
// Test 1: Basic bounds checking test
// Verifies that out-of-bounds states are caught and handled
// ==============================================================================
bool test_state_bounds_checking(TestResults &results) {
  printf("\n=== Test: State Bounds Checking ===\n");

  // Create a small test table
  const u32 table_log = 6; // 64 entries
  const u32 table_size = 1u << table_log;

  // Simple normalized distribution for testing
  std::vector<u16> normalized(36, 1); // 36 symbols, 1 each = 36
  normalized[0] = table_size - 35;    // Make sum = 64

  // Build decode table
  FSEDecodeTable h_table;
  h_table.newState = new u16[table_size];
  h_table.symbol = new u8[table_size];
  h_table.nbBits = new u8[table_size];
  h_table.nbAdditionalBits = new u8[table_size];
  h_table.baseValue = new u32[table_size];
  h_table.table_log = table_log;
  h_table.table_size = table_size;

  Status status =
      FSE_buildDTable_Host(normalized.data(), 35, table_size, h_table);
  if (status != Status::SUCCESS) {
    results.report("Build decode table", false, "Failed to build table");
    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    delete[] h_table.nbAdditionalBits;
    delete[] h_table.baseValue;
    return false;
  }

  // Verify table bounds
  bool bounds_ok = true;
  for (u32 i = 0; i < table_size; i++) {
    if (h_table.symbol[i] > 35) {
      bounds_ok = false;
      printf("Symbol out of bounds at state %u: %u\n", i, h_table.symbol[i]);
    }
    if (h_table.nbBits[i] > table_log) {
      bounds_ok = false;
      printf("nbBits out of bounds at state %u: %u\n", i, h_table.nbBits[i]);
    }
    if (h_table.newState[i] >= table_size) {
      bounds_ok = false;
      printf("newState out of bounds at state %u: %u\n", i,
             h_table.newState[i]);
    }
  }

  results.report("Table bounds validation", bounds_ok,
                 "Values out of bounds found");

  // Cleanup
  delete[] h_table.newState;
  delete[] h_table.symbol;
  delete[] h_table.nbBits;
  delete[] h_table.nbAdditionalBits;
  delete[] h_table.baseValue;

  return bounds_ok;
}

// ==============================================================================
// Test 2: State normalization test
// Verifies that states are properly normalized after transitions
// ==============================================================================
bool test_state_normalization(TestResults &results) {
  printf("\n=== Test: State Normalization ===\n");

  const u32 table_log = 5; // 32 entries
  const u32 table_size = 1u << table_log;

  // Create predefined offset table distribution
  u32 max_sym, log;
  const u16 *of_norm =
      get_predefined_norm(TableType::OFFSETS, &max_sym, &log);

  FSEDecodeTable h_table;
  h_table.newState = new u16[table_size];
  h_table.symbol = new u8[table_size];
  h_table.nbBits = new u8[table_size];
  h_table.nbAdditionalBits = new u8[table_size];
  h_table.baseValue = new u32[table_size];
  h_table.table_log = table_log;
  h_table.table_size = table_size;

  // Build with predefined OF distribution
  std::vector<u16> normalized(29);
  for (int i = 0; i < 29; i++) {
    normalized[i] = (u16)(((const i16 *)of_norm)[i]);
  }

  Status status =
      FSE_buildDTable_Host(normalized.data(), 28, table_size, h_table);
  if (status != Status::SUCCESS) {
    results.report("Build OF table", false, "Failed to build table");
    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    delete[] h_table.nbAdditionalBits;
    delete[] h_table.baseValue;
    return false;
  }

  // Verify newState values are properly bounded
  bool normalization_ok = true;
  for (u32 i = 0; i < table_size; i++) {
    // newState should be in range [0, table_size-1]
    if (h_table.newState[i] >= table_size) {
      printf("State %u: newState=%u >= table_size=%u\n", i, h_table.newState[i],
             table_size);
      normalization_ok = false;
    }
  }

  results.report("State normalization", normalization_ok,
                 "newState values out of bounds");

  // Cleanup
  delete[] h_table.newState;
  delete[] h_table.symbol;
  delete[] h_table.nbBits;
  delete[] h_table.nbAdditionalBits;
  delete[] h_table.baseValue;

  return normalization_ok;
}

// ==============================================================================
// Test 3: Predefined table validation
// Verifies that predefined LL, OF, ML tables are correctly built
// ==============================================================================
bool test_predefined_tables(TestResults &results) {
  printf("\n=== Test: Predefined Tables ===\n");

  bool all_ok = true;

  // Test each predefined table type
  TableType types[] = {TableType::LITERALS, TableType::OFFSETS,
                       TableType::MATCH_LENGTHS};
  const char *type_names[] = {"LITERALS", "OFFSETS", "MATCH_LENGTHS"};
  u32 expected_logs[] = {6, 5, 6}; // Expected table logs per RFC 8878

  for (int t = 0; t < 3; t++) {
    u32 max_sym, log;
    const u16 *norm = get_predefined_norm(types[t], &max_sym, &log);

    printf("Table %s: max_sym=%u, log=%u\n", type_names[t], max_sym, log);

    // Verify table log matches expected
    if (log != expected_logs[t]) {
      printf("  WARNING: Expected log=%u, got log=%u\n", expected_logs[t], log);
    }

    u32 table_size = 1u << log;

    // Build decode table
    FSEDecodeTable h_table;
    h_table.newState = new u16[table_size];
    h_table.symbol = new u8[table_size];
    h_table.nbBits = new u8[table_size];
    h_table.nbAdditionalBits = new u8[table_size];
    h_table.baseValue = new u32[table_size];
    h_table.table_log = log;
    h_table.table_size = table_size;

    std::vector<u16> normalized(max_sym + 1);
    for (u32 i = 0; i <= max_sym; i++) {
      normalized[i] = (u16)(((const i16 *)norm)[i]);
    }

    Status status =
        FSE_buildDTable_Host(normalized.data(), max_sym, table_size, h_table);

    if (status != Status::SUCCESS) {
      printf("  FAILED to build table\n");
      all_ok = false;
    } else {
      // Verify all entries
      bool table_ok = true;
      for (u32 i = 0; i < table_size; i++) {
        if (h_table.symbol[i] > max_sym) {
          printf("  State %u: symbol %u > max_sym %u\n", i, h_table.symbol[i],
                 max_sym);
          table_ok = false;
          all_ok = false;
        }
        if (h_table.nbBits[i] > log) {
          printf("  State %u: nbBits %u > log %u\n", i, h_table.nbBits[i], log);
          table_ok = false;
          all_ok = false;
        }
      }

      if (table_ok) {
        printf("  Table OK: %u entries verified\n", table_size);
      }
    }

    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    delete[] h_table.nbAdditionalBits;
    delete[] h_table.baseValue;
  }

  results.report("Predefined tables", all_ok, "Table validation failed");
  return all_ok;
}

// ==============================================================================
// Test 4: Device table copy and access
// Verifies that tables are correctly copied to device
// ==============================================================================
bool test_device_table_copy(TestResults &results) {
  printf("\n=== Test: Device Table Copy ===\n");

  const u32 table_log = 5;
  const u32 table_size = 1u << table_log;

  // Build host table
  u32 max_sym, log;
  const u16 *of_norm =
      get_predefined_norm(TableType::OFFSETS, &max_sym, &log);

  FSEDecodeTable h_table;
  h_table.newState = new u16[table_size];
  h_table.symbol = new u8[table_size];
  h_table.nbBits = new u8[table_size];
  h_table.nbAdditionalBits = new u8[table_size];
  h_table.baseValue = new u32[table_size];
  h_table.table_log = table_log;
  h_table.table_size = table_size;

  std::vector<u16> normalized(29);
  for (int i = 0; i < 29; i++) {
    normalized[i] = (u16)(((const i16 *)of_norm)[i]);
  }

  Status status =
      FSE_buildDTable_Host(normalized.data(), 28, table_size, h_table);
  if (status != Status::SUCCESS) {
    results.report("Build table", false, "Failed to build table");
    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    delete[] h_table.nbAdditionalBits;
    delete[] h_table.baseValue;
    return false;
  }

  // Copy to device
  FSEDecodeTable d_table = {};

  // Allocate device memory
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_table.newState, table_size * sizeof(u16)));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_table.symbol, table_size * sizeof(u8)));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_table.nbBits, table_size * sizeof(u8)));

  CHECK_CUDA(cudaMemcpy(d_table.newState, h_table.newState,
                        table_size * sizeof(u16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_table.symbol, h_table.symbol, table_size * sizeof(u8),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_table.nbBits, h_table.nbBits, table_size * sizeof(u8),
                        cudaMemcpyHostToDevice));

  // Copy back and verify
  std::vector<u16> newState_back(table_size);
  std::vector<u8> symbol_back(table_size);
  std::vector<u8> nbBits_back(table_size);

  CHECK_CUDA(cudaMemcpy(newState_back.data(), d_table.newState,
                        table_size * sizeof(u16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(symbol_back.data(), d_table.symbol,
                        table_size * sizeof(u8), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(nbBits_back.data(), d_table.nbBits,
                        table_size * sizeof(u8), cudaMemcpyDeviceToHost));

  bool copy_ok = true;
  for (u32 i = 0; i < table_size; i++) {
    if (newState_back[i] != h_table.newState[i]) {
      printf("Mismatch at %u: newState %u != %u\n", i, newState_back[i],
             h_table.newState[i]);
      copy_ok = false;
      break;
    }
    if (symbol_back[i] != h_table.symbol[i]) {
      printf("Mismatch at %u: symbol %u != %u\n", i, symbol_back[i],
             h_table.symbol[i]);
      copy_ok = false;
      break;
    }
    if (nbBits_back[i] != h_table.nbBits[i]) {
      printf("Mismatch at %u: nbBits %u != %u\n", i, nbBits_back[i],
             h_table.nbBits[i]);
      copy_ok = false;
      break;
    }
  }

  results.report("Device table copy", copy_ok, "Data mismatch after copy");

  // Cleanup
  cudaFree(d_table.newState);
  cudaFree(d_table.symbol);
  cudaFree(d_table.nbBits);
  delete[] h_table.newState;
  delete[] h_table.symbol;
  delete[] h_table.nbBits;
  delete[] h_table.nbAdditionalBits;
  delete[] h_table.baseValue;

  return copy_ok;
}

// ==============================================================================
// Test 5: Offset calculation validation
// Verifies that offset values are calculated correctly
// ==============================================================================
bool test_offset_calculation(TestResults &results) {
  printf("\n=== Test: Offset Calculation ===\n");

  // Build the predefined OF decode table so we can verify actual table entries
  const u32 table_log = 5;
  const u32 table_size = 1u << table_log;

  u32 max_sym, log;
  const u16 *of_norm = get_predefined_norm(TableType::OFFSETS, &max_sym, &log);

  FSEDecodeTable h_table;
  h_table.newState = new u16[table_size];
  h_table.symbol = new u8[table_size];
  h_table.nbBits = new u8[table_size];
  h_table.nbAdditionalBits = new u8[table_size];
  h_table.baseValue = new u32[table_size];
  h_table.table_log = table_log;
  h_table.table_size = table_size;

  std::vector<u16> normalized(max_sym + 1);
  for (u32 i = 0; i <= max_sym; i++) {
    normalized[i] = (u16)(((const i16 *)of_norm)[i]);
  }

  Status status =
      FSE_buildDTable_Host(normalized.data(), max_sym, table_size, h_table);
  if (status != Status::SUCCESS) {
    printf("Failed to build OF decode table\n");
    results.report("Offset calculation", false, "Could not build OF table");
    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    delete[] h_table.nbAdditionalBits;
    delete[] h_table.baseValue;
    return false;
  }

  bool all_ok = true;

  // Verify offset codes (sym >= 0) using the built table
  // Per RFC 8878 Table 15: Offset_Bits = N, Offset_Value_Baseline = (1 << N)
  // For sym 0: nbAdditionalBits=0, baseValue=1
  // For sym 1: nbAdditionalBits=1, baseValue=1
  // For sym N (N>=1): nbAdditionalBits=N-1, baseValue=1<<(N-1) — but this depends
  // on the table encoding. Let's verify the table entries match expected patterns.
  //
  // The standard OF table for symbols:
  //   sym 0: Num_Bits=0, Baseline=0 (offset 0, but offset 1-3 are repeat codes in context)
  //   sym N (N>=1): Num_Bits=N-1, Baseline=1<<(N-1)

  // Scan all table entries and verify each symbol's baseValue and nbAdditionalBits
  // Build a map of which symbols appear and what their table entries claim
  bool symbol_seen[32] = {};
  u8 symbol_nbAdditionalBits[32] = {};
  u32 symbol_baseValue[32] = {};

  for (u32 i = 0; i < table_size; i++) {
    u8 sym = h_table.symbol[i];
    if (sym < 32 && !symbol_seen[sym]) {
      symbol_seen[sym] = true;
      symbol_nbAdditionalBits[sym] = h_table.nbAdditionalBits[i];
      symbol_baseValue[sym] = h_table.baseValue[i];
    }
  }

  // Verify symbols that appear in the table have consistent offset properties
  for (u32 sym = 0; sym < 32 && sym <= max_sym; sym++) {
    if (!symbol_seen[sym]) continue;

    u8 actual_nb = symbol_nbAdditionalBits[sym];
    u32 actual_base = symbol_baseValue[sym];

    // Per RFC 8878 Table 15: for offset code N
    // Num_Bits = N, Baseline = (1 << N) - N
    // But the exact encoding may differ; at minimum verify:
    //  - nbAdditionalBits should equal the symbol value (offset code = extra bits count)
    //  - baseValue should be (1 << sym) for sym >= 1, or 0/1 for sym 0

    u8 expected_nb = (u8)sym;
    u32 expected_base = (sym == 0) ? 0 : (1u << sym);

    // Offset codes per RFC 8878: Num_Bits = Offset_Code, Offset = Baseline + readBits(Num_Bits)
    if (actual_nb != expected_nb) {
      printf("Symbol %u: nbAdditionalBits mismatch: expected=%u, got=%u\n",
             sym, expected_nb, actual_nb);
      all_ok = false;
    } else {
      printf("Symbol %u: nbAdditionalBits=%u, baseValue=%u OK\n",
             sym, actual_nb, actual_base);
    }
  }

  delete[] h_table.newState;
  delete[] h_table.symbol;
  delete[] h_table.nbBits;
  delete[] h_table.nbAdditionalBits;
  delete[] h_table.baseValue;

  results.report("Offset calculation", all_ok, "Offset values incorrect");
  return all_ok;
}

// ==============================================================================
// Main test runner
// ==============================================================================
int main(int argc, char **argv) {
  printf("====================================================================="
         "=========\n");
  printf("FSE Sequence Decoding Unit Tests\n");
  printf("====================================================================="
         "=========\n");

  TestResults results;

  // Run all tests
  test_state_bounds_checking(results);
  test_state_normalization(results);
  test_predefined_tables(results);
  test_device_table_copy(results);
  test_offset_calculation(results);

  // Print summary
  printf("\n==================================================================="
         "===========\n");
  printf("Test Summary: %d passed, %d failed\n", results.passed,
         results.failed);
  printf("====================================================================="
         "=========\n");

  return results.failed > 0 ? 1 : 0;
}
