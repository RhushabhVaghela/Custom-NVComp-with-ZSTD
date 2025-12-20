#include "cuda_zstd_fse.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>


using namespace cuda_zstd;
using namespace cuda_zstd::fse;

void check_table_consistency() {
  printf("=== Testing FSE Table Consistency (REAL FIX VERIFICATION) ===\n");

  u32 table_log = 12;
  u32 table_size = 1 << table_log;
  u32 max_symbol = 255;

  std::vector<u16> norm(max_symbol + 1, 1);
  u32 current_sum = max_symbol + 1;
  norm[0] += (table_size - current_sum);

  // Build ENCODER Table (Standard)
  FSEEncodeTable ctable;
  FSE_buildCTable_Host(norm.data(), max_symbol, table_log, &ctable);

  // Build DECODER Table (Standard)
  FSEDecodeTable dtable;
  FSE_buildDTable_Host(norm.data(), max_symbol, table_size, dtable);

  // APPLY FIX: Reorder Symbols
  reorder_symbols_for_gpu(ctable, norm.data(), max_symbol);

  // VERIFY
  int mismatches = 0;

  // Re-verify logic:
  // With Reordering (Sort Descending Indices),
  // nextState values are assigned to states such that:
  // Low nextState (High Bits) -> assigned to High Index (High Bits in Formula).
  // High nextState (Low Bits) -> assigned to Low Index (Low Bits in Formula).
  // Wait.
  // Encoder Formula: (Index + delta) >> 16.
  // Low Index -> 0 bits.
  // High Index -> 1 bit.

  // Decoder NextState:
  // Low k (Low nextState) -> 1 bit.
  // High k (High nextState) -> 0 bits.

  // Sorting Descending:
  // k=0 (Low, Needs 1 bit) -> gets Largest Index (Providers 1 bit).
  // k=max (High, Needs 0 bits) -> gets Smallest Index (Provides 0 bits).

  // SO IT SHOULD MATCH!

  // But we need to know WHICH `nextState` maps to WHICH `Index`.
  // ctable.d_next_state stores `table_size + Index`.
  // It is indexed by `k` (cumulative position).
  // So `d_next_state[cumul[s] + k]` tells us the Index assigned to `k`.

  // Ideally we iterate `k` for each symbol?
  // Or iterate `Index` and reverse lookup?
  // Encoder Kernel: `k = (Index >> nbBits)`.
  // `state` = `d_next_state[cumul + k]`.
  // We want `nbBits(state)` to equal `nbBits(Index)`. (Wait, forward or
  // backward?)

  // Let's stick to the bit count check for EVERY Index in the table.
  // Iterate `Index` 0..size-1.
  // Find what `nextState` corresponds to this Index.
  // This is hard because `d_next_state` is (Sym -> Index).
  // We need map (Index -> Sym, k).
  // We have `spread` (Index -> Sym).
  // We need (Index -> k).
  // Since `d_next_state` maps `k -> Index`.
  // We can build the reverse map.

  std::vector<u32> index_to_nextState(table_size);
  std::vector<u16> state_to_symbol(table_size); // from Spread

  // Rebuild Spread (Decoder Logic) to assume what symbol is at Index
  {
    u32 pos = 0;
    u32 step = (table_size >> 1) + (table_size >> 3) + 3;
    u32 mask = table_size - 1;
    for (u32 s = 0; s <= max_symbol; s++) {
      for (u32 i = 0; i < norm[s]; i++) {
        state_to_symbol[pos] = (u16)s;
        pos = (pos + step) & mask;
      }
    }
  }

  // Build Index -> nextState map from CTable
  // CTable `d_next_state` array contains `table_size + Index`.
  // It is laid out by Symbol, then k.
  // `k` runs 0..freq-1.
  // `nextState` value corresponding to `k` is `freq + k`. (Zstd def).

  u32 current_offset = 0;
  for (u32 s = 0; s <= max_symbol; s++) {
    for (u32 k = 0; k < norm[s]; k++) {
      u16 encoded_val = ctable.d_next_state[current_offset + k];
      u32 index = encoded_val - table_size;

      // This k corresponds to nextState = freq + k?
      // NO! `reorder` permutes ONLY `d_next_state` values (Indices).
      // It does NOT change `nextState` semantics!
      // `nextState` semantics are defined by Decoder table build order?
      // Decoder: `nextState = symbolNext++`. (k=0 -> freq).
      // So `k`-th entry in `d_next_state` MUST correspond to `k`-th entry in
      // Decoder build. So Yes: k corresponds to `nextState = freq + k`.

      u32 nextState = norm[s] + k; // freq + k
      index_to_nextState[index] = nextState;
    }
    current_offset += norm[s];
  }

  // Now Check consistency
  for (u32 index = 0; index < table_size; index++) {
    u16 symbol = state_to_symbol[index]; // Should match
    u32 nextState = index_to_nextState[index];

    // 1. Decoder Bits needed for this nextState
    u32 highBit = 0;
    if (nextState > 0) {
      u32 tmp = nextState;
      while (tmp >>= 1)
        highBit++;
    }
    u32 d_nbBits = table_log - highBit;

    // 2. Encoder Bits produced by this Index (Formula)
    u32 true_state = index + table_size;
    i32 deltaNbBits = ctable.d_symbol_table[symbol].deltaNbBits;
    u32 e_nbBits = (true_state + deltaNbBits) >> 16;

    if (d_nbBits != e_nbBits) {
      if (mismatches < 10) {
        printf(
            "Mismatch State %u (Sym %u): DecBits(NextSt %u)=%u, EncBits=%u\n",
            index, symbol, nextState, d_nbBits, e_nbBits);
      }
      mismatches++;
    }
  }

  if (mismatches == 0) {
    printf("✅ SUCCESS: Encoder (Reordered) matches Decoder perfectly!\n");
  } else {
    printf("❌ FAILURE: Found %d mismatches!\n", mismatches);
    exit(1);
  }

  // Cleanup
  cudaFree(ctable.d_symbol_table); // It was malloc'd in source?
                                   // No, source uses `new`. Wait.
                                   // FSE_buildCTable_Host uses `new`.
                                   // reorder uses it.
                                   // safe deletion is delete[].
                                   // But we need access to the pointers.
                                   // The struct is local.
}

int main() {
  check_table_consistency();
  return 0;
}
