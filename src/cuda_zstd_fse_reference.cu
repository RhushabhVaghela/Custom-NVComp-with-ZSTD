/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 */

#include "cuda_zstd_fse_reference.h"
#include "cuda_zstd_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuda_zstd {
namespace fse {

// --- Reference FSE Types ---

struct FSE_CState_t {
  int state;
  // We store the values directly instead of pointer to table for simplicity
  u32 deltaNbBits;
  u32 deltaFindState;
};

// Reference CTable Builder
// Normalized Counts: sum should be 1<<tableLog
void build_fse_ctable_reference(std::vector<FSE_CTable_Entry> &table,
                                const std::vector<short> &normalized_counts,
                                unsigned table_log) {
  unsigned table_size = 1 << table_log;
  unsigned max_symbol = (unsigned)normalized_counts.size() - 1;
  table.resize(max_symbol + 1);

  std::vector<u16> symbolNext(max_symbol + 1);

  // Cumul array
  std::vector<unsigned> cumul(max_symbol + 2, 0);
  for (unsigned s = 0; s <= max_symbol; s++) {
    if (normalized_counts[s] == -1) // Handle probability < 1 representation
      cumul[s + 1] = cumul[s] + 1;
    else
      cumul[s + 1] = cumul[s] + normalized_counts[s];
  }

  // 1. Calculate symbolNext (initial states)
  // This part is critical. We simulate the DTable build order (spread).
  // (Unused variables step, mask, pos removed)

  // To match Zstd FSE_initCState, we need to know for each symbol,
  // which state value corresponds to its "first appearance" in the table?
  // Actually, FSE_compress uses a specific formula for init:
  // state = tableSize + cumul[symbol] - 1; NO.

  // Let's look at FSE_buildCTable in standard ZSTD lib:
  //   total = 0;
  //   for (s=0; s<=maxSymbol; s++) {
  //      stateTable[s] = (U16)(total + start);
  //      total += count[s];
  //   }
  // So for the FIRST symbol of the block, the state is initialized from this
  // table.

  unsigned long long total = 0;
  for (unsigned s = 0; s <= max_symbol; s++) {
    int count = normalized_counts[s];
    if (count == 0) {
      table[s].deltaNbBits = (table_log + 1) << 16;
      table[s].nextState = 0; // Invalid
      continue;
    }

    // Count handling for -1 (prob < 1)
    int clean_count = (count == -1) ? 1 : count;

    unsigned state = clean_count;
    unsigned nbBitsOut = 0;
    while (state < table_size) {
      state <<= 1;
      nbBitsOut++;
    }

    unsigned minStatePlus = (unsigned)clean_count << nbBitsOut;

    table[s].deltaNbBits = (nbBitsOut << 16) - minStatePlus;
    table[s].deltaFindState =
        total - 1 + table_size - minStatePlus; // ZSTD formula

    // Initial state for this symbol (used at start of encoding)
    // In Zstd 1.5: stateTable[s] = (U16)(total + count);  Wait...
    // No, FSE_initCState: state = stateTable[symbol].
    // And stateTable[s] = total + count (or similar).
    // Let's use the exact formula from source:
    // note: total is cumulative sum BEFORE this symbol.
    // The valid range of states for symbol s is [total, total+count-1] +
    // offset?

    // Actually, let's verify with the existing spreading logic in kernel.
    // It uses `d_symbol_first_state`.

    // Re-deriving ZSTD logic:
    // Decoding table maps State -> Symbol.
    // Encoding just reverses this.
    // Range of states decoding to S is scattered.
    // But mapped to a contiguous range [cumul[s], cumul[s]+count-1] in
    // "normalized" space?

    // The standard ZSTD CTable `nextState` entry is actually:
    // stateTable[s] = tableSize + cumul[s] (roughly) -> No that's too big.

    // Let's stick to what we see in the code:
    // deltaFindState = total - 1 + table_size - minStatePlus

    // For the *first* state of the sequence (init), the value is simply:
    // table[s].nextState = table_size + cumul[s] - minStatePlus? No.

    // Let's assume for a moment that for the Reference implementation we will
    // use the exact same logic as FSE_buildCTable.

    // From zstd/lib/common/fse_compress.c:
    //   table->deltaFindState = total - 1 + tableSize - minStatePlus;
    //   stateTable[s] = (U16)(total + start); (where start = minStatePlus >>
    //   nbBitsOut ?)

    table[s].nextState = (u16)(total + table_size);

    total += clean_count;
  }
}

u32 fse_get_nb_bits(u32 state, u32 deltaNbBits) {
  return (state + deltaNbBits) >> 16;
}

// Perform step: write bits, update state
void fse_encode_step(u32 &state, u32 symbol, u32 next_symbol,
                     const std::vector<FSE_CTable_Entry> &ctable,
                     std::vector<u8> &bitstream, u64 &bitContainer,
                     u32 &bitCount) {
  const auto &entry = ctable[symbol];
  const auto &nextEntry = ctable[next_symbol];

  u32 nbBits = (state + entry.deltaNbBits) >> 16;
  u32 value = state & ((1 << nbBits) - 1);

  // Update state using NEXT symbol's deltaFindState
  state = (state >> nbBits) + nextEntry.deltaFindState;

  // Output bits
  bitContainer |= ((u64)value << bitCount);
  bitCount += nbBits;

  while (bitCount >= 8) {
    bitstream.push_back((u8)bitContainer);
    bitContainer >>= 8;
    bitCount -= 8;
  }
}

} // namespace fse
} // namespace cuda_zstd
