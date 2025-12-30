
/**
 * @brief (NEW) Custom CPU builder for FSE Decode Table.
 * Guarantees logic parity with FSE_buildCTable_Host.
 */
__host__ Status FSE_buildDTable_Custom(const u16 *normalized_freqs,
                                       u32 max_symbol, u32 table_size,
                                       FSEDecodeTable *d_table) {
  // 1. Calculate table log
  // Assumes table_size is power of 2
  u32 table_log = 0;
  while ((1u << table_log) < table_size)
    table_log++;

  // 2. Spread Symbols (Matches ZSTD)
  const u32 table_mask = table_size - 1;
  const u32 step = (table_size >> 1) + (table_size >> 3) + 3;

  std::vector<u8> spread_symbol(table_size);
  u32 position = 0;

  for (u32 s = 0; s <= max_symbol; s++) {
    for (u32 i = 0; i < normalized_freqs[s]; i++) {
      spread_symbol[position] = (u8)s;
      position = (position + step) & table_mask;
    }
  }

  // 3. Build Table Entries
  u16 symbol_next[256]; // Max symbols
  for (u32 s = 0; s <= max_symbol; s++) {
    symbol_next[s] = normalized_freqs[s];
  }

  for (u32 state = 0; state < table_size; state++) {
    u8 symbol = spread_symbol[state];
    u16 next_state = symbol_next[symbol]++;
    u32 high_bit = 31 - __builtin_clz(next_state); // Assuming GCC/CUDA compiler
    // For MSVC use _BitScanReverse or manual count if needed.
    // Since this is CUDA file, __clz is available (or __builtin_clz for host
    // gcc). Safer:
    /*
    u32 high_bit = 0;
    u32 tmp = next_state;
    while(tmp >>= 1) high_bit++;
    */

    u32 nb_bits = table_log - high_bit;
    u16 new_state = (u16)((next_state << nb_bits) - table_size);

    d_table->symbol[state] = symbol;
    d_table->nbBits[state] = (u8)nb_bits;
    d_table->newState[state] = new_state;
  }

  return Status::SUCCESS;
}
