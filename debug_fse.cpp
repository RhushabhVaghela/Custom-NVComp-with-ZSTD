#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <vector>


typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef short i16;

const i16 default_ml_norm[53] = {
    1,  4,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

const i16 default_of_norm[29] = {1, 1, 1, 1, 1,  1,  2,  2,  2, 1,
                                 1, 1, 1, 1, 1,  1,  1,  1,  1, 1,
                                 1, 1, 1, 1, -1, -1, -1, -1, -1};

u32 FSE_spread_symbols(const i16 *h_normalized, u32 max_symbol, u32 table_size,
                       u8 *spread_symbol) {
  const u32 SPREAD_STEP = (table_size >> 1) + (table_size >> 3) + 3;
  const u32 table_mask = table_size - 1;

  std::fill(spread_symbol, spread_symbol + table_size, 0);

  u32 high_thresh = table_size - 1;
  for (u32 s = 0; s <= max_symbol; s++) {
    if (h_normalized[s] == -1) {
      spread_symbol[high_thresh--] = (u8)s;
    }
  }

  printf("HighThresh: %u\n", high_thresh);

  u32 position = 0;
  for (u32 s = 0; s <= max_symbol; s++) {
    i16 count = h_normalized[s];
    if (count <= 0)
      continue;
    for (int i = 0; i < (int)count; i++) {
      spread_symbol[position] = (u8)s;
      do {
        position = (position + SPREAD_STEP) & table_mask;
      } while (position > high_thresh);
    }
  }
  return 0;
}

int main() {
  // ML Table
  u32 table_log = 6;
  u32 table_size = 1 << table_log;
  std::vector<u8> spread(table_size);
  FSE_spread_symbols(default_ml_norm, 52, table_size, spread.data());

  printf("ML Table (Size 64):\n");
  for (int i = 0; i < 64; i++) {
    printf("%d: %d\n", i, spread[i]);
  }

  printf("Check State 21: Symbol %d\n", spread[21]);

  // OF Table
  u32 of_log = 5;
  u32 of_size = 1 << of_log;
  std::vector<u8> of_spread(of_size);
  FSE_spread_symbols(default_of_norm, 28, of_size, of_spread.data());

  printf("OF Table (Size 32):\n");
  for (int i = 0; i < 32; i++) {
    printf("%d: %d\n", i, of_spread[i]);
  }
  printf("Check State 11: Symbol %d\n", of_spread[11]);

  return 0;
}
