#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Norm tables from RFC 8878 / libzstd
static const short LL_defaultNorm[36] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,  1,  2,  2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};

static const short OF_defaultNorm[29] = {1, 1, 1, 1, 1,  1,  2,  2,  2, 1,
                                         1, 1, 1, 1, 1,  1,  1,  1,  1, 1,
                                         1, 1, 1, 1, -1, -1, -1, -1, -1};

static const short ML_defaultNorm[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};

#define LL_LOG 6
#define OF_LOG 5
#define ML_LOG 6

typedef struct {
  uint8_t symbol[64];
  uint8_t nbBits[64];
  uint16_t newState[64];
} FSEDecodeTable;

void build_fse_table(const short *norm, int maxSymbol, int tableLog,
                     FSEDecodeTable *table, const char *name) {
  uint32_t tableSize = 1 << tableLog;
  uint32_t tableMask = tableSize - 1;
  uint32_t step = (tableSize >> 1) + (tableSize >> 3) + 3;
  uint32_t highThreshold = tableSize - 1;

  // Init table
  memset(table, 0, sizeof(FSEDecodeTable));

  // Phase 1: Place low-prob symbols at high threshold
  for (int s = 0; s <= maxSymbol; s++) {
    if (norm[s] == -1) {
      table->symbol[highThreshold] = s;
      highThreshold--;
    }
  }

  // Phase 2: Spread regular symbols
  uint32_t position = 0;
  for (int s = 0; s <= maxSymbol; s++) {
    if (norm[s] <= 0)
      continue;
    for (int i = 0; i < norm[s]; i++) {
      table->symbol[position] = s;
      position = (position + step) & tableMask;
      while (position > highThreshold) {
        position = (position + step) & tableMask;
      }
    }
  }

  printf("\n%s Table (tableLog=%d, size=%d):\n", name, tableLog, tableSize);
  for (uint32_t i = 0; i < tableSize; i++) {
    printf("  State %2d -> Symbol %2d\n", i, table->symbol[i]);
  }
}

int main() {
  FSEDecodeTable ll_table, of_table, ml_table;

  build_fse_table(LL_defaultNorm, 35, LL_LOG, &ll_table, "LL");
  build_fse_table(OF_defaultNorm, 28, OF_LOG, &of_table, "OF");
  build_fse_table(ML_defaultNorm, 52, ML_LOG, &ml_table, "ML");

  printf("\nVerified States from Bitstream:\n");
  printf("  LL State 19 -> Symbol %d\n", ll_table.symbol[19]);
  printf("  OF State 11 -> Symbol %d\n", of_table.symbol[11]);
  printf("  ML State 21 -> Symbol %d\n", ml_table.symbol[21]);

  return 0;
}
