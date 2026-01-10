/**
 * Dump libzstd's predefined FSE decoding tables.
 * This extracts the EXACT state->symbol mapping from libzstd.
 *
 * Compile: gcc -o dump_fse_tables dump_fse_tables.c -I/path/to/zstd/lib -lzstd
 * Or: gcc -o dump_fse_tables dump_fse_tables.c -I$(pwd)/zstd/lib
 * -L$(pwd)/zstd/lib -lzstd
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define FSE types matching libzstd's internal structure
typedef uint8_t BYTE;
typedef uint16_t U16;
typedef int16_t S16;
typedef uint32_t U32;

// FSE decode table entry (matches libzstd's ZSTD_seqSymbol)
typedef struct {
  U16 nextState;
  BYTE nbAdditionalBits;
  BYTE nbBits;
  U32 baseValue;
} FSE_decode_t;

// libzstd's predefined normalized counts (from zstd_internal.h)
static const S16 OF_defaultNorm[29] = {1, 1, 1, 1, 1,  1,  2,  2,  2, 1,
                                       1, 1, 1, 1, 1,  1,  1,  1,  1, 1,
                                       1, 1, 1, 1, -1, -1, -1, -1, -1};
#define OF_DEFAULTNORMLOG 5

static const S16 LL_defaultNorm[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,
                                       2, 1, 1, 1, 2, 2, 2, 2, 2,  2,  2,  2,
                                       2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
#define LL_DEFAULTNORMLOG 6

static const S16 ML_defaultNorm[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
#define ML_DEFAULTNORMLOG 6

// FSE_TABLESTEP from libzstd
#define FSE_TABLESTEP(tableSize) (((tableSize) >> 1) + ((tableSize) >> 3) + 3)

// Build FSE decode table using libzstd's algorithm
void build_fse_table(const S16 *norm, int maxSymbol, int tableLog,
                     BYTE *symbolTable, const char *name) {
  U32 tableSize = 1 << tableLog;
  U32 tableMask = tableSize - 1;
  U32 step = FSE_TABLESTEP(tableSize);

  // Phase 1: Place low-prob symbols at high threshold positions
  U32 highThreshold = tableSize - 1;
  for (int s = 0; s <= maxSymbol; s++) {
    if (norm[s] == -1) {
      symbolTable[highThreshold--] = (BYTE)s;
    }
  }

  // Phase 2: Spread regular symbols
  if (highThreshold == tableSize - 1) {
    // No low-prob symbols - two-phase spread
    BYTE *spread = (BYTE *)malloc(tableSize + 8);
    size_t pos = 0;
    for (int s = 0; s <= maxSymbol; s++) {
      S16 count = norm[s];
      if (count <= 0)
        continue;
      for (int i = 0; i < count; i++) {
        spread[pos++] = (BYTE)s;
      }
    }

    // Scatter
    U32 position = 0;
    for (size_t i = 0; i < tableSize; i++) {
      symbolTable[position] = spread[i];
      position = (position + step) & tableMask;
    }
    free(spread);
  } else {
    // Has low-prob - step and skip
    U32 position = 0;
    for (int s = 0; s <= maxSymbol; s++) {
      S16 count = norm[s];
      if (count <= 0)
        continue;
      for (int i = 0; i < count; i++) {
        symbolTable[position] = (BYTE)s;
        position = (position + step) & tableMask;
        while (position > highThreshold) {
          position = (position + step) & tableMask;
        }
      }
    }
  }

  // Print the table
  printf("\n%s FSE Table (tableLog=%d, tableSize=%u):\n", name, tableLog,
         tableSize);
  printf("State -> Symbol mapping:\n");
  for (U32 i = 0; i < tableSize; i++) {
    printf("  State %2u -> Symbol %2u\n", i, symbolTable[i]);
  }

  // Specifically check state 11 for OF table
  if (strcmp(name, "OF") == 0) {
    printf("\n*** CRITICAL: State 11 maps to Symbol %u ***\n", symbolTable[11]);
  }
}

int main() {
  printf("FSE Predefined Table Dump (using libzstd algorithm)\n");
  printf("====================================================\n");

  BYTE of_table[32];
  build_fse_table(OF_defaultNorm, 28, OF_DEFAULTNORMLOG, of_table, "OF");

  BYTE ll_table[64];
  build_fse_table(LL_defaultNorm, 35, LL_DEFAULTNORMLOG, ll_table, "LL");

  BYTE ml_table[64];
  build_fse_table(ML_defaultNorm, 52, ML_DEFAULTNORMLOG, ml_table, "ML");

  printf("\nOffset Code Table (libzstd logic):\n");
  printf("Code | Base (OF_base) | Bits (OF_bits) | Result Distance Example "
         "(extra=0)\n");
  printf(
      "-----|----------------|----------------|-------------------------------"
      "--\n");
  for (int i = 0; i <= 31; i++) {
    uint32_t base = (i <= 2) ? 0 : (1u << i) - 3;
    if (i == 1)
      base = 1;
    if (i == 2)
      base = 1;
    // Note: libzstd OF_base[1]=1, OF_base[2]=1, OF_base[3]=5.
    // Our code uses: table = {0, 1, 1, 5, 13, 29, 61, 125, 253, ...}
    printf("%4d | %14u | %14d | %u\n", i, base, i, base);
  }

  printf("\nVerification for Distance 256:\n");
  printf("Distance 256 in RFC means actual_offset = 256.\n");
  printf("Offset_Value = actual_offset + 3 = 259.\n");
  uint32_t code8_base = (1u << 8) - 3; // 253
  printf("Code 8 Base: %u\n", code8_base);
  printf("Needed Extra for Offset_Value 259: %u (259 - 253)\n",
         259 - code8_base);
  printf("Binary for Extra 6 (8 bits): 00000110\n");

  return 0;
}
