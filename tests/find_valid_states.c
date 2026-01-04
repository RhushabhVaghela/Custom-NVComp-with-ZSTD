// find_valid_states.c
// Compile with: gcc -o find_valid_states find_valid_states.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


// --- Helper Types ---
typedef int16_t i16;
typedef uint16_t u16;
typedef uint8_t u8;
typedef uint32_t u32;

// --- Predefined Norm Distributions (RFC 8878) ---
const i16 default_ll_norm[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,
                                 2, 1, 1, 1, 2, 2, 2, 2, 2,  2,  2,  2,
                                 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};

const i16 default_of_norm[29] = {1, 1, 1, 1, 1,  1,  2,  2,  2, 1,
                                 1, 1, 1, 1, 1,  1,  1,  1,  1, 1,
                                 1, 1, 1, 1, -1, -1, -1, -1, -1};

const i16 default_ml_norm[53] = {
    1,  4,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

// --- DTable Builder ---
#define FSE_TABLESTEP(tableSize) (((tableSize) >> 1) + ((tableSize) >> 3) + 3)

void buildDTable(const i16 *norm, int max_sym, int table_log, u8 *symbols) {
  int table_size = 1 << table_log;
  int highThreshold = table_size - 1;
  int tableMask = table_size - 1;
  int step = FSE_TABLESTEP(table_size);

  // Init
  for (int i = 0; i < table_size; i++)
    symbols[i] = 0;

  // Phase 1: -1
  for (int s = 0; s <= max_sym; s++) {
    if (norm[s] == -1) {
      symbols[highThreshold--] = s;
    }
  }

  // Phase 2: Spread
  int position = 0;
  for (int s = 0; s <= max_sym; s++) {
    if (norm[s] <= 0)
      continue;
    for (int i = 0; i < norm[s]; i++) {
      symbols[position] = s;
      position = (position + step) & tableMask;
      while (position > highThreshold) {
        position = (position + step) & tableMask;
      }
    }
  }
}

// --- Main Permutation Search ---
int main() {
  u8 ll_syms[64];
  u8 of_syms[32]; // log 5
  u8 ml_syms[64];

  buildDTable(default_ll_norm, 35, 6, ll_syms);
  buildDTable(default_of_norm, 28, 5, of_syms);
  buildDTable(default_ml_norm, 52, 6, ml_syms);

  printf("Tables Built.\n");

  u32 vals[3] = {19, 11, 21};

  // Permutations
  int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                     {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

  char *names[6] = {"19, 11, 21", "19, 21, 11", "11, 19, 21",
                    "11, 21, 19", "21, 19, 11", "21, 11, 19"};

  printf("Searching for valid (LL, OF, ML) tuple...\n");
  printf("Candidates from bits: 19, 11, 21\n\n");

  for (int i = 0; i < 6; i++) {
    u32 v1 = vals[perms[i][0]];
    u32 v2 = vals[perms[i][1]];
    u32 v3 = vals[perms[i][2]];

    // Assume mapping 1: LL=v1, OF=v2, ML=v3
    // Check validity
    u32 ll = ll_syms[v1];
    u32 of = of_syms[v2];
    u32 ml = ml_syms[v3];

    printf("Permutation [%s] -> LL=%d OF=%d ML=%d\n", names[i], ll, of, ml);
  }

  return 0;
}
