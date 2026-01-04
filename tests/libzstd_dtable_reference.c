// libzstd_dtable_reference.c
// Compile with: gcc -o dtable_ref libzstd_dtable_reference.c -lzstd
#include <stdint.h>
#include <stdio.h>


// Reproduce libzstd's predefined LL table spread
#define FSE_TABLESTEP(tableSize) (((tableSize) >> 1) + ((tableSize) >> 3) + 3)

const short default_ll_norm[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,
                                   2, 1, 1, 1, 2, 2, 2, 2, 2,  2,  2,  2,
                                   2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};

int main() {
  const int tableLog = 6;
  const int tableSize = 1 << tableLog;
  const int tableMask = tableSize - 1;
  const int step = FSE_TABLESTEP(tableSize);
  const int maxSV1 = 36;

  uint8_t symbols[64];
  int highThreshold = tableSize - 1;

  // Phase 1: Place -1 symbols at highThreshold
  for (int s = 0; s < maxSV1; s++) {
    if (default_ll_norm[s] == -1) {
      symbols[highThreshold--] = s;
      printf("Phase1: Placed sym=%d at pos=%d\n", s, highThreshold + 1);
    }
  }

  printf("highThreshold after lowprob = %d\n", highThreshold);

  // Phase 2: Spread regular symbols
  int position = 0;
  for (int s = 0; s < maxSV1; s++) {
    int n = default_ll_norm[s];
    if (n <= 0)
      continue;

    for (int i = 0; i < n; i++) {
      symbols[position] = s;
      if (position == 19) {
        printf("Placing sym=%d at pos=19\n", s);
      }
      position = (position + step) & tableMask;
      while (position > highThreshold) {
        position = (position + step) & tableMask;
      }
    }
  }

  printf("\nFull DTable symbols:\n");
  for (int i = 0; i < tableSize; i++) {
    printf("%d ", symbols[i]);
  }
  printf("\n");

  printf("\nState 19 = symbol %d\n", symbols[19]);

  return 0;
}
