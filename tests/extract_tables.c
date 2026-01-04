#include <stdint.h>
#include <stdio.h>


// Declare the external symbol
extern const int16_t matchLengths_defaultDistribution[53];
extern const int16_t show_defaultDistribution[53]; // Assuming name? No.
// RFC says "Offset Code Table" (Table 30).
// Source mentions "of_defaultNorm"?
// I'll try matchLengths first.

int main() {
  printf("Match Lengths Default:\n");
  for (int i = 0; i < 53; i++) {
    printf("%d, ", matchLengths_defaultDistribution[i]);
  }
  printf("\n");
  return 0;
}
