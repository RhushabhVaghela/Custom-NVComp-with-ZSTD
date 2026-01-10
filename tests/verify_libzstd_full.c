#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

int main() {
  // Create 1KB sequential pattern
  uint8_t input[1024];
  for (int i = 0; i < 1024; i++)
    input[i] = i % 256;

  // Compress with libzstd
  size_t max_comp = ZSTD_compressBound(1024);
  uint8_t *comp = malloc(max_comp);
  size_t comp_size = ZSTD_compress(comp, max_comp, input, 1024, 1);

  printf("Compressed size: %zu\n", comp_size);

  // Decompress to verify
  uint8_t *decomp = malloc(1024);
  size_t decomp_size = ZSTD_decompress(decomp, 1024, comp, comp_size);
  printf("Decompressed size: %zu\n", decomp_size);

  // Verify
  if (memcmp(input, decomp, 1024) == 0) {
    printf("Decompression verified OK\n");
  } else {
    printf("Decompression FAILED!\n");
    // Find first mismatch
    for (int i = 0; i < 1024; i++) {
      if (input[i] != decomp[i]) {
        printf("First mismatch at %d: expected 0x%02X, got 0x%02X\n", i,
               input[i], decomp[i]);
        break;
      }
    }
  }

  // Print what the decompressed content should look like
  printf("\nExpected pattern:\n");
  for (int i = 0; i < 20; i++)
    printf("%02X ", input[i]);
  printf("...\n");

  printf("\nActual decompressed:\n");
  for (int i = 0; i < 20; i++)
    printf("%02X ", decomp[i]);
  printf("...\n");

  free(comp);
  free(decomp);
  return 0;
}
