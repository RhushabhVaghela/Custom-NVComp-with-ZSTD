#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <zstd.h>


// Use ZSTD internal headers to decode sequences
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

int main() {
  uint8_t input[1024];
  for (int i = 0; i < 1024; i++)
    input[i] = i % 256;

  // Compress
  size_t max_comp = ZSTD_compressBound(1024);
  uint8_t *comp = malloc(max_comp);
  size_t comp_size = ZSTD_compress(comp, max_comp, input, 1024, 1);
  printf("Compressed size: %zu\n", comp_size);

  // Decompress and verify
  uint8_t *decomp = malloc(1024);
  size_t decomp_size = ZSTD_decompress(decomp, 1024, comp, comp_size);
  if (ZSTD_isError(decomp_size)) {
    printf("Decompress error: %s\n", ZSTD_getErrorName(decomp_size));
    return 1;
  }
  printf("Decompressed size: %zu\n", decomp_size);

  // Verify
  int match = 1;
  for (int i = 0; i < 1024; i++) {
    if (input[i] != decomp[i]) {
      printf("Mismatch at %d: expected %d, got %d\n", i, input[i], decomp[i]);
      match = 0;
      break;
    }
  }
  if (match)
    printf("Decompression verified OK\n");

  free(comp);
  free(decomp);
  return 0;
}
