#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <zstd.h>


int main() {
  // Create sequential pattern 0,1,2,...255,0,1,... (1024 bytes)
  uint8_t input[1024];
  for (int i = 0; i < 1024; i++) {
    input[i] = i % 256;
  }

  // Compress
  size_t max_comp = ZSTD_compressBound(1024);
  uint8_t output[max_comp];
  size_t comp_size = ZSTD_compress(output, max_comp, input, 1024, 1);

  if (ZSTD_isError(comp_size)) {
    printf("Error: %s\n", ZSTD_getErrorName(comp_size));
    return 1;
  }

  printf("Compressed size: %zu\n", comp_size);
  printf("Full hex dump:\n");
  for (size_t i = 0; i < comp_size; i++) {
    printf("%02X ", output[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n");

  // Show last 10 bytes (sequence section)
  printf("\nLast 10 bytes (sequence section):\n");
  for (size_t i = comp_size - 10; i < comp_size; i++) {
    printf("%02X ", output[i]);
  }
  printf("\n");

  return 0;
}
