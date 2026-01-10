#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <zstd.h>

int main() {
  uint8_t input[1024];
  for (int i = 0; i < 1024; i++)
    input[i] = i % 256;

  size_t max_comp = ZSTD_compressBound(1024);
  uint8_t *output = malloc(max_comp);
  size_t comp_size = ZSTD_compress(output, max_comp, input, 1024, 1);

  printf("Compressed size: %zu\n", comp_size);
  printf("Frame Descriptor (Byte 4): %02X\n", output[4]);

  printf("\nDump 0-20:\n");
  for (int i = 0; i < 20; i++)
    printf("[%d] %02X\n", i, output[i]);

  printf("\nRaw dump 260-%zu:\n", comp_size);
  for (int i = 260; i < comp_size; i++) {
    printf("[%3d] %02X", i, output[i]);
    if (i == 273)
      printf(" <- Expected End of Literals");
    if (i == 274)
      printf(" <- Expected Seq Header");
    printf("\n");
  }

  free(output);
  return 0;
}
