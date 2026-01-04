// libzstd_verify_dtable.c
// Compile with: gcc -o verify_dtable libzstd_verify_dtable.c -lzstd
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

void print_bits(unsigned char byte) {
  for (int i = 7; i >= 0; i--) {
    printf("%c", (byte & (1 << i)) ? '1' : '0');
  }
}

int main() {
  // Sawtooth pattern 0..255 repeated 4 times (1024 bytes)
  size_t data_size = 1024;
  unsigned char *data = malloc(data_size);
  for (int i = 0; i < data_size; i++)
    data[i] = i % 256;

  // Compress with level 1
  size_t bound = ZSTD_compressBound(data_size);
  unsigned char *compressed = malloc(bound);

  // Enforce predefined mode? Zstd might choose it automatically for this
  // pattern We can't easily force it via simple API, but let's see what it
  // produces.
  size_t compressed_size = ZSTD_compress(compressed, bound, data, data_size, 1);
  if (ZSTD_isError(compressed_size)) {
    printf("Compression error: %s\n", ZSTD_getErrorName(compressed_size));
    return 1;
  }

  printf("Compressed size: %zu bytes\n", compressed_size);

  // Dump the last 16 bytes where the sequence states are
  printf("Last 16 bytes (hex): ");
  for (size_t i = 0; i < 16 && i < compressed_size; i++) {
    size_t idx = compressed_size - 16 + i;
    if (idx < compressed_size)
      printf("%02x ", compressed[idx]);
  }
  printf("\n");

  // Dump binary of last 3 bytes
  printf("Last 3 bytes (binary):\n");
  for (size_t i = 0; i < 3 && i < compressed_size; i++) {
    size_t idx = compressed_size - 1 - i; // Reverse: Last, 2nd Last, ...
    printf("Byte -%zu (%02x): ", i + 1, compressed[idx]);
    print_bits(compressed[idx]);
    printf("\n");
  }

  free(data);
  free(compressed);
  return 0;
}
