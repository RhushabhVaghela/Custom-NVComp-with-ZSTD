/**
 * Dump libzstd's predefined FSE tables for comparison.
 *
 * Compile: gcc -o dump_libzstd_tables dump_libzstd_tables.c -lzstd
 * Run: ./dump_libzstd_tables
 */

#include <stdint.h>
#include <stdio.h>
#include <zstd.h>


// Access zstd internals (may need to include internal headers or copy
// definitions) These are the predefined tables from zstd's seqSymbol_header

// From zstd's zstd_decompress_block.c - the predefined tables
// We'll use ZSTD's API to decompress a minimal frame and inspect state

int main() {
  printf("libzstd version: %s\n", ZSTD_versionString());

  // Create a minimal compressed frame to force libzstd to use predefined tables
  // We'll compress empty data which uses predefined mode

  uint8_t input[1024];
  for (int i = 0; i < 1024; i++)
    input[i] = (uint8_t)(i % 256);

  uint8_t compressed[2048];
  size_t csize =
      ZSTD_compress(compressed, sizeof(compressed), input, sizeof(input), 1);

  if (ZSTD_isError(csize)) {
    printf("Compression error: %s\n", ZSTD_getErrorName(csize));
    return 1;
  }

  printf("Compressed size: %zu bytes\n", csize);
  printf("Compressed hex dump (first 50 bytes):\n");
  for (size_t i = 0; i < csize && i < 50; i++) {
    printf("%02X ", compressed[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n\n");

  // Now let's look at the sequence section
  // The frame header is typically 5-6 bytes
  // Block header is 3 bytes
  // Literals section varies
  // Sequences section starts with:
  //   - Number of sequences (1-2 bytes)
  //   - Symbol compression modes (1 byte)
  //   - FSE tables or predefined mode indicator

  printf("To dump predefined tables, we need to access zstd internals.\n");
  printf("The predefined Offset table (OF) from libzstd source:\n");
  printf("Symbol | State mappings need inspection via debugger.\n");

  // Let's decompress to verify it works
  uint8_t decompressed[1024];
  size_t dsize =
      ZSTD_decompress(decompressed, sizeof(decompressed), compressed, csize);

  if (ZSTD_isError(dsize)) {
    printf("Decompression error: %s\n", ZSTD_getErrorName(dsize));
    return 1;
  }

  printf("\nDecompressed size: %zu bytes\n", dsize);
  printf("Roundtrip: %s\n", (dsize == 1024) ? "SUCCESS" : "FAILED");

  return 0;
}
