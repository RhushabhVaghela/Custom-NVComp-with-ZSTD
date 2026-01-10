#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>


int main() {
  // Generate text data (LZ77 friendly)
  size_t size = 1024 * 64; // 64KB
  uint8_t *data = (uint8_t *)malloc(size);
  const char *pattern = "The quick brown fox jumps over the lazy dog. ";
  size_t pat_len = strlen(pattern);
  for (size_t i = 0; i < size; i++) {
    data[i] = pattern[i % pat_len];
  }

  // Write expected txt
  FILE *f_txt = fopen("test_interop.txt", "wb");
  if (!f_txt) {
    perror("fopen txt");
    return 1;
  }
  fwrite(data, 1, size, f_txt);
  fclose(f_txt);

  // Compress
  size_t bound = ZSTD_compressBound(size);
  uint8_t *compressed = (uint8_t *)malloc(bound);
  size_t comp_size = ZSTD_compress(compressed, bound, data, size, 1); // Level 1
  if (ZSTD_isError(comp_size)) {
    printf("Compression error: %s\n", ZSTD_getErrorName(comp_size));
    return 1;
  }

  // Write zst
  FILE *f_zst = fopen("test_interop.zst", "wb");
  if (!f_zst) {
    perror("fopen zst");
    return 1;
  }
  fwrite(compressed, 1, comp_size, f_zst);
  fclose(f_zst);

  printf("Generated test_interop.txt (%zu bytes) and test_interop.zst (%zu "
         "bytes)\n",
         size, comp_size);

  free(data);
  free(compressed);
  return 0;
}
