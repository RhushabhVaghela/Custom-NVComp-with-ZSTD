#include "zstd.h"
#include <stdio.h>
#include <string.h>
#include <vector>


int main() {
  size_t input_size = 1024;
  std::vector<unsigned char> input(input_size);
  for (size_t i = 0; i < input_size; i++)
    input[i] = (unsigned char)(i % 256);

  size_t max_comp = ZSTD_compressBound(input_size);
  std::vector<unsigned char> compressed(max_comp);
  size_t comp_size =
      ZSTD_compress(compressed.data(), max_comp, input.data(), input_size, 1);

  printf("Compressed Size: %zu\n", comp_size);
  printf("Full Dump: ");
  for (size_t i = 0; i < comp_size; i++)
    printf("%02x ", compressed[i]);
  printf("\n");

  return 0;
}
