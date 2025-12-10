#include <stdio.h>
#include <zstd.h>

int main() {
  printf("Error 6: %s\n", ZSTD_getErrorName((size_t)6));
  printf("Error 1: %s\n", ZSTD_getErrorName((size_t)1));
  printf("Error 10: %s\n", ZSTD_getErrorName((size_t)10));
  return 0;
}
