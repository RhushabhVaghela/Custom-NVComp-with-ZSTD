// CPU FSE Reference Encoder using Zstandard library
// Load exact input from GPU test and compare outputs

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Include Zstandard FSE headers
#define FSE_STATIC_LINKING_ONLY
#include "../build/_deps/zstd-src/lib/common/error_private.h"
#include "../build/_deps/zstd-src/lib/common/fse.h"
#include "../build/_deps/zstd-src/lib/compress/hist.h"


#define TABLE_LOG 9

int main() {
  printf("=== CPU FSE Reference Encoder ===\n\n");

  // Load input from GPU test
  FILE *f_in = fopen("gpu_test_input.bin", "rb");
  if (!f_in) {
    printf("Error: Could not open gpu_test_input.bin\n");
    printf("Run ./test_single_chunk first to generate it\n");
    return 1;
  }

  fseek(f_in, 0, SEEK_END);
  size_t data_size = ftell(f_in);
  fseek(f_in, 0, SEEK_SET);

  uint8_t *input = (uint8_t *)malloc(data_size);
  fread(input, 1, data_size, f_in);
  fclose(f_in);

  printf("Loaded input: %zu bytes\n", data_size);
  printf("First 20 bytes: ");
  for (int i = 0; i < 20; i++) {
    printf("%02x ", input[i]);
  }
  printf("\n\n");

  // Count symbol frequencies
  unsigned count[256] = {0};
  for (size_t i = 0; i < data_size; i++) {
    count[input[i]]++;
  }

  // Find max symbol
  unsigned maxSymbolValue = 0;
  for (int i = 255; i >= 0; i--) {
    if (count[i] > 0) {
      maxSymbolValue = i;
      break;
    }
  }

  printf("Max symbol value: %u\n", maxSymbolValue);
  printf("Normalizing frequencies for tableLog=%d...\n", TABLE_LOG);

  // Normalize frequencies
  short normalized[256];
  size_t const tableLog =
      FSE_normalizeCount(normalized, TABLE_LOG, count, data_size,
                         maxSymbolValue, /* useLowProbCount */ 1);

  if (FSE_isError(tableLog)) {
    printf("Error normalizing: %zu\n", tableLog);
    return 1;
  }

  printf("Actual tableLog used: %zu\n\n", tableLog);

  // Build compression table
  size_t const workspaceSize =
      FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbolValue, TABLE_LOG);
  void *workspace = malloc(workspaceSize);
  FSE_CTable *ctable = (FSE_CTable *)malloc(
      FSE_CTABLE_SIZE_U32(maxSymbolValue, TABLE_LOG) * sizeof(uint32_t));

  size_t err = FSE_buildCTable_wksp(ctable, normalized, maxSymbolValue,
                                    TABLE_LOG, workspace, workspaceSize);
  if (FSE_isError(err)) {
    printf("Error building CTable: %zu\n", err);
    return 1;
  }

  printf("CTable built successfully\n\n");

  // Compress
  size_t const max_compressed_size = FSE_compressBound(data_size);
  uint8_t *compressed = (uint8_t *)malloc(max_compressed_size);

  size_t compressed_size = FSE_compress_usingCTable(
      compressed, max_compressed_size, input, data_size, ctable);

  if (FSE_isError(compressed_size)) {
    printf("Compression failed: %zu\n", compressed_size);
    return 1;
  }

  printf("Compressed to %zu bytes (ratio: %.2f%%)\n\n", compressed_size,
         100.0 * compressed_size / data_size);

  printf("Compressed data (first 40 bytes):\n");
  for (size_t i = 0; i < 40 && i < compressed_size; i++) {
    printf("%02x ", compressed[i]);
    if ((i + 1) % 20 == 0)
      printf("\n");
  }
  printf("\n");

  printf("\nCompressed data (last 40 bytes):\n");
  size_t start = compressed_size > 40 ? compressed_size - 40 : 0;
  for (size_t i = start; i < compressed_size; i++) {
    printf("%02x ", compressed[i]);
    if ((i - start + 1) % 20 == 0)
      printf("\n");
  }
  printf("\n\n");

  // Save CPU output for comparison
  FILE *f = fopen("cpu_fse_output.bin", "wb");
  if (f) {
    fwrite(compressed, 1, compressed_size, f);
    fclose(f);
    printf("Saved CPU output to cpu_fse_output.bin (%zu bytes)\n",
           compressed_size);
  }

  printf("\n=== To compare with GPU output ===\n");
  printf("Run: ./test_single_chunk | grep 'Encoded data'\n");
  printf("Compare first/last bytes above with GPU encoder output\n");

  free(input);
  free(compressed);
  free(workspace);
  free(ctable);

  return 0;
}
