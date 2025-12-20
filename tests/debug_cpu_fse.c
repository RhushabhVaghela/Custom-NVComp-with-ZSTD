// Debug CPU FSE encoding step-by-step
// Purpose: Print exact state values to compare with GPU

#define FSE_STATIC_LINKING_ONLY
#include "../build/_deps/zstd-src/lib/common/bitstream.h"
#include "../build/_deps/zstd-src/lib/common/fse.h"
#include "../build/_deps/zstd-src/lib/compress/hist.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TABLE_LOG 6

int main() {
  printf("=== CPU FSE Debug - Step by Step ===\n\n");

  uint8_t input[] = {0x41, 0x41, 0x42, 0x41, 0x43,
                     0x41, 0x41, 0x42, 0x41, 0x44};
  size_t input_size = sizeof(input);

  // Build CTable
  unsigned count[256] = {0};
  for (size_t i = 0; i < input_size; i++)
    count[input[i]]++;

  unsigned maxSymbol = 0x44;
  short normalized[256];
  size_t tableLog = FSE_normalizeCount(normalized, TABLE_LOG, count, input_size,
                                       maxSymbol, 1);

  size_t workspaceSize = FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbol, tableLog);
  void *workspace = malloc(workspaceSize);
  FSE_CTable *ctable =
      (FSE_CTable *)malloc(FSE_CTABLE_SIZE(tableLog, maxSymbol));

  FSE_buildCTable_wksp(ctable, normalized, maxSymbol, tableLog, workspace,
                       workspaceSize);

  // Manual encoding with debug
  uint8_t output[256];
  BIT_CStream_t bitC;
  BIT_initCStream(&bitC, output, 256);

  FSE_CState_t state;

  // Initialize with LAST symbol
  printf("Initializing with symbol 0x%02x (index %zu)\n", input[input_size - 1],
         input_size - 1);
  FSE_initCState2(&state, ctable, input[input_size - 1]);
  printf("  Initial state value: %zu\n", (size_t)state.value);
  printf("  State log: %u\n\n", state.stateLog);

  // Encode backward
  printf("Encoding symbols:\n");
  for (int i = (int)input_size - 2; i >= 0; --i) {
    uint8_t symbol = input[i];
    size_t value_before = state.value;

    FSE_encodeSymbol(&bitC, &state, symbol);

    printf("  [%d] symbol=0x%02x  state: %zu -> %zu  bits=%u/%zu\n", i, symbol,
           value_before, (size_t)state.value, bitC.bitPos, bitC.bitContainer);

    if ((i & 3) == 0) {
      BIT_flushBits(&bitC);
      printf("    (flushed bits)\n");
    }
  }

  printf("\nFlushing final state:\n");
  printf("  State value: %zu\n", (size_t)state.value);
  printf("  State log: %u\n", state.stateLog);
  FSE_flushCState(&bitC, &state);

  size_t size = BIT_closeCStream(&bitC);

  printf("\nFinal output (%zu bytes): ", size);
  for (size_t i = 0; i < size; i++) {
    printf("%02x ", output[i]);
  }
  printf("\n");

  // Also print CTable structure details
  printf("\n=== CTable Structure ===\n");
  const uint16_t *u16ptr = (const uint16_t *)ctable;
  printf("tableLog = %u\n", u16ptr[0]);
  printf("First few StateTable entries:");
  for (int i = 0; i < 10; i++) {
    printf(" %04x", u16ptr[2 + i]);
  }
  printf("\n");

  free(workspace);
  free(ctable);

  return 0;
}
