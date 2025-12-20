// Minimal FSE Encoder Test - 10 Bytes with Variation
// Purpose: Test complete FSE encoding path with small varied input
// Compare GPU output with CPU Zstandard FSE reference

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>


// Use CPU Zstandard FSE for reference
#define FSE_STATIC_LINKING_ONLY
#include "../build/_deps/zstd-src/lib/common/bitstream.h"
#include "../build/_deps/zstd-src/lib/common/error_private.h"
#include "../build/_deps/zstd-src/lib/common/fse.h"
#include "../build/_deps/zstd-src/lib/compress/hist.h"


using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

#define TABLE_LOG 6

int main() {
  printf("=== Minimal FSE Encoder Test ===\n\n");

  // Test input: 10 bytes with some variation
  // Pattern: A A B A C A A B A D
  u8 cpu_input[] = {0x41, 0x41, 0x42, 0x41, 0x43, 0x41, 0x41, 0x42, 0x41, 0x44};
  size_t input_size = sizeof(cpu_input);

  printf("Input: %zu bytes\n", input_size);
  printf("Data: ");
  for (size_t i = 0; i < input_size; i++) {
    printf("%02x ", cpu_input[i]);
  }
  printf("\n");
  printf("ASCII: ");
  for (size_t i = 0; i < input_size; i++) {
    printf(" %c  ", cpu_input[i]);
  }
  printf("\n\n");

  // === CPU Reference using Zstandard FSE ===
  printf("=== CPU Reference (Zstandard FSE) ===\n");

  u8 cpu_output[256];

  // Count symbol frequencies
  unsigned count[256] = {0};
  for (size_t i = 0; i < input_size; i++) {
    count[cpu_input[i]]++;
  }

  // Find max symbol
  unsigned maxSymbol = 0;
  for (int i = 255; i >= 0; i--) {
    if (count[i] > 0) {
      maxSymbol = i;
      break;
    }
  }

  printf("Symbol frequencies:\n");
  for (u32 i = 0; i <= maxSymbol; i++) {
    if (count[i] > 0) {
      printf("  0x%02x ('%c'): %u times\n", i, i, count[i]);
    }
  }
  printf("\n");

  // Normalize
  short normalized[256];
  size_t tableLog = FSE_normalizeCount(normalized, TABLE_LOG, count, input_size,
                                       maxSymbol, 1);

  if (tableLog == 0) {
    printf("RLE mode detected (all same symbol)\n");
    return 0;
  }

  printf("Table log: %zu\n", tableLog);
  printf("Normalized counts:\n");
  for (u32 i = 0; i <= maxSymbol; i++) {
    if (count[i] > 0) {
      printf("  0x%02x: %d\n", i, normalized[i]);
    }
  }
  printf("\n");

  // Build compression table
  size_t workspaceSize = FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbol, tableLog);
  void *workspace = malloc(workspaceSize);
  FSE_CTable *ctable =
      (FSE_CTable *)malloc(FSE_CTABLE_SIZE(tableLog, maxSymbol));

  size_t err = FSE_buildCTable_wksp(ctable, normalized, maxSymbol, tableLog,
                                    workspace, workspaceSize);
  if (FSE_isError(err)) {
    printf("Error building CTable: %zu\n", err);
    return 1;
  }

  printf("CTable built successfully\n\n");

  // Compress
  size_t cpu_size =
      FSE_compress_usingCTable(cpu_output, 256, cpu_input, input_size, ctable);

  if (FSE_isError(cpu_size)) {
    printf("CPU compression failed: %zu\n", cpu_size);
    return 1;
  }

  printf("CPU compressed %zu bytes -> %zu bytes\n", input_size, cpu_size);
  printf("Compression ratio: %.2f%%\n", 100.0 * cpu_size / input_size);
  printf("\nCPU output (%zu bytes):\n", cpu_size);
  for (size_t i = 0; i < cpu_size; i++) {
    printf("%02x ", cpu_output[i]);
    if ((i + 1) % 20 == 0)
      printf("\n");
  }
  printf("\n\n");

  // Show bitstream structure
  printf("=== Bitstream Analysis ===\n");
  printf("This is the target format our GPU encoder must match.\n");
  printf("The bitstream contains:\n");
  printf("1. Encoded symbols (in reverse order)\n");
  printf("2. Final FSE state\n");
  printf("3. Terminator bit (1)\n");
  printf("4. Padding to byte boundary\n\n");

  printf("📌 GOAL: GPU encoder must produce these exact %zu bytes\n", cpu_size);
  printf("Next: Implement GPU kernel to match this output\n");

  free(workspace);
  free(ctable);

  return 0;
}
