// GPU FSE Encoder Kernel - Zstandard Compatible (Single State Version)
// Purpose: Match Zstandard FSE output for small inputs

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// CPU Zstandard FSE for reference
#define FSE_STATIC_LINKING_ONLY
#include "../build/_deps/zstd-src/lib/common/bitstream.h"
#include "../build/_deps/zstd-src/lib/common/error_private.h"
#include "../build/_deps/zstd-src/lib/common/fse.h"
#include "../build/_deps/zstd-src/lib/compress/hist.h"

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;

#define TABLE_LOG 6

// === GPU Bitstream (verified) ===
struct GPU_BitStream {
  u64 bitContainer;
  u32 bitPos;
  u8 *startPtr;
  u8 *ptr;
  u8 *endPtr;
};

__device__ inline u32 gpu_bit_init_stream(GPU_BitStream *bitC, u8 *dstBuffer,
                                          u32 dstCapacity) {
  if (dstCapacity < 1)
    return 1;
  bitC->bitContainer = 0;
  bitC->bitPos = 0;
  bitC->startPtr = dstBuffer;
  bitC->ptr = dstBuffer;
  bitC->endPtr = dstBuffer + dstCapacity;
  return 0;
}

__device__ inline void gpu_bit_add_bits(GPU_BitStream *bitC, u64 value,
                                        u32 nbBits) {
  u64 const mask = ((u64)1 << nbBits) - 1;
  value &= mask;
  bitC->bitContainer |= value << bitC->bitPos;
  bitC->bitPos += nbBits;
}

__device__ inline void gpu_bit_flush_bits(GPU_BitStream *bitC) {
  u32 const nbBytes = bitC->bitPos >> 3;
  if (bitC->ptr + nbBytes > bitC->endPtr)
    return;

  for (u32 i = 0; i < nbBytes; i++) {
    bitC->ptr[i] = (u8)(bitC->bitContainer >> (i * 8));
  }

  bitC->ptr += nbBytes;
  bitC->bitPos &= 7;
  bitC->bitContainer >>= (nbBytes * 8);
}

__device__ inline u32 gpu_bit_close_stream(GPU_BitStream *bitC) {
  gpu_bit_add_bits(bitC, 1, 1); // Terminator
  gpu_bit_flush_bits(bitC);

  if (bitC->bitPos > 0) {
    if (bitC->ptr < bitC->endPtr) {
      *bitC->ptr = (u8)bitC->bitContainer;
      bitC->ptr++;
    }
  }

  return (u32)(bitC->ptr - bitC->startPtr);
}

// === GPU FSE State ===
struct GPU_FSE_SymbolTransform {
  i32 deltaFindState;
  u32 deltaNbBits;
};

struct GPU_FSE_CState {
  u64 value; // Using u64 to match ptrdiff_t on 64-bit
  const u16 *stateTable;
  const GPU_FSE_SymbolTransform *symbolTT;
  u32 stateLog;
};

// Initialize FSE state (matches FSE_initCState2)
__device__ inline void
gpu_fse_init_state(GPU_FSE_CState *statePtr,
                   const u16 *ctable, // Points to FSE_CTable
                   u32 symbol) {
  // Read table log from first u16
  u32 const tableLog = ctable[0];

  // State table starts at offset +2 u16s
  const u16 *stateTable = ctable + 2;

  // Symbol transform table starts after: ct + 1 + (1 << (tableLog-1))
  // Since ct is u32*, offset in u32 units
  const u32 *ct_u32 = (const u32 *)ctable;
  const GPU_FSE_SymbolTransform *symbolTT =
      (const GPU_FSE_SymbolTransform *)(ct_u32 + 1 +
                                        (tableLog ? (1 << (tableLog - 1)) : 1));

  // Initialize state
  statePtr->value = (u64)1 << tableLog;
  statePtr->stateTable = stateTable;
  statePtr->symbolTT = symbolTT;
  statePtr->stateLog = tableLog;

  // Apply symbol-specific initialization (FSE_initCState2 logic)
  GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
  u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
  statePtr->value = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
  statePtr->value = stateTable[(statePtr->value >> nbBitsOut) +
                               symbolTransform.deltaFindState];
}

// Encode one symbol (matches FSE_encodeSymbol)
__device__ inline void gpu_fse_encode_symbol(GPU_BitStream *bitC,
                                             GPU_FSE_CState *statePtr,
                                             u32 symbol) {
  GPU_FSE_SymbolTransform const symbolTT = statePtr->symbolTT[symbol];
  u32 const nbBitsOut = (u32)((statePtr->value + symbolTT.deltaNbBits) >> 16);

  // Output low bits
  gpu_bit_add_bits(bitC, statePtr->value, nbBitsOut);

  // Update state
  statePtr->value = statePtr->stateTable[(statePtr->value >> nbBitsOut) +
                                         symbolTT.deltaFindState];
}

// Flush final state (matches FSE_flushCState)
__device__ inline void gpu_fse_flush_state(GPU_BitStream *bitC,
                                           const GPU_FSE_CState *statePtr) {
  gpu_bit_add_bits(bitC, statePtr->value, statePtr->stateLog);
  gpu_bit_flush_bits(bitC);
}

// === GPU FSE Encoder Kernel ===
__global__ void
gpu_fse_encode_kernel(const u8 *d_input, u32 input_size,
                      const u16 *d_ctable, // FSE_CTable uploaded to GPU
                      u8 *d_output, u32 *d_output_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  if (input_size <= 2) {
    *d_output_size = 0;
    return;
  }

  // Initialize bitstream
  GPU_BitStream bitC;
  if (gpu_bit_init_stream(&bitC, d_output, 256) != 0) {
    *d_output_size = 0;
    return;
  }

  // Initialize FSE state with LAST symbol
  GPU_FSE_CState state;
  gpu_fse_init_state(&state, d_ctable, d_input[input_size - 1]);

  // Encode symbols backward (from end-2 to start)
  for (i32 i = (i32)input_size - 2; i >= 0; --i) {
    gpu_fse_encode_symbol(&bitC, &state, d_input[i]);

    // Flush bits periodically
    if ((i & 3) == 0) {
      gpu_bit_flush_bits(&bitC);
    }
  }

  // Flush final state
  gpu_fse_flush_state(&bitC, &state);

  // Close bitstream
  *d_output_size = gpu_bit_close_stream(&bitC);

  printf("[GPU] Encoded %u bytes -> %u bytes\n", input_size, *d_output_size);
}

// === Main Test ===
int main() {
  printf("=== GPU FSE Encoder Test ===\n\n");

  // Test input
  u8 cpu_input[] = {0x41, 0x41, 0x42, 0x41, 0x43, 0x41, 0x41, 0x42, 0x41, 0x44};
  size_t input_size = sizeof(cpu_input);

  printf("Input: %zu bytes: ", input_size);
  for (size_t i = 0; i < input_size; i++)
    printf("%02x ", cpu_input[i]);
  printf("\n\n");

  // === Build CPU Reference ===
  printf("=== CPU Reference ===\n");

  unsigned count[256] = {0};
  for (size_t i = 0; i < input_size; i++)
    count[cpu_input[i]]++;

  unsigned maxSymbol = 0;
  for (int i = 255; i >= 0; i--) {
    if (count[i] > 0) {
      maxSymbol = i;
      break;
    }
  }

  short normalized[256];
  size_t tableLog = FSE_normalizeCount(normalized, TABLE_LOG, count, input_size,
                                       maxSymbol, 1);

  size_t workspaceSize = FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbol, tableLog);
  void *workspace = malloc(workspaceSize);

  size_t ctableSize = FSE_CTABLE_SIZE(tableLog, maxSymbol);
  FSE_CTable *ctable = (FSE_CTable *)malloc(ctableSize);

  FSE_buildCTable_wksp(ctable, normalized, maxSymbol, tableLog, workspace,
                       workspaceSize);

  u8 cpu_output[256];
  size_t cpu_size =
      FSE_compress_usingCTable(cpu_output, 256, cpu_input, input_size, ctable);

  printf("CPU output (%zu bytes): ", cpu_size);
  for (size_t i = 0; i < cpu_size; i++)
    printf("%02x ", cpu_output[i]);
  printf("\n\n");

  // === Upload CTable to GPU ===
  u16 *d_ctable;
  cudaMalloc(&d_ctable, ctableSize);
  cudaMemcpy(d_ctable, ctable, ctableSize, cudaMemcpyHostToDevice);

  // === Upload input ===
  u8 *d_input;
  u8 *d_output;
  u32 *d_output_size;
  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, 256);
  cudaMalloc(&d_output_size, sizeof(u32));
  cudaMemcpy(d_input, cpu_input, input_size, cudaMemcpyHostToDevice);

  // === Run GPU encoder ===
  printf("=== GPU Encoder ===\n");
  gpu_fse_encode_kernel<<<1, 1>>>(d_input, input_size, d_ctable, d_output,
                                  d_output_size);
  cudaDeviceSynchronize();

  u32 gpu_size;
  u8 gpu_output[256];
  cudaMemcpy(&gpu_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_output, d_output, gpu_size, cudaMemcpyDeviceToHost);

  printf("GPU output (%u bytes): ", gpu_size);
  for (u32 i = 0; i < gpu_size; i++)
    printf("%02x ", gpu_output[i]);
  printf("\n\n");

  // === Compare ===
  printf("=== Comparison ===\n");
  bool match = (cpu_size == gpu_size);
  if (match) {
    for (size_t i = 0; i < cpu_size; i++) {
      if (cpu_output[i] != gpu_output[i]) {
        printf("✗ Byte %zu: CPU=%02x GPU=%02x\n", i, cpu_output[i],
               gpu_output[i]);
        match = false;
        break;
      }
    }
  } else {
    printf("✗ Size mismatch: CPU=%zu GPU=%u\n", cpu_size, gpu_size);
  }

  if (match) {
    printf("✅ PERFECT MATCH! GPU encoder produces Zstandard-compatible "
           "output!\n");
  } else {
    printf("❌ Mismatch - debugging needed\n");
  }

  free(workspace);
  free(ctable);
  cudaFree(d_ctable);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_size);

  return match ? 0 : 1;
}
