// Test GPU FSE Encoder at Multiple Sizes
// Purpose: Verify encoder works for 100B, 1KB, 4KB inputs

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>


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

#define TABLE_LOG 9

// Include GPU encoder from test_gpu_fse_encoder.cu (just the kernel part)
// We'll compile this with the same utilities

// Copy GPU bitstream and FSE utilities here for simplicity
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
  gpu_bit_add_bits(bitC, 1, 1);
  gpu_bit_flush_bits(bitC);

  if (bitC->bitPos > 0) {
    if (bitC->ptr < bitC->endPtr) {
      *bitC->ptr = (u8)bitC->bitContainer;
      bitC->ptr++;
    }
  }

  return (u32)(bitC->ptr - bitC->startPtr);
}

struct GPU_FSE_SymbolTransform {
  i32 deltaFindState;
  u32 deltaNbBits;
};

struct GPU_FSE_CState {
  u64 value;
  const u16 *stateTable;
  const GPU_FSE_SymbolTransform *symbolTT;
  u32 stateLog;
};

__device__ inline void gpu_fse_init_state(GPU_FSE_CState *statePtr,
                                          const u16 *ctable, u32 symbol) {
  u32 const tableLog = ctable[0];
  const u16 *stateTable = ctable + 2;
  const u32 *ct_u32 = (const u32 *)ctable;
  const GPU_FSE_SymbolTransform *symbolTT =
      (const GPU_FSE_SymbolTransform *)(ct_u32 + 1 +
                                        (tableLog ? (1 << (tableLog - 1)) : 1));

  statePtr->value = (u64)1 << tableLog;
  statePtr->stateTable = stateTable;
  statePtr->symbolTT = symbolTT;
  statePtr->stateLog = tableLog;

  GPU_FSE_SymbolTransform const symbolTransform = symbolTT[symbol];
  u32 nbBitsOut = (symbolTransform.deltaNbBits + (1u << 15)) >> 16;
  statePtr->value = (nbBitsOut << 16) - symbolTransform.deltaNbBits;
  statePtr->value = stateTable[(statePtr->value >> nbBitsOut) +
                               symbolTransform.deltaFindState];
}

__device__ inline void gpu_fse_encode_symbol(GPU_BitStream *bitC,
                                             GPU_FSE_CState *statePtr,
                                             u32 symbol) {
  GPU_FSE_SymbolTransform const symbolTT = statePtr->symbolTT[symbol];
  u32 const nbBitsOut = (u32)((statePtr->value + symbolTT.deltaNbBits) >> 16);

  gpu_bit_add_bits(bitC, statePtr->value, nbBitsOut);
  statePtr->value = statePtr->stateTable[(statePtr->value >> nbBitsOut) +
                                         symbolTT.deltaFindState];
}

__device__ inline void gpu_fse_flush_state(GPU_BitStream *bitC,
                                           const GPU_FSE_CState *statePtr) {
  gpu_bit_add_bits(bitC, statePtr->value, statePtr->stateLog);
  gpu_bit_flush_bits(bitC);
}

// Dual-state encoder kernel
__global__ void gpu_fse_encode_kernel(const u8 *d_input, u32 input_size,
                                      const u16 *d_ctable, u8 *d_output,
                                      u32 *d_output_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  if (input_size <= 2) {
    *d_output_size = 0;
    return;
  }

  GPU_BitStream bitC;
  if (gpu_bit_init_stream(&bitC, d_output, 65536) != 0) {
    *d_output_size = 0;
    return;
  }

  GPU_FSE_CState CState1, CState2;
  const u8 *ip = d_input + input_size;

  if (input_size & 1) {
    gpu_fse_init_state(&CState1, d_ctable, *--ip);
    gpu_fse_init_state(&CState2, d_ctable, *--ip);
    gpu_fse_encode_symbol(&bitC, &CState1, *--ip);
    gpu_bit_flush_bits(&bitC);
  } else {
    gpu_fse_init_state(&CState2, d_ctable, *--ip);
    gpu_fse_init_state(&CState1, d_ctable, *--ip);
  }

  while (ip > d_input) {
    gpu_fse_encode_symbol(&bitC, &CState2, *--ip);
    if ((ip - d_input) & 1)
      gpu_bit_flush_bits(&bitC);
    if (ip <= d_input)
      break;
    gpu_fse_encode_symbol(&bitC, &CState1, *--ip);
    gpu_bit_flush_bits(&bitC);
  }

  gpu_fse_flush_state(&bitC, &CState2);
  gpu_fse_flush_state(&bitC, &CState1);
  *d_output_size = gpu_bit_close_stream(&bitC);
}

bool test_size(size_t size) {
  printf("\n=== Testing %zu bytes ===\n", size);

  // Generate random data
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<u8> input(size);
  for (size_t i = 0; i < size; i++) {
    input[i] = (u8)dist(rng);
  }

  // CPU reference
  unsigned count[256] = {0};
  for (size_t i = 0; i < size; i++)
    count[input[i]]++;

  unsigned maxSymbol = 0;
  for (int i = 255; i >= 0; i--) {
    if (count[i] > 0) {
      maxSymbol = i;
      break;
    }
  }

  short normalized[256];
  size_t tableLog =
      FSE_normalizeCount(normalized, TABLE_LOG, count, size, maxSymbol, 1);

  if (tableLog == 0) {
    printf("RLE mode (skipping)\n");
    return true;
  }

  size_t workspaceSize = FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbol, tableLog);
  void *workspace = malloc(workspaceSize);
  size_t ctableSize = FSE_CTABLE_SIZE(tableLog, maxSymbol);
  FSE_CTable *ctable = (FSE_CTable *)malloc(ctableSize);

  FSE_buildCTable_wksp(ctable, normalized, maxSymbol, tableLog, workspace,
                       workspaceSize);

  std::vector<u8> cpu_output(size * 2);
  size_t cpu_size = FSE_compress_usingCTable(
      cpu_output.data(), cpu_output.size(), input.data(), size, ctable);

  printf("CPU: %zu -> %zu bytes (%.1f%%)\n", size, cpu_size,
         100.0 * cpu_size / size);

  // GPU test
  u8 *d_input, *d_output;
  u16 *d_ctable;
  u32 *d_output_size;

  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size * 2);
  cudaMalloc(&d_ctable, ctableSize);
  cudaMalloc(&d_output_size, sizeof(u32));

  cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ctable, ctable, ctableSize, cudaMemcpyHostToDevice);

  gpu_fse_encode_kernel<<<1, 1>>>(d_input, size, d_ctable, d_output,
                                  d_output_size);
  cudaDeviceSynchronize();

  u32 gpu_size;
  std::vector<u8> gpu_output(size * 2);
  cudaMemcpy(&gpu_size, d_output_size, sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_output.data(), d_output, gpu_size, cudaMemcpyDeviceToHost);

  printf("GPU: %zu -> %u bytes (%.1f%%)\n", size, gpu_size,
         100.0 * gpu_size / size);

  // Compare
  bool match = (cpu_size == gpu_size);
  if (match) {
    for (size_t i = 0; i < cpu_size; i++) {
      if (cpu_output[i] != gpu_output[i]) {
        printf("❌ Mismatch at byte %zu: CPU=%02x GPU=%02x\n", i, cpu_output[i],
               gpu_output[i]);
        match = false;
        break;
      }
    }
  } else {
    printf("❌ Size mismatch: CPU=%zu GPU=%u\n", cpu_size, gpu_size);
  }

  if (match) {
    printf("✅ PASS\n");
  }

  free(workspace);
  free(ctable);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_ctable);
  cudaFree(d_output_size);

  return match;
}

int main() {
  printf("=== GPU FSE Encoder Scale Test ===\n");

  bool all_pass = true;
  all_pass &= test_size(10);   // Already tested
  all_pass &= test_size(100);  // Small
  all_pass &= test_size(1024); // 1KB
  all_pass &= test_size(4096); // 4KB (original test case)

  printf("\n=== Final Result ===\n");
  if (all_pass) {
    printf("✅ ALL TESTS PASSED!\n");
    printf("GPU FSE encoder is Zstandard-compatible at all tested sizes!\n");
  } else {
    printf("❌ Some tests failed\n");
  }

  return all_pass ? 0 : 1;
}
