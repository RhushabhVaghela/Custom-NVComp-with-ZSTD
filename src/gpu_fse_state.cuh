// GPU FSE State Machine - Zstandard Compatible
// Purpose: Match Zstandard's fse.h for FSE encoding
// Reference: build/_deps/zstd-src/lib/common/fse.h

#ifndef GPU_FSE_STATE_H
#define GPU_FSE_STATE_H

#include "cuda_zstd_types.h"
#include "gpu_bitstream.cuh"

namespace cuda_zstd {
namespace fse {

// GPU FSE Compression State (matches FSE_CState_t)
struct GPU_FSE_CState {
  u32 value;              // Current state value
  const void *stateTable; // Pointer to compression table (symbol transform)
};

// Symbol compression transform (matches FSE_symbolCompressionTransform)
struct GPU_FSE_SymbolTransform {
  i32 deltaFindState; // Signed offset for state transition
  u32 deltaNbBits;    // Packed: (maxBitsOut << 16) - minStatePlus
};

// Initialize FSE compression state (matches FSE_initCState2)
__device__ inline void
gpu_fse_init_state(GPU_FSE_CState *statePtr,
                   const GPU_FSE_SymbolTransform *ct, // Compression table
                   u32 symbol) {
  const GPU_FSE_SymbolTransform symbolTT = ct[symbol];
  const u16 *stateTable = (const u16 *)ct;

  // Initial state is at the beginning of the symbol's range
  // This matches FSE_initCState2 logic
  statePtr->value = (u32)stateTable[0]; // Start state (will be adjusted)
  statePtr->stateTable = ct;
}

// Encode one symbol (matches FSE_encodeSymbol)
// This is the CORE function that must match Zstandard exactly!
__device__ inline void
gpu_fse_encode_symbol(GPU_BitStream *bitC, GPU_FSE_CState *statePtr, u32 symbol,
                      const GPU_FSE_SymbolTransform *symbolTable,
                      const u16 *stateTable) {
  // Get symbol transformation info
  GPU_FSE_SymbolTransform const symbolTT = symbolTable[symbol];

  // Calculate number of bits to output
  // Formula from Zstandard FSE_encodeSymbol: nbBitsOut = (state + deltaNbBits) >> 16
  u32 const nbBitsOut = (statePtr->value + symbolTT.deltaNbBits) >> 16;

  // Output low bits of current state
  gpu_bit_add_bits(bitC, statePtr->value, nbBitsOut);

  // Calculate next state
  // Formula: stateTable[(value >> nbBitsOut) + deltaFindState]
  u32 const stateIndex =
      (statePtr->value >> nbBitsOut) + symbolTT.deltaFindState;
  statePtr->value = stateTable[stateIndex];
}

// Flush final state (matches FSE_flushCState)
__device__ inline void gpu_fse_flush_state(GPU_BitStream *bitC,
                                           const GPU_FSE_CState *statePtr,
                                           u32 tableLog) {
  // Write final state minus table size
  // The table size is (1 << tableLog)
  u32 const tableSize = 1u << tableLog;
  u32 const stateToWrite = statePtr->value - tableSize;

  gpu_bit_add_bits(bitC, stateToWrite, tableLog);
}

} // namespace fse
} // namespace cuda_zstd

#endif // GPU_FSE_STATE_H
