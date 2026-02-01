# CUDA ZSTD FSE Implementation - RFC 8878 Compliant

This directory contains a clean, RFC 8878-compliant FSE implementation.

## Design Principles

1. **Spec Compliance**: Follow RFC 8878 Section 4.1 exactly
2. **Simplicity**: Clear, maintainable code over micro-optimizations
3. **Correctness**: Extensive validation at each step
4. **Modularity**: Separate encode/decode with well-defined interfaces

## Files

- `cuda_zstd_fse_rfc.cu` - Main implementation
- `cuda_zstd_fse_rfc.h` - Interface header
- `FSE_TABLES.md` - Table construction details

## RFC 8878 Section 4.1 Summary

### FSE Encoding
1. Normalize symbol frequencies (powers of 2 table size)
2. Build distribution table using "spread" algorithm
3. For each symbol, track:
   - Current state (0 to TableSize-1)
   - Bits to write: `state & ((1 << nbBits) - 1)`
   - Next state: `newState = state >> nbBits + offset`

### FSE Decoding
1. Read initial state from bitstream (TableLog bits)
2. While not done:
   - Decode symbol from state via table lookup
   - Read bits: `bits = read(nbBits)`
   - Update state: `state = newState[state] + bits`

## Key Differences from Current Implementation

1. **Table Structure**: Single unified table (symbol, nbBits, newState) vs fragmented
2. **State Initialization**: Uses RFC-compliant starting state calculation
3. **Bitstream Layout**: Strict LSB-first as per spec
4. **Validation**: Checks at every step

## Testing

Run with: `./test_correctness`
Expected: All tests pass including 4KB
