# Fix Plan for 25 Failing Tests - CUDA ZSTD Implementation

## Root Cause Analysis

Based on detailed test output analysis, the following root causes have been identified:

### PRIMARY ISSUE: FSE Sequence Decode Producing Zeros
**Files Affected:** `src/cuda_zstd_fse.cu`, `src/cuda_zstd_manager.cu`
**Evidence:**
```
[DEBUG_BUILD] Seq 0: LL=0, ML=0, OF=0, Ptr=0x90be20100
[DEBUG_COMPUTE] Finished. NumSeq=13, TotalLit=0, InputLit=768, TotalOut=768
```
The sequences are being decoded as all zeros instead of the correct values (e.g., LL=547, ML=256, OF=512).

**Root Cause:** The `k_decode_sequences_interleaved` kernel is not properly:
1. Reading initial states from the bitstream
2. Decoding FSE symbols from the interleaved streams
3. Outputting decoded values to the sequence arrays

### SECONDARY ISSUES:

1. **CUDA Illegal Memory Access in FSE Kernels** - Out-of-bounds access in FSE tables
2. **Memory Allocation Tracking Errors** - Workspace size calculations incorrect
3. **Streaming Context Initialization** - Stream pool creation failures

## Detailed Fix Plan

### Phase 1: Fix FSE Sequence Decode Kernel (CRITICAL)

#### 1.1 Fix Bitstream Reading
- Verify bitstream position initialization
- Check bit extraction for tableLog bits
- Ensure proper byte boundary handling

#### 1.2 Fix State Initialization
- Correct initialization of stateLL, stateOF, stateML
- Verify table selection based on mode byte
- Add bounds checking for state values

#### 1.3 Fix Symbol Decoding
- Correct FSE decoding formula: `symbol = d_state_to_symbol[state]`
- Proper state update: `state = nextStateTable[nbBits][state]`
- Verify repcode offset handling

#### 1.4 Add Bounds Checking
- Check bitstream position before each read
- Validate state values against table sizes
- Prevent out-of-bounds memory access

### Phase 2: Fix Memory Allocation Tracking

#### 2.1 Fix get_compress_temp_size
- Remove duplicate padding calculations
- Correct sequence buffer size estimation
- Fix per-block workspace scaling

#### 2.2 Fix get_decompress_temp_size
- Correct sequence buffer allocation
- Fix F