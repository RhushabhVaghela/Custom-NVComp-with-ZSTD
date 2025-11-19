# Test Results Summary

**Date:** 2025-11-19  
**Total Tests:** 31  
**Passed:** 11 (35%)  
**Failed:** 16 (52%)  
**Timeout:** 4 (13%)  

---

## ✅ PASSED Tests (11)

1. `level_nvcomp_demo` - NVCOMP integration demo
2. `simple_test` - Basic functionality test
3. `test_adaptive_level` - Adaptive compression level
4. `test_basic_profile` - Basic profiling
5. `test_memory_pool_deallocate_timeout` - Memory pool deallocation timeout handling
6. `test_memory_pool_lock_ordering` - Memory pool lock ordering
7. `test_memory_pool_lock_timeout` - Memory pool lock timeout handling
8. `test_metadata_roundtrip` - RFC 8878 frame header metadata
9. `test_multitable_fse` - Multi-table FSE compression
10. `test_stream_pool` - Stream pool management
11. `test_stream_pool_timeout` - Stream pool timeout handling

---

## ❌ FAILED Tests (16)

### Critical Issues (Segmentation Faults - 7 tests)

1. **test_correctness** (exit 139 - SIGSEGV)
   - Segfault after compress_sequences
   - Issue: Core dump during compression pipeline

2. **test_error_handling** (exit 139 - SIGSEGV)
   - Segfault during cleanup
   - CUDA errors: 700 (illegal memory access), 400 (invalid resource handle)

3. **test_fse_advanced** (exit 139 - SIGSEGV)
   - Segfault during FSE roundtrip test
   - Issue: Memory corruption in FSE encoding

4. **test_integration** (exit 139 - SIGSEGV)
   - Segfault during cleanup
   - CUDA errors: 716, 400 during stream destruction

5. **test_memory_pool** (exit 139 - SIGSEGV)
   - Segfault after stream-based allocations
   - Issue: Memory pool corruption

6. **test_performance** (exit 139 - SIGSEGV)
   - Segfault during cleanup
   - CUDA errors: 700, 400 during stream destruction

7. **test_find_matches_small** (exit 134 - SIGABRT)
   - malloc(): invalid size (unsorted)
   - Issue: Memory corruption in LZ77 matching

### Logic/Functionality Issues (5 tests)

8. **test_c_api** (exit 1)
   - StreamPool cleanup errors (CUDA error 4)
   - Issue: Stream destruction failures

9. **test_comprehensive_fallback** (exit 1)
   - 1/2 tests passed
   - Issue: Fallback strategies not working correctly

10. **test_dictionary** (exit 1)
    - Dictionary training failed
    - Issue: Dictionary training implementation incomplete

11. **test_dictionary_compression** (exit 1)
    - Dictionary training failed
    - Issue: Same as test_dictionary

12. **test_fallback_strategies** (exit 1)
    - 3/9 tests passed
    - Issue: Multiple fallback strategy failures

### FSE/Encoding Issues (2 tests)

13. **test_fse_advanced_function** (exit 134 - SIGABRT)
    - Assertion failed at line 63
    - Issue: FSE encoding/decoding mismatch

### NVCOMP/Stream Issues (2 tests)

14. **test_nvcomp_batch** (exit 1)
    - StreamPool cleanup errors (CUDA error 4)
    - Issue: Stream destruction failures

15. **test_roundtrip** (exit 1)
    - StreamPool cleanup errors (CUDA error 4)
    - Issue: Stream destruction failures

16. **test_streaming** (exit 1)
    - StreamPool cleanup errors (CUDA error 4)
    - Issue: Stream destruction failures

---

## ⏱️ TIMEOUT Tests (4)

1. **test_memory_pool_double_free** - Infinite loop/deadlock
2. **test_memory_pool_double_free_fixed** - Infinite loop/deadlock
3. **test_memory_pool_double_free_race** - Infinite loop/deadlock
4. **test_memory_pool_double_free_race_fixed** - Infinite loop/deadlock

---

## Root Cause Analysis

### Primary Issues:

1. **StreamPool Cleanup Errors (CUDA error 4)**
   - Affects: test_c_api, test_nvcomp_batch, test_roundtrip, test_streaming
   - Root cause: Streams being destroyed after CUDA context is torn down
   - Fix: Ensure proper cleanup order

2. **Memory Corruption in Compression Pipeline**
   - Affects: test_correctness, test_error_handling, test_integration, test_performance
   - Root cause: CUDA errors 700 (illegal memory access), 400 (invalid resource handle)
   - Fix: Review workspace allocation and stream synchronization

3. **Dictionary Training Not Implemented**
   - Affects: test_dictionary, test_dictionary_compression
   - Root cause: Dictionary training function returns error
   - Fix: Implement or stub dictionary training

4. **Memory Pool Double-Free Detection**
   - Affects: 4 timeout tests
   - Root cause: Infinite loops in double-free detection tests
   - Fix: Review test logic or disable if intentionally testing edge cases

5. **FSE Encoding Issues**
   - Affects: test_fse_advanced, test_fse_advanced_function
   - Root cause: Encoding/decoding mismatch or memory corruption
   - Fix: Review FSE implementation

---

## Recommended Fix Priority

### High Priority (Blocking Core Functionality):
1. Fix StreamPool cleanup order (affects 4 tests)
2. Fix memory corruption in compression pipeline (affects 4 tests)
3. Fix FSE encoding issues (affects 2 tests)

### Medium Priority (Feature Completeness):
4. Implement dictionary training (affects 2 tests)
5. Fix fallback strategies (affects 2 tests)

### Low Priority (Edge Case Testing):
6. Fix or disable double-free detection tests (affects 4 timeout tests)

---

## Next Steps

1. Fix StreamPool cleanup order
2. Add proper CUDA error checking and synchronization
3. Review workspace allocation logic
4. Implement dictionary training or provide stub
5. Fix FSE encoding/decoding
6. Review and fix fallback strategies
7. Address double-free detection test timeouts
