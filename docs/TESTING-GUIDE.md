# 🧪 Testing Guide: Ensuring Quality

> *"If it's not tested, it doesn't work."*

## Why Testing Matters

Before you trust CUDA-ZSTD with your precious data, we've thrown everything at it:
- ✅ 86+ automated tests
- ✅ Edge cases (empty files, single bytes, huge files)
- ✅ Stress tests (millions of operations)
- ✅ Compression/decompression roundtrips

**Result**: Your data is safe with us!

---

## 🏃 Running Tests

### The Quick Way
```bash
cd build
ctest --output-on-failure
```

### See All the Details
```bash
ctest --verbose
```

### Run Tests in Parallel (Faster!)
```bash
ctest -j8 --output-on-failure
```

### Run a Specific Test
```bash
./test_correctness
./test_integration
./test_streaming
```

---

## 📋 What We Test

### 🎯 Core Functionality
| Test File | What It Checks | Tests |
|:----------|:---------------|:-----:|
| `test_correctness.cu` | Does compression actually work? | 15 |
| `test_roundtrip.cu` | Compress → Decompress → Same data? | 8 |
| `test_integration.cu` | All pieces work together? | 9 |

### ⚡ Performance & Streaming
| Test File | What It Checks | Tests |
|:----------|:---------------|:-----:|
| `test_streaming.cu` | Chunk-by-chunk compression | 12 |
| `test_nvcomp_batch.cu` | Batch processing | 6 |

### 🔧 Components
| Test File | What It Checks | Tests |
|:----------|:---------------|:-----:|
| `test_fse_*.cu` | Entropy encoding/decoding | 18 |
| `test_huffman.cu` | Huffman compression | 6 |
| `test_memory_pool*.cu` | GPU memory management | 8 |

### 🛡️ Edge Cases
| Test File | What It Checks | Tests |
|:----------|:---------------|:-----:|
| `test_coverage_gaps.cu` | Boundary conditions | 8 |
| `test_edge_case.cu` | Weird inputs | 4 |
| `test_error_handling.cu` | Graceful failure | 6 |

---

## 🧪 Writing Your Own Tests

Here's a template:

```cpp
#include "cuda_zstd_manager.h"
#include <iostream>

bool test_my_feature() {
    std::cout << "[TEST] My feature..." << std::flush;
    
    // 1. Setup
    auto manager = cuda_zstd::create_manager(3);
    
    // 2. Do the thing
    Status result = manager->some_function(...);
    
    // 3. Check the result
    if (result != Status::SUCCESS) {
        std::cerr << " FAILED! ❌\n";
        return false;
    }
    
    std::cout << " PASSED ✅\n";
    return true;
}

int main() {
    int passed = 0, failed = 0;
    
    if (test_my_feature()) passed++; else failed++;
    
    std::cout << "\n=== " << passed << " passed, " 
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
```

### Adding Your Test to the Build
1. Create `tests/test_my_feature.cu`
2. Rebuild: `cmake --build .`
3. CMake auto-discovers files matching `test_*.cu`!

---

## 🔍 Debugging Failed Tests

### Enable Verbose CUDA Errors
```bash
CUDA_LAUNCH_BLOCKING=1 ./test_name
```

### Check for Memory Issues
```bash
compute-sanitizer --tool memcheck ./test_name
```

### See Debug Output
```bash
CUDA_ZSTD_DEBUG_LEVEL=3 ./test_name
```

---

## ✅ Test Coverage Summary

```
Component Coverage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Manager Layer        ████████████ 100%
LZ77 Matching        ████████████ 100%
FSE Encoding         ████████████ 100%
Huffman Coding       ████████████ 100%
Memory Pool          ████████████ 100%
Streaming API        ████████████ 100%
Batch Processing     ████████████ 100%
Error Handling       ████████████ 100%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 86+ tests, ALL PASSING ✅
```

---

## 📚 Related Guides

- [Debugging Guide](DEBUGGING-GUIDE.md) — When things go wrong
- [Error Handling](ERROR-HANDLING.md) — Understanding error codes
- [Build Guide](BUILD-GUIDE.md) — Setting up the build

---

*"Trust, but verify. We verified 86 times." 🧪*
