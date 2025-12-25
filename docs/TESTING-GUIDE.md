# CUDA-ZSTD Testing Guide

## Overview

The testing infrastructure provides comprehensive validation of all components with 86+ unit tests, covering correctness, performance, edge cases, and stress scenarios.

## Test Categories

### 1. Core Functionality Tests

| Test File | Coverage | Tests |
|:----------|:---------|:-----:|
| `test_correctness.cu` | RFC 8878 compliance | 15 |
| `test_roundtrip.cu` | Compress/decompress cycle | 8 |
| `test_integration.cu` | Full E2E workflow | 9 |
| `test_streaming.cu` | Streaming API | 12 |

### 2. Component Tests

| Test File | Coverage | Tests |
|:----------|:---------|:-----:|
| `test_fse_*.cu` | FSE encoding/decoding | 18 |
| `test_huffman.cu` | Huffman compression | 6 |
| `test_lz77_comprehensive.cu` | LZ77 matching | 8 |
| `test_sequence_encoder.cu` | Sequence encoding | 5 |
| `test_hash_comprehensive.cu` | Hash tables | 6 |

### 3. Infrastructure Tests

| Test File | Coverage | Tests |
|:----------|:---------|:-----:|
| `test_memory_pool*.cu` | Memory management | 8 |
| `test_stream_pool.cu` | Stream management | 4 |
| `test_error_handling.cu` | Error codes | 6 |
| `test_c_api.c` | C API bindings | 5 |

### 4. Edge Case Tests

| Test File | Coverage | Tests |
|:----------|:---------|:-----:|
| `test_coverage_gaps.cu` | Boundary conditions | 8 |
| `test_edge_case.cu` | Corner cases | 4 |
| `test_data_integrity_comprehensive.cu` | Data validation | 6 |

## Running Tests

### All Tests
```bash
cd build
ctest --output-on-failure
```

### Specific Test
```bash
./test_correctness
./test_integration
```

### Parallel Execution
```bash
ctest -j8 --output-on-failure
```

### By Label
```bash
ctest -L unittest          # Unit tests only
ctest -L benchmark         # Benchmarks only
```

## Test Structure

### Standard Test Pattern
```cpp
#include "cuda_zstd_manager.h"
#include <iostream>

bool test_basic_compression() {
    std::cout << "[TEST] Basic compression..." << std::flush;
    
    // Setup
    auto manager = cuda_zstd::create_manager(3);
    
    // Execute
    size_t compressed_size;
    Status status = manager->compress(...);
    
    // Verify
    if (status != Status::SUCCESS) {
        std::cerr << " FAILED: " << status_to_string(status) << std::endl;
        return false;
    }
    
    std::cout << " PASSED" << std::endl;
    return true;
}

int main() {
    int passed = 0, failed = 0;
    
    if (test_basic_compression()) passed++; else failed++;
    // ... more tests ...
    
    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
```

## Test Data Generation

```cpp
// Random data
std::vector<uint8_t> generate_random(size_t size) {
    std::vector<uint8_t> data(size);
    std::mt19937 rng(42);  // Reproducible
    std::generate(data.begin(), data.end(), 
                  [&]() { return rng() % 256; });
    return data;
}

// Compressible data (repeating patterns)
std::vector<uint8_t> generate_compressible(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = "The quick brown fox "[i % 20];
    }
    return data;
}

// Edge case: all zeros
std::vector<uint8_t> generate_zeros(size_t size) {
    return std::vector<uint8_t>(size, 0);
}
```

## Debugging Failed Tests

### Enable Verbose Output
```bash
CUDA_ZSTD_DEBUG_LEVEL=3 ./test_name
```

### CUDA Error Checking
```bash
CUDA_LAUNCH_BLOCKING=1 ./test_name
```

### Memory Checking
```bash
compute-sanitizer --tool memcheck ./test_name
```

## Adding New Tests

1. Create file: `tests/test_feature_name.cu`
2. Follow naming convention: `test_*.cu`
3. CMake auto-discovers via glob pattern
4. Rebuild: `cmake --build . --target test_feature_name`

## Source Files

| File | Description |
|:-----|:------------|
| `tests/test_*.cu` | All test files |
| `tests/cuda_error_checking.h` | CUDA test utilities |
| `CMakeLists.txt:138-180` | Test registration |

## Related Documentation
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
