# CUDA-ZSTD Memory Pool Implementation Guide

## Overview

The Memory Pool system provides high-performance GPU memory allocation and reuse, reducing allocation overhead by 20-30% in high-throughput scenarios.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryPoolManager                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Small Pool  │  │ Medium Pool │  │ Large Pool  │         │
│  │  (<64KB)    │  │ (64KB-1MB)  │  │  (>1MB)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  - Thread-safe allocation/deallocation                      │
│  - Automatic coalescing of adjacent free blocks             │
│  - Best-fit allocation strategy                             │
│  - Configurable pool sizes and growth policies              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. MemoryPoolManager Class
**Location:** `src/cuda_zstd_memory_pool_complex.cu`

```cpp
class MemoryPoolManager {
public:
    // Singleton access
    static MemoryPoolManager& get_instance();
    
    // Core allocation
    void* allocate(size_t size, cudaStream_t stream = 0);
    void deallocate(void* ptr, cudaStream_t stream = 0);
    
    // Pool management
    void reset();                    // Clear all pools
    void trim();                     // Release unused memory
    size_t get_allocated_size();     // Total allocated bytes
    size_t get_pool_size();          // Total pool capacity
};
```

### 2. Allocation Strategies

| Strategy | Description | Use Case |
|:---------|:------------|:---------|
| **Best-Fit** | Find smallest block that fits | Default, minimizes fragmentation |
| **First-Fit** | Use first available block | Faster, higher fragmentation |
| **Segregated** | Size-class pools | Predictable allocation patterns |

### 3. Block Management

```cpp
struct PoolBlock {
    void* ptr;           // GPU memory pointer
    size_t size;         // Block size in bytes
    bool is_free;        // Availability flag
    uint64_t timestamp;  // For LRU eviction
};
```

## Usage Examples

### Basic Allocation
```cpp
#include "cuda_zstd_memory_pool.h"

auto& pool = cuda_zstd::MemoryPoolManager::get_instance();

// Allocate 1MB
void* buffer = pool.allocate(1024 * 1024);

// Use buffer...

// Return to pool (not freed, reused later)
pool.deallocate(buffer);
```

### Stream-Ordered Allocation
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Allocations tied to stream ordering
void* a = pool.allocate(1024, stream);
void* b = pool.allocate(2048, stream);

// Deallocate in any order - pool handles dependencies
pool.deallocate(b, stream);
pool.deallocate(a, stream);
```

### Bulk Operations
```cpp
// Pre-warm pool for known workload
pool.reserve(100 * 1024 * 1024);  // Reserve 100MB

// After batch processing, release excess
pool.trim();
```

## Configuration

### Pool Size Limits
```cpp
// Set via environment variables
CUDA_ZSTD_POOL_MAX_SIZE=2147483648    // 2GB max pool
CUDA_ZSTD_POOL_INITIAL_SIZE=268435456 // 256MB initial
CUDA_ZSTD_POOL_GROWTH_FACTOR=2.0      // Double on growth
```

### Thread Safety
- All operations are mutex-protected
- Per-stream allocation queues for async safety
- Lock-free fast path for common sizes

## Performance Characteristics

| Operation | Time Complexity | Typical Latency |
|:----------|:---------------:|:---------------:|
| Allocate (cache hit) | O(1) | <1µs |
| Allocate (cache miss) | O(n) | ~10µs |
| Deallocate | O(1) | <1µs |
| Trim | O(n) | ~100µs |

### Memory Overhead
- Per-block metadata: 32 bytes
- Pool management: ~1% of total size
- Fragmentation: Typically <5% with best-fit

## Debugging

### Enable Pool Diagnostics
```cpp
// Print pool statistics
pool.print_stats();

// Output:
// MemoryPool Stats:
//   Total Allocated: 128.5 MB
//   Total Capacity:  256.0 MB
//   Free Blocks:     47
//   Fragmentation:   3.2%
```

### Detect Leaks
```cpp
// At application exit
if (pool.get_allocated_size() > 0) {
    fprintf(stderr, "WARNING: %zu bytes still allocated\n",
            pool.get_allocated_size());
    pool.print_allocated_blocks();  // Show leak sources
}
```

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_memory_pool.h` | Public API header |
| `include/cuda_zstd_memory_pool_enhanced.h` | Extended features |
| `src/cuda_zstd_memory_pool_complex.cu` | Implementation |
| `tests/test_memory_pool.cu` | Unit tests |
| `tests/test_memory_pool_*.cu` | Edge case tests |

## Related Documentation
- [ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md](ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
