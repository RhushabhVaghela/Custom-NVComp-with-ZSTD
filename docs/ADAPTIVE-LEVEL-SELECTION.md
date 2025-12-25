# CUDA-ZSTD Adaptive Compression Level Selection

## Overview

The Adaptive Level Selector automatically chooses the optimal compression level based on data characteristics, balancing throughput and compression ratio.

## Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                   AdaptiveLevelSelector                          │
├─────────────────────────────────────────────────────────────────┤
│  Input Data Sample (first 64KB)                                  │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Entropy Analysis                                         ││
│  │    - Byte frequency histogram                               ││
│  │    - Shannon entropy estimation                             ││
│  └─────────────────────────────────────────────────────────────┘│
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 2. Pattern Detection                                        ││
│  │    - Repeated substring analysis                            ││
│  │    - Run-length encoding potential                          ││
│  └─────────────────────────────────────────────────────────────┘│
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 3. Level Mapping                                            ││
│  │    - High entropy → Low level (1-3)                         ││
│  │    - Medium entropy → Medium level (4-9)                    ││
│  │    - Low entropy → High level (10-15)                       ││
│  └─────────────────────────────────────────────────────────────┘│
│         ↓                                                        │
│  Recommended Level (1-22)                                        │
└─────────────────────────────────────────────────────────────────┘
```

## API

```cpp
#include "cuda_zstd_adaptive.h"

class AdaptiveLevelSelector {
public:
    // Analyze data and recommend level
    static int recommend_level(
        const void* d_input,
        size_t input_size,
        cudaStream_t stream = 0
    );
    
    // Set target ratio (2.0-10.0)
    static void set_target_ratio(float ratio);
    
    // Set speed constraint (1-5, 1=fastest)
    static void set_speed_mode(int mode);
};
```

## Usage

```cpp
#include "cuda_zstd_adaptive.h"
#include "cuda_zstd_manager.h"

void compress_with_adaptive_level(const void* d_input, size_t size) {
    // Get recommended level
    int level = cuda_zstd::AdaptiveLevelSelector::recommend_level(
        d_input, size
    );
    
    printf("Recommended level: %d\n", level);
    
    // Create manager with recommended level
    auto manager = cuda_zstd::create_manager(level);
    
    // Compress...
}
```

## Level Mapping Table

| Entropy | Repetition | Recommended | Expected Ratio |
|:-------:|:----------:|:-----------:|:--------------:|
| High | Low | 1-2 | 1.5-2.0x |
| High | Medium | 3-4 | 2.0-2.5x |
| Medium | Low | 5-6 | 2.5-3.0x |
| Medium | Medium | 7-9 | 3.0-4.0x |
| Low | Medium | 10-12 | 4.0-5.0x |
| Low | High | 13-15 | 5.0-8.0x |

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_adaptive.cu` | Implementation |
| `include/cuda_zstd_adaptive.h` | Public API |
| `tests/test_adaptive_level.cu` | Tests |
| `tests/test_fse_adaptive.cu` | FSE adaptive tests |

## Related Documentation
- [PERFORMANCE-TUNING.md](PERFORMANCE-TUNING.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
