# CUDA-ZSTD Checksum and Data Integrity

## Overview

CUDA-ZSTD uses XXHash64 for fast data integrity verification, producing 8-byte checksums at GPU speeds.

## XXHash64 Algorithm

XXHash64 is a non-cryptographic hash function optimized for speed:
- **Throughput**: 15+ GB/s on GPU
- **Collision Resistance**: Excellent for data integrity
- **Output**: 64-bit hash

## API Reference

```cpp
#include "cuda_zstd_xxhash.h"

namespace cuda_zstd {
namespace xxhash {

// Compute hash on GPU
Status compute_xxhash64(
    const void* d_input,
    size_t size,
    uint64_t seed,
    uint64_t* d_result,
    cudaStream_t stream = 0
);

// Verify data against expected hash
Status verify_xxhash64(
    const void* d_input,
    size_t size,
    uint64_t expected_hash,
    bool* match,
    cudaStream_t stream = 0
);

} // namespace xxhash
} // namespace cuda_zstd
```

## Checksum Policies

```cpp
enum class ChecksumPolicy {
    NO_COMPUTE_NO_VERIFY,  // Fastest, no integrity check
    COMPUTE_NO_VERIFY,     // Compute on compress, skip on decompress
    NO_COMPUTE_VERIFY,     // Skip on compress, verify on decompress
    COMPUTE_AND_VERIFY     // Full integrity (default)
};
```

## Usage

### Enable Checksums (Default)
```cpp
CompressionConfig config;
config.checksum = ChecksumPolicy::COMPUTE_AND_VERIFY;

auto manager = cuda_zstd::create_manager(5);
manager->set_config(config);

// Compression appends checksum to frame
manager->compress(d_input, size, d_output, &out_size, ...);

// Decompression verifies checksum
Status status = manager->decompress(d_compressed, comp_size,
                                    d_output, &decom_size, ...);
if (status == Status::ERROR_CHECKSUM_MISMATCH) {
    // Data corrupted!
}
```

### Disable for Speed
```cpp
CompressionConfig config;
config.checksum = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
manager->set_config(config);
// ~5% throughput improvement
```

## Frame Structure with Checksum

```
ZSTD Frame with Checksum:
┌─────────────────────────┐
│ Magic Number (4 bytes)   │
├─────────────────────────┤
│ Frame Header             │
├─────────────────────────┤
│ Block 1                  │
├─────────────────────────┤
│ Block 2                  │
├─────────────────────────┤
│ ...                      │
├─────────────────────────┤
│ Checksum (4 bytes low)   │  ← XXHash64 truncated to 32-bit
└─────────────────────────┘
```

## Performance Impact

| Policy | Compress | Decompress |
|:-------|:--------:|:----------:|
| NO_COMPUTE_NO_VERIFY | 100% | 100% |
| COMPUTE_AND_VERIFY | 95% | 95% |

## Source Files

| File | Description |
|:-----|:------------|
| `src/cuda_zstd_xxhash.cu` | GPU XXHash implementation |
| `include/cuda_zstd_xxhash.h` | Public API |
| `tests/test_checksum_validation.cu` | Checksum tests |

## Related Documentation
- [ERROR-HANDLING.md](ERROR-HANDLING.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
