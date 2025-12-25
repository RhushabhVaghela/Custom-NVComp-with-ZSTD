# CUDA-ZSTD GPU Kernel Reference

## Overview

CUDA-ZSTD implements 47 GPU kernels for parallel compression and decompression. This document catalogs all kernels with their purpose and configuration.

## Kernel Categories

### 1. Hash Table Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `init_hash_table_kernel` | `cuda_zstd_lz77.cu` | Initialize hash table to 0xFFFFFFFF |
| `build_hash_chains_kernel` | `cuda_zstd_lz77.cu` | Build hash chains for match finding |
| `radix_sort_bucket_kernel` | `cuda_zstd_lz77.cu` | Sort hash entries for coalesced access |

**Configuration:**
```cpp
dim3 blocks((num_entries + 255) / 256);
dim3 threads(256);
```

### 2. LZ77 Match Finding Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `parallel_find_all_matches_kernel` | `cuda_zstd_lz77.cu` | Find matches at all positions |
| `validate_matches_kernel` | `cuda_zstd_lz77.cu` | Verify match integrity |
| `extend_matches_kernel` | `cuda_zstd_lz77.cu` | Extend match lengths |

**Configuration:**
```cpp
dim3 blocks((input_size + 127) / 128);  // 2KB tiles
dim3 threads(128);
__shared__ uint8_t tile[2048 + 256];     // Tile + overlap
```

### 3. Optimal Parsing Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `initialize_costs_kernel` | `cuda_zstd_lz77.cu` | Set cost array to infinity |
| `optimal_parse_kernel` | `cuda_zstd_lz77.cu` | Forward DP pass |
| `backtrack_kernel` | `cuda_zstd_lz77.cu` | Extract optimal sequence |

**Configuration:**
```cpp
// Sequential per block (wavefront pattern)
dim3 blocks(1);
dim3 threads(256);
```

### 4. Sequence Encoding Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `count_sequences_kernel` | `cuda_zstd_sequence.cu` | Count literals/matches |
| `compress_sequences_kernel` | `cuda_zstd_sequence.cu` | Build sequence array |
| `extract_literals_kernel` | `cuda_zstd_sequence.cu` | Copy literal bytes |

### 5. FSE Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `fse_count_symbols_kernel` | `cuda_zstd_fse.cu` | Symbol frequency histogram |
| `fse_normalize_kernel` | `cuda_zstd_fse.cu` | Normalize probabilities |
| `fse_build_table_kernel` | `cuda_zstd_fse.cu` | Build encoding table |
| `fse_encode_kernel` | `cuda_zstd_fse.cu` | Encode symbols to bits |
| `fse_decode_kernel` | `cuda_zstd_fse.cu` | Decode bits to symbols |
| `fse_compute_states_kernel` | `cuda_zstd_fse.cu` | State machine computation |

### 6. Huffman Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `huffman_count_symbols_kernel` | `cuda_zstd_huffman.cu` | Byte histogram |
| `huffman_build_tree_kernel` | `cuda_zstd_huffman.cu` | Build Huffman tree |
| `huffman_build_codes_kernel` | `cuda_zstd_huffman.cu` | Generate code table |
| `huffman_encode_kernel` | `cuda_zstd_huffman.cu` | Encode literals |
| `huffman_decode_kernel` | `cuda_zstd_huffman.cu` | Decode literals |

### 7. XXHash Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `xxhash64_kernel` | `cuda_zstd_xxhash.cu` | Compute 64-bit hash |
| `xxhash64_block_kernel` | `cuda_zstd_xxhash.cu` | Block-level hashing |

### 8. Dictionary Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `dmer_extract_kernel` | `cuda_zstd_dictionary.cu` | Extract d-mers |
| `cover_score_kernel` | `cuda_zstd_dictionary.cu` | Score segments |
| `select_segments_kernel` | `cuda_zstd_dictionary.cu` | Select best segments |

### 9. Utility Kernels

| Kernel | File | Description |
|:-------|:-----|:------------|
| `memset_kernel` | `cuda_zstd_utils.cu` | Fast GPU memset |
| `memcpy_kernel` | `cuda_zstd_utils.cu` | Device-to-device copy |
| `prefix_sum_kernel` | `cuda_zstd_utils.cu` | Parallel prefix sum |

## Shared Memory Usage

| Kernel | Shared Memory | Purpose |
|:-------|:-------------:|:--------|
| `parallel_find_all_matches` | 2304 B | Input tile |
| `huffman_build_tree` | 4096 B | Tree nodes |
| `fse_build_table` | 8192 B | State table |
| `count_sequences` | 1024 B | Histogram |

## Occupancy Guidelines

| Compute Capability | Recommended Threads | Registers/Thread |
|:------------------:|:-------------------:|:----------------:|
| 7.0 (Volta) | 128-256 | 32 |
| 7.5 (Turing) | 128-256 | 32 |
| 8.0 (Ampere) | 256-512 | 64 |
| 8.6 (Ada) | 256-512 | 64 |

## Source Files

| File | Kernels |
|:-----|:-------:|
| `cuda_zstd_lz77.cu` | 12 |
| `cuda_zstd_fse.cu` | 10 |
| `cuda_zstd_huffman.cu` | 8 |
| `cuda_zstd_sequence.cu` | 5 |
| `cuda_zstd_dictionary.cu` | 6 |
| `cuda_zstd_xxhash.cu` | 3 |
| `cuda_zstd_utils.cu` | 3 |

## Related Documentation
- [LZ77-IMPLEMENTATION.md](LZ77-IMPLEMENTATION.md)
- [FSE-IMPLEMENTATION.md](FSE-IMPLEMENTATION.md)
- [HUFFMAN-IMPLEMENTATION.md](HUFFMAN-IMPLEMENTATION.md)
