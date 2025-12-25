# CUDA-ZSTD Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
│                (User Code: C++/C API, Single/Streaming/Batch)               │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
┌─────────────────────────────────────────────────┴───────────────────────────┐
│                            MANAGER LAYER                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ DefaultZstdMgr  │  │ StreamingMgr    │  │ BatchManager    │            │
│  │ (Single-shot)   │  │ (Chunked)       │  │ (Parallel)      │            │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                    │                       │
│  ┌────────┴────────────────────┴────────────────────┴───────┐              │
│  │              AdaptiveLevelSelector                        │              │
│  │              MemoryPoolManager                            │              │
│  │              StreamPoolManager                            │              │
│  └──────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
┌─────────────────────────────────────────────────┴───────────────────────────┐
│                         COMPRESSION PIPELINE                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    LZ77      │→ │   Optimal    │→ │  Sequence    │→ │ FSE/Huffman  │   │
│  │  Matching    │  │   Parsing    │  │  Encoding    │  │  Encoding    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  Dictionary  │  │   XXHash64   │  │    Frame     │                     │
│  │   Training   │  │  Checksum    │  │   Assembly   │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────┬───────────────────────────┘
                                                  │
┌─────────────────────────────────────────────────┴───────────────────────────┐
│                           CUDA KERNEL LAYER                                  │
│                    47 GPU Kernels: Parallel Operations                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Hash/Chain Tables │ Match Finding │ Cost DP │ Entropy Coding │ I/O  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
cuda-zstd/
├── include/                    # Public headers
│   ├── cuda_zstd_manager.h     # Main API
│   ├── cuda_zstd_types.h       # Core types, Status enum
│   ├── cuda_zstd_fse.h         # FSE API
│   ├── cuda_zstd_huffman.h     # Huffman API
│   ├── cuda_zstd_lz77.h        # LZ77 API
│   ├── cuda_zstd_dictionary.h  # Dictionary API
│   ├── cuda_zstd_memory_pool.h # Memory pool
│   ├── cuda_zstd_nvcomp.h      # nvCOMP compatibility
│   └── error_context.h         # Error handling
│
├── src/                        # Implementation
│   ├── cuda_zstd_manager.cu    # Manager implementations
│   ├── cuda_zstd_fse.cu        # FSE kernels
│   ├── cuda_zstd_huffman.cu    # Huffman kernels
│   ├── cuda_zstd_lz77.cu       # LZ77 kernels
│   ├── cuda_zstd_sequence.cu   # Sequence encoding
│   ├── cuda_zstd_dictionary.cu # Dictionary training
│   ├── cuda_zstd_xxhash.cu     # XXHash implementation
│   ├── cuda_zstd_memory_pool_complex.cu
│   ├── cuda_zstd_nvcomp.cpp    # nvCOMP layer
│   └── cuda_zstd_c_api.cpp     # C API bindings
│
├── tests/                      # Test suite (86+ tests)
│   ├── test_correctness.cu
│   ├── test_integration.cu
│   ├── test_streaming.cu
│   └── ...
│
├── benchmarks/                 # Performance tests
│   ├── benchmark_batch_throughput.cu
│   └── run_performance_suite.cu
│
├── docs/                       # Documentation
│   ├── FSE-IMPLEMENTATION.md
│   ├── LZ77-IMPLEMENTATION.md
│   └── ...
│
└── CMakeLists.txt              # Build system
```

## Data Flow

### Compression Pipeline
```
Input Buffer (GPU)
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING                                                │
│    - Partition into 128KB blocks                                │
│    - Initialize workspace buffers                               │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. LZ77 MATCHING (per block)                                    │
│    - Build hash/chain tables (parallel)                         │
│    - Find all matches at each position                          │
│    - Store in d_matches[]                                       │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. OPTIMAL PARSING                                              │
│    - Initialize cost table (∞)                                  │
│    - Forward DP: compute minimum bit costs                      │
│    - Backtrack: extract literal/match sequence                  │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. SEQUENCE ENCODING                                            │
│    - Build sequences: (literal_len, match_len, offset)          │
│    - Apply repeat offset optimization                           │
│    - Count symbol frequencies                                   │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. ENTROPY CODING                                               │
│    - FSE encode: offsets, match lengths, literal lengths        │
│    - Huffman encode: literal bytes                              │
│    - Parallel chunked encoding                                  │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ 6. FRAME ASSEMBLY                                               │
│    - Write frame header (magic, sizes)                          │
│    - Concatenate block outputs                                  │
│    - Append XXHash64 checksum                                   │
└────────────────────────────────────────────────────────────────┘
     ↓
Output Buffer (GPU) - Valid ZSTD Frame
```

## Memory Layout

### Workspace Structure
```
┌─────────────────────────────────────────────────────────────┐
│ Workspace Partition (per compression call)                   │
├──────────────────────┬──────────────────────────────────────┤
│ Hash Table           │ 512 KB (131072 × 4 bytes)            │
├──────────────────────┼──────────────────────────────────────┤
│ Chain Table          │ 512 KB (131072 × 4 bytes)            │
├──────────────────────┼──────────────────────────────────────┤
│ Match Array          │ 2 MB (up to 131072 matches × 16B)    │
├──────────────────────┼──────────────────────────────────────┤
│ Cost Array           │ 2 MB (131073 costs × 16B)            │
├──────────────────────┼──────────────────────────────────────┤
│ Sequence Buffers     │ 1.5 MB (reverse buffers for backtrack)│
├──────────────────────┼──────────────────────────────────────┤
│ FSE Tables           │ 256 KB (3 tables × ~80KB each)       │
├──────────────────────┼──────────────────────────────────────┤
│ Huffman Table        │ 64 KB                                 │
├──────────────────────┼──────────────────────────────────────┤
│ Output Buffer        │ 1.5 × input size                      │
└──────────────────────┴──────────────────────────────────────┘
```

## Key Classes

| Class | Header | Responsibility |
|:------|:-------|:---------------|
| `DefaultZstdManager` | `cuda_zstd_manager.h` | Single-shot compress/decompress |
| `ZstdStreamingManager` | `cuda_zstd_manager.h` | Chunk-based streaming |
| `ZstdBatchManager` | `cuda_zstd_manager.h` | Parallel batch processing |
| `MemoryPoolManager` | `cuda_zstd_memory_pool.h` | GPU allocation reuse |
| `AdaptiveLevelSelector` | `cuda_zstd_adaptive.h` | Auto-select compression level |

## Source Files Summary

| Category | Files | Lines |
|:---------|:-----:|:-----:|
| Headers | 26 | ~8,000 |
| Sources | 30 | ~35,000 |
| Tests | 86 | ~15,000 |
| Docs | 18 | ~4,000 |
| **Total** | **160** | **~62,000** |

## Related Documentation
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
- [FSE-IMPLEMENTATION.md](FSE-IMPLEMENTATION.md)
- [LZ77-IMPLEMENTATION.md](LZ77-IMPLEMENTATION.md)
