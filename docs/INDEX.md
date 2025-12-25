# CUDA-ZSTD Documentation Index

## 📚 Documentation Overview

Welcome to the CUDA-ZSTD documentation. This comprehensive guide covers all aspects of the GPU-accelerated Zstandard compression library.

---

## Quick Start
- [**Quick Reference Card**](QUICK-REFERENCE.md) - One-page cheat sheet
- [**Build Guide**](BUILD-GUIDE.md) - Installation and compilation

---

## Architecture & Design

| Document | Description |
|:---------|:------------|
| [Architecture Overview](ARCHITECTURE-OVERVIEW.md) | System design, data flow, directory structure |
| [Frame Format](FRAME-FORMAT.md) | RFC 8878 compliant ZSTD frame structure |
| [Kernel Reference](KERNEL-REFERENCE.md) | All 47 GPU kernel descriptions |

---

## Core Components

| Document | Description |
|:---------|:------------|
| [Manager Implementation](MANAGER-IMPLEMENTATION.md) | DefaultZstdManager, StreamingManager, BatchManager |
| [LZ77 Implementation](LZ77-IMPLEMENTATION.md) | Hash tables, match finding, optimal parsing |
| [FSE Implementation](FSE-IMPLEMENTATION.md) | Finite State Entropy encoding/decoding |
| [Huffman Implementation](HUFFMAN-IMPLEMENTATION.md) | Huffman tree and coding |
| [Sequence Implementation](SEQUENCE-IMPLEMENTATION.md) | Sequence encoding |
| [Dictionary Implementation](DICTIONARY-IMPLEMENTATION.md) | COVER algorithm, dictionary training |
| [XXHash Implementation](XXHASH-IMPLEMENTATION.md) | Fast checksumming |

---

## APIs & Integration

| Document | Description |
|:---------|:------------|
| [C API Reference](C-API-REFERENCE.md) | Complete C language bindings |
| [Streaming API](STREAMING-API.md) | Chunk-based streaming compression |
| [Batch Processing](BATCH-PROCESSING.md) | Parallel batch operations (>60 GB/s) |
| [NVComp Integration](NVCOMP-INTEGRATION.md) | nvCOMP compatibility layer |

---

## Performance & Optimization

| Document | Description |
|:---------|:------------|
| [Performance Tuning](PERFORMANCE-TUNING.md) | Optimization strategies |
| [Memory Pool Implementation](MEMORY-POOL-IMPLEMENTATION.md) | GPU memory management |
| [Adaptive Level Selection](ADAPTIVE-LEVEL-SELECTION.md) | Auto compression level selection |
| [Stream Optimization](STREAM-OPTIMIZATION.md) | CUDA stream best practices |
| [Hash Table Optimization](HASH_TABLE_OPTIMIZATION.md) | Hash table tuning |
| [Checksum Implementation](CHECKSUM-IMPLEMENTATION.md) | XXHash64 data integrity |

---

## Development

| Document | Description |
|:---------|:------------|
| [Testing Guide](TESTING-GUIDE.md) | 86+ test suite documentation |
| [Debugging Guide](DEBUGGING-GUIDE.md) | Troubleshooting and profiling |
| [Error Handling](ERROR-HANDLING.md) | 18 error codes and recovery |

---

## Advanced Topics

| Document | Description |
|:---------|:------------|
| [Fallback Strategies](FALLBACK_STRATEGIES_IMPLEMENTATION.md) | Error recovery mechanisms |
| [Alternative Allocation Strategies](ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md) | Memory allocation alternatives |

---

## Document Count: 24 comprehensive guides

**Total Documentation**: ~4,000+ lines covering all features, APIs, and implementation details.
