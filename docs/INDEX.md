# CUDA-ZSTD Documentation

> Complete documentation for the GPU-accelerated ZSTD compression library.

Welcome! Whether you're a developer integrating CUDA-ZSTD or just curious how GPU compression works, you're in the right place.

---

## Quick Start

| I want to... | Start here |
|:-------------|:-----------|
| Get started in 5 minutes | [**Quick Reference**](QUICK-REFERENCE.md) |
| Build from source | [**Build Guide**](BUILD-GUIDE.md) |
| Understand how it works | [**Architecture Overview**](ARCHITECTURE-OVERVIEW.md) |

---

## Core Guides

### For Everyone
| Guide | What You'll Learn |
|:------|:------------------|
| [Architecture Overview](ARCHITECTURE-OVERVIEW.md) | How the whole system works |
| [Performance Tuning](PERFORMANCE-TUNING.md) | Optimize compression throughput |
| [Quick Reference](QUICK-REFERENCE.md) | One-page cheat sheet |

### For Developers
| Guide | What You'll Learn |
|:------|:------------------|
| [Batch Processing](BATCH-PROCESSING.md) | Compress many files in milliseconds |
| [Streaming API](STREAMING-API.md) | Process data as it flows |
| [C API Reference](C-API-REFERENCE.md) | Using the C interface (11 functions) |
| [NVComp Integration](NVCOMP-INTEGRATION.md) | Drop-in nvCOMP v5 replacement |

### For Debugging
| Guide | What You'll Learn |
|:------|:------------------|
| [Error Handling](ERROR-HANDLING.md) | What errors mean and how to fix them |
| [Debugging Guide](DEBUGGING-GUIDE.md) | When things go wrong |
| [Testing Guide](TESTING-GUIDE.md) | Run and write tests |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |

---

## Algorithm Deep-Dives

| Algorithm | What It Does |
|:----------|:-------------|
| [LZ77 Implementation](LZ77-IMPLEMENTATION.md) | Finding patterns in data |
| [FSE Implementation](FSE-IMPLEMENTATION.md) | Finite State Entropy encoding |
| [Huffman Implementation](HUFFMAN-IMPLEMENTATION.md) | Huffman coding for literals |
| [Dictionary Implementation](DICTIONARY-IMPLEMENTATION.md) | Pre-trained compression dictionaries |
| [XXHash Implementation](XXHASH-IMPLEMENTATION.md) | Fast checksumming |
| [Sequence Implementation](SEQUENCE-IMPLEMENTATION.md) | Sequence encoding |

---

## Reference

| Document | Purpose |
|:---------|:--------|
| [Frame Format](FRAME-FORMAT.md) | ZSTD file structure |
| [Kernel Reference](KERNEL-REFERENCE.md) | All 47 GPU kernels |
| [Checksum Implementation](CHECKSUM-IMPLEMENTATION.md) | Data integrity |
| [Adaptive Level Selection](ADAPTIVE-LEVEL-SELECTION.md) | Auto-tuning compression levels |
| [Memory Pool](MEMORY-POOL-IMPLEMENTATION.md) | GPU memory management |
| [Manager Implementation](MANAGER-IMPLEMENTATION.md) | Core manager classes |

---

## Optimization and Advanced Topics

| Document | Purpose |
|:---------|:--------|
| [Stream Optimization](STREAM-OPTIMIZATION.md) | CUDA stream optimization techniques |
| [Stream-Based Parallelism](stream_based_parallelism.md) | Parallelism via CUDA streams |
| [Hash Table Optimization](HASH_TABLE_OPTIMIZATION.md) | Hash table GPU optimization |
| [Fallback Strategies](FALLBACK_STRATEGIES_IMPLEMENTATION.md) | CPU/GPU fallback implementation |
| [Alternative Allocation Strategies](ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md) | Memory allocation strategies |

---

## Python Package

CUDA-ZSTD includes a Python wrapper (`cuda-zstd`) providing access to compress, decompress, batch, and manager APIs from Python. See the `python/` directory or install with:

```bash
cd python/
pip install -e .
```

---

## By the Numbers

| Metric | Value |
|:-------|:-----:|
| Documentation files | 31 |
| Test files | 67 (100% passing) |
| Benchmark executables | 30 |
| GPU Kernels | 47 |
| Header files | 29 |
| Source files | 28 |
| Compression levels | 1-22 |
| C API functions | 11 |
| NVComp v5 C API functions | 7 |

---

## Can't Find What You Need?

- **General questions**: Start with [Architecture Overview](ARCHITECTURE-OVERVIEW.md)
- **Code examples**: Check [Quick Reference](QUICK-REFERENCE.md)
- **Something broke**: See [Error Handling](ERROR-HANDLING.md) or [Debugging Guide](DEBUGGING-GUIDE.md)
- **Build problems**: See [Build Guide](BUILD-GUIDE.md) or [Troubleshooting](TROUBLESHOOTING.md)
