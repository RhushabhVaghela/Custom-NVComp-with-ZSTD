# 📚 CUDA-ZSTD Documentation

> *"Good documentation is like a good map—it gets you where you need to go."*

Welcome! Whether you're a developer integrating CUDA-ZSTD or just curious how GPU compression works, you're in the right place.

---

## 🚀 Quick Start

| I want to... | Start here |
|:-------------|:-----------|
| Get started in 5 minutes | [**Quick Reference**](QUICK-REFERENCE.md) |
| Build from source | [**Build Guide**](BUILD-GUIDE.md) |
| Understand how it works | [**Architecture Overview**](ARCHITECTURE-OVERVIEW.md) |

---

## 📖 Core Guides

### For Everyone
| Guide | What You'll Learn |
|:------|:------------------|
| 🏛️ [Architecture Overview](ARCHITECTURE-OVERVIEW.md) | How the whole system works |
| ⚡ [Performance Tuning](PERFORMANCE-TUNING.md) | Make your compression fly |
| 📋 [Quick Reference](QUICK-REFERENCE.md) | One-page cheat sheet |

### For Developers
| Guide | What You'll Learn |
|:------|:------------------|
| 🚀 [Batch Processing](BATCH-PROCESSING.md) | Compress 1000 files in milliseconds |
| 🌊 [Streaming API](STREAMING-API.md) | Process data as it flows |
| 📄 [C API Reference](C-API-REFERENCE.md) | Using the C interface |
| 🔗 [NVComp Integration](NVCOMP-INTEGRATION.md) | Drop-in nvCOMP replacement |

### For Debugging
| Guide | What You'll Learn |
|:------|:------------------|
| 🚨 [Error Handling](ERROR-HANDLING.md) | What errors mean & how to fix them |
| 🔍 [Debugging Guide](DEBUGGING-GUIDE.md) | When things go wrong |
| 🧪 [Testing Guide](TESTING-GUIDE.md) | Run and write tests |

---

## 🔧 Algorithm Deep-Dives

Want to understand the magic under the hood?

| Algorithm | What It Does |
|:----------|:-------------|
| [LZ77 Implementation](LZ77-IMPLEMENTATION.md) | Finding patterns in data |
| [FSE Implementation](FSE-IMPLEMENTATION.md) | Smart symbol encoding |
| [Huffman Implementation](HUFFMAN-IMPLEMENTATION.md) | Classic compression magic |
| [Dictionary Implementation](DICTIONARY-IMPLEMENTATION.md) | Pre-trained compression |
| [XXHash Implementation](XXHASH-IMPLEMENTATION.md) | Fast checksumming |
| [Sequence Implementation](SEQUENCE-IMPLEMENTATION.md) | Sequence encoding |

---

## 📊 Reference

| Document | Purpose |
|:---------|:--------|
| [Frame Format](FRAME-FORMAT.md) | ZSTD file structure |
| [Kernel Reference](KERNEL-REFERENCE.md) | All 47 GPU kernels |
| [Checksum Implementation](CHECKSUM-IMPLEMENTATION.md) | Data integrity |
| [Adaptive Level Selection](ADAPTIVE-LEVEL-SELECTION.md) | Auto-tuning |
| [Memory Pool](MEMORY-POOL-IMPLEMENTATION.md) | GPU memory management |
| [Manager Implementation](MANAGER-IMPLEMENTATION.md) | Core classes |

---

## 📊 By the Numbers

| Metric | Value |
|:-------|:-----:|
| Documentation files | 28 |
| Total lines | ~5,000+ |
| Code examples | 50+ |
| Diagrams | 25+ |

---

## 🎯 Can't Find What You Need?

- **General questions**: Start with [Architecture Overview](ARCHITECTURE-OVERVIEW.md)
- **Code examples**: Check [Quick Reference](QUICK-REFERENCE.md)
- **Something broke**: See [Error Handling](ERROR-HANDLING.md) or [Debugging Guide](DEBUGGING-GUIDE.md)

---

*Happy compressing! 🚀*
