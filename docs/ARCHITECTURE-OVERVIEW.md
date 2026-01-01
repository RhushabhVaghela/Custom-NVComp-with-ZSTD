# 🏛️ Architecture Overview: How CUDA-ZSTD Works

> *"Understanding the machine helps you use it better."*

## The Big Picture

CUDA-ZSTD is like a **high-speed compression factory** that runs on your graphics card. Here's how data flows through the system:

```
Your Data  ──▶  [  GPU Compression Factory  ]  ──▶  Compressed Output
                        │
                ┌───────┴───────┐
                │ 10,000+ Workers │
                │  (CUDA cores)   │
                └─────────────────┘
```

## 🏭 The Three-Layer Architecture

Think of it like a well-organized company:

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎯 LAYER 1: MANAGEMENT                                          │
│     "The Executives"                                             │
│                                                                  │
│     You talk to managers. They handle the details.              │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│     │  Default   │  │ Streaming  │  │   Batch    │            │
│     │  Manager   │  │  Manager   │  │  Manager   │            │
│     │ (1 file)   │  │ (chunks)   │  │(1000 files)│            │
│     └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⚙️ LAYER 2: COMPRESSION PIPELINE                                │
│     "The Assembly Line"                                          │
│                                                                  │
│     Data flows through specialized stations:                     │
│                                                                  │
│     [Find Patterns] → [Optimize] → [Encode] → [Package]        │
│         (LZ77)         (Parse)      (FSE)     (Frame)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 🔧 LAYER 3: GPU KERNELS                                         │
│     "The Workers"                                                │
│                                                                  │
│     47 specialized GPU programs doing the actual work           │
│     • Hash table builders                                       │
│     • Match finders                                             │
│     • Encoders & decoders                                       │
└─────────────────────────────────────────────────────────────────┘
```


## 🧠 Smart Router (Hybrid Execution)

To achieve the best latency *and* throughput, the Manager intelligently routes workloads:

```
Input Data
    │
    ▼
[ < 1MB? ] ──YES──▶ [ CPU (libzstd) ] ──▶ Low Latency for Small Files
    │
    NO
    │
    ▼
[ GPU Pipeline ] ──▶ High Throughput for Large Files
```

- **Small Files (<1MB)**: Processed on CPU to avoid PCIe/Kernel launch overhead.
- **Large Files (>1MB)**: Processed on GPU to leverage massive parallelism.
- **Batches**: Always processed on GPU for aggregate throughput.

---

## 🔄 The Compression Journey

When you compress data, it goes through **6 steps**:

### Step 1: 📥 Arrival
Your data arrives on the GPU. We divide it into manageable chunks (like cutting a long rope into pieces).

### Step 2: 🔍 Pattern Finding (LZ77)
The GPU searches for repeated patterns. If "hello" appears 100 times, we only need to store it once and point to it later!

```
Original:  "hello hello hello hello"
Optimized: "hello" + [copy 3 more times]
```

### Step 3: 🧮 Optimal Parsing
We figure out the **best** way to represent the data. Sometimes a longer match saves more space than two short ones.

### Step 4: 📝 Sequence Encoding
Convert our compression decisions into a compact format.

### Step 5: 🎲 Entropy Coding (FSE + Huffman)
This is where the magic happens! Common patterns get short codes, rare ones get longer codes.

```
Very common:  "e" → 2 bits
Common:       "t" → 3 bits
Rare:         "q" → 8 bits
Result: Much smaller file!
```

### Step 6: 📦 Frame Assembly
Wrap everything in a proper ZSTD frame with headers and checksums.

---

## 📁 Where to Find Things

```
cuda-zstd/
├── 📂 include/              ← Headers (the public API)
│   ├── cuda_zstd_manager.h  ← Start here!
│   ├── cuda_zstd_types.h    ← Data types & error codes
│   └── ...
│
├── 📂 src/                  ← Implementation (the secret sauce)
│   ├── cuda_zstd_manager.cu ← Manager implementations
│   ├── cuda_zstd_lz77.cu    ← Pattern finding
│   ├── cuda_zstd_fse.cu     ← Entropy coding
│   └── ...
│
├── 📂 tests/                ← Test suite (86+ tests)
│   └── test_*.cu
│
├── 📂 benchmarks/           ← Performance tests
│   └── benchmark_*.cu
│
├── 📂 docs/                 ← You are here! 📍
│   └── *.md
│
└── CMakeLists.txt           ← Build configuration
```

---

## 🧩 The Key Players

| Component | What It Does | Analogy |
|:----------|:-------------|:--------|
| **Manager** | Orchestrates compression | The conductor of an orchestra |
| **LZ77** | Finds patterns | A detective finding clues |
| **FSE** | Encodes symbols | A translator (common = short, rare = long) |
| **Huffman** | Encodes literals | Another translator (for raw bytes) |
| **Memory Pool** | Manages GPU memory | A warehouse manager |
| **XXHash** | Verifies integrity | A quality inspector |

---

## 📊 By the Numbers

| Metric | Count |
|:-------|:-----:|
| **Source Files** | 30 |
| **Header Files** | 26 |
| **GPU Kernels** | 47 |
| **Test Cases** | 86+ |
| **Lines of Code** | ~62,000 |

---

## 🎓 Where to Go Next

| I want to... | Read this |
|:-------------|:----------|
| Start using the library | [Quick Reference](QUICK-REFERENCE.md) |
| Compress many files fast | [Batch Processing](BATCH-PROCESSING.md) |
| Understand the algorithms | [FSE Implementation](FSE-IMPLEMENTATION.md) |
| Debug a problem | [Debugging Guide](DEBUGGING-GUIDE.md) |

---

*"The best code is code you understand. Now you understand CUDA-ZSTD!" 🎉*
