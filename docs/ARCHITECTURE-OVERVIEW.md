# Architecture Overview: How CUDA-ZSTD Works

> *"Understanding the machine helps you use it better."*

## The Big Picture

CUDA-ZSTD is a high-performance compression library that runs ZSTD compression on your graphics card. Here's how data flows through the system:

```
Your Data  -->  [  GPU Compression Factory  ]  -->  Compressed Output
                        |
                +-----------------+
                | 10,000+ Workers |
                |  (CUDA cores)   |
                +-----------------+
```

## The Four-Layer Architecture

```
+------------------------------------------------------------------+
| LAYER 1: LANGUAGE BINDINGS AND C API                             |
|     "The Front Doors"                                            |
|                                                                  |
|     Multiple entry points for different languages and systems.   |
|     +------------+  +------------+  +------------+               |
|     |  Python    |  |  C API     |  | NVComp v5  |               |
|     |  Bindings  |  | (11 funcs) |  | (7 funcs)  |               |
|     +------------+  +------------+  +------------+               |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| LAYER 2: MANAGEMENT                                              |
|     "The Executives"                                             |
|                                                                  |
|     You talk to managers. They handle the details.               |
|     +------------+  +------------+  +------------+               |
|     |  Default   |  | Streaming  |  |   Batch    |               |
|     |  Manager   |  |  Manager   |  |  Manager   |               |
|     | (1 file)   |  | (chunks)   |  |(1000 files)|               |
|     +------------+  +------------+  +------------+               |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| LAYER 3: COMPRESSION PIPELINE                                    |
|     "The Assembly Line"                                          |
|                                                                  |
|     Data flows through specialized stations:                     |
|                                                                  |
|     [Find Patterns] -> [Optimize] -> [Encode] -> [Package]      |
|         (LZ77)          (Parse)      (FSE)     (Frame)           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| LAYER 4: GPU KERNELS                                             |
|     "The Workers"                                                |
|                                                                  |
|     47 specialized GPU programs doing the actual work            |
|     - Hash table builders                                        |
|     - Match finders                                              |
|     - Encoders and decoders                                      |
+------------------------------------------------------------------+
```


## Smart Router (Hybrid Execution)

To achieve the best latency *and* throughput, the Manager intelligently routes workloads:

```
Input Data
    |
    v
[ < 1MB? ] --YES--> [ CPU (libzstd) ] --> Low Latency for Small Files
    |
    NO
    |
    v
[ GPU Pipeline ] --> High Throughput for Large Files
```

- **Small Files (<1MB)**: Processed on CPU to avoid PCIe/Kernel launch overhead.
- **Large Files (>1MB)**: Processed on GPU to leverage massive parallelism.
- **Batches**: Always processed on GPU for aggregate throughput.

---

## The Compression Journey

When you compress data, it goes through **6 steps**:

### Step 1: Arrival
Your data arrives on the GPU. It is divided into manageable chunks (like cutting a long rope into pieces).

### Step 2: Pattern Finding (LZ77)
The GPU searches for repeated patterns. If "hello" appears 100 times, we only need to store it once and point to it later.

```
Original:  "hello hello hello hello"
Optimized: "hello" + [copy 3 more times]
```

### Step 3: Optimal Parsing
The best representation is determined. Sometimes a longer match saves more space than two short ones.

### Step 4: Sequence Encoding
Compression decisions are converted into a compact format.

### Step 5: Entropy Coding (FSE + Huffman)
Common patterns get short codes, rare ones get longer codes.

```
Very common:  "e" -> 2 bits
Common:       "t" -> 3 bits
Rare:         "q" -> 8 bits
Result: Much smaller file
```

### Step 6: Frame Assembly
Everything is wrapped in a proper ZSTD frame with headers and checksums.

---

## Where to Find Things

```
cuda-zstd/
+-- include/              <- Headers (the public API)
|   +-- cuda_zstd_manager.h  <- Start here! (C++ and C API)
|   +-- cuda_zstd_nvcomp.h   <- NVComp v5 C API
|   +-- cuda_zstd_types.h    <- Data types and error codes
|   +-- cuda_zstd_stream_pool.h <- Stream pool for concurrency
|   +-- ...                  <- 29 headers total
|
+-- src/                  <- Implementation
|   +-- cuda_zstd_manager.cu <- Manager implementations
|   +-- cuda_zstd_c_api.cpp  <- C API wrapper (212 lines)
|   +-- cuda_zstd_lz77.cu    <- Pattern finding
|   +-- cuda_zstd_fse.cu     <- Entropy coding
|   +-- ...                  <- 28 source files total
|
+-- python/               <- Python package (cuda-zstd)
|   +-- cuda_zstd/           <- Python bindings
|   +-- pyproject.toml       <- pip install -e .
|
+-- tests/                <- Test suite (67 tests, 100% passing)
|   +-- test_*.cu / test_*.cpp
|
+-- benchmarks/           <- Performance tests (30 executables)
|   +-- benchmark_*.cu
|
+-- docs/                 <- Documentation (31 files)
|   +-- *.md
|
+-- CMakeLists.txt        <- Build configuration
```

---

## The Key Players

| Component | What It Does | Analogy |
|:----------|:-------------|:--------|
| **Manager** | Orchestrates compression | The conductor of an orchestra |
| **LZ77** | Finds patterns | A detective finding clues |
| **FSE** | Encodes symbols | A translator (common = short, rare = long) |
| **Huffman** | Encodes literals | Another translator (for raw bytes) |
| **Memory Pool** | Manages GPU memory | A warehouse manager |
| **Stream Pool** | Manages CUDA streams | A traffic controller for concurrency |
| **XXHash** | Verifies integrity | A quality inspector |
| **C API** | Exposes C interface | The front desk for external callers |

---

## By the Numbers

| Metric | Count |
|:-------|:-----:|
| **Source Files** | 28 |
| **Header Files** | 29 |
| **GPU Kernels** | 47 |
| **Test Cases** | 67 (100% passing) |
| **Benchmark Executables** | 30 |
| **Documentation Files** | 31 |
| **Lines of Code** | ~62,000 |

---

## Where to Go Next

| I want to... | Read this |
|:-------------|:----------|
| Start using the library | [Quick Reference](QUICK-REFERENCE.md) |
| Compress many files fast | [Batch Processing](BATCH-PROCESSING.md) |
| Use the C API | [C API Reference](C-API-REFERENCE.md) |
| Use from Python | [Quick Reference - Python](QUICK-REFERENCE.md) |
| Understand the algorithms | [FSE Implementation](FSE-IMPLEMENTATION.md) |
| Debug a problem | [Debugging Guide](DEBUGGING-GUIDE.md) |

---

*"The best code is code you understand. Now you understand CUDA-ZSTD."*
