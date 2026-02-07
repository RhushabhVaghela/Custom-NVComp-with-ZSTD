# CUDA-ZSTD: GPU-Accelerated Zstandard Compression

```
   ____  _   _  ____    _      _________ _____ ____  
  / ___|| | | ||  _ \  / \    |__  / ___|_   _|  _ \ 
 | |    | | | || | | |/ _ \     / /\___ \ | | | | | |
 | |___ | |_| || |_| / ___ \   / /_ ___) || | | |_| |
  \____| \___/ |____/_/   \_\ /____|____/ |_| |____/ 
                                                       
        GPU-Accelerated Zstandard Compression
```

[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake)](https://cmake.org/)
[![RFC 8878](https://img.shields.io/badge/RFC-8878-orange.svg)](https://datatracker.ietf.org/doc/html/rfc8878)
[![Tests](https://img.shields.io/badge/Tests-67%2F67%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A GPU-accelerated implementation of the Zstandard (RFC 8878) compression algorithm, built from the ground up in CUDA C++. The entire compression pipeline -- LZ77 match finding, optimal parsing, FSE entropy coding, Huffman coding, and frame assembly -- runs as native CUDA kernels. This library is a drop-in replacement for CPU `libzstd` with massive batch throughput gains, reaching **9.81 GB/s** batch compression on an RTX 5080 Laptop GPU.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [C++ Simple Compress/Decompress](#c-simple-compressdecompress)
  - [C++ Batch Processing](#c-batch-processing)
  - [C++ Streaming](#c-streaming)
  - [C++ Dictionary Compression](#c-dictionary-compression)
  - [C API](#c-api-usage)
  - [Python](#python-usage)
- [API Reference Summary](#api-reference-summary)
- [Compression Levels](#compression-levels)
- [Benchmark Results](#benchmark-results)
- [Python Package](#python-package)
- [Configuration Options](#configuration-options)
- [How It Works](#how-it-works)
- [User Options](#user-options)
- [Build Options](#build-options)
- [Testing](#testing)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Key Features

### Core Compression

| Feature | Description |
|---------|-------------|
| **Full RFC 8878 Compliance** | Frame headers, magic numbers, block types, FSE state ordering, RepCode logic -- output is decompressible by standard `libzstd` |
| **22 Compression Levels** | Complete level 1-22 range with configurable speed/ratio tradeoff |
| **8 Compression Strategies** | FAST, DFAST, GREEDY, LAZY, LAZY2, BTLAZY2, BTOPT, BTULTRA |
| **GPU/CPU Hybrid Routing** | Auto-selects execution path based on input size (CPU for small, GPU for large) |
| **Batch Processing** | Compress/decompress N independent items in parallel on the GPU |
| **Streaming API** | Chunked frame compression for arbitrarily large data |
| **Dictionary Compression** | COVER algorithm training on the GPU; per-dictionary ID tracking |
| **Adaptive Level Selection** | Automatically picks compression level based on data characteristics |
| **XXHash64/32 Checksumming** | Optional data integrity verification per RFC 8878 |

### APIs and Integration

| Feature | Description |
|---------|-------------|
| **C++ API** | `ZstdManager`, `ZstdBatchManager`, `ZstdStreamingManager` with RAII resource management |
| **C API (FFI-Ready)** | 11 `extern "C"` functions for calling from C, Python, Rust, Go, etc. |
| **NVCOMP v5 Compatible API** | `NvcompV5BatchManager` with 7 C functions for drop-in nvCOMP replacement |
| **Inference API** | Pre-allocated buffers, zero-malloc decompression, async no-sync for ML pipelines |
| **Python Package** | `cuda_zstd` module with `compress()`, `decompress()`, batch, and Manager class |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **47 CUDA Kernels** | Across 7 categories: LZ77, FSE, Huffman, Sequence, Dictionary, XXHash, Utils |
| **GPU Memory Pooling** | Allocation reuse with fallback strategies to reduce `cudaMalloc` overhead |
| **RAII Everywhere** | `CudaDevicePtr<T>`, `SequenceContext`, `StreamPool::Guard` prevent resource leaks |
| **Performance Profiler** | Per-stage timing (LZ77, FSE, Huffman) with CSV and JSON export |
| **29 Error Codes** | Detailed status reporting with `ErrorContext` (file, line, function, message) |
| **67 Tests, 30 Benchmarks** | Comprehensive test suite with 100% pass rate |
| **Cross-Platform** | Linux, Windows, WSL2 |

---

## Architecture Overview

CUDA-ZSTD is organized into three layers:

### Layer 1: Management

The management layer provides the public API surface. Users interact exclusively with these three manager classes:

- **ZstdManager** -- Single-shot compression and decompression
- **ZstdBatchManager** -- Parallel batch operations with inference API
- **ZstdStreamingManager** -- Chunked streaming for large files

### Layer 2: Compression Pipeline

Each compression call flows through a five-stage pipeline:

```
LZ77 Match Finding --> Optimal Parsing --> Sequence Encoding --> FSE+Huffman Entropy Coding --> Frame Assembly
```

### Layer 3: GPU Kernels

47 CUDA kernels implement all compute-intensive operations:

| Category | Kernels | Purpose |
|----------|---------|---------|
| LZ77 | Hash table build, tiled match search, chain extension | Pattern discovery |
| FSE | Table build, parallel chunk encode, bitstream merge | Finite State Entropy coding |
| Huffman | Frequency count, tree build, canonical codes, batch encode | Literal compression |
| Sequence | Count, compress, extract, reverse buffer | Match/literal management |
| Dictionary | COVER training, dictionary application | Trained compression |
| XXHash | XXHash64 bulk, XXHash32 frame checksum | Data integrity |
| Utils | Prefix sum, reduction, memset, validation | Shared primitives |

### System Diagram

```
+------------------------------------------------------------------+
|                      APPLICATION LAYER                            |
|         (User Code: C++/C/Python, Single/Streaming/Batch)        |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                       MANAGER LAYER                               |
|  +--------------+  +-------------+  +----------------+           |
|  |   Default    |  |  Streaming  |  |     Batch      |           |
|  |   Manager    |  |   Manager   |  |    Manager     |           |
|  |  (Single)    |  |  (Chunks)   |  |  (Parallel)    |           |
|  +--------------+  +-------------+  +----------------+           |
|  +--------------+  +-------------+  +----------------+           |
|  |  Adaptive    |  | Memory Pool |  |  Inference     |           |
|  |  Selector    |  |   Manager   |  |    API         |           |
|  +--------------+  +-------------+  +----------------+           |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                   COMPRESSION PIPELINE                            |
|  +----------+  +----------+  +--------+  +-------------+        |
|  |   LZ77   |->| Optimal  |->| Seq.   |->| FSE/Huffman |        |
|  | Matching |  | Parsing  |  | Encode |  |  Encoding   |        |
|  +----------+  +----------+  +--------+  +-------------+        |
|  +----------+  +----------+  +---------+                         |
|  |   Dict   |  |  XXHash  |  |  Frame  |                        |
|  | Training |  | Checksum |  | Assembly|                         |
|  +----------+  +----------+  +---------+                         |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                     CUDA KERNEL LAYER                             |
|            47 GPU Kernels across 7 categories                    |
+------------------------------------------------------------------+
```

### Data Flow

```
INPUT DATA (Host or Device Memory)
    |
    v
1. Memory Preparation    -- Allocate workspace, init hash/chain tables
    |
    v
2. LZ77 Match Finding    -- Parallel hash chain build + tiled match search
    |
    v
3. Optimal Parsing       -- GPU dynamic programming + parallel backtracking
    |
    v
4. Sequence Encoding     -- Count sequences, compress, extract literals
    |
    v
5. Entropy Coding        -- FSE encode sequences, Huffman encode literals
    |
    v
6. Frame Assembly        -- Frame header, block assembly, optional checksum
    |
    v
OUTPUT DATA (RFC 8878 Compressed ZSTD Stream)
```

---

## Quick Start

### Prerequisites

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| **CUDA Toolkit** | 12.0+ | Developed with CUDA 12.8 |
| **CMake** | 3.24+ | Required for `CMAKE_CUDA_ARCHITECTURES "native"` |
| **C++ Compiler** | C++17 capable | GCC 11+, Clang 14+, or MSVC 19.29+ |
| **libzstd-dev** | 1.4.0+ | Host-side ZSTD reference for validation |
| **pkg-config** | Any | Library detection |

### Build from Source

```bash
git clone https://github.com/RhushabhVaghela/Custom-NVComp-with-ZSTD.git
cd Custom-NVComp-with-ZSTD
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Platform-Specific Setup

<details>
<summary><b>Ubuntu / Debian / WSL2</b></summary>

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config libzstd-dev

# Install CUDA Toolkit from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

nvcc --version        # Should show 12.0+
cmake --version       # Should show 3.24+
```

</details>

<details>
<summary><b>Fedora / RHEL</b></summary>

```bash
sudo dnf install -y gcc-c++ cmake pkgconf libzstd-devel
# Install CUDA Toolkit from NVIDIA repo
```

</details>

<details>
<summary><b>Windows (Visual Studio)</b></summary>

1. Install Visual Studio 2019/2022 with "Desktop development with C++"
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.0+
3. Install zstd via vcpkg:
   ```cmd
   vcpkg install zstd:x64-windows
   ```

</details>

### Linking Against the Library

**Direct compilation:**

```bash
g++ -std=c++17 my_app.cpp \
    -I/path/to/cuda-zstd/include \
    -L/path/to/cuda-zstd/build/lib \
    -lcuda_zstd \
    -L${CUDA_HOME}/lib64 -lcudart \
    -lzstd \
    -o my_app
```

**CMake subdirectory:**

```cmake
add_subdirectory(path/to/cuda-zstd)
target_link_libraries(my_app cuda_zstd)
```

**CMake find_package (after install):**

```cmake
find_package(cuda_zstd REQUIRED)
target_link_libraries(my_app cuda_zstd::cuda_zstd)
```

---

## Usage Examples

### C++ Simple Compress/Decompress

```cpp
#include <cuda_zstd_manager.h>
#include <vector>
#include <iostream>

int main() {
    using namespace cuda_zstd;

    // Prepare input data
    std::vector<uint8_t> input_data(1024 * 1024, 'A'); // 1 MB

    // Allocate GPU memory
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_data.size());
    cudaMemcpy(d_input, input_data.data(), input_data.size(),
               cudaMemcpyHostToDevice);

    // Create compression manager (level 5)
    auto manager = create_manager(5);

    // Query workspace and output sizes
    size_t temp_size = manager->get_compress_temp_size(input_data.size());
    size_t max_output = manager->get_max_compressed_size(input_data.size());

    cudaMalloc(&d_temp, temp_size);
    cudaMalloc(&d_output, max_output);

    // Compress
    size_t compressed_size = 0;
    Status status = manager->compress(
        d_input, input_data.size(),
        d_output, &compressed_size,
        d_temp, temp_size,
        nullptr, 0,  // No dictionary
        0            // Default CUDA stream
    );

    if (status == Status::SUCCESS) {
        std::cout << "Compressed " << input_data.size()
                  << " -> " << compressed_size << " bytes ("
                  << get_compression_ratio(input_data.size(), compressed_size)
                  << "x)\n";
    }

    // Decompress
    void *d_decompressed;
    cudaMalloc(&d_decompressed, input_data.size());
    size_t decompressed_size = 0;

    manager->decompress(
        d_output, compressed_size,
        d_decompressed, &decompressed_size,
        d_temp, temp_size, 0
    );

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_decompressed);
    return 0;
}
```

**Convenience one-liner (manages GPU memory internally):**

```cpp
#include <cuda_zstd_manager.h>

size_t out_size = 0;
cuda_zstd::compress_simple(d_input, input_size, d_output, &out_size, /*level=*/3);
cuda_zstd::decompress_simple(d_output, out_size, d_decompressed, &decompressed_size);
```

### C++ Batch Processing

Compress hundreds of buffers in a single GPU launch:

```cpp
#include <cuda_zstd_manager.h>

void batch_example() {
    using namespace cuda_zstd;

    auto batch_mgr = create_batch_manager(5);

    // Prepare batch items
    const size_t num_items = 250;
    const size_t item_size = 256 * 1024;  // 256 KB each (sweet spot)

    std::vector<BatchItem> items(num_items);
    for (auto& item : items) {
        cudaMalloc(&item.input_ptr, item_size);
        cudaMalloc(&item.output_ptr, item_size * 2);
        item.input_size = item_size;
        // ... copy data to item.input_ptr ...
    }

    // Allocate temp workspace
    std::vector<size_t> sizes(num_items, item_size);
    size_t temp_size = batch_mgr->get_batch_compress_temp_size(sizes);
    void* d_temp;
    cudaMalloc(&d_temp, temp_size);

    // Compress all items in parallel
    Status status = batch_mgr->compress_batch(items, d_temp, temp_size);

    // Check results
    for (size_t i = 0; i < items.size(); i++) {
        if (items[i].status == Status::SUCCESS) {
            printf("Item %zu: %zu -> %zu bytes\n",
                   i, items[i].input_size, items[i].output_size);
        }
    }

    // Decompress batch
    batch_mgr->decompress_batch(items, d_temp, temp_size);

    // Cleanup
    for (auto& item : items) {
        cudaFree(item.input_ptr);
        cudaFree(item.output_ptr);
    }
    cudaFree(d_temp);
}
```

**Inference API (zero-malloc decompression for ML pipelines):**

```cpp
// Pre-allocate output buffer once at initialization
void* d_output_buffer;
cudaMalloc(&d_output_buffer, max_decompressed_size);

// During inference: zero-malloc decompress into pre-allocated buffer
size_t actual_size = 0;
batch_mgr->decompress_to_preallocated(
    d_compressed, compressed_size,
    d_output_buffer, max_decompressed_size,
    &actual_size, d_workspace, workspace_size, stream
);

// Async version: decompress layer N+1 while computing layer N
batch_mgr->decompress_async_no_sync(
    d_compressed, compressed_size,
    d_output_buffer, max_decompressed_size,
    d_actual_size,  // device pointer -- written asynchronously
    d_workspace, workspace_size, stream
);
```

### C++ Streaming

Process arbitrarily large data in chunks:

```cpp
#include <cuda_zstd_manager.h>
#include <fstream>

void compress_large_file(const std::string& filename) {
    using namespace cuda_zstd;

    auto stream_mgr = create_streaming_manager(5);
    stream_mgr->init_compression();

    const size_t chunk_size = 128 * 1024; // 128 KB chunks
    std::vector<uint8_t> chunk(chunk_size);

    void *d_input, *d_output;
    cudaMalloc(&d_input, chunk_size);
    cudaMalloc(&d_output, chunk_size * 2);

    std::ifstream input(filename, std::ios::binary);
    std::ofstream output(filename + ".zst", std::ios::binary);

    while (input.read((char*)chunk.data(), chunk_size) || input.gcount() > 0) {
        size_t bytes_read = input.gcount();
        bool is_last = input.eof();

        cudaMemcpy(d_input, chunk.data(), bytes_read, cudaMemcpyHostToDevice);

        size_t compressed_size;
        stream_mgr->compress_chunk(d_input, bytes_read, d_output,
                                   &compressed_size, is_last);

        std::vector<uint8_t> compressed(compressed_size);
        cudaMemcpy(compressed.data(), d_output, compressed_size,
                   cudaMemcpyDeviceToHost);
        output.write((char*)compressed.data(), compressed_size);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}
```

**With window history for better ratios:**

```cpp
stream_mgr->init_compression_with_history();
stream_mgr->compress_chunk_with_history(d_input, bytes_read, d_output,
                                        &compressed_size, is_last);
```

### C++ Dictionary Compression

Train a dictionary on similar data for significantly better compression ratios:

```cpp
#include <cuda_zstd_dictionary.h>
#include <cuda_zstd_manager.h>

void dictionary_example() {
    using namespace cuda_zstd;
    using namespace cuda_zstd::dictionary;

    // Prepare training samples on device
    std::vector<const void*> sample_ptrs;
    std::vector<size_t> sample_sizes;
    // ... fill with device pointers to training data (e.g., JSON logs) ...

    // Train dictionary using COVER algorithm
    void* d_dict = nullptr;
    size_t dict_size = 64 * 1024; // 64 KB dictionary
    DictionaryTrainingParams params;

    cudaStream_t stream = 0;
    train_dictionary(sample_ptrs.data(), sample_sizes.data(),
                     sample_ptrs.size(),
                     &d_dict, &dict_size,
                     params, stream);

    // Use dictionary for compression
    auto manager = create_manager(5);
    size_t compressed_size = 0;
    manager->compress(
        d_input, input_size,
        d_output, &compressed_size,
        d_temp, temp_size,
        d_dict, dict_size,  // Pass dictionary
        stream
    );
}
```

### C API Usage

The C API is suitable for FFI integration from any language:

```c
#include <cuda_zstd_manager.h>
#include <stdio.h>

int main() {
    /* Create manager at level 5 */
    cuda_zstd_manager_t* manager = cuda_zstd_create_manager(5);

    size_t input_size = 1024 * 1024;
    void *d_input, *d_output, *d_temp;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);

    size_t temp_size = cuda_zstd_get_compress_workspace_size(manager, input_size);
    cudaMalloc(&d_temp, temp_size);

    /* Copy data to GPU ... */

    /* Compress */
    size_t compressed_size = 0;
    int status = cuda_zstd_compress(
        manager, d_input, input_size,
        d_output, &compressed_size,
        d_temp, temp_size, 0
    );

    if (status == 0) {
        printf("Compressed: %zu -> %zu bytes\n", input_size, compressed_size);
    } else {
        printf("Error: %s\n", cuda_zstd_get_error_string(status));
    }

    /* Decompress */
    size_t decompressed_size = 0;
    cuda_zstd_decompress(
        manager, d_output, compressed_size,
        d_input, &decompressed_size,
        d_temp, temp_size, 0
    );

    cuda_zstd_destroy_manager(manager);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    return 0;
}
```

### Python Usage

```python
import cuda_zstd

# One-shot compression
compressed = cuda_zstd.compress(b"hello world" * 10000, level=5)
original   = cuda_zstd.decompress(compressed)
assert original == b"hello world" * 10000

# Reusable manager (recommended for repeated calls)
with cuda_zstd.Manager(level=3) as mgr:
    c = mgr.compress(data)
    d = mgr.decompress(c)
    print(f"Ratio: {len(data) / len(c):.1f}x")

# Batch API -- compress many buffers in a single GPU launch
buffers = [chunk1, chunk2, chunk3]
compressed_list = cuda_zstd.compress_batch(buffers, level=5)
decompressed_list = cuda_zstd.decompress_batch(compressed_list)

# Check GPU availability
if cuda_zstd.is_cuda_available():
    info = cuda_zstd.get_cuda_device_info()
    print(f"GPU: {info}")
```

---

## API Reference Summary

### Factory Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `create_manager(int level)` | `unique_ptr<ZstdManager>` | Create single-shot manager |
| `create_manager(CompressionConfig)` | `unique_ptr<ZstdManager>` | Create manager with full config |
| `create_batch_manager(int level)` | `unique_ptr<ZstdBatchManager>` | Create batch processing manager |
| `create_streaming_manager(int level)` | `unique_ptr<ZstdStreamingManager>` | Create streaming manager |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `compress_simple(d_in, size, d_out, &out_size, level)` | One-call compression |
| `decompress_simple(d_in, size, d_out, &out_size)` | One-call decompression |
| `compress_with_dict(d_in, size, d_out, &out_size, dict)` | Compress with dictionary |
| `decompress_with_dict(d_in, size, d_out, &out_size, dict)` | Decompress with dictionary |

### ZstdManager Methods

| Method | Description |
|--------|-------------|
| `configure(config)` | Apply a `CompressionConfig` |
| `get_config()` | Return current configuration |
| `get_compress_temp_size(size)` | Query workspace size for compression |
| `get_decompress_temp_size(size)` | Query workspace size for decompression |
| `get_max_compressed_size(size)` | Upper bound on compressed output |
| `compress(...)` | Compress a buffer (supports dictionary and stream) |
| `decompress(...)` | Decompress a buffer |
| `set_dictionary(dict)` | Load a trained dictionary |
| `clear_dictionary()` | Remove loaded dictionary |
| `set_compression_level(level)` | Change level (1-22) |
| `get_stats()` | Return `CompressionStats` |
| `reset_stats()` | Zero all statistics |
| `preallocate_tables(stream)` | Pre-allocate internal tables for reuse |
| `select_execution_path(size)` | Determine CPU/GPU/GPU_CHUNK path |

### ZstdBatchManager Methods (extends ZstdManager)

| Method | Description |
|--------|-------------|
| `compress_batch(items, workspace, size)` | Compress N items in parallel |
| `decompress_batch(items, workspace, size)` | Decompress N items in parallel |
| `get_batch_compress_temp_size(sizes)` | Query workspace for batch compression |
| `get_batch_decompress_temp_size(sizes)` | Query workspace for batch decompression |
| `decompress_to_preallocated(...)` | Zero-malloc decompress into existing buffer |
| `decompress_batch_preallocated(...)` | Batch zero-malloc decompress |
| `decompress_async_no_sync(...)` | Async decompress (caller manages sync) |
| `allocate_inference_workspace(...)` | Allocate reusable inference workspace |
| `free_inference_workspace(ptr)` | Free inference workspace |
| `get_inference_workspace_size(...)` | Query inference workspace size |

### ZstdStreamingManager Methods

| Method | Description |
|--------|-------------|
| `init_compression(stream, chunk_size)` | Initialize for streaming compression |
| `init_compression_with_history(stream, chunk_size)` | Init with sliding window history |
| `init_decompression(stream)` | Initialize for streaming decompression |
| `compress_chunk(in, size, out, &out_size, is_last)` | Compress one chunk |
| `compress_chunk_with_history(...)` | Compress with cross-chunk history |
| `decompress_chunk(in, size, out, &out_size, &is_last)` | Decompress one chunk |
| `reset()` | Reset streaming state |
| `flush(stream)` | Flush pending data |
| `set_config(config)` | Update configuration |
| `set_dictionary(dict)` | Load dictionary for streaming |

### C API Functions (11)

| Function | Description |
|----------|-------------|
| `cuda_zstd_create_manager(level)` | Create opaque manager handle |
| `cuda_zstd_destroy_manager(mgr)` | Destroy manager |
| `cuda_zstd_compress(mgr, src, src_size, dst, &dst_size, ws, ws_size, stream)` | Compress |
| `cuda_zstd_decompress(mgr, src, src_size, dst, &dst_size, ws, ws_size, stream)` | Decompress |
| `cuda_zstd_get_compress_workspace_size(mgr, size)` | Query workspace |
| `cuda_zstd_get_decompress_workspace_size(mgr, size)` | Query workspace |
| `cuda_zstd_train_dictionary(samples, sizes, n, dict_size)` | Train dictionary |
| `cuda_zstd_destroy_dictionary(dict)` | Free dictionary |
| `cuda_zstd_set_dictionary(mgr, dict)` | Attach dictionary |
| `cuda_zstd_get_error_string(code)` | Error code to string |
| `cuda_zstd_is_error(code)` | Check if code is an error |

### NVCOMP v5 API Functions (7)

| Function | Description |
|----------|-------------|
| `nvcomp_zstd_create_manager_v5(level)` | Create nvCOMP-style handle |
| `nvcomp_zstd_destroy_manager_v5(handle)` | Destroy handle |
| `nvcomp_zstd_compress_async_v5(...)` | Async compress |
| `nvcomp_zstd_decompress_async_v5(...)` | Async decompress |
| `nvcomp_zstd_get_compress_temp_size_v5(handle, size)` | Query workspace |
| `nvcomp_zstd_get_decompress_temp_size_v5(handle, size)` | Query workspace |
| `nvcomp_zstd_get_metadata_v5(data, size, meta, stream)` | Extract metadata |

### Configuration Types

```cpp
enum class Strategy : u32 {
    FAST = 0, DFAST = 1, GREEDY = 2, LAZY = 3,
    LAZY2 = 4, BTLAZY2 = 5, BTOPT = 6, BTULTRA = 7
};

enum class CompressionMode : u32 {
    LEVEL_BASED = 0,    // Use exact level (1-22)
    STRATEGY_BASED = 1  // Library picks level from strategy
};

enum class ChecksumPolicy : u32 {
    NO_COMPUTE_NO_VERIFY = 0,
    COMPUTE_NO_VERIFY = 1,
    COMPUTE_AND_VERIFY = 2
};

enum class BitstreamKind : u32 {
    NATIVE = 0,  // GPU-optimized (internal use)
    RAW = 1      // RFC 8878 compatible (default)
};
```

---

## Compression Levels

CUDA-ZSTD supports the full ZSTD level range (1-22). Higher levels produce smaller output at the cost of compression speed. Decompression speed is largely independent of the compression level used.

| Level | Strategy | Speed | Ratio | Use Case |
|------:|----------|-------|-------|----------|
| 1 | FAST | Fastest | Lowest | Real-time streaming, logging |
| 2 | DFAST | Very Fast | Low | High-throughput pipelines |
| 3 | DFAST | Fast | Moderate | **Default** -- good balance |
| 4 | GREEDY | Fast | Moderate | General purpose |
| 5 | GREEDY | Moderate | Moderate | General purpose |
| 6 | GREEDY | Moderate | Good | File compression |
| 7 | LAZY | Moderate | Good | File compression |
| 8 | LAZY | Moderate | Good | Archival |
| 9 | LAZY | Slower | Better | Archival |
| 10 | LAZY | Slower | Better | Archival |
| 11 | LAZY | Slower | Better | Archival |
| 12 | LAZY | Slow | High | Storage optimization |
| 13 | LAZY2 | Slow | High | Storage optimization |
| 14 | LAZY2 | Slow | High | Cold storage |
| 15 | LAZY2 | Slow | Very High | Cold storage |
| 16 | BTLAZY2 | Very Slow | Very High | Maximum compression |
| 17 | BTLAZY2 | Very Slow | Very High | Maximum compression |
| 18 | BTLAZY2 | Very Slow | Very High | Maximum compression |
| 19 | BTOPT | Extremely Slow | Excellent | Archival, distribution |
| 20 | BTOPT | Extremely Slow | Excellent | Archival, distribution |
| 21 | BTULTRA | Slowest | Maximum | Maximum compression |
| 22 | BTULTRA | Slowest | Maximum | Maximum compression |

**Tip:** For most GPU batch workloads, levels 1-6 offer the best throughput. Levels 7-12 provide a good balance for offline processing. Levels 13+ are best suited for cold storage where compression time is not critical.

---

## Benchmark Results

All benchmarks measured on an **RTX 5080 Laptop GPU** (Blackwell architecture, sm_120) with CUDA 12.8.

### Peak Performance

| Metric | Value |
|--------|-------|
| Peak Compress Throughput | **800.3 MB/s** (256 KB input) |
| Peak Decompress Throughput | **1.77 GB/s** (256 KB input) |
| Peak Batch Throughput | **9.81 GB/s** (250 x 256 KB) |
| Peak FSE GPU Encode | **5.95 GB/s** (16 MB input) |
| Peak Huffman Encode | **1.19 GB/s** (16 MB input) |
| Peak Inference Decompress | **1.14 GB/s** (256 KB input) |
| Stream Pool Speedup | **4.7x** vs `cudaStreamCreate` |

### GPU vs CPU Comparison

| Metric | CPU libzstd | GPU CUDA-ZSTD | Advantage |
|--------|-------------|---------------|-----------|
| Compress (single buffer) | 400-600 MB/s | 800 MB/s | ~1.5x |
| Decompress (single buffer) | 1.0-1.8 GB/s | 1.77 GB/s | ~1x |
| Batch compress (parallel) | ~2.4 GB/s (8 cores) | 9.81 GB/s | **~4x** |

### Where GPU Wins

The GPU's primary advantage is **batch processing**. When compressing or decompressing many independent buffers in parallel, the GPU achieves throughput that scales far beyond what CPU threading can match:

- **Sweet spot**: 64 KB - 256 KB per item, 50-500 items per batch
- **Batch advantage**: Up to 4x over 8-core CPU parallel compression
- **Single-buffer parity**: For individual large buffers, GPU and CPU are comparable
- **Small buffers**: Below ~4 KB, CPU is preferred due to kernel launch overhead (handled automatically by the hybrid router)

### Throughput by Input Size (Single Buffer)

| Input Size | Compress | Decompress |
|------------|----------|------------|
| 4 KB | CPU fallback | CPU fallback |
| 16 KB | ~200 MB/s | ~400 MB/s |
| 64 KB | ~450 MB/s | ~900 MB/s |
| 256 KB | **800 MB/s** | **1.77 GB/s** |
| 1 MB | ~750 MB/s | ~1.6 GB/s |
| 16 MB | ~700 MB/s | ~1.5 GB/s |

### Profiling

```bash
# Profile with NVIDIA Nsight Systems
nsys profile --stats=true ./build/bin/benchmark_batch_throughput

# Detailed kernel analysis with Nsight Compute
ncu --set full ./build/bin/benchmark_batch_throughput
```

---

## Python Package

### Installation

```bash
# From PyPI (when published)
pip install cuda-zstd

# From source
cd python && pip install .
```

### Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed and on PATH
- (Optional) NumPy for array support
- (Optional) CuPy for zero-copy GPU arrays

### Python API

```python
import cuda_zstd

# --- One-shot functions ---
compressed = cuda_zstd.compress(data, level=5)
original   = cuda_zstd.decompress(compressed)

# --- Batch processing ---
results = cuda_zstd.compress_batch([buf1, buf2, buf3], level=3)
originals = cuda_zstd.decompress_batch(results)

# --- Manager for repeated use ---
with cuda_zstd.Manager(level=5) as mgr:
    c = mgr.compress(data)
    d = mgr.decompress(c)
    print(mgr.stats)         # CompressionStats
    print(mgr.config)        # CompressionConfig
    mgr.level = 10           # Change level on the fly

# --- Validation & estimation ---
cuda_zstd.validate_compressed_data(compressed)              # True/False
cuda_zstd.estimate_compressed_size(len(data), level=5)      # upper-bound size

# --- GPU info ---
cuda_zstd.is_cuda_available()     # True/False
cuda_zstd.get_cuda_device_info()  # Device details

# --- Constants ---
cuda_zstd.MIN_LEVEL       # 1
cuda_zstd.MAX_LEVEL       # 22
cuda_zstd.DEFAULT_LEVEL   # 3
cuda_zstd.__version__     # "1.0.0"
```

### Python Exported Symbols

| Symbol | Type | Description |
|--------|------|-------------|
| `compress(data, level=3)` | function | Compress bytes on GPU |
| `decompress(data)` | function | Decompress bytes on GPU |
| `compress_batch(inputs, level=3)` | function | Batch compress |
| `decompress_batch(inputs)` | function | Batch decompress |
| `validate_compressed_data(data, check_checksum=True)` | function | Validate compressed data integrity |
| `estimate_compressed_size(uncompressed_size, level=3)` | function | Upper-bound compressed size estimate |
| `Manager(level=3, config=None)` | class | Reusable compression manager |
| `CompressionConfig` | class | Fine-grained configuration |
| `CompressionStats` | class | Statistics |
| `Status` | enum | Status codes |
| `Strategy` | enum | Compression strategies |
| `ChecksumPolicy` | enum | Checksum options |
| `is_cuda_available()` | function | Check GPU availability |
| `get_cuda_device_info()` | function | GPU device information |

---

## Configuration Options

### CompressionConfig Struct

```cpp
struct CompressionConfig {
    // Mode selection
    CompressionMode compression_mode;  // LEVEL_BASED or STRATEGY_BASED

    // Level-based mode
    int level;               // 1-22, exact compression level
    bool use_exact_level;    // Force exact level parameters

    // Strategy-based mode
    Strategy strategy;       // FAST through BTULTRA

    // Advanced parameters
    u32 window_log;          // Window size = 1 << window_log (10-31, default 20)
    u32 hash_log;            // Hash table size log (default 17)
    u32 chain_log;           // Chain table size log (default 17)
    u32 search_log;          // Search depth log (default 8)
    u32 min_match;           // Minimum match length (3-7, default 3)
    u32 target_length;       // Target match length (0 = auto)
    u32 block_size;          // Block size in bytes (default 128 KB)
    bool enable_ldm;         // Enable Long Distance Matching
    u32 ldm_hash_log;        // LDM hash table size (default 20)
    ChecksumPolicy checksum; // Checksum computation/verification

    // Hybrid routing
    u32 cpu_threshold;       // Below this size, use CPU (default 1 MB)

    // Helper functions
    static CompressionConfig from_level(int level);
    static CompressionConfig optimal(size_t input_size);
    static CompressionConfig get_default();
    Status validate() const;
};
```

### Compression Strategies Explained

| Strategy | Levels | Algorithm | Description |
|----------|--------|-----------|-------------|
| `FAST` | 1 | Greedy hash | Fastest, no match optimization |
| `DFAST` | 2-3 | Double hash | Two hash tables for better matches |
| `GREEDY` | 4-6 | Greedy parse | Take first acceptable match |
| `LAZY` | 7-12 | Lazy evaluation | Check if next position has better match |
| `LAZY2` | 13-15 | Double lazy | Two-position look-ahead |
| `BTLAZY2` | 16-18 | Binary tree + lazy | Binary tree match finder with lazy eval |
| `BTOPT` | 19-20 | Binary tree + optimal | Optimal parsing with binary tree |
| `BTULTRA` | 21-22 | Binary tree + ultra | Maximum effort optimal parsing |

### ChecksumPolicy

| Policy | Compute | Verify | Description |
|--------|---------|--------|-------------|
| `NO_COMPUTE_NO_VERIFY` | No | No | Fastest, no integrity check |
| `COMPUTE_NO_VERIFY` | Yes | No | Embed checksum, skip verification on decompress |
| `COMPUTE_AND_VERIFY` | Yes | Yes | Full integrity: embed and verify |

---

## How It Works

### ZSTD Frame Format (RFC 8878)

Every compressed output follows the ZSTD frame format:

```
+------------------+------------------+------------------+------------------+
| Magic Number     | Frame Header     | Data Blocks      | Content Checksum |
| (4 bytes)        | (2-14 bytes)     | (variable)       | (0 or 4 bytes)   |
| 0xFD2FB528       |                  |                  |                  |
+------------------+------------------+------------------+------------------+
```

**Frame Header** contains:
- Frame Header Descriptor (window size, single segment flag, content size flag, checksum flag, dictionary ID flag)
- Window Descriptor (window size)
- Dictionary ID (0, 1, 2, or 4 bytes)
- Frame Content Size (0, 1, 2, 4, or 8 bytes)

### Block Types

Each frame contains one or more blocks:

| Block Type | Code | Description |
|------------|------|-------------|
| Raw | 0 | Uncompressed data (stored as-is) |
| RLE | 1 | Run-Length Encoded (single repeated byte) |
| Compressed | 2 | LZ77 + entropy coded data |
| Reserved | 3 | Not used |

### Compression Pipeline Detail

**1. LZ77 Match Finding (GPU)**

The LZ77 stage finds repeated patterns in the input data:
- A hash table maps 4-byte prefixes to positions in the input
- Chain tables link positions with the same hash (hash chain)
- GPU threads search chains in parallel, each thread handling a tile of the input
- Output: array of `Match` structs `(offset, length, literal_length)`

**2. Optimal Parsing (GPU)**

For levels >= 4, optimal parsing replaces greedy selection:
- Dynamic programming over the match array to minimize output size
- Cost model considers literal costs, match costs, and FSE state transitions
- Parallel backtracking extracts the optimal sequence of matches and literals
- Output: optimized sequence of `(literal_length, match_length, offset)` triples

**3. Sequence Encoding**

Sequences are split into three parallel streams per RFC 8878:
- Literal Lengths (LL): how many literal bytes precede each match
- Match Lengths (ML): how long each match is
- Offsets: distance back in the window to the match source

RepCodes (offset shortcuts for recently used offsets) are applied to reduce entropy.

**4. FSE Entropy Coding (GPU)**

Finite State Entropy coding compresses the sequence streams:
- Symbol frequency analysis (GPU parallel histogram)
- Normalized count computation (total = power of two)
- Encoding table construction
- Parallel chunk encoding: input is divided into chunks, each encoded by a separate GPU thread
- Bitstream merging: chunk bitstreams are concatenated with correct bit alignment

**5. Huffman Coding (GPU)**

Literals are compressed with Huffman coding:
- GPU frequency counting over the literal buffer
- Tree construction (canonical Huffman)
- Code length assignment (max 11 bits per RFC 8878)
- Batch bit writing with prefix-sum-based offset calculation

**6. Frame Assembly**

The final stage assembles all components:
- Frame magic number (0xFD2FB528)
- Frame header with window size, content size, checksum flag
- Compressed block header (block type, block size, last block flag)
- Literals section (Huffman-compressed or raw)
- Sequences section (FSE-compressed LL, ML, Offset streams)
- Optional XXHash64 content checksum

### Decompression Pipeline

Decompression reverses the process:
1. Parse frame header and validate magic number
2. For each block: read block header, determine type
3. Decode FSE tables from the compressed bitstream
4. Decode sequences (LL, ML, Offset) using FSE states
5. Decode literals (Huffman or raw)
6. Execute sequences: copy literals, then copy match data from window
7. Verify checksum if present

---

## User Options

### What Users Can Configure

| Option | Type | Range/Values | Default | Description |
|--------|------|-------------|---------|-------------|
| Compression Level | `int` | 1-22 | 3 | Speed/ratio tradeoff |
| Strategy | `Strategy` | FAST to BTULTRA | Level-dependent | Algorithm selection |
| Window Size | `u32` | 10-31 (log2) | 20 (1 MB) | Lookback distance |
| Block Size | `u32` | bytes | 131072 (128 KB) | Processing block size |
| Checksum Policy | `ChecksumPolicy` | 3 options | NO_COMPUTE_NO_VERIFY | Integrity checking |
| CPU Threshold | `u32` | bytes | 1048576 (1 MB) | Hybrid routing cutoff |
| Dictionary | pointer+size | any | none | Pre-trained dictionary |
| Adaptive Level | bool | on/off | off | Auto-select level |
| LDM Enable | `bool` | on/off | off | Long Distance Matching |
| Min Match | `u32` | 3-7 | 3 | Minimum match length |
| Hash Log | `u32` | log2 | 17 | Hash table size |
| Chain Log | `u32` | log2 | 17 | Chain table size |
| Search Log | `u32` | log2 | 8 | Search depth |

### Smart Path Selection

The library automatically routes operations based on input size:

| Input Size | Path | Rationale |
|------------|------|-----------|
| < cpu_threshold | CPU (libzstd) | Kernel launch overhead dominates |
| >= cpu_threshold, < 1 MB | GPU Block Parallel | Standard GPU path |
| >= 1 MB | GPU Chunk Parallel | Intra-block parallelism |

Override with `CompressionConfig::cpu_threshold` to tune the crossover point for your hardware.

---

## Build Options

### CMake Configuration

```bash
# Standard release build (auto-detect GPU architecture)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Specify exact GPU architecture (e.g., sm_120 for Blackwell)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120

# Multiple architectures for portable binary
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"

# Debug build with CUDA device debug symbols
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Enable verbose decompress debug output
cmake -B build -DCUDA_ZSTD_DEBUG=ON

# Enable verbose PTX register/spill info
cmake -B build -DCUDA_ZSTD_VERBOSE_PTX=ON
```

### CMake Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | String | (none) | `Release`, `Debug`, `RelWithDebInfo` |
| `CMAKE_CUDA_ARCHITECTURES` | String | `native` | GPU compute capability targets |
| `CUDA_ZSTD_DEBUG` | Bool | `OFF` | Enable verbose ZSTD decompress tracing |
| `CUDA_ZSTD_VERBOSE_PTX` | Bool | `OFF` | Print register usage and spill info (`-Xptxas=-v`) |

### Build Output

```
build/
  lib/
    libcuda_zstd.a          # Static library
  bin/
    test_correctness        # Test executables (67 total)
    test_streaming
    test_roundtrip
    benchmark_batch_throughput  # Benchmark executables (30 total)
    benchmark_fse
    benchmark_huffman
    ...
```

---

## Testing

### Running Tests

```bash
# Run all 67 CTest targets
ctest --test-dir build --output-on-failure

# Parallel execution (recommended)
ctest --test-dir build -j8 --output-on-failure

# Run a specific test by name
ctest --test-dir build -R test_correctness -V

# Run a test executable directly
./build/bin/test_correctness
./build/bin/test_streaming
./build/bin/test_roundtrip
```

All tests have a 120-second timeout configured via CTest.

### Test Status

**67 of 67 CTest targets pass (100%)**

### Test Coverage Overview

| Test Category | Tests | What They Validate |
|---------------|-------|--------------------|
| Correctness | `test_correctness`, `test_compressible_data` | RFC 8878 compliance, format correctness |
| Roundtrip | `test_roundtrip`, `test_dual_mode_roundtrip` | Compress then decompress, verify data integrity |
| Streaming | `test_streaming`, `test_streaming_manager`, `test_streaming_unit`, `test_streaming_integration` | Chunked frame streaming operations |
| FSE | `test_fse_comprehensive`, `test_fse_canonical` | FSE encoding/decoding correctness |
| Dictionary | `test_dictionary`, `test_dictionary_compression`, `test_dictionary_memory` | Dictionary training, usage, memory |
| Memory | `test_workspace_usage`, `test_workspace_patterns`, `test_alternative_allocation_strategies` | GPU memory management |
| Error Handling | `test_error_handling`, `test_error_context` | Error codes, context, callbacks |
| APIs | `test_c_api`, `test_c_api_edge_cases` | C API compatibility |
| Validation | `test_data_integrity_comprehensive`, `test_checksum_validation` | Data integrity and checksums |
| Edge Cases | `test_coverage_gaps`, `test_extended_validation`, `test_fallback_strategies` | Boundary conditions, fallbacks |
| Concurrency | `test_stream_pool`, `test_concurrency_repro` | Multi-stream, concurrent access |
| Components | `test_sequence_encoder`, `test_find_matches_small`, `test_adaptive_level`, `test_utils`, `test_two_phase_unit`, `test_scale_repro` | Individual pipeline components |

### Running Benchmarks

```bash
# List available benchmarks
ls build/bin/benchmark_*

# Run specific benchmarks
./build/bin/benchmark_batch_throughput
./build/bin/benchmark_streaming
./build/bin/benchmark_lz77
./build/bin/benchmark_fse
./build/bin/benchmark_huffman
./build/bin/benchmark_dictionary_compression
./build/bin/benchmark_c_api
./build/bin/benchmark_nvcomp_interface
./build/bin/benchmark_all_levels
./build/bin/benchmark_inference_api
./build/bin/benchmark_zstd_gpu_comparison

# Run via CMake custom target
cmake --build build --target run_benchmark_batch_throughput
```

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Core Guides

| Document | Description |
|----------|-------------|
| [INDEX.md](docs/INDEX.md) | Documentation index and navigation |
| [QUICK-REFERENCE.md](docs/QUICK-REFERENCE.md) | One-page API cheat sheet |
| [BUILD-GUIDE.md](docs/BUILD-GUIDE.md) | Detailed build instructions |
| [ARCHITECTURE-OVERVIEW.md](docs/ARCHITECTURE-OVERVIEW.md) | System architecture deep-dive |
| [PERFORMANCE-TUNING.md](docs/PERFORMANCE-TUNING.md) | Optimization guide |

### API Documentation

| Document | Description |
|----------|-------------|
| [C-API-REFERENCE.md](docs/C-API-REFERENCE.md) | C API reference |
| [STREAMING-API.md](docs/STREAMING-API.md) | Streaming API guide |
| [BATCH-PROCESSING.md](docs/BATCH-PROCESSING.md) | Batch processing guide |
| [NVCOMP-INTEGRATION.md](docs/NVCOMP-INTEGRATION.md) | NVCOMP v5 compatibility layer |

### Algorithm Deep-Dives

| Document | Description |
|----------|-------------|
| [LZ77-IMPLEMENTATION.md](docs/LZ77-IMPLEMENTATION.md) | LZ77 match finding on the GPU |
| [FSE-IMPLEMENTATION.md](docs/FSE-IMPLEMENTATION.md) | Finite State Entropy coding |
| [HUFFMAN-IMPLEMENTATION.md](docs/HUFFMAN-IMPLEMENTATION.md) | Huffman coding implementation |
| [SEQUENCE-IMPLEMENTATION.md](docs/SEQUENCE-IMPLEMENTATION.md) | Sequence encoding pipeline |
| [DICTIONARY-IMPLEMENTATION.md](docs/DICTIONARY-IMPLEMENTATION.md) | COVER dictionary training |
| [XXHASH-IMPLEMENTATION.md](docs/XXHASH-IMPLEMENTATION.md) | XXHash checksumming |
| [FRAME-FORMAT.md](docs/FRAME-FORMAT.md) | ZSTD frame format reference |

### Infrastructure

| Document | Description |
|----------|-------------|
| [MEMORY-POOL-IMPLEMENTATION.md](docs/MEMORY-POOL-IMPLEMENTATION.md) | GPU memory pool design |
| [KERNEL-REFERENCE.md](docs/KERNEL-REFERENCE.md) | All 47 CUDA kernel signatures |
| [STREAM-OPTIMIZATION.md](docs/STREAM-OPTIMIZATION.md) | CUDA stream optimization |
| [HASH_TABLE_OPTIMIZATION.md](docs/HASH_TABLE_OPTIMIZATION.md) | Hash table design for GPU |
| [MANAGER-IMPLEMENTATION.md](docs/MANAGER-IMPLEMENTATION.md) | Manager class internals |

### Debugging and Testing

| Document | Description |
|----------|-------------|
| [ERROR-HANDLING.md](docs/ERROR-HANDLING.md) | Error codes and handling |
| [DEBUGGING-GUIDE.md](docs/DEBUGGING-GUIDE.md) | Debugging techniques |
| [TESTING-GUIDE.md](docs/TESTING-GUIDE.md) | Testing guide |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and fixes |
| [BENCHMARKING-GUIDE.md](docs/BENCHMARKING-GUIDE.md) | How to run benchmarks |
| [CHECKSUM-IMPLEMENTATION.md](docs/CHECKSUM-IMPLEMENTATION.md) | Checksum verification |
| [ADAPTIVE-LEVEL-SELECTION.md](docs/ADAPTIVE-LEVEL-SELECTION.md) | Adaptive level algorithm |
| [FALLBACK_STRATEGIES_IMPLEMENTATION.md](docs/FALLBACK_STRATEGIES_IMPLEMENTATION.md) | GPU-to-CPU fallback |
| [ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md](docs/ALTERNATIVE_ALLOCATION_STRATEGIES_IMPLEMENTATION.md) | Memory allocation strategies |

---

## Project Structure

```
Custom-NVComp-with-ZSTD/
|
|-- CMakeLists.txt                  # Build system (CMake 3.24+)
|-- LICENSE                         # MIT License
|-- README.md                       # This file
|-- DEBUGLOG.md                     # Development debug log
|
|-- include/                        # Public headers (29 files)
|   |-- cuda_zstd.h                 #   Umbrella header
|   |-- cuda_zstd_types.h           #   Status, Config, BatchItem, constants
|   |-- cuda_zstd_manager.h         #   Manager classes, factory functions, C API
|   |-- cuda_zstd_dictionary.h      #   Dictionary training and structures
|   |-- cuda_zstd_nvcomp.h          #   NVCOMP v5 compatibility layer
|   |-- cuda_zstd_fse.h             #   FSE encode/decode
|   |-- cuda_zstd_huffman.h         #   Huffman encode/decode
|   |-- cuda_zstd_sequence.h        #   Sequence management
|   |-- cuda_zstd_lz77.h            #   LZ77 match finding
|   |-- cuda_zstd_stream_pool.h     #   CUDA stream pooling
|   |-- cuda_zstd_cuda_ptr.h        #   RAII CudaDevicePtr<T>
|   |-- cuda_zstd_memory_pool.h     #   GPU memory pool
|   |-- cuda_zstd_xxhash.h          #   XXHash checksums
|   |-- cuda_zstd_adaptive.h        #   Adaptive level selection
|   |-- performance_profiler.h      #   Profiling and metrics
|   |-- error_context.h             #   Error context struct
|   |-- workspace_manager.h         #   Workspace allocation
|   `-- ...                         #   (and more internal headers)
|
|-- src/                            # Implementation (33 files)
|   |-- cuda_zstd_manager.cu        #   Manager implementations
|   |-- cuda_zstd_lz77.cu           #   LZ77 kernels
|   |-- cuda_zstd_fse.cu            #   FSE kernels
|   |-- cuda_zstd_huffman.cu        #   Huffman kernels
|   |-- cuda_zstd_sequence.cu       #   Sequence kernels
|   |-- cuda_zstd_dictionary.cu     #   Dictionary training
|   |-- cuda_zstd_xxhash.cu         #   XXHash kernels
|   |-- cuda_zstd_c_api.cpp         #   C API implementation
|   |-- cuda_zstd_nvcomp.cpp        #   NVCOMP compatibility
|   |-- cuda_zstd_stream_pool.cpp   #   Stream pool
|   |-- pipeline_manager.cu         #   Pipeline orchestration
|   |-- workspace_manager.cu        #   Workspace management
|   `-- ...                         #   (and more source files)
|
|-- tests/                          # Test suite (67 test files)
|   |-- test_correctness.cu
|   |-- test_roundtrip.cu
|   |-- test_streaming.cu
|   |-- test_c_api.cpp
|   |-- cuda_error_checking.h       #   Test utilities
|   `-- ...
|
|-- benchmarks/                     # Benchmarks (30 files)
|   |-- benchmark_batch_throughput.cu
|   |-- benchmark_fse.cu
|   |-- benchmark_huffman.cu
|   |-- benchmark_results.h         #   Benchmark utilities
|   |-- throughput_display.h        #   Display helpers
|   `-- ...
|
|-- docs/                           # Documentation (31 files)
|   |-- INDEX.md
|   |-- ARCHITECTURE-OVERVIEW.md
|   |-- KERNEL-REFERENCE.md
|   `-- ...
|
|-- python/                         # Python package
|   |-- CMakeLists.txt              #   Python build
|   |-- pyproject.toml              #   Package metadata
|   |-- README.md                   #   Python README
|   |-- cuda_zstd/
|   |   |-- __init__.py             #   Python API
|   |   |-- _core.pyi               #   Type stubs
|   |   `-- py.typed                #   PEP 561 marker
|   |-- src/
|   |   `-- binding.cpp             #   pybind11 bindings
|   `-- tests/
|       |-- test_basic.py
|       `-- conftest.py
|
`-- cmake/
    `-- cuda_zstdConfig.cmake.in    # CMake package config template
```

### Code Statistics

| Metric | Count |
|--------|-------|
| Lines of code (src + include) | ~31,000 |
| Public header files | 29 |
| Source files | 33 |
| Test files | 67 |
| Benchmark files | 30 |
| Documentation files | 31 |
| CUDA kernels | 47 |
| CTest targets | 67 |
| Tests passing | 67 (100%) |
| Compression levels | 1-22 |
| Error codes | 29 |

---

## Dependencies

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | 12.0+ | GPU compute runtime, `nvcc` compiler |
| [CMake](https://cmake.org/) | 3.24+ | Build system |
| [libzstd](https://github.com/facebook/zstd) | 1.4.0+ | Host-side reference, validation, CPU fallback |
| C++17 Compiler | GCC 11+, Clang 14+, MSVC 19.29+ | Host code compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| [OpenMP](https://www.openmp.org/) | Any | Benchmark multi-threaded CPU baselines |
| [Python](https://www.python.org/) | 3.8+ | Python bindings |
| [pybind11](https://github.com/pybind/pybind11) | 2.10+ | Python/C++ bridge (for Python package) |
| [NumPy](https://numpy.org/) | Any | Python array support |
| [CuPy](https://cupy.dev/) | Any | Python zero-copy GPU array support |
| [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | Any | Performance profiling |
| [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) | Any | Kernel-level analysis |

### Platform Support

| OS | Status | Notes |
|----|--------|-------|
| Ubuntu 20.04+ / Debian 11+ | Supported | Primary development platform (WSL2) |
| Fedora / RHEL / CentOS | Supported | |
| Windows 10/11 | Supported | Visual Studio 2019+ |
| WSL2 | Supported | |
| macOS | Not supported | No CUDA on macOS |

---

## License

This project is licensed under the **MIT License**.

Copyright (c) 2025 Rhushabh Vaghela

See [LICENSE](LICENSE) for the full license text.

---

## Contributing

Contributions are welcome. To contribute:

1. **Fork** the repository on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** and ensure all tests pass:
   ```bash
   cmake --build build -j$(nproc)
   ctest --test-dir build --output-on-failure
   ```
4. **Commit** with a descriptive message:
   ```bash
   git commit -m "Add: description of what you changed"
   ```
5. **Push** to your fork and open a **Pull Request**

### Development Guidelines

- All new features should include tests
- All 67 existing tests must continue to pass
- Follow the existing code style (C++17, CUDA C++)
- Update documentation in `docs/` if you change public APIs
- Run benchmarks before and after performance-sensitive changes
- Prefer RAII wrappers (`CudaDevicePtr<T>`, `StreamPool::Guard`) over raw `cudaMalloc`/`cudaFree`

### Areas Where Contributions Are Especially Welcome

- Performance optimization of existing CUDA kernels
- Additional test coverage for edge cases
- CI/CD pipeline setup (GitHub Actions)
- Multi-GPU support
- Shared library build target
- Additional language bindings (Rust, Go, Java)
- Real-world benchmark datasets

---

## Future Work

| Feature | Priority | Description |
|---------|----------|-------------|
| Multi-GPU support | High | Distribute batch across multiple GPUs |
| Shared library build | High | `libcuda_zstd.so` / `cuda_zstd.dll` |
| CI/CD pipeline | High | GitHub Actions with CUDA runner |
| Long Distance Matching (LDM) | Medium | Cross-block match finding for large windows |
| Decompression verification | Medium | Automated cross-validation against `libzstd` |
| Python wheels on PyPI | Medium | Pre-built binary wheels for `pip install` |
| Rust bindings | Medium | `cuda-zstd-rs` crate |
| Adaptive block sizing | Low | Auto-tune block size based on GPU occupancy |
| Multi-frame parallelism | Low | Decompress multiple frames concurrently |
| ZSTD v2 format support | Low | Future format extensions |

---

## Acknowledgments

- [Zstandard](https://facebook.github.io/zstd/) by Meta -- the reference implementation and RFC 8878 specification
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) -- GPU computing platform
- [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878) -- Zstandard Compression and the `application/zstd` Media Type
- [nvCOMP](https://developer.nvidia.com/nvcomp) -- NVIDIA's GPU compression library (API compatibility target)
