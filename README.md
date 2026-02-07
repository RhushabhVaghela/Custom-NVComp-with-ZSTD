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
[![Tests](https://img.shields.io/badge/Tests-52%2F58%20passing-yellowgreen.svg)]()

**Experimental** GPU-accelerated implementation of the Zstandard (RFC 8878) compression algorithm, built from the ground up in CUDA C++. The entire compression pipeline -- LZ77 match finding, optimal parsing, FSE entropy coding, Huffman coding, and frame assembly -- runs as native CUDA kernels.

> **Status**: This library is under active development. Compression produces RFC 8878 compliant output. Decompression, Huffman encoding, and some FSE edge cases still have known failures (6 of 58 test targets). No authoritative throughput numbers are published yet -- run the benchmark suite on your own hardware to measure.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Building](#building)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Project Status](#project-status)
- [Documentation](#documentation)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

CUDA-ZSTD implements the complete Zstandard compression pipeline as GPU kernels:

- **LZ77 match finding** with parallel hash table construction and tiled match search
- **Optimal parsing** via GPU-parallel dynamic programming with backtracking
- **FSE (Finite State Entropy)** encoding and decoding with RFC 8878 compliant bitstream order
- **Huffman coding** with canonical code generation and batch bit writing
- **Dictionary compression** using the COVER training algorithm
- **Streaming** via chunked independent frames
- **Batch processing** for compressing many buffers in parallel
- **NVCOMP v5 compatible** API surface for drop-in use

The library provides C++ and C APIs, supports compression levels 1-22, and targets RFC 8878 format compliance so that output can be decompressed by standard `libzstd`.

### What This Is

- A research/experimental GPU compression library
- A complete from-scratch CUDA implementation (not a wrapper around `libzstd`)
- RFC 8878 compliant frame format output
- Actively developed, with known issues being fixed

### What This Is Not

- A production-hardened library (yet)
- A drop-in replacement for `libzstd` with guaranteed compatibility for all edge cases
- Benchmarked with published throughput numbers (run benchmarks yourself)

---

## Key Features

### Core Compression

| Feature | Description |
|---------|-------------|
| **Single-shot compress/decompress** | Compress or decompress an entire buffer in one call |
| **Streaming** | Chunked frame compression with `compress_chunk()` / `decompress_chunk()` |
| **Batch processing** | Compress multiple buffers in parallel via `compress_batch()` |
| **Dictionary compression** | Train dictionaries with COVER algorithm; use for better ratios on similar data |
| **Levels 1-22** | Full range of ZSTD compression levels with configurable parameters |
| **Adaptive level selection** | Automatically select compression level based on data characteristics |
| **RFC 8878 compliance** | Frame headers, magic numbers, block types, FSE state ordering, RepCode logic |
| **XXHash64 checksums** | Optional data integrity verification |

### APIs and Integration

| Feature | Description |
|---------|-------------|
| **C++ API** | `ZstdManager`, `ZstdBatchManager`, `ZstdStreamingManager` classes |
| **C API** | `cuda_zstd_compress()`, `cuda_zstd_decompress()`, etc. (in `cuda_zstd_manager.h`) |
| **NVCOMP v5 API** | `NvcompV5BatchManager` for nvCOMP-compatible integration |
| **Inference-ready API** | Pre-allocated workspace, async no-sync decompression for ML pipelines |
| **Factory functions** | `create_manager()`, `create_batch_manager()`, `create_streaming_manager()` |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **GPU memory pool** | Allocation reuse to reduce `cudaMalloc` overhead |
| **Performance profiler** | Per-stage timing (LZ77, FSE, Huffman) and throughput metrics |
| **29 error codes** | Detailed status reporting with `ErrorContext` (file, line, message) |
| **Metadata frames** | Custom skippable frames for embedding compression metadata |
| **Cross-platform** | Linux, Windows, WSL2 |

---

## Architecture

### System Architecture

```
+------------------------------------------------------------------+
|                      APPLICATION LAYER                            |
|         (User Code: C++/C API, Single/Streaming/Batch)           |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                       MANAGER LAYER                               |
|  +--------------+  +-------------+  +----------------+           |
|  |   Default    |  |  Streaming  |  |     Batch      |           |
|  |   Manager    |  |   Manager   |  |    Manager     |           |
|  |  (Single)    |  |  (Chunks)   |  |  (Parallel)    |           |
|  +--------------+  +-------------+  +----------------+           |
|  +--------------+  +-------------+                                |
|  |  Adaptive    |  | Memory Pool |                                |
|  |  Selector    |  |   Manager   |                                |
|  +--------------+  +-------------+                                |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                   COMPRESSION PIPELINE                            |
|  +----------+  +----------+  +--------+  +-------------+        |
|  |   LZ77   |->| Optimal  |->| Seq.   |->| FSE/Huffman |        |
|  | Matching |  | Parsing  |  | Encode |  |  Encoding   |        |
|  +----------+  +----------+  +--------+  +-------------+        |
|  +----------+  +----------+                                       |
|  |   Dict   |  |  XXHash  |                                       |
|  | Training |  | Checksum |                                       |
|  +----------+  +----------+                                       |
+-------------------------------+----------------------------------+
                                |
+-------------------------------+----------------------------------+
|                     CUDA KERNEL LAYER                             |
|            40+ GPU Kernels for all pipeline stages                |
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
3. Optimal Parsing       -- GPU dynamic programming + backtracking
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

### Workspace Memory Layout

```
CompressionWorkspace (7-10 MB for 128KB block):
+--------------------------------------------------+
| Hash Table (512 KB)         131072 entries x 4B   |
+--------------------------------------------------+
| Chain Table (512 KB)        131072 entries x 4B   |
+--------------------------------------------------+
| Match Array (2 MB)          131072 matches x 16B  |
+--------------------------------------------------+
| Cost Array (2 MB)           131073 costs x 16B    |
+--------------------------------------------------+
| Sequence Buffers (1.5 MB)   Reverse buffers       |
+--------------------------------------------------+
| Huffman/FSE Tables (var.)   Symbol tables         |
+--------------------------------------------------+
```

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support
  - The build system uses `CMAKE_CUDA_ARCHITECTURES=native`, so it compiles for whatever GPU is installed
  - Developed and tested on RTX 5080 Laptop (Compute Capability 12.0)
- **VRAM**: Sized to your workload
- **System RAM**: 4 GB minimum

### Software

| Dependency | Minimum Version | Notes |
|------------|-----------------|-------|
| **CUDA Toolkit** | 12.0+ | Developed with CUDA 12.8 |
| **CMake** | 3.24+ | Required for `CMAKE_CUDA_ARCHITECTURES "native"` |
| **C++ Compiler** | C++17 capable | GCC 11+, Clang 14+, or MSVC 19.29+ |
| **libzstd-dev** | 1.4.0+ | Host-side ZSTD dependency |
| **pkg-config** | Any | Library detection |

### Platform Support

| OS | Status |
|----|--------|
| Ubuntu 20.04+ / Debian 11+ | Supported (primary dev platform: WSL2) |
| Fedora / RHEL / CentOS | Supported |
| Windows 10/11 | Supported (Visual Studio 2019+) |
| WSL2 | Supported |
| macOS | Not supported (no CUDA) |

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

---

## Building

```bash
# Clone and build
git clone <repository-url>
cd cuda-zstd

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### CMake Options

```bash
# Release build (default recommendation)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug build with symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Explicit CUDA architectures (instead of native auto-detect)
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"

# Enable debug logging in the library
cmake .. -DCUDA_ZSTD_DEBUG=ON

# Enable verbose PTX output
cmake .. -DCUDA_ZSTD_VERBOSE_PTX=ON
```

### Build Output

```
build/
  libcuda_zstd.a          # Static library
  test_correctness        # Test executables
  test_streaming
  test_roundtrip
  benchmark_*             # Benchmark executables
  ...
```

The build produces a **static library** (`libcuda_zstd.a` / `cuda_zstd.lib`). There is no shared library target or install target currently.

### Linking

```bash
# Link against the static library
g++ -std=c++17 my_app.cpp \
    -I/path/to/cuda-zstd/include \
    -L/path/to/cuda-zstd/build \
    -lcuda_zstd \
    -L${CUDA_HOME}/lib64 -lcudart \
    -lzstd \
    -o my_app
```

Or add as a CMake subdirectory:

```cmake
add_subdirectory(path/to/cuda-zstd)
target_link_libraries(my_app cuda_zstd)
```

---

## Quick Start

### Example 1: Basic Compression (C++)

```cpp
#include <cuda_zstd_manager.h>
#include <vector>
#include <iostream>

int main() {
    using namespace cuda_zstd;

    // 1. Prepare input data
    std::vector<uint8_t> input_data(1024 * 1024, 'A'); // 1 MB

    // 2. Allocate GPU memory
    void *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, input_data.size());
    cudaMemcpy(d_input, input_data.data(), input_data.size(),
               cudaMemcpyHostToDevice);

    // 3. Create compression manager (level 5)
    auto manager = create_manager(5);

    // 4. Query workspace and output sizes
    size_t temp_size = manager->get_compress_temp_size(input_data.size());
    size_t max_output = manager->get_max_compressed_size(input_data.size());

    cudaMalloc(&d_temp, temp_size);
    cudaMalloc(&d_output, max_output);

    // 5. Compress
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
                  << " -> " << compressed_size << " bytes\n";
    }

    // 6. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    return 0;
}
```

### Example 2: Streaming Compression

```cpp
#include <cuda_zstd_manager.h>
#include <fstream>

void compress_large_file(const std::string& filename) {
    using namespace cuda_zstd;

    auto stream_mgr = create_streaming_manager(5);
    stream_mgr->init_compression();
    // Each chunk produces an independent ZSTD frame.

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

### Example 3: C API

```c
#include <cuda_zstd_manager.h>  /* C API is declared here */
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

    /* Compress */
    size_t compressed_size = 0;
    int status = cuda_zstd_compress(
        manager, d_input, input_size,
        d_output, &compressed_size,
        d_temp, temp_size, 0
    );

    if (status == 0) {
        printf("Compressed: %zu bytes\n", compressed_size);
    } else {
        printf("Error: %s\n", cuda_zstd_get_error_string(status));
    }

    cuda_zstd_destroy_manager(manager);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    return 0;
}
```

---

## Advanced Usage

### Batch Processing

Compress multiple buffers in parallel:

```cpp
#include <cuda_zstd_manager.h>

void batch_example() {
    using namespace cuda_zstd;

    auto batch_mgr = create_batch_manager(5);

    // Prepare batch items
    std::vector<BatchItem> items(4);
    for (auto& item : items) {
        size_t size = 64 * 1024; // 64 KB each
        cudaMalloc(&item.input_ptr, size);
        cudaMalloc(&item.output_ptr, size * 2);
        item.input_size = size;
        // ... copy data to item.input_ptr ...
    }

    // Allocate temp workspace
    std::vector<size_t> sizes(items.size(), 64 * 1024);
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

    // Cleanup
    for (auto& item : items) {
        cudaFree(item.input_ptr);
        cudaFree(item.output_ptr);
    }
    cudaFree(d_temp);
}
```

### Dictionary Compression

Train and use dictionaries for better ratios on similar data:

```cpp
#include <cuda_zstd_dictionary.h>
#include <cuda_zstd_manager.h>

void dictionary_example() {
    using namespace cuda_zstd;
    using namespace cuda_zstd::dictionary;

    // Prepare training samples on device
    // Each sample is a pointer + size pair
    std::vector<const void*> sample_ptrs;
    std::vector<size_t> sample_sizes;
    // ... fill with device pointers to training data ...

    // Train dictionary
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

### Performance Profiling

```cpp
#include <performance_profiler.h>
#include <cuda_zstd_manager.h>

void profile_example() {
    using namespace cuda_zstd;

    // Enable profiling
    PerformanceProfiler::enable_profiling(true);

    // ... perform compression ...

    // Get metrics
    const auto& metrics = PerformanceProfiler::get_metrics();

    printf("Total time:      %.2f ms\n", metrics.total_time_ms);
    printf("  LZ77:          %.2f ms\n", metrics.lz77_time_ms);
    printf("  FSE encode:    %.2f ms\n", metrics.fse_encode_time_ms);
    printf("  Huffman:       %.2f ms\n", metrics.huffman_encode_time_ms);
    printf("Throughput:      %.2f MB/s\n", metrics.compression_throughput_mbps);

    // Export to CSV for analysis
    metrics.export_csv("profile.csv");
}
```

---

## API Reference

### Headers

All public API is in `include/`:

| Header | Contents |
|--------|----------|
| `cuda_zstd.h` | Umbrella header (includes everything below) |
| `cuda_zstd_types.h` | `Status` enum (29 codes), `CompressionConfig`, `BatchItem`, `CompressionStats`, constants |
| `cuda_zstd_manager.h` | `ZstdManager`, `ZstdBatchManager`, `ZstdStreamingManager`, factory functions, C API |
| `cuda_zstd_dictionary.h` | Dictionary training (`train_dictionary`), `Dictionary` struct, `DictionaryTrainingParams` |
| `cuda_zstd_nvcomp.h` | `NvcompV5BatchManager`, nvCOMP v5-compatible C API |
| `cuda_zstd_fse.h` | FSE encode/decode functions and table types |
| `cuda_zstd_huffman.h` | Huffman encode/decode functions |
| `cuda_zstd_sequence.h` | `SequenceContext` for match/literal sequence management |
| `cuda_zstd_stream_pool.h` | CUDA stream pool management |
| `cuda_zstd_cuda_ptr.h` | RAII `CudaDevicePtr<T>` wrapper |
| `performance_profiler.h` | `PerformanceProfiler` static class, `DetailedPerformanceMetrics` |

**Note**: There is no separate `cuda_zstd_c_api.h`. The C API functions are declared at the bottom of `cuda_zstd_manager.h` inside `extern "C"` blocks.

### Core Types

```cpp
namespace cuda_zstd {

// Status codes (29 values)
enum class Status {
    SUCCESS = 0,
    ERROR_INVALID_PARAMETER,
    ERROR_OUT_OF_MEMORY,
    ERROR_CUDA_ERROR,
    ERROR_BUFFER_TOO_SMALL,
    ERROR_CORRUPT_DATA,
    ERROR_UNSUPPORTED_FORMAT,
    // ... see cuda_zstd_types.h for full list
};

// Compression configuration
struct CompressionConfig {
    int level = 3;                    // 1-22
    Strategy strategy = Strategy::FAST;
    u32 window_log = 20;
    u32 hash_log = 17;
    u32 chain_log = 17;
    u32 min_match = 3;
    u32 block_size = 131072;          // 128 KB default
    ChecksumPolicy checksum = ChecksumPolicy::NO_COMPUTE_NO_VERIFY;

    static CompressionConfig from_level(int level);
    static CompressionConfig optimal();
    static CompressionConfig get_default();
};

// Factory functions
std::unique_ptr<ZstdManager> create_manager(int level);
std::unique_ptr<ZstdManager> create_manager(const CompressionConfig& config);
std::unique_ptr<ZstdBatchManager> create_batch_manager(int level);
std::unique_ptr<ZstdStreamingManager> create_streaming_manager(int level);

// Convenience functions
Status compress_simple(const void* d_in, size_t in_size,
                       void* d_out, size_t* out_size, int level);
Status decompress_simple(const void* d_in, size_t in_size,
                         void* d_out, size_t* out_size);

} // namespace cuda_zstd
```

### C API

```c
/* Declared in cuda_zstd_manager.h */

cuda_zstd_manager_t* cuda_zstd_create_manager(int level);
void                 cuda_zstd_destroy_manager(cuda_zstd_manager_t* manager);

int cuda_zstd_compress(cuda_zstd_manager_t* manager,
                       const void* d_input, size_t input_size,
                       void* d_output, size_t* output_size,
                       void* d_temp, size_t temp_size,
                       cudaStream_t stream);

int cuda_zstd_decompress(cuda_zstd_manager_t* manager,
                         const void* d_input, size_t input_size,
                         void* d_output, size_t* output_size,
                         void* d_temp, size_t temp_size,
                         cudaStream_t stream);

size_t      cuda_zstd_get_compress_workspace_size(cuda_zstd_manager_t* m, size_t input_size);
const char* cuda_zstd_get_error_string(int status);
int         cuda_zstd_train_dictionary(/* ... */);
```

---

## Testing

### Running Tests

```bash
cd build

# Run all 58 CTest targets
ctest --output-on-failure

# Parallel execution
ctest -j8 --output-on-failure

# Run a specific test by name
ctest -R test_correctness -V

# Run a test executable directly
./test_correctness
./test_streaming
./test_roundtrip
```

All tests have a 120-second timeout configured via CTest.

### Test Status

**52 of 58 CTest targets pass (90%)**. The 6 failures are pre-existing issues in Huffman encoding, FSE encoding, and one performance sub-test:

| Failing Test | Root Cause |
|---|---|
| `test_fse_encoding_gpu` | GPU bitstream differs from CPU reference |
| `test_fse_encoding_host` | CTable symbol entries not fully initialized |
| `test_huffman` | `encode_huffman` fails for 64KB inputs |
| `test_huffman_four_way` | GPU Huffman encode fails (most sub-tests skip) |
| `test_huffman_simple` | GPU Huffman encode fails for certain sizes |
| `test_performance` | Decompression timing sub-test fails |

### Key Test Files

| Test | What It Validates |
|------|-------------------|
| `test_correctness` | RFC 8878 compliance and format correctness |
| `test_roundtrip` | Compress -> decompress -> verify data integrity |
| `test_streaming` | Chunked frame streaming operations |
| `test_integration` | End-to-end compress/decompress workflows |
| `test_fse_*` | FSE encoding/decoding correctness |
| `test_memory_pool*` | GPU memory pool management |
| `test_error_handling` | Error code and status validation |
| `test_c_api` | C API compatibility |
| `test_nvcomp_*` | NVCOMP v5 API surface |
| `test_dictionary` | Dictionary training and compression |

---

## Benchmarks

The `benchmarks/` directory contains 34 benchmark programs covering individual pipeline stages and end-to-end throughput:

```bash
cd build

# Run individual benchmarks
./benchmark_batch_throughput
./benchmark_streaming
./benchmark_lz77
./benchmark_fse
./benchmark_huffman
./benchmark_dictionary_compression
./benchmark_c_api
./benchmark_nvcomp_interface
./benchmark_all_levels
# ... and more (34 total)
```

No pre-computed benchmark results are included. Run on your hardware to get accurate numbers.

### Profiling

```bash
# Profile with NVIDIA Nsight Systems
nsys profile --stats=true ./benchmark_batch_throughput

# Detailed kernel analysis with Nsight Compute
ncu --set full ./benchmark_batch_throughput
```

---

## Project Status

### Code Statistics

| Metric | Count |
|--------|-------|
| Lines of code (src + include) | ~31,000 |
| Header files | 32 |
| Source files | 29 |
| Test files | 67 |
| Benchmark files | 34 |
| CTest targets | 58 |
| Tests passing | 52 (90%) |
| Compression levels | 1-22 |
| Error codes | 29 |

### Implementation Status

| Component | Status |
|-----------|--------|
| LZ77 match finding | Complete |
| Optimal parsing | Complete |
| Sequence encoding | Complete |
| FSE encoding/decoding | Complete (6 edge-case test failures) |
| Huffman coding | Complete (3 test failures in GPU encode path) |
| Frame format (RFC 8878) | Complete |
| Dictionary training (COVER) | Complete |
| Streaming (chunked frames) | Complete |
| Batch processing | Complete |
| C++ API | Complete |
| C API | Complete |
| NVCOMP v5 API | Complete |
| Memory pool | Complete |
| Performance profiler | Complete |
| GPU path enforcement | Complete |

---

## Documentation

Additional documentation is in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [DEBUGLOG.md](DEBUGLOG.md) | Log of bugs found and fixed |
| [docs/QUICK-REFERENCE.md](docs/QUICK-REFERENCE.md) | Quick API reference |
| [docs/BATCH-PROCESSING.md](docs/BATCH-PROCESSING.md) | Batch processing guide |
| [docs/FSE-IMPLEMENTATION.md](docs/FSE-IMPLEMENTATION.md) | FSE algorithm details |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Troubleshooting guide |
| [docs/BENCHMARKING-GUIDE.md](docs/BENCHMARKING-GUIDE.md) | How to run benchmarks |

---

## Future Work

- Fix remaining Huffman GPU encode failures
- Fix FSE encoding edge cases
- Add decompression verification against `libzstd` reference
- Long Distance Matching (LDM)
- Multi-GPU support
- CI/CD pipeline with automated testing
- `cmake install` target and `find_package` support
- Shared library build option
- Python bindings

---

## Acknowledgments

- [Zstandard](https://facebook.github.io/zstd/) by Meta -- the reference implementation and RFC 8878 specification
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) -- GPU computing platform
- [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878) -- Zstandard Compression and the `application/zstd` Media Type
