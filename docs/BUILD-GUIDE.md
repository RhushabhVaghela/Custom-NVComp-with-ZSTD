# CUDA-ZSTD Build System Guide

## Overview

The build system uses CMake 3.18+ with CUDA support, producing static and shared libraries plus a comprehensive test suite.

## Quick Start

```bash
# Clone, configure, build
git clone https://github.com/your-org/cuda-zstd.git
cd cuda-zstd
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## CMake Configuration

### Basic Options

```bash
# Library type
-DCUDA_ZSTD_BUILD_SHARED=ON    # Build shared library (default: ON)
-DCUDA_ZSTD_BUILD_STATIC=ON    # Build static library (default: ON)

# Components
-DCUDA_ZSTD_BUILD_TESTS=ON     # Build tests (default: ON)
-DCUDA_ZSTD_BUILD_BENCHMARKS=ON # Build benchmarks (default: ON)

# Build type
-DCMAKE_BUILD_TYPE=Release     # Release/Debug/RelWithDebInfo
```

### CUDA Configuration

```bash
# GPU architectures (compute capability)
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

# CUDA toolkit path (if not auto-detected)
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Separable compilation (for device linking)
-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON
```

### Advanced Options

```bash
# Debug features
-DCUDA_ZSTD_ENABLE_DEBUG_LOGS=ON    # Debug output
-DCUDA_ZSTD_ENABLE_PROFILING=ON     # Perf timing
-DCUDA_ZSTD_ENABLE_SANITIZERS=ON    # Address sanitizer

# Optimization
-DCMAKE_CUDA_FLAGS="-O3 --use_fast_math"

# Installation prefix
-DCMAKE_INSTALL_PREFIX=/usr/local
```

## Build Targets

| Target | Description |
|:-------|:------------|
| `cuda_zstd_static` | Static library |
| `cuda_zstd_shared` | Shared library |
| `test_*` | Individual test executables |
| `benchmark_*` | Benchmark executables |
| `all` | All targets |

### Building Specific Targets

```bash
cmake --build . --target cuda_zstd_static
cmake --build . --target test_correctness
cmake --build . --target benchmark_batch_throughput
```

## Platform-Specific Notes

### Linux
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Windows (Visual Studio)
```cmd
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release -j 8
```

### WSL2
```bash
# Ensure CUDA toolkit is installed in WSL
sudo apt install nvidia-cuda-toolkit
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86" ..
make -j$(nproc)
```

## Build Artifacts

```
build/
├── libcuda_zstd_static.a       # Static library
├── libcuda_zstd_shared.so      # Shared library (Linux)
├── cuda_zstd.dll               # Shared library (Windows)
├── libzstd_static.a            # Bundled ZSTD (CPU)
├── test_*                      # Test executables
├── benchmark_*                 # Benchmark executables
└── CMakeCache.txt              # Build configuration
```

## Running Tests

```bash
# All tests
ctest --output-on-failure

# Parallel execution
ctest -j8

# Specific test
./test_correctness

# With verbose output
ctest --verbose
```

## Installing

```bash
# System-wide installation
sudo cmake --install .

# Custom prefix
cmake --install . --prefix /opt/cuda-zstd
```

## Troubleshooting

### CUDA not found
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc ..
```

### Architecture mismatch
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Use matching architecture
cmake -DCMAKE_CUDA_ARCHITECTURES="86" ..
```

### Linking errors
```bash
# Ensure CUDA runtime is linked
cmake -DCMAKE_CUDA_FLAGS="-lcudart" ..
```

## Source Files

| File | Description |
|:-----|:------------|
| `CMakeLists.txt` | Main build configuration |
| `cmake/FindCUDA.cmake` | CUDA detection module |

## Related Documentation
- [TESTING-GUIDE.md](TESTING-GUIDE.md)
- [ARCHITECTURE-OVERVIEW.md](ARCHITECTURE-OVERVIEW.md)
