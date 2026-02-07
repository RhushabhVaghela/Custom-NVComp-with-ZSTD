# CUDA-ZSTD Build System Guide

## Overview

The build system uses CMake 3.24+ with CUDA support, producing a static library plus tests and benchmarks.

## Requirements

| Requirement | Minimum Version | Notes |
|:------------|:----------------|:------|
| CMake | 3.24+ | Required for CUDA language support |
| CUDA Toolkit | 12.0+ | CUDAToolkit CMake component |
| C++ Standard | C++17 | Set automatically by CMake |
| libzstd | System install | CPU fallback and dictionary training |
| OpenMP | Optional | Enables multi-threaded batch patterns |

## Quick Start

```bash
# Clone, configure, build
git clone https://github.com/RhushabhVaghela/Custom-NVComp-with-ZSTD.git
cd cuda-zstd
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## CMake Configuration

### Basic Options

```bash
# Build type
-DCMAKE_BUILD_TYPE=Release     # Release/Debug/RelWithDebInfo
```

### CUDA Configuration

```bash
# GPU architectures (compute capability)
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120"

# CUDA toolkit path (if not auto-detected)
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Separable compilation (for device linking)
-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON
```

### Project-Specific Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `CUDA_ZSTD_DEBUG` | `OFF` | Enable debug-level logging and assertions |
| `CUDA_ZSTD_VERBOSE_PTX` | `OFF` | Emit verbose PTX output during compilation |
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration (Release/Debug/RelWithDebInfo) |

```bash
# Example: debug build with verbose PTX
cmake -DCUDA_ZSTD_DEBUG=ON -DCUDA_ZSTD_VERBOSE_PTX=ON -DCMAKE_BUILD_TYPE=Debug ..
```

### Advanced Options

```bash
# Optimization
-DCMAKE_CUDA_FLAGS="-O3 --use_fast_math"

# Installation prefix
-DCMAKE_INSTALL_PREFIX=/usr/local
```

## Build Targets

| Target | Description |
|:-------|:------------|
| `cuda_zstd` | Static library |
| `test_*` | Individual test executables (67 tests) |
| `benchmark_*` | Benchmark executables (30 benchmarks) |
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

### Windows (Visual Studio 2019/2022)
```cmd
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release -j 8
```

For Visual Studio 2019:
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
+-- lib/
|   +-- libcuda_zstd.a           # Static library
+-- bin/
|   +-- test_*                   # Test executables (67)
|   +-- benchmark_*              # Benchmark executables (30)
+-- CMakeCache.txt               # Build configuration
```

## Running Tests

```bash
# All tests (67 tests, 100% passing)
ctest --output-on-failure

# Parallel execution
ctest -j8

# Specific test
./test_correctness

# With verbose output
ctest --verbose
```

## Python Package

The project includes a Python package with bindings to the CUDA-ZSTD library.

```bash
# Install in development mode
cd python/
pip install -e .

# Verify installation
python -c "import cuda_zstd; print(cuda_zstd.__version__)"
```

The Python package supports `compress()`, `decompress()`, `compress_batch()`, and a `Manager` context manager. See [QUICK-REFERENCE.md](QUICK-REFERENCE.md) for Python examples.

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

### libzstd not found
```bash
# Ubuntu/Debian
sudo apt install libzstd-dev

# RHEL/CentOS
sudo yum install libzstd-devel

# macOS
brew install zstd
```

## Source Files

| File | Description |
|:-----|:------------|
| `CMakeLists.txt` | Main build configuration |
| `cmake/cuda_zstdConfig.cmake.in` | CMake package config template |

## Related Documentation

- [TESTING-GUIDE.md](TESTING-GUIDE.md)
- [BENCHMARKING-GUIDE.md](BENCHMARKING-GUIDE.md)
- [ARCHITECTURE-OVERVIEW.md](ARCHITECTURE-OVERVIEW.md)
