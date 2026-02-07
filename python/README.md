# cuda-zstd

GPU-accelerated Zstandard (ZSTD) compression and decompression using NVIDIA CUDA.

This is the Python package for the [cuda-zstd](https://github.com/RhushabhVaghela/Custom-NVComp-with-ZSTD)
library.  It provides a simple, Pythonic API on top of a high-performance CUDA C++
implementation of the ZSTD compression algorithm.

## Requirements

| Requirement        | Version   | Notes                                              |
|--------------------|-----------|----------------------------------------------------|
| NVIDIA GPU         | Maxwell+  | Compute Capability ≥ 5.0                           |
| CUDA Toolkit       | ≥ 12.0    | `nvcc` must be on `PATH`                           |
| CMake              | ≥ 3.24    | For building the native extension                  |
| C++ compiler       | C++17     | GCC ≥ 9, Clang ≥ 10, or MSVC ≥ 2019               |
| zstd (C library)   | ≥ 1.5     | `libzstd-dev` / `zstd-devel` / Conda `zstd`       |
| Python             | ≥ 3.9     |                                                    |

## Installation

### From source (recommended)

```bash
# Clone the repo and install from the repository root:
git clone https://github.com/RhushabhVaghela/Custom-NVComp-with-ZSTD.git
cd Custom-NVComp-with-ZSTD
pip install .

# Or with test dependencies:
pip install ".[test]"
```

### With specific CUDA architectures

```bash
pip install . -C cmake.define.CMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

## Quick Start

```python
import cuda_zstd

# Simple one-shot API
compressed = cuda_zstd.compress(b"hello world" * 10000, level=3)
original = cuda_zstd.decompress(compressed)
assert original == b"hello world" * 10000

# Reusable manager (faster for repeated calls)
with cuda_zstd.Manager(level=5) as mgr:
    c1 = mgr.compress(data1)
    c2 = mgr.compress(data2)

# Batch API — process many buffers in a single GPU launch
compressed_batch = cuda_zstd.compress_batch([buf1, buf2, buf3], level=3)

# NumPy integration
import numpy as np
arr = np.random.bytes(1024 * 1024)
compressed = cuda_zstd.compress(arr)
```

## API Reference

See the docstrings in `cuda_zstd.__init__` or the
[main repository documentation](https://github.com/RhushabhVaghela/Custom-NVComp-with-ZSTD).

## License

MIT
