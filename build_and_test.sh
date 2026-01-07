#!/bin/bash
export PATH="/usr/local/cuda-13.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH"

echo "Check NVCC:"
nvcc --version

echo "Cleaning build_wsl..."
rm -rf build_wsl
mkdir build_wsl
cd build_wsl

echo "Running CMake..."
cmake ..

echo "Running Make..."
make -j

echo "List binaries:"
ls -l

echo "Running Interop Test..."
./test_libzstd_interop

echo "Running Roundtrip Test..."
./test_huffman_roundtrip
