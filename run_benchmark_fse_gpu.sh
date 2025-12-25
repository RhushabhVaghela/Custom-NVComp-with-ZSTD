#!/bin/bash
# Run FSE GPU-to-GPU benchmark

echo "=============================================="
echo "  Building GPU-to-GPU FSE Benchmark..."
echo "=============================================="

nvcc -O3 -std=c++17 -arch=sm_89 \
    -I./include \
    -o benchmark_fse_gpu \
    benchmarks/benchmark_fse_gpu.cu \
    src/cuda_zstd_fse.cu \
    src/cuda_zstd_types.cpp \
    -lcuda 2>&1 | head -30

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Running GPU-to-GPU FSE Benchmark..."
echo ""
./benchmark_fse_gpu
