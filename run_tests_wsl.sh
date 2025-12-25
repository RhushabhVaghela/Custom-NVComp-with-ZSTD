#!/bin/bash
set -e

# Define project root (assuming script is run from project root or checks current dir)
PROJECT_ROOT="/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD"
cd "$PROJECT_ROOT"

echo "Compiling test_fse_decode_correctness in WSL..."
nvcc -Iinclude -Icommon -I. \
    tests/test_fse_decode_correctness.cu \
    src/cuda_zstd_fse.cu \
    src/cuda_zstd_utils.cu \
    src/cuda_zstd_xxhash.cu \
    src/linker_stubs.cpp \
    -o test_fse_decode_correctness_wsl \
    -std=c++17 -lcuda -lcudart --extended-lambda -O3 -arch=sm_80

echo "Running test..."
./test_fse_decode_correctness_wsl
