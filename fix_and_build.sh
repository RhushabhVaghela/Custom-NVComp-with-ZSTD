#!/bin/bash
cd "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD"
sed -i '1485,1600d' src/cuda_zstd_manager.cu
echo "✓ Removed problematic lines from cuda_zstd_manager.cu"
rm -rf build && mkdir build && cd build && cmake .. && make -j8
