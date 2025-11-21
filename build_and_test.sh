#!/bin/bash
set -e

# Clean and create build directory
rm -rf build/
mkdir build
cd build

# Configure
cmake ..

# Build
make -j8

# Run tests
ctest --verbose 2>&1 | tee ../build_and_test_output.txt
