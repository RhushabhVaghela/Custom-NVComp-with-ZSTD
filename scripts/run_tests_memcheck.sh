#!/usr/bin/env bash
set -euo pipefail

# Script to run unit tests that are likely to trigger GPU OOB with cuda-memcheck.
# Run from repository root and make sure build/test binaries exist.
# Usage: ./scripts/run_tests_memcheck.sh [<test_pattern>]

PATTERN=${1:-test_roundtrip}
DEBUG_KERNEL_VERIFY=${2:-0}

# Prefer cuda-memcheck, fall back to compute-sanitizer memcheck if available
MEMCHECK_CMD=""
if command -v cuda-memcheck >/dev/null 2>&1; then
    MEMCHECK_CMD="cuda-memcheck"
elif command -v compute-sanitizer >/dev/null 2>&1; then
    MEMCHECK_CMD="compute-sanitizer --tool memcheck"
else
    echo "cuda-memcheck or compute-sanitizer not found in PATH. Install CUDA dev tools or run this from a cuda-enabled system." >&2
    exit 1
fi

# Find tests matching pattern and run them with memcheck
cd $(dirname "$0")/.. >/dev/null
ctest -N | grep -E "${PATTERN}" | awk '{print $2}' | while read -r name; do
    # Find test binary
    TESTBIN="build/${name}"
    if [ -x "${TESTBIN}" ]; then
        echo "Running memcheck for ${TESTBIN} (CUDA_ZSTD_DEBUG_KERNEL_VERIFY=${DEBUG_KERNEL_VERIFY}) using ${MEMCHECK_CMD}"
        CUDA_LAUNCH_BLOCKING=1 CUDA_ZSTD_DEBUG_KERNEL_VERIFY=$DEBUG_KERNEL_VERIFY ${MEMCHECK_CMD} ${TESTBIN} || true
    else
        echo "Test binary ${TESTBIN} not found or not executable" >&2
    fi
done
