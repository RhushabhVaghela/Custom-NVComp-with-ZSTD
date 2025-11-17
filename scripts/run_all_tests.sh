#!/usr/bin/env bash
set -euo pipefail

# Single script to build and run each test sequentially with a timeout.
# Usage (from Windows PowerShell):
# wsl bash -lc 'bash scripts/run_all_tests.sh'

ROOT_DIR="$(pwd)"
BUILD_DIR="$ROOT_DIR/build"
LOG_DIR="$ROOT_DIR/build/test_logs"
SUMMARY_FILE="$ROOT_DIR/build/test_results.txt"

# Configurable timeout (seconds)
TEST_TIMEOUT=${TEST_TIMEOUT:-100}
# Optionally make kernel launches synchronous for debugging
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# Build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
make -j8

# Ensure logs and summary
mkdir -p "$LOG_DIR"
: > "$SUMMARY_FILE"

echo "Build completed at: $(date)" | tee -a "$SUMMARY_FILE"

echo "====== GPU INFO ======" | tee -a "$SUMMARY_FILE"
# nvidia-smi may not exist on CI; ignore errors
nvidia-smi 2>&1 | tee -a "$SUMMARY_FILE" || true

echo "====== TESTS LIST (ctest --show-only=json-v1) ======" | tee -a "$SUMMARY_FILE"
# Use ctest JSON if available to enumerate tests (ctest 3.19+), fallback to find
TESTS=""
# Prefer ctest JSON if available (ctest 3.19+), otherwise use `ctest -N`.
if ctest --show-only=json-v1 >/dev/null 2>&1; then
        TESTS=$(ctest --show-only=json-v1 2>/dev/null | grep -Po '"name"\s*:\s*"\K[^"]+' | grep -v -E '^(WORKING_DIRECTORY|TIMEOUT)$' || true)
else
        echo "ctest JSON not available, falling back to enumerating tests with 'ctest -N'" | tee -a "$SUMMARY_FILE"
        TESTS=$(ctest -N 2>/dev/null | sed -n 's/.*: //p' | grep -v -E '^(TIMEOUT|WORKING_DIRECTORY)$' || true)
fi

# Run tests sequentially
for T in $TESTS; do
    echo "\n=== RUNNING: $T ===" | tee -a "$SUMMARY_FILE"
    LOG_FILE="$LOG_DIR/${T}.log"
    echo "Start: $(date)" > "$LOG_FILE"
    echo "Running: $T with timeout ${TEST_TIMEOUT}s" | tee -a "$LOG_FILE" "$SUMMARY_FILE"

    # Run test with timeout; store exit code
    if [ -x "$T" ]; then
            # Wait a fraction to allow GPU to reset in some environments
            sleep 0.5
        # Use timeout to kill hanging tests. We also export CUDA_LAUNCH_BLOCKING to help debug kernel hangs.
        # Run inside a non-fatal context so a failing or timed-out test does not abort the whole script.
        set +e
        CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING timeout ${TEST_TIMEOUT}s bash -lc "./\"$T\"" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
    else
        echo "Executable not found or not executable: $T" | tee -a "$LOG_FILE" "$SUMMARY_FILE"
        EXIT_CODE=127
    fi

    echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE" "$SUMMARY_FILE"
    echo "$T:$EXIT_CODE" >> "$SUMMARY_FILE"

    # If the test timed out or crashed, collect additional diagnostics
    if [ "$EXIT_CODE" -eq 124 ]; then
        echo "Test $T timed out. Capturing GPU and dmesg output..." | tee -a "$LOG_FILE" "$SUMMARY_FILE"
        nvidia-smi 2>&1 | tee -a "$LOG_FILE" "$SUMMARY_FILE" || true
        if command -v dmesg >/dev/null 2>&1; then
            echo "Recent dmesg output:" | tee -a "$LOG_FILE" "$SUMMARY_FILE"
            dmesg -T | tail -n 200 | tee -a "$LOG_FILE" "$SUMMARY_FILE" || true
        fi
    fi

    # Quick pause between tests to clean up the GPU state
    sleep 1
done

# Summary
echo "\n==== TEST SUMMARY ====" | tee -a "$SUMMARY_FILE"
cat "$SUMMARY_FILE" | tee -a "$SUMMARY_FILE"

# Exit non-zero if any test failed (helpful for CI)
FAILED=$(awk -F: '$2!=0{print $0}' "$SUMMARY_FILE" | wc -l)
if [ "$FAILED" -gt 0 ]; then
    echo "Detected $FAILED failed tests" | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "All tests passed successfully." | tee -a "$SUMMARY_FILE"
exit 0
