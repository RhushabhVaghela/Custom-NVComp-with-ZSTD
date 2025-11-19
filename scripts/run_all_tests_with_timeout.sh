#!/bin/bash

# ==============================================================================
# run_all_tests_with_timeout.sh - Run all tests individually with timeout
# ==============================================================================

set +e  # Don't exit on error, continue running all tests

# Configuration
TIMEOUT=100  # 100 seconds timeout per test
BUILD_DIR="build"
LOG_DIR="test_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Summary variables
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TIMEOUT_TESTS=0

echo "========================================================================"
echo "  CUDA ZSTD - Test Suite Runner with Timeout"
echo "  Timeout: ${TIMEOUT} seconds per test"
echo "  Timestamp: ${TIMESTAMP}"
echo "========================================================================"
echo ""

# Get list of all test executables
cd "$BUILD_DIR" || exit 1
TEST_EXECUTABLES=$(find . -maxdepth 1 -type f -executable -name "test_*" -o -name "*_test" -o -name "simple_test" -o -name "level_nvcomp_demo" | sort)

if [ -z "$TEST_EXECUTABLES" ]; then
    echo -e "${RED}ERROR: No test executables found in $BUILD_DIR${NC}"
    exit 1
fi

echo "Found $(echo "$TEST_EXECUTABLES" | wc -l) test executables"
echo ""

# Function to run a single test with timeout
run_test() {
    local test_exe=$1
    local test_name=$(basename "$test_exe")
    local log_file="../${LOG_DIR}/${test_name}_${TIMESTAMP}.log"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo "========================================================================"
    echo -e "${BLUE}[${TOTAL_TESTS}] Running: ${test_name}${NC}"
    echo "------------------------------------------------------------------------"
    
    # Run test with timeout
    timeout ${TIMEOUT}s "$test_exe" > "$log_file" 2>&1
    local exit_code=$?
    
    # Check result
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [ $exit_code -eq 124 ]; then
        echo -e "${YELLOW}⏱ TIMEOUT (exceeded ${TIMEOUT}s)${NC}"
        TIMEOUT_TESTS=$((TIMEOUT_TESTS + 1))
        echo "TIMEOUT: Test exceeded ${TIMEOUT} seconds" >> "$log_file"
    else
        echo -e "${RED}✗ FAILED (exit code: $exit_code)${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        # Show last 10 lines of error
        echo "Last 10 lines of output:"
        tail -n 10 "$log_file" | sed 's/^/  /'
    fi
    
    echo "Log: $log_file"
    echo ""
}

# Run all tests
for test_exe in $TEST_EXECUTABLES; do
    run_test "$test_exe"
done

# Print summary
echo "========================================================================"
echo "  TEST SUMMARY"
echo "========================================================================"
echo -e "Total Tests:    ${TOTAL_TESTS}"
echo -e "${GREEN}Passed:         ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed:         ${FAILED_TESTS}${NC}"
echo -e "${YELLOW}Timeout:        ${TIMEOUT_TESTS}${NC}"
echo "------------------------------------------------------------------------"

if [ $FAILED_TESTS -eq 0 ] && [ $TIMEOUT_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED OR TIMED OUT${NC}"
    echo ""
    echo "Failed/Timeout test logs are in: $LOG_DIR/"
    exit 1
fi
