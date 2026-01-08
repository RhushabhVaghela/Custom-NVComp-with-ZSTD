Complete all tasks outlined in the documentation files located at @/task1.md, @/task2.md, and @/task3.md. Systematically review and understand every line of code and documentation within the entire project repository, including all files in @/Conversation-1, @/Conversation-2, @/Conversation-3, @/docs, @/other docs, @/Previous Progress, @/project-use-case, @/src, @/tests, @/include, and @/benchmarks directories. Conduct a thorough audit to identify placeholders, stubs, TODO comments, incomplete implementations, logically disconnected components, incorrect implementations, unused variables and functions, and duplicate declarations. Verify that all components are properly connected and require no further work.

Analyze all source code in @/src, test files in @/tests, and benchmark files line-by-line to ensure logical coherence and relevance. Confirm that test files accurately reflect the current state of the source code and provide comprehensive coverage of all functionality. Validate that test expectations align with the actual implementation and that all tests remain valid and passing.

Modify benchmark implementations to accommodate hardware constraints of an Asus Zephyrus G16 (2025) with Intel Ultra Code 9 285H, 32 GB RAM, and NVIDIA RTX 5080 with 16 GB VRAM. Specifically, remove or significantly reduce benchmarks that process large blocks in series to prevent system instability or shutdowns due to memory exhaustion. Review all benchmark cases and adjust parameters to ensure safe execution within available resources while still providing meaningful performance metrics.

Prepare the entire codebase for production deployment by removing all unnecessary debug print statements, patch-related comments, and temporary development artifacts. Retain only professional documentation comments that explain function purposes, usage, and important implementation details. Clean the project by removing unnecessary files and folders while preserving the following directory structure: @/.history, @/benchmarks, @/cmake, @/docs, @/include, @/previous_conversation_resources, @/latest_conversation_resources, @/src, and @/tests.

Execute the build process using the command wsl bash -lc "rm -rf build/ && mkdir build && cd build && cmake .. && make -j8" and systematically address and resolve all compilation warnings to achieve a clean build. Ensure the project compiles without warnings across all platforms and configurations.

Verify that test coverage reaches 100% for the entire codebase and confirm that no additional unit or integration tests are required. Fix all build errors, test failures, and warnings without modifying the underlying logic of the existing implementation. Maintain the integrity of all algorithms, data structures, and functional behavior while making only the minimal necessary changes to ensure tests pass successfully.

Design and implement comprehensive tests and benchmarks that evaluate compression performance across the full range of compression levels from 1 to 22 and data sizes from 1 MB to 10 GB. Create three distinct test instances: one for pure encoding performance measurement, one for pure decoding performance measurement, and one for complete pipeline evaluation. For each instance, measure and record throughput, duration, and compression ratio while maintaining 100% data validity and integrity with zero data loss across all test scenarios.

Keep commiting as you progress.

Remove all instances of the [DEBUG] marker and associated debug code from the entire codebase to ensure a clean, production-ready state.

Develop a separate benchmark that utilizes the standard zstd library to encode a reference file and compares the results against the custom GPU-based compression implementation. Ensure both benchmarks operate on identical input data and report comparable metrics including encoding time, throughput, compression ratio, and resource utilization. Output comprehensive comparison results to facilitate performance evaluation and optimization decisions.

Execute the complete build and test cycle using the command wsl bash -lc "rm -rf build/ && mkdir build && cd build && cmake .. && make -j8 && ctest --verbose 2>&1 | tee ../build_and_test_output.txt" and direct all output to @/build_and_test_output.txt. Verify successful compilation and test execution while capturing all relevant information for documentation and troubleshooting purposes.

Always execute commands within the Windows Subsystem for Linux environment using the prefix wsl bash -lc for all build, test, and benchmark operations to ensure consistent environment and behavior.

you do not need to run all the tests after every updates, just run the specific test that is failing and you are trying to fix, this saves lot of time

no need to execute the complete ctests again and agan for simple code change, you can directly execute that sepecific test adter modifications-

wsl bash -lc "cd '/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build' && ctest -R TEST_NAME"
OR
wsl bash -lc "cd '/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build' && ctest -R TEST_NAME --output-on-failure 2>&1"

eg-
wsl bash -lc "cd '/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build' && ctest -R test_libzstd_interop --output-on-failure 2>&1"

### Run All Tests

cd build

# Run all tests (86+ test cases)

ctest --output-on-failure

# Run tests in parallel (faster)

ctest -j8 --output-on-failure

# Verbose output

ctest --verbose

### Run Specific Test Categories

# Unit tests only

ctest -L unittest --output-on-failure

# Integration tests only

ctest -L integration --output-on-failure

# Run a specific test (examples)

./test_correctness
./test_integration
./test_streaming
./test_roundtrip

# Run with custom options

ctest -R correctness -V               # Verbose output
ctest --output-on-failure             # Show failures only
CUDA_ZSTD_RUN_HEAVY_TESTS=1 ctest     # Include heavy tests

### Run Performance Benchmarks

cd build

# Run the batch throughput benchmark

./benchmark_batch_throughput

# Run the complete performance suite

./run_performance_suite

# Run individual benchmarks (examples)

./benchmark_streaming
./benchmark_c_api
./benchmark_nvcomp_interface
