cmake_minimum_required(VERSION 3.18)
project(cuda_zstd LANGUAGES CXX CUDA C)

# ==============================================================================
# DEPENDENCIES: ZSTD (for CPU Smart Router)
# ==============================================================================
# Suppress ZSTD compiler flag checks that fail on GCC/NVCC
set(C_FLAG_QUNUSED_ARGUMENTS 0 CACHE INTERNAL "Suppress check")
set(CXX_FLAG_QUNUSED_ARGUMENTS 0 CACHE INTERNAL "Suppress check")

# Suppress deprecation warnings (e.g. from ZSTD legacy CMake)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        v1.5.5
    SOURCE_SUBDIR  build/cmake
)
FetchContent_MakeAvailable(zstd)
FetchContent_GetProperties(zstd)
include_directories(${zstd_SOURCE_DIR}/lib)

# Suppress warnings from ZSTD legacy code
if(TARGET libzstd_static)
    target_compile_options(libzstd_static PRIVATE -Wno-maybe-uninitialized)
endif()
if(TARGET libzstd_shared)
    target_compile_options(libzstd_shared PRIVATE -Wno-maybe-uninitialized)
endif()

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Options
option(CUDA_ZSTD_BUILD_TESTS "Build test programs" ON)
option(CUDA_ZSTD_BUILD_EXAMPLES "Build example programs" ON)
option(CUDA_ZSTD_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CUDA_ZSTD_DEBUG_BOUNDS "Enable debug bounds checks in CUDA kernels" OFF)
option(CUDA_ZSTD_DEBUG "Enable debug logging" OFF)
option(BUILD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(CUDA_ZSTD_DEBUG)
    add_compile_definitions(CUDA_ZSTD_DEBUG)
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100 /wd4101")
endif()

if(BUILD_WARNINGS_AS_ERRORS)
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Source files
set(CUDA_ZSTD_SOURCES
    src/cuda_zstd_c_api.cpp
    src/cuda_zstd_types.cpp
    src/cuda_zstd_manager.cu
    src/cuda_zstd_lz77.cu
    src/cuda_zstd_fse.cu
    src/cuda_zstd_huffman.cu
    src/cuda_zstd_dictionary.cu
    src/cuda_zstd_sequence.cu
    src/cuda_zstd_hash.cu
    src/cuda_zstd_xxhash.cu
    src/cuda_zstd_nvcomp.cpp
    src/cuda_zstd_utils.cu
    src/cuda_zstd_utils.cpp
    src/cuda_zstd_memory_pool_complex.cu
    src/cuda_zstd_stacktrace.cpp
    src/cuda_zstd_adaptive.cu
    src/cuda_zstd_stream_pool.cpp

    # Workspace pattern implementations
    src/workspace_manager.cu
    src/error_context.cpp
    src/lz77_parallel.cu
    src/pipeline_manager.cu
)

# Static library
add_library(cuda_zstd_static STATIC ${CUDA_ZSTD_SOURCES})
set_target_properties(cuda_zstd_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "cuda_zstd"
)
target_link_libraries(cuda_zstd_static PUBLIC CUDA::cudart Threads::Threads libzstd_static)
if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(cuda_zstd_static PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Shared library
add_library(cuda_zstd_shared SHARED ${CUDA_ZSTD_SOURCES})
set_target_properties(cuda_zstd_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "cuda_zstd"
    VERSION 1.0.0
)
target_link_libraries(cuda_zstd_shared PUBLIC CUDA::cudart Threads::Threads libzstd_static)
if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(cuda_zstd_shared PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(CUDA_ZSTD_DEBUG_BOUNDS)
    target_compile_definitions(cuda_zstd_static PUBLIC CUDA_ZSTD_DEBUG_BOUNDS)
    target_compile_definitions(cuda_zstd_shared PUBLIC CUDA_ZSTD_DEBUG_BOUNDS)
endif()

# Examples
if(CUDA_ZSTD_BUILD_EXAMPLES)
    add_executable(simple_test tests/simple_test.cu)
    target_link_libraries(simple_test cuda_zstd_static libzstd_static)
endif()

# Tests
if(CUDA_ZSTD_BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_CU_FILES "tests/test_*.cu")
    # NOTE: All tests are now enabled - removed exclusion filter

    foreach(test_file ${TEST_CU_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_link_libraries(${test_name} cuda_zstd_static)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_property(TEST ${test_name} PROPERTY LABELS "unittest")

        # Set timeout for tests (300 seconds)
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 300)
    endforeach()




    # C-API test
    add_executable(test_c_api tests/test_c_api.c)
    set_source_files_properties(tests/test_c_api.c PROPERTIES LANGUAGE CXX)
    target_link_libraries(test_c_api cuda_zstd_static)
    add_test(NAME test_c_api COMMAND test_c_api)
    set_property(TEST test_c_api PROPERTY LABELS "unittest")
    set_tests_properties(test_c_api PROPERTIES TIMEOUT 100)

    # level_nvcomp_demo test
    add_executable(level_nvcomp_demo tests/level_nvcomp_demo.cu)
    set_target_properties(level_nvcomp_demo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(level_nvcomp_demo cuda_zstd_static)
    add_test(NAME level_nvcomp_demo COMMAND level_nvcomp_demo)
    set_property(TEST level_nvcomp_demo PROPERTY LABELS "unittest")
    set_tests_properties(level_nvcomp_demo PROPERTIES TIMEOUT 100)
endif()

# Benchmarks
if(CUDA_ZSTD_BUILD_BENCHMARKS)
    # Include tests directory for shared headers if needed
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tests)

    add_executable(fixed_benchmark benchmarks/fixed_benchmark.cu)
    set_target_properties(fixed_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(fixed_benchmark cuda_zstd_static)
    add_test(NAME fixed_benchmark COMMAND fixed_benchmark)
    set_property(TEST fixed_benchmark PROPERTY LABELS "benchmark")
    

    
    add_executable(limited_benchmark benchmarks/limited_benchmark.cu)
    set_target_properties(limited_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(limited_benchmark cuda_zstd_static)
    add_test(NAME limited_benchmark COMMAND limited_benchmark)
    set_property(TEST limited_benchmark PROPERTY LABELS "benchmark")
    
    add_executable(comprehensive_benchmark benchmarks/comprehensive_benchmark.cu)
    set_target_properties(comprehensive_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(comprehensive_benchmark cuda_zstd_static)
    add_test(NAME comprehensive_benchmark COMMAND comprehensive_benchmark)
    set_property(TEST comprehensive_benchmark PROPERTY LABELS "benchmark")

    add_executable(benchmark_lz77 benchmarks/benchmark_lz77.cu)
    set_target_properties(benchmark_lz77 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_lz77 cuda_zstd_static)
    add_test(NAME benchmark_lz77 COMMAND benchmark_lz77)
    set_property(TEST benchmark_lz77 PROPERTY LABELS "benchmark")


    add_executable(benchmark_parallel_throughput benchmarks/benchmark_parallel_throughput.cu)
    set_target_properties(benchmark_parallel_throughput PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_parallel_throughput cuda_zstd_static)
    add_test(NAME benchmark_parallel_throughput COMMAND benchmark_parallel_throughput)
    set_property(TEST benchmark_parallel_throughput PROPERTY LABELS "benchmark")

    add_executable(benchmark_memory_alloc benchmarks/benchmark_memory_alloc.cu)
    set_target_properties(benchmark_memory_alloc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_memory_alloc cuda_zstd_static)
    add_test(NAME benchmark_memory_alloc COMMAND benchmark_memory_alloc)
    set_property(TEST benchmark_memory_alloc PROPERTY LABELS "benchmark")

    add_executable(benchmark_pipeline tests/benchmark_pipeline.cu)
    set_target_properties(benchmark_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_pipeline cuda_zstd_static)
    add_test(NAME benchmark_pipeline COMMAND benchmark_pipeline)
    set_property(TEST benchmark_pipeline PROPERTY LABELS "benchmark")

    add_executable(benchmark_fse_parallel tests/benchmark_fse_parallel.cu)
    set_target_properties(benchmark_fse_parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_fse_parallel cuda_zstd_static)
    add_test(NAME benchmark_fse_parallel COMMAND benchmark_fse_parallel)
    set_property(TEST benchmark_fse_parallel PROPERTY LABELS "benchmark")

    # Batch encoding benchmark (test_batch_fse is auto-discovered via glob)
    add_executable(benchmark_batch_fse tests/benchmark_batch_fse.cu)
    set_target_properties(benchmark_batch_fse PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_batch_fse cuda_zstd_static)
    add_test(NAME benchmark_batch_fse COMMAND benchmark_batch_fse)
    set_property(TEST benchmark_batch_fse PROPERTY LABELS "benchmark")

    # Phase 3 Benchmark
    # Compile dependencies directly to access internal functions (encode_fse_advanced)
    add_executable(benchmark_phase3 benchmarks/benchmark_phase3.cu 
        src/cuda_zstd_fse.cu 
        src/cuda_zstd_utils.cu 
        src/cuda_zstd_manager.cu
        src/cuda_zstd_c_api.cpp
        src/cuda_zstd_types.cpp
        src/cuda_zstd_dictionary.cu
        src/cuda_zstd_sequence.cu
        src/cuda_zstd_hash.cu
        src/cuda_zstd_xxhash.cu
        src/cuda_zstd_nvcomp.cpp
        src/cuda_zstd_utils.cpp
        src/cuda_zstd_memory_pool_complex.cu
        src/cuda_zstd_stacktrace.cpp
        src/cuda_zstd_adaptive.cu
        src/cuda_zstd_stream_pool.cpp
        src/workspace_manager.cu
        src/error_context.cpp
        src/lz77_parallel.cu
        src/pipeline_manager.cu 
        src/cuda_zstd_lz77.cu
        src/cuda_zstd_huffman.cu
    )
    target_link_libraries(benchmark_phase3 PRIVATE libzstd_static ${CUDA_LIBRARIES})
    set_target_properties(benchmark_phase3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME benchmark_phase3 COMMAND benchmark_phase3)
    set_property(TEST benchmark_phase3 PROPERTY LABELS "benchmark")

    # GPU-to-GPU FSE Benchmark (pure kernel performance)
    add_executable(benchmark_fse_gpu benchmarks/benchmark_fse_gpu.cu)
    set_target_properties(benchmark_fse_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_fse_gpu cuda_zstd_static)
    add_test(NAME benchmark_fse_gpu COMMAND benchmark_fse_gpu)
    set_property(TEST benchmark_fse_gpu PROPERTY LABELS "benchmark")

    # Pipelined Streaming Benchmark
    add_executable(benchmark_pipeline_streaming benchmarks/benchmark_pipeline.cu)
    set_target_properties(benchmark_pipeline_streaming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_pipeline_streaming cuda_zstd_static)
    add_test(NAME benchmark_pipeline_streaming COMMAND benchmark_pipeline_streaming)
    set_property(TEST benchmark_pipeline_streaming PROPERTY LABELS "benchmark")

    # Dictionary Compression Benchmark
    add_executable(benchmark_dictionary_compression benchmarks/benchmark_dictionary_compression.cu)
    set_target_properties(benchmark_dictionary_compression PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_dictionary_compression cuda_zstd_static)
    add_test(NAME benchmark_dictionary_compression COMMAND benchmark_dictionary_compression)
    set_property(TEST benchmark_dictionary_compression PROPERTY LABELS "benchmark")


endif()


# Installation
install(TARGETS cuda_zstd_static cuda_zstd_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/cuda_zstd)

# Summary
message(STATUS "")
message(STATUS "=========================================")
message(STATUS "CUDA Zstandard Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Test Timeout: 100 seconds")
message(STATUS "=========================================")
