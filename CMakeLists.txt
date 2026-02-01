cmake_minimum_required(VERSION 3.18)

# Fix for PDB path with spaces: Redirect PDBs to safe path
file(MAKE_DIRECTORY "C:/Temp/NVComp_PDBs")
set(CMAKE_PDB_OUTPUT_DIRECTORY "C:/Temp/NVComp_PDBs")
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY "C:/Temp/NVComp_PDBs")

# Fix for PDB path with spaces: Use /Z7 via modern CMake abstraction
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

project(cuda_zstd LANGUAGES CXX CUDA C)

# ==============================================================================
# DEPENDENCIES: ZSTD (for CPU Smart Router)
# ==============================================================================
# Use system-installed zstd library (faster builds, no git dependency issues)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD QUIET libzstd)
endif()

if(NOT ZSTD_FOUND)
    message(STATUS "Zstandard not found via pkg-config. Attempting to download and build via FetchContent...")
    
    include(FetchContent)
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.5
        SOURCE_SUBDIR build/cmake
    )
    
    # ZSTD options to minimize build time/artifacts
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(zstd)
    
    # Set variables expected by the rest of the script
    if(TARGET libzstd_static)
        set(ZSTD_LIBRARIES libzstd_static)
        # Manually set include dir as target property might be missing or generator expression
        set(ZSTD_INCLUDE_DIRS "${zstd_SOURCE_DIR}/lib")
        message(STATUS "Built zstd via FetchContent at ${zstd_SOURCE_DIR}")
        set(ZSTD_FOUND TRUE)
    else()
        message(FATAL_ERROR "Failed to build zstd via FetchContent")
    endif()
endif()

message(STATUS "Found zstd: ${ZSTD_VERSION}")
include_directories(${ZSTD_INCLUDE_DIRS})
link_directories(${ZSTD_LIBRARY_DIRS})

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Fix for MSVC/NVCC space-in-path issue: Use /Z7 (Embed Debug Info) instead of /Zi (PDB)
if(MSVC)
    message(STATUS "Patching flags to use /Z7 instead of /Zi")
    foreach(lang C CXX CUDA)
        # Global flags
        set(var "CMAKE_${lang}_FLAGS")
        string(REPLACE "/Zi" "/Z7" new_flags "${${var}}")
        set(${var} "${new_flags}" CACHE STRING "Flags used by the compiler during all build types." FORCE)
        
        # Config-specific flags
        foreach(type DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
             set(var "CMAKE_${lang}_FLAGS_${type}")
             # Replace /Zi with /Z7
             string(REPLACE "/Zi" "/Z7" new_flags "${${var}}")
             # Also replace -Zi with -Z7 (common variation)
             string(REPLACE "-Zi" "-Z7" new_flags "${new_flags}")
             set(${var} "${new_flags}" CACHE STRING "Flags used by the compiler during ${type} builds." FORCE)
        endforeach()
    endforeach()
    message(STATUS "Fixed CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Options
option(CUDA_ZSTD_BUILD_TESTS "Build test programs" ON)
option(CUDA_ZSTD_BUILD_EXAMPLES "Build example programs" ON)
option(CUDA_ZSTD_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CUDA_ZSTD_DEBUG_BOUNDS "Enable debug bounds checks in CUDA kernels" OFF)
option(CUDA_ZSTD_DEBUG "Enable debug logging" ON)
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
    src/cuda_zstd_fse_reference.cu
    src/cuda_zstd_fse_encoding_kernel.cu
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
    
    # New implementations
    src/streaming_implementation.cpp
    src/ldm_implementation.cu
)

# Static library
add_library(cuda_zstd_static STATIC ${CUDA_ZSTD_SOURCES})
set_target_properties(cuda_zstd_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "cuda_zstd"
)
target_link_libraries(cuda_zstd_static PUBLIC CUDA::cudart Threads::Threads ${ZSTD_LIBRARIES})
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
target_link_libraries(cuda_zstd_shared PUBLIC CUDA::cudart Threads::Threads ${ZSTD_LIBRARIES})
if(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(cuda_zstd_shared PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(CUDA_ZSTD_DEBUG_BOUNDS)
    target_compile_definitions(cuda_zstd_static PUBLIC CUDA_ZSTD_DEBUG_BOUNDS)
    target_compile_definitions(cuda_zstd_shared PUBLIC CUDA_ZSTD_DEBUG_BOUNDS)
endif()

# Examples (simple_test.cu was removed during test cleanup)
# if(CUDA_ZSTD_BUILD_EXAMPLES)
#     add_executable(simple_test tests/simple_test.cu)
#     target_link_libraries(simple_test cuda_zstd_static libzstd_static)
# endif()

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

    # level_nvcomp_demo was removed during test cleanup
    # add_executable(level_nvcomp_demo tests/level_nvcomp_demo.cu)
    # set_target_properties(level_nvcomp_demo PROPERTIES
    #     CUDA_SEPARABLE_COMPILATION ON
    #     CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # )
    # target_link_libraries(level_nvcomp_demo cuda_zstd_static)
    # add_test(NAME level_nvcomp_demo COMMAND level_nvcomp_demo)
    # set_property(TEST level_nvcomp_demo PROPERTY LABELS "unittest")
    # set_tests_properties(level_nvcomp_demo PROPERTIES TIMEOUT 100)
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
    # Export PROJECT_ROOT so CSV files go to project root, not build folder
    target_compile_definitions(comprehensive_benchmark PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
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

    # FSE Decoding Benchmark (Phase 3)
    add_executable(benchmark_fse_decode benchmarks/benchmark_fse_decode.cu)
    set_target_properties(benchmark_fse_decode PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_fse_decode cuda_zstd_static)
    add_test(NAME benchmark_fse_decode COMMAND benchmark_fse_decode)
    set_property(TEST benchmark_fse_decode PROPERTY LABELS "benchmark")

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
        src/cuda_zstd_fse_encoding_kernel.cu
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
    target_link_libraries(benchmark_phase3 PRIVATE ${ZSTD_LIBRARIES} ${CUDA_LIBRARIES})
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

    # FSE Host Fallback Benchmark (measures encode_sequences_with_predefined_fse)
    add_executable(benchmark_fse_host_fallback benchmarks/benchmark_fse_host_fallback.cu)
    set_target_properties(benchmark_fse_host_fallback PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_fse_host_fallback cuda_zstd_static)
    add_test(NAME benchmark_fse_host_fallback COMMAND benchmark_fse_host_fallback)
    set_property(TEST benchmark_fse_host_fallback PROPERTY LABELS "benchmark")


    # Dictionary Compression Benchmark
    add_executable(benchmark_dictionary_compression benchmarks/benchmark_dictionary_compression.cu)
    set_target_properties(benchmark_dictionary_compression PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_dictionary_compression cuda_zstd_static)
    add_test(NAME benchmark_dictionary_compression COMMAND benchmark_dictionary_compression)
    set_property(TEST benchmark_dictionary_compression PROPERTY LABELS "benchmark")

    # XXHash Checksum Benchmark (100% coverage)
    add_executable(benchmark_xxhash benchmarks/benchmark_xxhash.cu)
    set_target_properties(benchmark_xxhash PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_xxhash cuda_zstd_static)
    add_test(NAME benchmark_xxhash COMMAND benchmark_xxhash)
    set_property(TEST benchmark_xxhash PROPERTY LABELS "benchmark")

    # Adaptive Level Selection Benchmark (100% coverage)
    add_executable(benchmark_adaptive benchmarks/benchmark_adaptive.cu)
    set_target_properties(benchmark_adaptive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_adaptive cuda_zstd_static)
    add_test(NAME benchmark_adaptive COMMAND benchmark_adaptive)
    set_property(TEST benchmark_adaptive PROPERTY LABELS "benchmark")

    # Stream Pool Performance Benchmark (100% coverage)
    add_executable(benchmark_stream_pool benchmarks/benchmark_stream_pool.cu)
    set_target_properties(benchmark_stream_pool PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_stream_pool cuda_zstd_static)
    add_test(NAME benchmark_stream_pool COMMAND benchmark_stream_pool)
    set_property(TEST benchmark_stream_pool PROPERTY LABELS "benchmark")

    # Performance Suite Runner
    add_executable(run_performance_suite benchmarks/run_performance_suite.cu)
    set_target_properties(run_performance_suite PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(run_performance_suite cuda_zstd_static)
    add_test(NAME run_performance_suite COMMAND run_performance_suite)
    set_property(TEST run_performance_suite PROPERTY LABELS "benchmark")

    # nvComp Interface Throughput Benchmark (100% coverage)
    add_executable(benchmark_nvcomp_interface benchmarks/benchmark_nvcomp_interface.cu)
    set_target_properties(benchmark_nvcomp_interface PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_nvcomp_interface cuda_zstd_static)
    add_test(NAME benchmark_nvcomp_interface COMMAND benchmark_nvcomp_interface)
    set_property(TEST benchmark_nvcomp_interface PROPERTY LABELS "benchmark")

    # C API Performance Benchmark (100% coverage)
    add_executable(benchmark_c_api benchmarks/benchmark_c_api.cu)
    set_target_properties(benchmark_c_api PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_c_api cuda_zstd_static)
    add_test(NAME benchmark_c_api COMMAND benchmark_c_api)
    set_property(TEST benchmark_c_api PROPERTY LABELS "benchmark")

    # Streaming Manager Performance Benchmark (100% coverage)
    add_executable(benchmark_streaming benchmarks/benchmark_streaming.cu)
    set_target_properties(benchmark_streaming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_streaming cuda_zstd_static)
    add_test(NAME benchmark_streaming COMMAND benchmark_streaming)
    set_property(TEST benchmark_streaming PROPERTY LABELS "benchmark")

    # Streaming Comparison Benchmark (basic vs with history)
    add_executable(benchmark_streaming_comparison benchmarks/benchmark_streaming_comparison.cu)
    set_target_properties(benchmark_streaming_comparison PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_streaming_comparison cuda_zstd_static)
    add_test(NAME benchmark_streaming_comparison COMMAND benchmark_streaming_comparison)
    set_property(TEST benchmark_streaming_comparison PROPERTY LABELS "benchmark")

    # Pipeline Streaming Benchmark (Phase 2b)
    add_executable(benchmark_pipeline_streaming benchmarks/benchmark_pipeline_streaming.cu)
    set_target_properties(benchmark_pipeline_streaming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_pipeline_streaming cuda_zstd_static)
    add_test(NAME benchmark_pipeline_streaming COMMAND benchmark_pipeline_streaming)
    set_property(TEST benchmark_pipeline_streaming PROPERTY LABELS "benchmark")


    # Batch Throughput Benchmark
    add_executable(benchmark_batch_throughput benchmarks/benchmark_batch_throughput.cu)
    set_target_properties(benchmark_batch_throughput PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    find_package(OpenMP REQUIRED)
    target_link_libraries(benchmark_batch_throughput cuda_zstd_static OpenMP::OpenMP_CXX)
    add_test(NAME benchmark_batch_throughput COMMAND benchmark_batch_throughput)
    set_property(TEST benchmark_batch_throughput PROPERTY LABELS "benchmark")

    # Comprehensive Levels Benchmark (Levels 1-22, multiple data sizes)
    add_executable(benchmark_comprehensive_levels benchmarks/benchmark_comprehensive_levels.cu)
    set_target_properties(benchmark_comprehensive_levels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_comprehensive_levels cuda_zstd_static)
    add_test(NAME benchmark_comprehensive_levels COMMAND benchmark_comprehensive_levels)
    set_property(TEST benchmark_comprehensive_levels PROPERTY LABELS "benchmark")

    # GPU vs Standard ZSTD Comparison Benchmark
    add_executable(benchmark_zstd_comparison benchmarks/benchmark_zstd_comparison.cu)
    set_target_properties(benchmark_zstd_comparison PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(benchmark_zstd_comparison cuda_zstd_static)
    add_test(NAME benchmark_zstd_comparison COMMAND benchmark_zstd_comparison)
    set_property(TEST benchmark_zstd_comparison PROPERTY LABELS "benchmark")

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


