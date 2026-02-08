cmake_minimum_required(VERSION 3.24)
project(cuda_zstd VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==============================================================================
# Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Enable Position Independent Code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# Build Options
# ==============================================================================
option(BUILD_SHARED_LIBS "Build cuda_zstd as a shared library instead of static" OFF)
option(CUDA_ZSTD_DEBUG "Enable verbose ZSTD decompress debug output" OFF)
option(CUDA_ZSTD_VERBOSE_PTX "Enable verbose PTX register/spill info (-Xptxas=-v)" OFF)

# ==============================================================================
# Dependencies
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

find_library(ZSTD_LIBRARY zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)

find_package(OpenMP)

# ==============================================================================
# Library Target
# ==============================================================================
file(GLOB_RECURSE CUDA_ZSTD_SOURCES 
    "src/*.cu" 
    "src/*.cpp"
)
# Exclude temporary or backup files if any remain
list(FILTER CUDA_ZSTD_SOURCES EXCLUDE REGEX ".*\\.new$")
list(FILTER CUDA_ZSTD_SOURCES EXCLUDE REGEX ".*\\.bak$")

add_library(cuda_zstd ${CUDA_ZSTD_SOURCES})

target_include_directories(cuda_zstd PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ZSTD_INCLUDE_DIR}
)

target_include_directories(cuda_zstd PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cuda_zstd PUBLIC CUDA::cudart ${ZSTD_LIBRARY})

# Debug flag for ZSTD decompression tracing
if(CUDA_ZSTD_DEBUG)
    target_compile_definitions(cuda_zstd PRIVATE CUDA_ZSTD_DEBUG)
endif()

# CUDA Compiler flags
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(cuda_zstd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        # Release: aggressive optimization
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
        # Debug: device debug info
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
        # RelWithDebInfo: line info for profilers (nsight etc.)
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
    )
    # Verbose PTX output (register usage, spills) — opt-in only
    if(CUDA_ZSTD_VERBOSE_PTX)
        target_compile_options(cuda_zstd PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        )
    endif()
endif()

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()

# FIX: Include .cpp test files alongside .cu and .c
file(GLOB TEST_SOURCES "tests/test_*.cu" "tests/test_*.c" "tests/test_*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda_zstd)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 300)
endforeach()

# Heavy GPU tests need longer timeout — especially under parallel execution
# (GPU contention with -j4 can 3-5x wall time vs isolated runs).
# These tests all pass in <60s when run alone but need headroom for ctest -j4.
set(HEAVY_GPU_TESTS
    test_pipeline_integration   # processes up to 50MB
    test_performance            # sweeps all compression levels
    test_extended_validation    # includes 128MB data
    test_integration            # large integration workload
    test_fallback_strategies    # memory pool stress testing
    test_hybrid                 # exercises GPU+CPU paths with transfers
)
foreach(heavy_test ${HEAVY_GPU_TESTS})
    if(TEST ${heavy_test})
        set_tests_properties(${heavy_test} PROPERTIES TIMEOUT 900)
    endif()
endforeach()

# test_pipeline_integration returns exit code 77 when zstd CLI is unavailable
if(TEST test_pipeline_integration)
    set_tests_properties(test_pipeline_integration PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ==============================================================================
# Benchmarks
# ==============================================================================
file(GLOB BENCHMARK_SOURCES "benchmarks/benchmark_*.cu")

foreach(bench_src ${BENCHMARK_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE cuda_zstd)
    target_include_directories(${bench_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${bench_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Add a custom target to run the benchmark easily
    add_custom_target(run_${bench_name}
        COMMAND ${bench_name}
        DEPENDS ${bench_name}
        COMMENT "Running benchmark ${bench_name}"
    )
endforeach()

# ==============================================================================
# Installation
# ==============================================================================
include(CMakePackageConfigHelpers)

install(TARGETS cuda_zstd
    EXPORT cuda_zstdTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers directly under include/ (not include/cuda_zstd/)
# This matches internal #include "cuda_zstd_types.h" paths
install(DIRECTORY include/ DESTINATION include)

# Generate and install config-mode find_package support
install(EXPORT cuda_zstdTargets
    FILE cuda_zstdTargets.cmake
    NAMESPACE cuda_zstd::
    DESTINATION lib/cmake/cuda_zstd
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_zstdConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_zstdConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_zstdConfig.cmake"
    INSTALL_DESTINATION lib/cmake/cuda_zstd
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_zstdConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_zstdConfigVersion.cmake"
    DESTINATION lib/cmake/cuda_zstd
)
