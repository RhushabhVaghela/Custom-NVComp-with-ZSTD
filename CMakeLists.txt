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
# Exclude superseded simple Huffman implementation (ODR conflict with cuda_zstd_huffman.cu
# and contains host memcpy to device pointers — undefined behavior)
list(FILTER CUDA_ZSTD_SOURCES EXCLUDE REGEX ".*huffman_simple.*")

add_library(cuda_zstd STATIC ${CUDA_ZSTD_SOURCES})

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
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()

# ==============================================================================
# Benchmarks
# ==============================================================================
file(GLOB BENCHMARK_SOURCES "benchmarks/benchmark_*.cu")

foreach(bench_src ${BENCHMARK_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE cuda_zstd)
    target_include_directories(${bench_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

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
