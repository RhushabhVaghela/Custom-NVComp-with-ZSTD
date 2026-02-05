cmake_minimum_required(VERSION 3.18)
project(cuda_zstd LANGUAGES CXX CUDA)

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
# Avoid compiling legacy stream pool implementation
list(REMOVE_ITEM CUDA_ZSTD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/stream_pool.cu")
# Exclude temporary or backup files if any remain
list(FILTER CUDA_ZSTD_SOURCES EXCLUDE REGEX ".*\\.new$")
list(FILTER CUDA_ZSTD_SOURCES EXCLUDE REGEX ".*\\.bak$")

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

# Optimization flags
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(cuda_zstd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        $<$<CONFIG:Release>:-O3>
    )
endif()

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()

file(GLOB TEST_SOURCES "tests/test_*.cu" "tests/test_*.c")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    
    # Handle C files separately if needed, but nvcc can handle .c via -x c
    # For simplicity, compile all as part of CUDA target or CXX executable linked with cuda_zstd
    
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda_zstd)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    
    add_test(NAME ${test_name} COMMAND ${test_name})
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
# Installation (Optional)
# ==============================================================================
install(TARGETS cuda_zstd
    EXPORT cuda_zstdTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/cuda_zstd)
