cmake_minimum_required(VERSION 3.24)
project(cuda_zstd_python VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==============================================================================
# Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to native GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# pybind11 — prefer scikit-build's bundled copy, fall back to system
find_package(pybind11 CONFIG REQUIRED)

# zstd (CPU reference library, needed by the parent CUDA library)
find_library(ZSTD_LIBRARY zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)

# OpenMP (optional, used by some parent source files)
find_package(OpenMP QUIET)

# ==============================================================================
# Build the parent cuda_zstd library as a static dependency
# ==============================================================================
# We add the parent project's CMakeLists.txt as a subdirectory so that
# `pip install .` builds everything from source.  The parent target is
# called `cuda_zstd` (a STATIC library).
#
# Guard: only add_subdirectory if the target doesn't already exist (avoids
# double-add when the user is building from the repo root).
if(NOT TARGET cuda_zstd)
    # Collect all source files exactly like the parent CMakeLists.txt does
    file(GLOB_RECURSE _PARENT_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp"
    )
    list(FILTER _PARENT_SOURCES EXCLUDE REGEX ".*\\.new$")
    list(FILTER _PARENT_SOURCES EXCLUDE REGEX ".*\\.bak$")

    add_library(cuda_zstd STATIC ${_PARENT_SOURCES})

    target_include_directories(cuda_zstd PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ${ZSTD_INCLUDE_DIR}
    )
    target_include_directories(cuda_zstd PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    )
    target_link_libraries(cuda_zstd PUBLIC CUDA::cudart ${ZSTD_LIBRARY})

    # CUDA compiler flags matching the parent project
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(cuda_zstd PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
        )
    endif()
endif()

# ==============================================================================
# pybind11 extension module: cuda_zstd._core
# ==============================================================================
pybind11_add_module(_core MODULE src/binding.cpp)

target_include_directories(_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    ${ZSTD_INCLUDE_DIR}
)

target_link_libraries(_core PRIVATE
    cuda_zstd
    CUDA::cudart
    ${ZSTD_LIBRARY}
)

# On Linux, hide all symbols except the PyInit entry point
if(UNIX AND NOT APPLE)
    target_link_options(_core PRIVATE "-Wl,--exclude-libs,ALL")
endif()

# Install the extension into the cuda_zstd package directory so that
# scikit-build-core picks it up automatically.
install(TARGETS _core LIBRARY DESTINATION cuda_zstd)
