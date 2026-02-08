// Copyright (c) 2025 Rhushabh Vaghela. All rights reserved.
// SPDX-License-Identifier: MIT
//
// cuda_zstd_safe_alloc.h -- Safe GPU/host memory allocation wrappers
//
// These wrappers check available memory against the safety buffer constants
// before allocating. This prevents WSL crashes and system instability by
// ensuring the OS and GPU driver always have enough headroom.
//
// Usage:
//   Replace  cudaMalloc(&ptr, size)
//   With     safe_cuda_malloc(&ptr, size)
//
//   Both return cudaError_t, so they work with CUDA_CHECK() unchanged.

#ifndef CUDA_ZSTD_SAFE_ALLOC_H
#define CUDA_ZSTD_SAFE_ALLOC_H

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

#include "cuda_zstd_types.h"  // VRAM_SAFETY_BUFFER_BYTES, RAM_SAFETY_BUFFER_BYTES

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

namespace cuda_zstd {

// ============================================================================
// Safe GPU Memory Allocation
// ============================================================================

/**
 * @brief Allocate GPU memory with safety buffer enforcement.
 *
 * Queries cudaMemGetInfo() to verify that after the allocation at least
 * VRAM_SAFETY_BUFFER_BYTES of free VRAM remain. If not, the allocation
 * is refused and cudaErrorMemoryAllocation is returned without calling
 * cudaMalloc.
 *
 * Returns cudaSuccess on success, a cudaError_t on failure -- compatible
 * with the CUDA_CHECK() macro.
 *
 * @param ptr    Output pointer (same semantics as cudaMalloc).
 * @param size   Bytes to allocate.
 * @return cudaError_t
 */
inline cudaError_t safe_cuda_malloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;

    // Zero-size allocations succeed trivially (matches cudaMalloc semantics)
    if (size == 0) {
        return cudaSuccess;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t query_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (query_err != cudaSuccess) {
        return query_err;
    }

    // Check: after this allocation, would we still have enough headroom?
    if (free_mem < size + VRAM_SAFETY_BUFFER_BYTES) {
#ifdef CUDA_ZSTD_DEBUG
        std::fprintf(stderr,
                     "[safe_cuda_malloc] REFUSED: requested %zu bytes, "
                     "free VRAM %zu bytes, safety buffer %zu bytes, "
                     "shortfall %zu bytes\n",
                     size, free_mem, VRAM_SAFETY_BUFFER_BYTES,
                     (size + VRAM_SAFETY_BUFFER_BYTES) - free_mem);
#endif
        return cudaErrorMemoryAllocation;
    }

    return cudaMalloc(ptr, size);
}

/**
 * @brief Typed overload for convenience with non-void pointers.
 *
 * Example: safe_cuda_malloc(&d_buffer, count * sizeof(int))
 */
template <typename T>
inline cudaError_t safe_cuda_malloc(T** ptr, size_t size) {
    return safe_cuda_malloc(reinterpret_cast<void**>(ptr), size);
}

// ============================================================================
// Safe GPU Memory Allocation (Async / Stream-ordered)
// ============================================================================

/**
 * @brief Allocate GPU memory asynchronously with safety buffer enforcement.
 *
 * Same safety check as safe_cuda_malloc(), but uses cudaMallocAsync() to
 * perform a stream-ordered allocation.  This is the drop-in replacement for
 * every raw cudaMallocAsync() call in the codebase.
 *
 * @param ptr    Output pointer (same semantics as cudaMallocAsync).
 * @param size   Bytes to allocate.
 * @param stream CUDA stream for the async allocation.
 * @return cudaError_t
 */
inline cudaError_t safe_cuda_malloc_async(void** ptr, size_t size,
                                          cudaStream_t stream) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;

    if (size == 0) {
        return cudaSuccess;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t query_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (query_err != cudaSuccess) {
        return query_err;
    }

    if (free_mem < size + VRAM_SAFETY_BUFFER_BYTES) {
#ifdef CUDA_ZSTD_DEBUG
        std::fprintf(stderr,
                     "[safe_cuda_malloc_async] REFUSED: requested %zu bytes, "
                     "free VRAM %zu bytes, safety buffer %zu bytes, "
                     "shortfall %zu bytes\n",
                     size, free_mem, VRAM_SAFETY_BUFFER_BYTES,
                     (size + VRAM_SAFETY_BUFFER_BYTES) - free_mem);
#endif
        return cudaErrorMemoryAllocation;
    }

    return cudaMallocAsync(ptr, size, stream);
}

/**
 * @brief Typed overload for convenience with non-void pointers.
 */
template <typename T>
inline cudaError_t safe_cuda_malloc_async(T** ptr, size_t size,
                                          cudaStream_t stream) {
    return safe_cuda_malloc_async(reinterpret_cast<void**>(ptr), size, stream);
}

// ============================================================================
// Safe Host Pinned Memory Allocation
// ============================================================================

/**
 * @brief Allocate host pinned memory with safety buffer enforcement.
 *
 * Wraps cudaMallocHost() with a free-RAM check (Linux only) to prevent
 * the host from running out of memory in WSL / constrained environments.
 *
 * Returns cudaError_t, compatible with CUDA_CHECK().
 *
 * @param ptr    Output pointer (same semantics as cudaMallocHost).
 * @param size   Bytes to allocate.
 * @return cudaError_t
 */
inline cudaError_t safe_cuda_malloc_host(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;

    if (size == 0) {
        return cudaSuccess;
    }

#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        size_t available_ram =
            static_cast<size_t>(si.freeram) * static_cast<size_t>(si.mem_unit);
        if (available_ram < size + RAM_SAFETY_BUFFER_BYTES) {
#ifdef CUDA_ZSTD_DEBUG
            std::fprintf(stderr,
                         "[safe_cuda_malloc_host] REFUSED: requested %zu bytes, "
                         "free RAM %zu bytes, safety buffer %zu bytes, "
                         "shortfall %zu bytes\n",
                         size, available_ram, RAM_SAFETY_BUFFER_BYTES,
                         (size + RAM_SAFETY_BUFFER_BYTES) - available_ram);
#endif
            return cudaErrorMemoryAllocation;
        }
    }
#endif

    return cudaMallocHost(ptr, size);
}

/**
 * @brief Typed overload for non-void host pointers.
 */
template <typename T>
inline cudaError_t safe_cuda_malloc_host(T** ptr, size_t size) {
    return safe_cuda_malloc_host(reinterpret_cast<void**>(ptr), size);
}

// ============================================================================
// Safe Host Memory Allocation (malloc-based)
// ============================================================================

/**
 * @brief Allocate host (CPU) memory with safety buffer enforcement.
 *
 * On Linux, queries sysinfo() to verify that after the allocation at least
 * RAM_SAFETY_BUFFER_BYTES of free RAM remain. On other platforms, falls
 * through to malloc() without the check (since sysinfo is Linux-specific).
 *
 * @param ptr    Output pointer (set to nullptr on failure).
 * @param size   Bytes to allocate.
 * @return true on success, false on failure.
 */
inline bool safe_host_malloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return false;
    }
    *ptr = nullptr;

    if (size == 0) {
        return true;
    }

#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        size_t available_ram =
            static_cast<size_t>(si.freeram) * static_cast<size_t>(si.mem_unit);
        if (available_ram < size + RAM_SAFETY_BUFFER_BYTES) {
#ifdef CUDA_ZSTD_DEBUG
            std::fprintf(stderr,
                         "[safe_host_malloc] REFUSED: requested %zu bytes, "
                         "free RAM %zu bytes, safety buffer %zu bytes, "
                         "shortfall %zu bytes\n",
                         size, available_ram, RAM_SAFETY_BUFFER_BYTES,
                         (size + RAM_SAFETY_BUFFER_BYTES) - available_ram);
#endif
            return false;
        }
    }
    // If sysinfo() fails, fall through to malloc (best-effort)
#endif

    *ptr = std::malloc(size);
    return (*ptr != nullptr);
}

/**
 * @brief Typed overload for non-void host pointers.
 */
template <typename T>
inline bool safe_host_malloc(T** ptr, size_t size) {
    return safe_host_malloc(reinterpret_cast<void**>(ptr), size);
}

// ============================================================================
// Query Helpers (for benchmarks and diagnostics)
// ============================================================================

/**
 * @brief Return the usable VRAM (free minus safety buffer), or 0 if
 *        the safety buffer exceeds free memory.
 */
inline size_t get_usable_vram() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 0;
    }
    if (free_mem <= VRAM_SAFETY_BUFFER_BYTES) {
        return 0;
    }
    return free_mem - VRAM_SAFETY_BUFFER_BYTES;
}

/**
 * @brief Return the usable host RAM (free minus safety buffer), or 0.
 *        Linux only; returns SIZE_MAX on other platforms (no-op).
 */
inline size_t get_usable_host_ram() {
#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        size_t available =
            static_cast<size_t>(si.freeram) * static_cast<size_t>(si.mem_unit);
        if (available <= RAM_SAFETY_BUFFER_BYTES) {
            return 0;
        }
        return available - RAM_SAFETY_BUFFER_BYTES;
    }
#endif
    return SIZE_MAX;  // Unknown platform -- no limit enforced
}

}  // namespace cuda_zstd

#endif  // CUDA_ZSTD_SAFE_ALLOC_H
