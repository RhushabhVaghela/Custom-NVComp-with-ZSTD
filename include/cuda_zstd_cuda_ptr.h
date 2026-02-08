// ============================================================================
// cuda_zstd_cuda_ptr.h - RAII Wrapper for CUDA Device Memory
// ============================================================================
// Move-only smart pointer that calls cudaFree on destruction.
// Host-side only — the wrapper lives on host, the pointed-to memory is device.
//
// Usage:
//   CudaDevicePtr<float> buf;
//   cudaMalloc(&buf.ptr_ref(), n * sizeof(float));
//   // ... use buf.get() ...
//   // automatically freed on scope exit / destruction
//
// Or with the factory:
//   auto buf = CudaDevicePtr<float>::alloc(n);
// ============================================================================

#ifndef CUDA_ZSTD_CUDA_PTR_H_
#define CUDA_ZSTD_CUDA_PTR_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <utility> // std::exchange
#include "cuda_zstd_safe_alloc.h"

namespace cuda_zstd {

/// @brief RAII wrapper for device memory allocated with cudaMalloc.
///
/// - Move-only (non-copyable) to enforce unique ownership.
/// - Null-safe: reset()/destructor skip cudaFree on nullptr.
/// - Host-only: do NOT use this type in __device__ code.
///
/// @tparam T  Element type of the device allocation.
template <typename T>
class CudaDevicePtr {
public:
  // -- Constructors -----------------------------------------------------------

  /// Default: null pointer, owns nothing.
  CudaDevicePtr() noexcept = default;

  /// Take ownership of an existing device pointer.
  explicit CudaDevicePtr(T *p) noexcept : ptr_(p) {}

  /// Move constructor: steals ownership from @p other.
  CudaDevicePtr(CudaDevicePtr &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)) {}

  /// Move assignment: releases current, steals from @p other.
  CudaDevicePtr &operator=(CudaDevicePtr &&other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }

  // -- Deleted copy (unique ownership) ----------------------------------------

  CudaDevicePtr(const CudaDevicePtr &) = delete;
  CudaDevicePtr &operator=(const CudaDevicePtr &) = delete;

  // -- Destructor -------------------------------------------------------------

  ~CudaDevicePtr() { reset(); }

  // -- Observers --------------------------------------------------------------

  /// Raw device pointer (may be nullptr).
  T *get() const noexcept { return ptr_; }

  /// Implicit boolean: true if non-null.
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  // -- Modifiers --------------------------------------------------------------

  /// Release ownership and return the raw pointer (caller must cudaFree).
  T *release() noexcept { return std::exchange(ptr_, nullptr); }

  /// Free current allocation (if any) and optionally take ownership of @p p.
  void reset(T *p = nullptr) noexcept {
    if (ptr_) {
      cudaFree(ptr_);
    }
    ptr_ = p;
  }

  /// Mutable reference to the internal pointer — for use with cudaMalloc:
  ///   CudaDevicePtr<float> buf;
  ///   cudaMalloc(&buf.ptr_ref(), size);
  ///
  /// CAUTION: If ptr_ is already non-null this will leak. Call reset() first.
  T *&ptr_ref() noexcept { return ptr_; }

  // -- Factory ----------------------------------------------------------------

  /// Allocate device memory for @p count elements of type T.
  /// Returns a null CudaDevicePtr if cudaMalloc fails.
  static CudaDevicePtr alloc(size_t count) {
    T *p = nullptr;
    cudaError_t err = safe_cuda_malloc(&p, count * sizeof(T));
    if (err != cudaSuccess) {
      return CudaDevicePtr{};
    }
    return CudaDevicePtr{p};
  }

  // -- Swap -------------------------------------------------------------------

  void swap(CudaDevicePtr &other) noexcept { std::swap(ptr_, other.ptr_); }

private:
  T *ptr_ = nullptr;
};

/// Non-member swap.
template <typename T>
void swap(CudaDevicePtr<T> &a, CudaDevicePtr<T> &b) noexcept {
  a.swap(b);
}

} // namespace cuda_zstd

#endif // CUDA_ZSTD_CUDA_PTR_H_
