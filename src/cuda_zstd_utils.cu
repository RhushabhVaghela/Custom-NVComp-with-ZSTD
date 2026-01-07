// ============================================================================
// cuda_zstd_utils.cu - Shared Utility Function Implementations// This file contains the implementation for common, reusable components
// like parallel scan, using Thrust for robustness.
// ============================================================================

#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_utils.h"
#include <algorithm> // for std::min
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace cuda_zstd {
namespace utils {

namespace {
// Helper struct for Thrust compatibility (device code visibility)
struct LocalThrustDmer {
  u64 hash;
  u32 position;
  u32 length;

  __device__ __host__ bool operator<(const LocalThrustDmer &other) const {
    return hash < other.hash;
  }
};
} // namespace

// Unused constant (reserved for future scan operations)
// constexpr u32 SCAN_THREADS = 256;

// Functor for DictSegment
struct GetLength {
  __host__ __device__ u32 operator()(const DictSegment &s) const {
    return s.length;
  }
};

// ============================================================================
// Host Function (Replaced with Thrust)
// ============================================================================
template <typename T>
__host__ Status parallel_scan(const T *d_input, u32 *d_output, u32 num_elements,
                              cudaStream_t stream) {
  if (num_elements == 0)
    return Status::SUCCESS;

  try {
    if constexpr (std::is_same<T, u32>::value) {
      thrust::device_ptr<const u32> dev_in((const u32 *)d_input);
      thrust::device_ptr<u32> dev_out(d_output);

      thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_in,
                             dev_in + num_elements, dev_out);
    } else if constexpr (std::is_same<T, DictSegment>::value) {
      thrust::device_ptr<const DictSegment> dev_in(d_input);
      thrust::device_ptr<u32> dev_out(d_output);

      // Use transform iterator for compatibility
      auto first = thrust::make_transform_iterator(dev_in, GetLength());
      auto last =
          thrust::make_transform_iterator(dev_in + num_elements, GetLength());

      thrust::exclusive_scan(thrust::cuda::par.on(stream), first, last,
                             dev_out);
    } else {
      // Fallback or error if type not supported
      return Status::ERROR_GENERIC;
    }
  } catch (thrust::system_error &e) {
    // Log error if possible
    return Status::ERROR_GENERIC;
  } catch (...) {
    return Status::ERROR_GENERIC;
  }

  cuda_zstd::utils::debug_kernel_verify("utils::parallel_scan (thrust)");
  return Status::SUCCESS;
}

// Explicit template instantiation
// Removed __host__ qualifier from explicit instantiation to comply with
// standard
template Status parallel_scan<u32>(const u32 *, u32 *, u32, cudaStream_t);
template Status parallel_scan<DictSegment>(const DictSegment *, u32 *, u32,
                                           cudaStream_t);

__host__ cudaError_t debug_kernel_verify(const char *where) {
  const char *env = std::getenv("CUDA_ZSTD_DEBUG_KERNEL_VERIFY");
  if (!env || env[0] == '\0') {
    return cudaSuccess;
  }

  // Force synchronous behavior and print an informative message
  cudaError_t err = cudaDeviceSynchronize();
  return err;
}

// ============================================================================
// Parallel Sort (Using Thrust for Robustness)
// ============================================================================

/**
 * @brief Host-side launcher for the parallel sorting of Dmers.
 * Replaces custom Radix Sort with highly optimized Thrust sort.
 */
__host__ Status parallel_sort_dmers(dictionary::Dmer *d_dmers, u32 num_dmers,
                                    cudaStream_t stream) {
  if (num_dmers == 0)
    return Status::SUCCESS;

  // Use LocalThrustDmer to ensure visibility in device code
  static_assert(sizeof(LocalThrustDmer) == sizeof(dictionary::Dmer),
                "Dmer layout mismatch");

  try {
    // Cast the pointer - safe because layout is identical (POD)
    auto *d_local = reinterpret_cast<LocalThrustDmer *>(d_dmers);
    thrust::device_ptr<LocalThrustDmer> dev_dmers(d_local);

    thrust::sort(thrust::cuda::par.on(stream), dev_dmers,
                 dev_dmers + num_dmers);

  } catch (thrust::system_error &e) {
    return Status::ERROR_GENERIC;
  } catch (...) {
    return Status::ERROR_GENERIC;
  }

  cuda_zstd::utils::debug_kernel_verify("utils::parallel_sort (thrust)");
  return Status::SUCCESS;
}

} // namespace utils
} // namespace cuda_zstd
