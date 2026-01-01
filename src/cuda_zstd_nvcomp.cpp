// ============================================================================
// cuda_zstd_nvcomp.cpp - NVCOMP v5.0 Compatibility Layer Implementation
//
// This file provides the "glue" layer that implements the C-API and C++
// batching API defined in 'cuda_zstd_nvcomp.h'. It translates
//
// This file provides the "glue" layer that implements the C-API and C++
// batching API defined in 'cuda_zstd_nvcomp.h'. It translates
// NVcomp-style calls into calls to the 'ZstdManager' and 'ZstdBatchManager'.
// ============================================================================

#include "cuda_zstd_nvcomp.h"
#include "cuda_zstd_manager.h"

#include <algorithm> // For std::min
#include <cmath>     // For log2f
#include <cstdlib>   // For malloc/free
#include <cstring>   // For memcpy
#include <numeric>   // for std::iota
#include <stdexcept>
#include <vector>

// For benchmark timing
#include <cuda_runtime.h>

namespace cuda_zstd {
namespace nvcomp_v5 {

// ZSTD skippable frame magic number
constexpr u32 ZSTD_MAGIC_SKIPPABLE_START = 0x184D2A50;

// Use CUDA_CHECK_RETURN for functions returning size_t
#define CUDA_CHECK_SIZE_T(call)                                                \
  do {                                                                         \
    cudaError_t __err = (call);                                                \
    if (__err != cudaSuccess) {                                                \
      return 0;                                                                \
    }                                                                          \
  } while (0)

class SynchronizedMemcpy {
private:
  cudaEvent_t event;
  cudaStream_t sync_stream;

public:
  SynchronizedMemcpy(cudaStream_t stream) : sync_stream(stream) {
    cudaEventCreate(&event);
  }

  ~SynchronizedMemcpy() { cudaEventDestroy(event); }

  // Copy with proper synchronization using events
  Status async_copy_with_sync(void *dst, const void *src, size_t size,
                              cudaMemcpyKind kind, cudaStream_t src_stream) {
    // Record event in source stream
    CUDA_CHECK(cudaEventRecord(event, src_stream));

    // Wait for event in sync stream (or create a barrier stream)
    cudaStream_t sync_stream_local = 0;
    CUDA_CHECK(cudaStreamWaitEvent(sync_stream_local, event, 0));

    // Now perform memcpy in sync stream
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, sync_stream_local));

    return Status::SUCCESS;
  }
};

// ============================================================================
// NVCOMP v5.0 Error Handling
// ============================================================================

int status_to_nvcomp_error(Status status) {
  switch (status) {
  case Status::SUCCESS:
    return 0;
  case Status::ERROR_INVALID_PARAMETER:
    return 2;
  case Status::ERROR_OUT_OF_MEMORY:
    return 3;
  case Status::ERROR_CUDA_ERROR:
    return 4;
  case Status::ERROR_CORRUPT_DATA:
    return 6;
  case Status::ERROR_BUFFER_TOO_SMALL:
    return 7;
  case Status::ERROR_CHECKSUM_FAILED:
    return 10;
  case Status::ERROR_COMPRESSION:
    return 12;
  default:
    return 1; // ERROR_GENERIC
  }
}

Status nvcomp_error_to_status(int nvcomp_error) {
  switch (nvcomp_error) {
  case 0:
    return Status::SUCCESS;
  case 2:
    return Status::ERROR_INVALID_PARAMETER;
  case 3:
    return Status::ERROR_OUT_OF_MEMORY;
  case 4:
    return Status::ERROR_CUDA_ERROR;
  case 6:
    return Status::ERROR_CORRUPT_DATA;
  case 7:
    return Status::ERROR_BUFFER_TOO_SMALL;
  case 10:
    return Status::ERROR_CHECKSUM_FAILED;
  case 12:
    return Status::ERROR_COMPRESSION;
  default:
    return Status::ERROR_GENERIC;
  }
}

const char *get_nvcomp_v5_error_string(int error_code) {
  return status_to_string(nvcomp_error_to_status(error_code));
}

// ============================================================================
// NVCOMP v5.0 Format Compatibility
// ============================================================================

bool is_nvcomp_v5_zstd_format(const void *compressed_data,
                              size_t compressed_size) {
  if (compressed_size < 4)
    return false;

  u32 magic = 0;
  // We assume this might be called with a host pointer for metadata checks
  // If it's device, this memcpy is invalid, but is_nvcomp_* usually implies
  // host access or specific API For safety in this glue layer, we'll assume
  // standard ZSTD magic behavior.
  memcpy(&magic, compressed_data, sizeof(u32));

  if (magic == ZSTD_MAGIC)
    return true;
  if ((magic & 0xFFFFFFF0) == ZSTD_MAGIC_SKIPPABLE_START)
    return true;

  return false;
}

bool is_compatible_with_nvcomp_v5(u32 format_version) {
  return format_version == 0x00050000;
}

// ============================================================================
// NVCOMP v5.0 Options Conversion
// ============================================================================

NvcompV5Options to_nvcomp_v5_opts(const CompressionConfig &config) {
  NvcompV5Options opts;
  opts.level = config.level;
  opts.chunk_size = config.block_size;
  opts.enable_checksum =
      (config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY);
  return opts;
}

CompressionConfig from_nvcomp_v5_opts(const NvcompV5Options &opts) {
  CompressionConfig config = CompressionConfig::from_level(opts.level);
  config.block_size = opts.chunk_size;
  config.checksum = opts.enable_checksum ? ChecksumPolicy::COMPUTE_AND_VERIFY
                                         : ChecksumPolicy::NO_COMPUTE_NO_VERIFY;
  return config;
}

// ============================================================================
// NVCOMP v5.0 Manager Factory
// ============================================================================

std::unique_ptr<ZstdManager>
create_nvcomp_v5_manager(const NvcompV5Options &opts) {
  CompressionConfig config = from_nvcomp_v5_opts(opts);
  return create_manager(config);
}

// ============================================================================
// NVCOMP v5.0 Batch Manager (PIMPL Implementation)
// ============================================================================

class NvcompV5BatchManager::Impl {
public:
  std::unique_ptr<ZstdBatchManager> manager;

  explicit Impl(const NvcompV5Options &opts) {
    CompressionConfig config = from_nvcomp_v5_opts(opts);
    manager = std::make_unique<ZstdBatchManager>(config);
  }
};

NvcompV5BatchManager::NvcompV5BatchManager(const NvcompV5Options &opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

NvcompV5BatchManager::~NvcompV5BatchManager() = default;

size_t NvcompV5BatchManager::get_compress_temp_size(const size_t *d_chunk_sizes,
                                                    size_t num_chunks,
                                                    cudaStream_t stream) const {
  if (num_chunks == 0)
    return 0;

  std::vector<size_t> h_chunk_sizes(num_chunks);

  // If the caller passed a device pointer then the GPU must be synchronized
  // before we copy device memory back to the host. If the pointer is a
  // host pointer (tests sometimes pass host arrays) then do a plain memcpy
  // instead.
  {
    cudaPointerAttributes patts;
    cudaError_t attr_err = cudaPointerGetAttributes(&patts, d_chunk_sizes);
    bool is_device_ptr = false;

    if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
      is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
      is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
    }

    if (is_device_ptr) {
      if (stream != 0) {
        CUDA_CHECK_SIZE_T(cudaStreamSynchronize(stream));
      } else {
        CUDA_CHECK_SIZE_T(cudaDeviceSynchronize());
      }

      // Now it's safe to copy the completed results from device
      CUDA_CHECK_SIZE_T(cudaMemcpy(h_chunk_sizes.data(), d_chunk_sizes,
                                   num_chunks * sizeof(size_t),
                                   cudaMemcpyDeviceToHost));
    } else {
      memcpy(h_chunk_sizes.data(), d_chunk_sizes, num_chunks * sizeof(size_t));
    }
  }

  return pimpl_->manager->get_batch_compress_temp_size(h_chunk_sizes);
}

size_t
NvcompV5BatchManager::get_decompress_temp_size(const size_t *d_compressed_sizes,
                                               size_t num_chunks,
                                               cudaStream_t stream) const {
  if (num_chunks == 0)
    return 0;

  std::vector<size_t> h_compressed_sizes(num_chunks);

  {
    cudaPointerAttributes patts;
    cudaError_t attr_err = cudaPointerGetAttributes(&patts, d_compressed_sizes);
    bool is_device_ptr = false;

    if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
      is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
      is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
    }

    if (is_device_ptr) {
      if (stream != 0) {
        CUDA_CHECK_SIZE_T(cudaStreamSynchronize(stream));
      } else {
        CUDA_CHECK_SIZE_T(cudaDeviceSynchronize());
      }
      CUDA_CHECK_SIZE_T(
          cudaMemcpy(h_compressed_sizes.data(), d_compressed_sizes,
                     num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
    } else {
      memcpy(h_compressed_sizes.data(), d_compressed_sizes,
             num_chunks * sizeof(size_t));
    }
  }

  return pimpl_->manager->get_batch_decompress_temp_size(h_compressed_sizes);
}

size_t NvcompV5BatchManager::get_max_compressed_chunk_size(
    size_t uncompressed_chunk_size) const {
  return pimpl_->manager->get_max_compressed_size(uncompressed_chunk_size);
}

const CompressionStats &NvcompV5BatchManager::get_stats() const {
  return pimpl_->manager->get_stats();
}

Status NvcompV5BatchManager::compress_async(
    const void *const *d_uncompressed_ptrs, const size_t *d_uncompressed_sizes,
    size_t num_chunks, void *const *d_compressed_ptrs,
    size_t *d_compressed_sizes, void *d_temp_storage, size_t temp_storage_bytes,
    cudaStream_t stream) {
  if (num_chunks == 0)
    return Status::SUCCESS;

  if (!d_uncompressed_ptrs || !d_uncompressed_sizes || !d_compressed_ptrs ||
      !d_compressed_sizes) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // DEBUG PRINT
  // printf("[NvcompV5] compress_async: chunks=%zu temp=%zu stream=%p\n",
  // num_chunks, temp_storage_bytes, (void *)stream);

  std::vector<BatchItem> items(num_chunks);
  std::vector<size_t> h_uncompressed_sizes(num_chunks);
  std::vector<void *> h_uncompressed_ptrs(num_chunks);
  std::vector<void *> h_compressed_ptrs(num_chunks);

  // === FIX: Use async memcpy with explicit synchronization ===
  // Check if d_uncompressed_sizes is device or host
  cudaPointerAttributes patts;
  cudaError_t attr_err = cudaPointerGetAttributes(&patts, d_uncompressed_sizes);
  bool is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_uncompressed_sizes.data(), d_uncompressed_sizes,
                          num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_uncompressed_sizes.data(), d_uncompressed_sizes,
           num_chunks * sizeof(size_t));
  }

  // Pointers array is usually on host for batch API, but let's check
  attr_err = cudaPointerGetAttributes(&patts, d_uncompressed_ptrs);
  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_uncompressed_ptrs.data(), d_uncompressed_ptrs,
                          num_chunks * sizeof(void *), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_uncompressed_ptrs.data(), d_uncompressed_ptrs,
           num_chunks * sizeof(void *));
  }

  // Output pointers
  attr_err = cudaPointerGetAttributes(&patts, d_compressed_ptrs);
  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_compressed_ptrs.data(), d_compressed_ptrs,
                          num_chunks * sizeof(void *), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_compressed_ptrs.data(), d_compressed_ptrs,
           num_chunks * sizeof(void *));
  }

  // === FIX: CRITICAL - Synchronize stream before using host copies ===
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Output sizes (used as input capacity)
  std::vector<size_t> h_compressed_sizes(num_chunks);
  attr_err = cudaPointerGetAttributes(&patts, d_compressed_sizes);
  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_compressed_sizes.data(), d_compressed_sizes,
                          num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_compressed_sizes.data(), d_compressed_sizes,
           num_chunks * sizeof(size_t));
  }

  // Now safe to access h_* arrays
  for (size_t i = 0; i < num_chunks; ++i) {
    items[i].input_ptr = const_cast<void *>(h_uncompressed_ptrs[i]);
    items[i].input_size = h_uncompressed_sizes[i];
    items[i].output_ptr = h_compressed_ptrs[i];
    items[i].output_size = h_compressed_sizes[i]; // Set capacity
    // printf("[NvcompV5] Chunk %zu: input=%p size=%zu output=%p cap=%zu\n", i,
    // items[i].input_ptr, items[i].input_size, items[i].output_ptr,
    // items[i].output_size);
  }

  // Call the batch manager
  Status status = pimpl_->manager->compress_batch(items, d_temp_storage,
                                                  temp_storage_bytes, stream);

  // === FIX: Update output sizes with proper synchronization ===
  // std::vector<size_t> h_compressed_sizes(num_chunks); // Already declared
  for (size_t i = 0; i < num_chunks; ++i) {
    h_compressed_sizes[i] = items[i].output_size;
  }

  // Ensure compression is complete before copying sizes back
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result sizes to device
  attr_err = cudaPointerGetAttributes(&patts, d_compressed_sizes);
  // Clear sticky error if pointer is invalid/host
  if (attr_err != cudaSuccess)
    cudaGetLastError();

  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(d_compressed_sizes, h_compressed_sizes.data(),
                          num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
  } else {
    memcpy(d_compressed_sizes, h_compressed_sizes.data(),
           num_chunks * sizeof(size_t));
  }

  return status;
}

Status NvcompV5BatchManager::decompress_async(
    const void *const *d_compressed_ptrs, const size_t *d_compressed_sizes,
    size_t num_chunks, void *const *d_uncompressed_ptrs,
    size_t *d_uncompressed_sizes, void *d_temp_storage,
    size_t temp_storage_bytes, cudaStream_t stream) {
  if (num_chunks == 0)
    return Status::SUCCESS;

  std::vector<BatchItem> items(num_chunks);
  std::vector<size_t> h_compressed_sizes(num_chunks);
  std::vector<void *> h_compressed_ptrs(num_chunks);
  std::vector<void *> h_uncompressed_ptrs(num_chunks);

  // === FIX: Use async memcpy with explicit synchronization and pointer
  // attributes check ===

  // 1. Compressed Sizes
  cudaPointerAttributes patts;
  cudaError_t attr_err = cudaPointerGetAttributes(&patts, d_compressed_sizes);
  // Clear sticky error if pointer is invalid/host
  if (attr_err != cudaSuccess)
    cudaGetLastError();

  bool is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_compressed_sizes.data(), d_compressed_sizes,
                          num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_compressed_sizes.data(), d_compressed_sizes,
           num_chunks * sizeof(size_t));
  }

  // 2. Compressed Pointers
  attr_err = cudaPointerGetAttributes(&patts, d_compressed_ptrs);
  if (attr_err != cudaSuccess)
    cudaGetLastError();

  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_compressed_ptrs.data(), d_compressed_ptrs,
                          num_chunks * sizeof(void *), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_compressed_ptrs.data(), d_compressed_ptrs,
           num_chunks * sizeof(void *));
  }

  // 3. Uncompressed Pointers
  attr_err = cudaPointerGetAttributes(&patts, d_uncompressed_ptrs);
  if (attr_err != cudaSuccess)
    cudaGetLastError();

  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_uncompressed_ptrs.data(), d_uncompressed_ptrs,
                          num_chunks * sizeof(void *), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_uncompressed_ptrs.data(), d_uncompressed_ptrs,
           num_chunks * sizeof(void *));
  }

  // 4. Output Capacities (Initialize from d_uncompressed_sizes)
  // Although nvcomp usually treats this as OUT, some tests (and ZSTD) treat it
  // as IN/OUT (Capacity/Result)
  std::vector<size_t> h_capacities(num_chunks);
  attr_err = cudaPointerGetAttributes(&patts, d_uncompressed_sizes);
  if (attr_err != cudaSuccess)
    cudaGetLastError();

  is_device_ptr = false;
  if (attr_err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr = (patts.type == cudaMemoryTypeDevice);
#else
    is_device_ptr = (patts.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr) {
    CUDA_CHECK(cudaMemcpy(h_capacities.data(), d_uncompressed_sizes,
                          num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
  } else {
    memcpy(h_capacities.data(), d_uncompressed_sizes,
           num_chunks * sizeof(size_t));
  }

  // === FIX: CRITICAL - Synchronize before using host arrays ===
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Now safe to access h_* arrays
  for (size_t i = 0; i < num_chunks; ++i) {
    items[i].input_ptr = const_cast<void *>(h_compressed_ptrs[i]);
    items[i].input_size = h_compressed_sizes[i];
    items[i].output_ptr = h_uncompressed_ptrs[i];
    items[i].output_size = h_capacities[i]; // Set capacity
  }

  // Call batch decompression
  Status status = pimpl_->manager->decompress_batch(items, d_temp_storage,
                                                    temp_storage_bytes, stream);

  // === FIX: Synchronize after decompression ===
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Update output sizes
  std::vector<size_t> h_uncompressed_sizes_out(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    h_uncompressed_sizes_out[i] = items[i].output_size;
  }

  // Copy result sizes to device
  // Copy result sizes to device
  cudaPointerAttributes patts_out;
  cudaError_t attr_err_out =
      cudaPointerGetAttributes(&patts_out, d_uncompressed_sizes);
  if (attr_err_out != cudaSuccess)
    cudaGetLastError(); // Clear error

  bool is_device_ptr_out = false;
  if (attr_err_out == cudaSuccess) {
#if CUDART_VERSION >= 10000
    is_device_ptr_out = (patts_out.type == cudaMemoryTypeDevice);
#else
    is_device_ptr_out = (patts_out.memoryType == cudaMemoryTypeDevice);
#endif
  }

  if (is_device_ptr_out) {
    CUDA_CHECK(cudaMemcpy(d_uncompressed_sizes, h_uncompressed_sizes_out.data(),
                          num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
  } else {
    memcpy(d_uncompressed_sizes, h_uncompressed_sizes_out.data(),
           num_chunks * sizeof(size_t));
  }

  return status;
}

// ============================================================================
// NVCOMP v5.0 Metadata Functions
// ============================================================================

Status get_metadata_async(const void *d_compressed_data, size_t compressed_size,
                          NvcompV5Metadata *h_metadata, cudaStream_t stream) {
  if (!h_metadata)
    return Status::ERROR_INVALID_PARAMETER;

  // (FIX) metadata extraction requires reading the header.
  // If d_compressed_data is on device, we must copy the header to host.
  // We copy enough for skippable frames + Zstd header.
  size_t header_peek_size = std::min(compressed_size, (size_t)128);
  // Use RAII container to avoid raw malloc/free and exception-safety issues.
  std::vector<byte_t> h_compressed_header(header_peek_size);

  if (h_compressed_header.empty())
    return Status::ERROR_OUT_OF_MEMORY;
  CUDA_CHECK(cudaMemcpy(h_compressed_header.data(), d_compressed_data,
                        header_peek_size, cudaMemcpyDeviceToHost));

  NvcompMetadata internal_meta;
  Status status = extract_metadata(h_compressed_header.data(), header_peek_size,
                                   internal_meta);

  // h_compressed_header is freed automatically when the vector goes out of
  // scope

  // Even if partial extract failed (due to small buffer), check what we got.
  // If magic was valid, populate what we can.
  if (status != Status::SUCCESS && status != Status::ERROR_BUFFER_TOO_SMALL)
    return status;

  h_metadata->format_version = internal_meta.format_version;
  h_metadata->library_version = get_format_version();
  h_metadata->compression_level = internal_meta.compression_level;
  h_metadata->uncompressed_size = internal_meta.uncompressed_size;
  h_metadata->compressed_size = compressed_size;
  h_metadata->num_chunks = internal_meta.num_chunks;
  h_metadata->chunk_size = internal_meta.chunk_size;
  h_metadata->dictionary_id = internal_meta.dictionary_id;
  h_metadata->checksum_policy = internal_meta.checksum_policy;
  h_metadata->has_dictionary = (internal_meta.dictionary_id != 0);
  h_metadata->checksum = 0;

  return Status::SUCCESS;
}

Status get_metadata(const void *d_compressed_data, size_t compressed_size,
                    NvcompV5Metadata &metadata) {
  return get_metadata_async(d_compressed_data, compressed_size, &metadata, 0);
}

bool validate_metadata(const NvcompV5Metadata &metadata) {
  return metadata.format_version == get_nvcomp_v5_format_version();
}

// ============================================================================
// NVCOMP v5.0 Utility Functions
// ============================================================================

Status get_decompressed_size_async(const void *d_compressed_data,
                                   size_t compressed_size,
                                   size_t *h_decompressed_size,
                                   cudaStream_t stream) {
  if (!h_decompressed_size)
    return Status::ERROR_INVALID_PARAMETER;

  NvcompV5Metadata metadata;
  Status status =
      get_metadata_async(d_compressed_data, compressed_size, &metadata, stream);

  if (status == Status::SUCCESS) {
    *h_decompressed_size = metadata.uncompressed_size;
  }
  return status;
}

Status get_num_chunks(const void *d_compressed_data, size_t compressed_size,
                      size_t *num_chunks) {
  NvcompV5Metadata metadata;
  Status status = get_metadata(d_compressed_data, compressed_size, metadata);
  if (status == Status::SUCCESS) {
    *num_chunks = metadata.num_chunks;
  }
  return status;
}

Status get_chunk_sizes(const void *d_compressed_data, size_t compressed_size,
                       size_t *chunk_sizes, size_t max_chunks) {
  NvcompV5Metadata metadata;
  Status status = get_metadata(d_compressed_data, compressed_size, metadata);

  if (status != Status::SUCCESS) {
    return status;
  }

  if (max_chunks < metadata.num_chunks) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  if (metadata.num_chunks == 0) {
    return Status::SUCCESS;
  }

  for (size_t i = 0; i < metadata.num_chunks - 1; ++i) {
    chunk_sizes[i] = metadata.chunk_size;
  }

  size_t last_chunk_size = metadata.uncompressed_size -
                           (metadata.chunk_size * (metadata.num_chunks - 1));
  chunk_sizes[metadata.num_chunks - 1] = last_chunk_size;

  return Status::SUCCESS;
}

// ============================================================================
// NVCOMP v5.0 C API (Implementation)
// ============================================================================

extern "C" {

nvcompZstdManagerHandle nvcomp_zstd_create_manager_v5(int compression_level) {
  try {
    auto manager = create_manager(compression_level);
    return manager.release();
  } catch (...) {
    return nullptr;
  }
}

void nvcomp_zstd_destroy_manager_v5(nvcompZstdManagerHandle handle) {
  delete static_cast<ZstdManager *>(handle);
}

int nvcomp_zstd_compress_async_v5(nvcompZstdManagerHandle handle,
                                  const void *d_uncompressed,
                                  size_t uncompressed_size, void *d_compressed,
                                  size_t *compressed_size, void *d_temp,
                                  size_t temp_size, cudaStream_t stream) {
  ZstdManager *manager = static_cast<ZstdManager *>(handle);
  if (!manager)
    return status_to_nvcomp_error(Status::ERROR_INVALID_PARAMETER);

  // Compress without dictionary
  Status status = manager->compress(d_uncompressed, uncompressed_size,
                                    d_compressed, compressed_size, d_temp,
                                    temp_size, nullptr, 0, // No dictionary
                                    stream);

  return status_to_nvcomp_error(status);
}

int nvcomp_zstd_decompress_async_v5(nvcompZstdManagerHandle handle,
                                    const void *d_compressed,
                                    size_t compressed_size,
                                    void *d_uncompressed,
                                    size_t *uncompressed_size, void *d_temp,
                                    size_t temp_size, cudaStream_t stream) {
  ZstdManager *manager = static_cast<ZstdManager *>(handle);
  if (!manager)
    return status_to_nvcomp_error(Status::ERROR_INVALID_PARAMETER);

  Status status =
      manager->decompress(d_compressed, compressed_size, d_uncompressed,
                          uncompressed_size, d_temp, temp_size, stream);
  return status_to_nvcomp_error(status);
}

size_t nvcomp_zstd_get_compress_temp_size_v5(nvcompZstdManagerHandle handle,
                                             size_t uncompressed_size) {
  ZstdManager *manager = static_cast<ZstdManager *>(handle);
  if (!manager)
    return 0;
  return manager->get_compress_temp_size(uncompressed_size);
}

size_t nvcomp_zstd_get_decompress_temp_size_v5(nvcompZstdManagerHandle handle,
                                               size_t compressed_size) {
  ZstdManager *manager = static_cast<ZstdManager *>(handle);
  if (!manager)
    return 0;
  return manager->get_decompress_temp_size(compressed_size);
}

int nvcomp_zstd_get_metadata_v5(const void *d_compressed_data,
                                size_t compressed_size,
                                NvcompV5Metadata *h_metadata,
                                cudaStream_t stream) {
  Status status = get_metadata_async(d_compressed_data, compressed_size,
                                     h_metadata, stream);
  return status_to_nvcomp_error(status);
}

} // extern "C"

// ============================================================================
// NVCOMP v5.0 Benchmark Helpers
// ============================================================================

NvcompV5BenchmarkResult benchmark_level(const void *d_input, size_t input_size,
                                        int level, int iterations,
                                        cudaStream_t stream) {
  NvcompV5BenchmarkResult result;
  result.level = level;

  auto manager = create_manager(level);
  size_t temp_size = manager->get_compress_temp_size(input_size);
  size_t max_comp_size = manager->get_max_compressed_size(input_size);

  void *d_temp;
  void *d_compressed;
  void *d_decompressed;
  cudaMalloc(&d_temp, temp_size);
  cudaMalloc(&d_compressed, max_comp_size);
  cudaMalloc(&d_decompressed, input_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- Benchmark Compression ---
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    manager->compress(d_input, input_size, d_compressed,
                      &result.compressed_size, d_temp, temp_size, nullptr,
                      0, // No dictionary
                      stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);
  result.compress_time_ms = time_ms / iterations;
  result.compress_throughput_mbps =
      (input_size / (1024.0 * 1024.0)) / (result.compress_time_ms / 1000.0);

  // --- Benchmark Decompression ---
  size_t decomp_temp_size =
      manager->get_decompress_temp_size(result.compressed_size);
  if (decomp_temp_size > temp_size) {
    cudaFree(d_temp);
    cudaMalloc(&d_temp, decomp_temp_size);
  }

  size_t decompressed_size = input_size;

  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    manager->decompress(d_compressed, result.compressed_size, d_decompressed,
                        &decompressed_size, d_temp, decomp_temp_size, stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time_ms, start, stop);
  result.decompress_time_ms = time_ms / iterations;
  result.decompress_throughput_mbps =
      (input_size / (1024.0 * 1024.0)) / (result.decompress_time_ms / 1000.0);

  result.compression_ratio = (float)input_size / result.compressed_size;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_temp);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);

  return result;
}

std::vector<NvcompV5BenchmarkResult> benchmark_all_levels(const void *d_input,
                                                          size_t input_size,
                                                          int iterations,
                                                          cudaStream_t stream) {
  std::vector<NvcompV5BenchmarkResult> results;
  for (int level = 1; level <= 22; ++level) {
    results.push_back(
        benchmark_level(d_input, input_size, level, iterations, stream));
  }
  return results;
}

void print_benchmark_results(
    const std::vector<NvcompV5BenchmarkResult> &results) {
  //     printf("| Level | Comp (ms) | Decomp (ms) | Comp (MB/s) | Decomp (MB/s)
  //     | Ratio |\n");
  //     printf("|-------|-----------|-------------|-------------|---------------|-------|\n");
  for (const auto &r : results) {
    //         printf("| %-5d | %-9.3f | %-11.3f | %-11.1f | %-13.1f | %-5.2f
    //         |\n",
    //             r.level,
    //             r.compress_time_ms,
    //             r.decompress_time_ms,
    //             r.compress_throughput_mbps,
    //             r.decompress_throughput_mbps,
    //             r.compression_ratio
    //         );
  }
}

} // namespace nvcomp_v5
} // namespace cuda_zstd