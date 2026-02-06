// ==============================================================================
// cuda_zstd_manager.cpp - COMPLETE Manager Implementation with Full Pipeline
// =============================================================================

#include "cuda_zstd_ldm.h"
#include "cuda_zstd_debug.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_fse_rfc.h"  // NEW: RFC 8878 compliant implementation
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_lz77.h" // Ensure Match and ParseCost are defined
#include "cuda_zstd_manager.h"
#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_stream_pool.h"
#include "cuda_zstd_types.h" // Also include for workspace struct
#include "cuda_zstd_xxhash.h"
#include "lz77_parallel.h" // For V2 pipeline
#include "workspace_manager.h"
#include <algorithm>
#include <chrono> // For performance timing
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <future> // Added for std::async
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <vector>
#include <zstd.h> // Libzstd for CPU fallback

// Fix for Windows macro collision with Status::ERROR_INVALID_PARAMETER
#ifdef ERROR_INVALID_PARAMETER
#undef ERROR_INVALID_PARAMETER
#endif

namespace cuda_zstd {

namespace lz77 {
void test_linkage_v2(int x) {}
} // namespace lz77

// Add alignment constants
constexpr u32 GPU_MEMORY_ALIGNMENT = 256; // Most GPU requirements

// Predefined Mode Conversion Kernel
// Converts raw Match Lengths to FSE Codes and detects "Gap" matches.
__global__ void convert_sequences_to_fse_codes_kernel(
    const u32 *d_ll_in, const u32 *d_ml_in, const u32 *d_of_in,
    u32 num_sequences, unsigned char *d_ll_out, u32 *d_ll_extra_out,
    unsigned char *d_ll_bits_out, unsigned char *d_ml_out, u32 *d_ml_extra_out,
    unsigned char *d_ml_bits_out, unsigned char *d_of_out, u32 *d_of_extra_out,
    unsigned char *d_of_bits_out, u32 *d_incompatible_flag) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_sequences)
    return;

  u32 ml = d_ml_in[idx];
  u32 ll = d_ll_in[idx];
  u32 of = d_of_in[idx];

#ifdef CUDA_ZSTD_DEBUG
  if (idx < 10) {
    printf("[FSE_CODES] seq[%u]: ll=%u, ml=%u, of=%u\n", idx, ll, ml, of);
  }
#endif

  // Match Length
  u32 ml_code = sequence::ZstdSequence::get_match_len_code(ml);
  
  

  u32 ml_base = sequence::ZstdSequence::get_ml_base_predefined(ml_code);
  u32 ml_bits = sequence::ZstdSequence::get_match_len_extra_bits(ml_code);
  u32 ml_max = (ml_bits >= 32) ? 0xFFFFFFFF : (ml_base + (1u << ml_bits) - 1);
  if (ml < ml_base || ml > ml_max) {
    *d_incompatible_flag = 1;
#ifdef CUDA_ZSTD_DEBUG
    printf("[FSE_CODES] seq[%u] ML INCOMPATIBLE: ml=%u, ml_code=%u, ml_base=%u, ml_bits=%u, ml_max=%u\n",
           idx, ml, ml_code, ml_base, ml_bits, ml_max);
#endif
  }

  d_ml_out[idx] = (unsigned char)ml_code;
  d_ml_bits_out[idx] = (unsigned char)ml_bits;
  d_ml_extra_out[idx] = ml - ml_base;

  u32 ll_code = sequence::ZstdSequence::get_lit_len_code(ll);
  u32 ll_base = sequence::ZstdSequence::get_ll_base_predefined(ll_code);
  u32 ll_bits = sequence::ZstdSequence::get_lit_len_extra_bits(ll_code);
  u32 ll_max = (ll_bits >= 32) ? 0xFFFFFFFF : (ll_base + (1u << ll_bits) - 1);
  if (ll < ll_base || ll > ll_max) {
    *d_incompatible_flag = 1;
#ifdef CUDA_ZSTD_DEBUG
    printf("[FSE_CODES] seq[%u] LL INCOMPATIBLE: ll=%u, ll_code=%u, ll_base=%u, ll_bits=%u, ll_max=%u\n",
           idx, ll, ll_code, ll_base, ll_bits, ll_max);
#endif
  }

  d_ll_out[idx] = (unsigned char)ll_code;
  d_ll_bits_out[idx] = (unsigned char)ll_bits;
  d_ll_extra_out[idx] = ll - ll_base;


  // Offset
  // RFC 8878: Offset Code encodes bits to read. Extra Value is remaining bits.
  // Zstd requires adding a bias of 3 to the literal offset value before
  // encoding.
  u32 of_code = sequence::ZstdSequence::get_offset_code(of);
  u32 of_bits = sequence::ZstdSequence::get_offset_code_extra_bits(of_code);
  u32 of_extra = sequence::ZstdSequence::get_offset_extra_bits(of, of_code);

  d_of_out[idx] = (unsigned char)of_code;
  d_of_bits_out[idx] = (unsigned char)of_bits;
  d_of_extra_out[idx] = of_extra;
}

inline size_t align_to_boundary(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

// ==============================================================================
// EXTREME PERFORMANCE ADAPTIVE BUFFER SIZING
// Uses official Zstd bound formula + GPU-friendly pool sizes
// Reduces memory usage by 50-87% compared to 8x approach
// ==============================================================================

// Official Zstd compress bound - EXACT formula from zstd.h
// Guarantees sufficient buffer for any input data
__host__ __device__ inline size_t zstd_compress_bound(size_t src_size) {
  // Margin for small inputs (< 128KB): gradual reduction from 64 to 0
  size_t margin =
      (src_size < (128 << 10)) ? (((128 << 10) - src_size) >> 11) : 0;
  return src_size + (src_size >> 8) + margin;
}

// GPU-aligned buffer sizing with cache-friendly pooling
// Snaps to optimal pool sizes for memory reuse and cache efficiency
inline size_t calculate_adaptive_output_size(size_t input_size) {
  // Tier 1: Calculate exact Zstd bound
  size_t zstd_bound = zstd_compress_bound(input_size);

  // Tier 2: Add safety margin based on size tier
  size_t safety_margin;
  if (input_size < 1024) {
    safety_margin = 512; // Small inputs need relative margin
  } else if (input_size < 64 * 1024) {
    safety_margin = 1024; // 1KB for small-medium
  } else if (input_size < 1024 * 1024) {
    safety_margin = 4 * 1024; // 4KB for medium
  } else if (input_size < 100 * 1024 * 1024) {
    safety_margin = 16 * 1024; // 16KB for large
  } else {
    safety_margin = 64 * 1024; // 64KB for very large
  }

  size_t required = zstd_bound + safety_margin;

  // Tier 3: Snap to GPU-friendly pool sizes for cache reuse
  // Pool sizes optimized for CUDA memory allocator efficiency
  constexpr size_t pool_sizes[] = {
      4 * 1024,             // 4KB
      16 * 1024,            // 16KB
      64 * 1024,            // 64KB
      256 * 1024,           // 256KB
      1024 * 1024,          // 1MB
      4 * 1024 * 1024,      // 4MB
      16 * 1024 * 1024,     // 16MB
      64 * 1024 * 1024,     // 64MB
      256 * 1024 * 1024,    // 256MB
      1024ULL * 1024 * 1024 // 1GB
  };
  constexpr int num_pools = sizeof(pool_sizes) / sizeof(pool_sizes[0]);

  for (int i = 0; i < num_pools; ++i) {
    if (required <= pool_sizes[i]) {
      return pool_sizes[i];
    }
  }

  // Tier 4: For huge inputs (>1GB), align to 256MB boundaries
  return ((required + (256ULL * 1024 * 1024 - 1)) / (256ULL * 1024 * 1024)) *
         (256ULL * 1024 * 1024);
}

// Helper: Align pointer to boundary
inline unsigned char *align_ptr(unsigned char *ptr, size_t alignment) {
  uintptr_t addr = (uintptr_t)ptr;
  addr = (addr + alignment - 1) & ~(alignment - 1);
  return (unsigned char *)addr;
}

// ==============================================================================
// Kernel to add bias to offsets (ZSTD requires Offset + 3)
__global__ void add_offset_bias_kernel(const u32 *src, u32 *dst, u32 count) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = src[idx] + 3;
  }
}

// ==============================================================================
// ASYNC VALUE SETTER KERNEL (Avoids H2D Sync from Stack)
__global__ void k_record_size(u32 *ptr, u32 val) {
  if (ptr)
    *ptr = val;
}

// ==============================================================================
// READ 3 BYTES SAFELY (Fixes Unaligned Access)
// ==============================================================================
__global__ void k_read_3bytes(const unsigned char *src, u32 *dst) {
  if (threadIdx.x == 0) {
    uintptr_t addr = (uintptr_t)src;
    uintptr_t aligned_addr = addr & ~3; // Align to 4 bytes
    u32 offset = addr & 3;

    const u32 *base = (const u32 *)aligned_addr;
    // Read two words to cover all offset cases (0, 1, 2, 3)
    // Little Endian assumption
    u32 w0 = base[0];
    u32 w1 = base[1];

    u64 combined = ((u64)w1 << 32) | w0;
    u64 shifted = combined >> (offset * 8);

    *dst = (u32)(shifted & 0xFFFFFF);
  }
}

// ==============================================================================
// ASYNC SUM REDUCE KERNEL (replaces sync thrust::reduce)
// Reduces array of u32 to single sum, writes to device memory (no sync needed)
// ==============================================================================
__global__ void async_sum_reduce_kernel(const u32 *input, u32 count,
                                        u32 *output) {
  __shared__ u32 sdata[256];

  u32 tid = threadIdx.x;
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load and accumulate
  u32 sum = 0;
  for (u32 idx = i; idx < count; idx += blockDim.x * gridDim.x) {
    sum += input[idx];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Reduce within block
  for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

// Launch async sum reduce (no sync, result in d_output)
inline void launch_async_sum_reduce(const u32 *input, u32 count, u32 *d_output,
                                    cudaStream_t stream) {
  if (count == 0)
    return;
  // Clear output first
  cudaMemsetAsync(d_output, 0, sizeof(u32), stream);
  // Launch kernel
  const u32 block_size = 256;
  const u32 num_blocks = std::min((count + block_size - 1) / block_size, 256u);
  async_sum_reduce_kernel<<<num_blocks, block_size, 0, stream>>>(input, count,
                                                                 d_output);
}

// (REMOVED) PERFORMANCE PROFILER IMPLEMENTATION - Now in cuda_zstd_utils.cpp
// ==============================================================================

// ==============================================================================
// TWO-PHASE COMPRESSION: BlockMetadata struct
// Stores per-block LZ77 results for batched encoding
// ==============================================================================
struct BlockMetadata {
  u32 num_sequences;  // Number of LZ77 sequences
  u32 total_literals; // Sum of literal lengths
  bool has_dummy;     // Trailing literal indicator
  u32 block_size;     // Original block size
  u32 block_offset;   // Offset in input
};

// ==============================================================================
// ZSTANDARD FRAME CONSTANTS (RFC 8878)
// ==============================================================================

constexpr u32 ZSTD_MAGIC_NUMBER = 0xFD2FB528;
// Use internal names to avoid conflict with zstd.h macros

// constexpr u32 CUDA_ZSTD_MAGIC_DICTIONARY = 0xEC30A437;
constexpr u32 CUDA_ZSTD_BLOCKSIZE_MAX = 128 * 1024; // 128 KB
constexpr u32 CUDA_ZSTD_WINDOWLOG_MAX = 27;
// constexpr u32 CUDA_ZSTD_MIN_CLEVEL = 1;
// constexpr u32 CUDA_ZSTD_MAX_CLEVEL = 22;
constexpr u32 CUDA_ZSTD_DEFAULT_CLEVEL = 3;
constexpr u32 CUDA_ZSTD_WINDOWLOG_MIN = 10;
// constexpr u32 CUDA_ZSTD_CHAINLOG_MAX = 29;
// constexpr u32 CUDA_ZSTD_CHAINLOG_MIN = 6;
// constexpr u32 CUDA_ZSTD_HASHLOG_MAX = 30;
// constexpr u32 CUDA_ZSTD_HASHLOG_MIN = 6;
// constexpr u32 CUDA_ZSTD_SEARCHLOG_MAX = 26;
// constexpr u32 CUDA_ZSTD_SEARCHLOG_MIN = 18;

// Frame header sizes
constexpr u32 FRAME_HEADER_SIZE_MIN = 2;
// constexpr u32 FRAME_HEADER_SIZE_MAX = 18;

// ==============================================================================
//  CUSTOM METADATA DEFINITIONS
// ==============================================================================

// Defines the standard Zstd Skippable Frame Header
struct SkippableFrameHeader {
  u32 magic_number;
  u32 frame_size; // Size of the data that follows (CustomMetadataFrame)
};

// This is our custom data. We can add more fields here later.
struct CustomMetadataFrame {
  u32 custom_magic;      // Set to CUSTOM_METADATA_MAGIC
  i32 compression_level; // The level we want to save
};

// ==============================================================================
// HELPER KERNELS AND FUNCTIONS
// ==============================================================================

// Forward declaration
Status parse_frame_header(const unsigned char *input, u32 input_size,
                          u32 *header_size, u32 *content_size,
                          bool *is_single_segment, bool *has_checksum);

__global__ void debug_inspect_memory(const unsigned char *ptr, u32 size,
                                     const char *label) {
  // Debug output removed for production - noop
}

// Kernel: Check if a block is RLE (all bytes identical)
// Returns: *d_is_rle = 1 if RLE, 0 otherwise
//          *d_rle_byte = value of the repeated byte
__global__ void check_rle_kernel(const unsigned char *input, size_t size,
                                 int *d_is_rle, unsigned char *d_rle_byte) {
  if (size == 0)
    return;

  // Use shared memory to broadcast the first byte
  __shared__ unsigned char s_target;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // First thread reads the reference byte
  if (tid == 0) {
    s_target = input[0];
    *d_is_rle = 1;          // Initialize result
    *d_rle_byte = input[0]; // Store result
  }
  __syncthreads();

  unsigned char target = s_target;

  // Each thread checks its chunk
  // Use a local flag to reduce global writes
  int local_mismatch = 0;

  for (size_t i = tid; i < size; i += stride) {
    if (input[i] != target) {
      local_mismatch = 1;
      // We can stop checking this thread's chunk, but others might still be
      // running Optimization: No break, to avoid divergence, or break? simple
      // break is fine.
      break;
    }
  }

  // If any mismatch found, write to global memory.
  // Race condition on write is fine (all write 0).
  if (local_mismatch) {
    *d_is_rle = 0;
  }
}

// Kernel: Write RLE Block Header (Type 1)
// Header layout:
// - Block Header (3 bytes):
//   - bit 0: last_block
//   - bit 1-2: block_type (1 = RLE)
//   - bit 3-20: RLE_Size (21 bits) - usually checked against header
// For RLE block: Header = (RLE_Size << 3) | (1 << 1) | last_block
// Content: 1 byte (the repeated value)
// Total output: 4 bytes
__global__ void write_rle_block_kernel(unsigned char *output, size_t rle_size,
                                       int last_block) {
  if (threadIdx.x > 0)
    return;

  u32 header = 0;
  header |= (last_block & 0x1);   // Last block bit
  header |= (1 << 1);             // Block Type 1 (RLE_BLOCK)
  header |= (u32)(rle_size << 3); // Size (RLE length)

  // Write 3-byte header (Little Endian)
  output[0] = (unsigned char)(header & 0xFF);
  output[1] = (unsigned char)((header >> 8) & 0xFF);
  output[2] = (unsigned char)((header >> 16) & 0xFF);
}

// Kernel to write skippable header and metadata directly on device
// This avoids cudaMemcpy invalid argument errors on specific platforms/drivers
__global__ void write_skippable_header_kernel(unsigned char *d_output,
                                              u32 skippable_magic,
                                              u32 frame_size, u32 custom_magic,
                                              u32 compression_level) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Write SkippableFrameHeader
    // Struct layout: magic (4), size (4)
    u32 *header_u32 = (u32 *)d_output;
    header_u32[0] = skippable_magic;
    header_u32[1] = frame_size;

    // Write CustomMetadataFrame
    // Struct layout: magic (4), level (4)
    // Offset 8 bytes (after header)
    unsigned char *meta_ptr = d_output + 8;

    // We cast to appropriate types
    u32 *meta_u32 = (u32 *)meta_ptr;
    meta_u32[0] = custom_magic;
    meta_u32[1] = compression_level;
  }
}

/**
 * @brief Expands a unsigned char[] array to a u32[] array.
 * This is used for 'Raw' sequence streams.
 */
__global__ void expand_bytes_to_u32_kernel(const unsigned char *d_input,
                                           u32 *d_output, u32 num_sequences) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < num_sequences; i += stride) {
    d_output[i] = (u32)d_input[i];
  }
}

/**
 * @brief Expands a single byte (RLE) to a full block.
 */
__global__ void expand_rle_kernel(unsigned char *d_output,
                                  u32 decompressed_size, unsigned char value) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < decompressed_size; i += stride) {
    d_output[i] = value;
  }
}

/**
 * @brief Expands a single u32 value (RLE) for sequence components.
 */
__global__ void expand_rle_u32_kernel(u32 *d_output, u32 num_sequences,
                                      u32 value) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < num_sequences; i += stride) {
    d_output[i] = value;
  }
}

/**
 * @brief Aligns a pointer to the specified byte alignment.
 */
template <typename T> T *align_ptr(T *ptr, size_t alignment) {
  uintptr_t int_ptr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned_ptr = (int_ptr + alignment - 1) & ~(alignment - 1);
  return reinterpret_cast<T *>(aligned_ptr);
}

// ============================================================================
// Helper Class: BlockBufferWriter
// Manages output buffer writing with bounds checking to prevent corruption
// ============================================================================
class BlockBufferWriter {
private:
  unsigned char *_base;
  size_t _capacity;
  size_t _offset;
  bool _overflow;

public:
  __host__ BlockBufferWriter(unsigned char *base, size_t capacity)
      : _base(base), _capacity(capacity), _offset(0), _overflow(false) {}

  __host__ bool write_byte(unsigned char value, cudaStream_t stream) {
    if (_offset + 1 > _capacity) {
      _overflow = true;
      return false;
    }
    
    // Use synchronous copy for single byte to ensure safety of stack variable
    // address
    if (cudaMemcpy(_base + _offset, &value, 1, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      return false;
    }
    _offset++;
    return true;
  }

  __host__ bool write_bytes(const void *src, size_t size, cudaStream_t stream,
                            bool is_device_ptr = false) {
    if (size == 0)
      return true;
    if (_offset + size > _capacity) {
      _overflow = true;
      return false;
    }

    if (is_device_ptr) {
      if (cudaMemcpyAsync(_base + _offset, src, size, cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess) {
        return false;
      }
    } else {
      // Use synchronous copy for host pointers to support stack/temporary
      // buffers
      if (cudaMemcpy(_base + _offset, src, size, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        return false;
      }
    }

    _offset += size;
    return true;
  }

  __host__ size_t get_offset() const { return _offset; }
  __host__ bool has_overflowed() const { return _overflow; }

  // Aligns current offset to 4-byte boundary (if needed for raw blocks)
  __host__ void align4() {
    size_t remainder = _offset & 3;
    if (remainder != 0) {
      size_t padding = 4 - remainder;
      if (_offset + padding <= _capacity) {
        // Zero pad? Or just advance? Usually just advance for alignment
        // but explicit zeroing is safer if buffer is reused.
        // For now, just advance.
        _offset += padding;
      } else {
        _overflow = true;
      }
    }
  }

  __host__ unsigned char *get_current_ptr() const { return _base + _offset; }
  __host__ void advance(size_t amount) {
    if (_offset + amount > _capacity) {
      _overflow = true;
    } else {
      _offset += amount;
    }
  }
};

// ============================================================================
// Internal Helper Functions
// ============================================================================

__global__ void
copy_block_literals_kernel(const unsigned char *input,
                           const u32 *literal_lengths, const u32 *match_lengths,
                           u32 num_sequences, unsigned char *literals_buffer,
                           u32 block_idx, u32 capacity, u32 input_size,
                           u32 *d_extracted_size) {
  // Simple sequential copy per block (launched with 1 thread)
  // Optimization: Could be parallelized with prefix sums, but this is
  // functional
  u32 in_pos = 0;
  u32 out_pos = 0;

  for (u32 i = 0; i < num_sequences; ++i) {
    u32 ll = literal_lengths[i];
    u32 ml = match_lengths[i];

    // Safety Check
    if (out_pos + ll > capacity) {
#ifdef CUDA_ZSTD_DEBUG
      printf("[FATAL GPU] Block %u: Literals Buffer OOB! Seq %u, LL %u, Out "
             "%u, Cap %u\n",
             block_idx, i, ll, out_pos, capacity);
#endif
      if (d_extracted_size) *d_extracted_size = out_pos;
      return; // Prevent crash
    }
    // Note: in_pos check is harder without knowing input size, but we assume
    // input valid

    // Valid input bounds check
    if (in_pos + ll > input_size) {
#ifdef CUDA_ZSTD_DEBUG
      printf("[FATAL GPU] Block %u: Input OOB! Seq %u, LL %u, In %u, Size %u\n",
             block_idx, i, ll, in_pos, input_size);
#endif
      if (d_extracted_size) *d_extracted_size = out_pos;
      return;
    }

    // Copy literals
    for (u32 k = 0; k < ll; ++k) {
      literals_buffer[out_pos + k] = input[in_pos + k];

    }

    in_pos += ll + ml;
    out_pos += ll;
  }

  // Copy trailing literals
  if (in_pos < input_size) {
      u32 trailing = input_size - in_pos;
      if (out_pos + trailing <= capacity) {
          for(u32 k=0; k<trailing; ++k) {
              literals_buffer[out_pos + k] = input[in_pos + k];
          }
          out_pos += trailing;
      } else {
#ifdef CUDA_ZSTD_DEBUG
          printf("[FATAL GPU] Block %u: Literals Buffer OOB (Trailing)! Need %u, Cap %u\n",
                 block_idx, out_pos + trailing, capacity);
#endif
      }
  }

  if (d_extracted_size) *d_extracted_size = out_pos;
}

// Helper to launch the kernel
void launch_copy_literals(const unsigned char *input, u32 input_size,
                          const u32 *literal_lengths, const u32 *match_lengths,
                          u32 num_sequences, unsigned char *literals_buffer,
                          cudaStream_t stream, u32 block_idx,
                          u32 *d_extracted_size) {
  copy_block_literals_kernel<<<1, 1, 0, stream>>>(
      input, literal_lengths, match_lengths, num_sequences, literals_buffer,
      block_idx, 131072, input_size, d_extracted_size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    // Log synchronization error silently
  }
}

// ==============================================================================
// INTERNAL STRUCTURES
// ==============================================================================

struct BlockInfo {
  unsigned char *data;
  u32 size;
  bool is_compressed;
  bool is_last;
};

struct CompressionContext {
  // LZ77 matching
  lz77::LZ77Context *lz77_ctx;

  // Sequence encoding
  sequence::SequenceContext *seq_ctx;

  // FSE tables
  fse::FSEEncodeTable *lit_fse_table;
  fse::FSEEncodeTable *ml_fse_table;
  fse::FSEEncodeTable *of_fse_table;

  // Huffman
  huffman::HuffmanTable *huff_ctx;

  //  Workspace for all temporary allocations
  CompressionWorkspace workspace;

  // Temporary buffers
  unsigned char *d_temp_buffer; // Persistent temp buffer
  u32 temp_buffer_size;

  //  Multi-stream support for pipelining
  cudaStream_t *streams;  // Array of CUDA streams
  cudaEvent_t *events;    // Array of CUDA events
  u32 num_streams;        // Number of streams in pool
  u32 current_stream_idx; // Round-robin stream selection

  // Statistics
  u64 total_matches;
  u64 total_literals;
  u64 total_sequences;
};

struct StreamingContext {
  // Window history from previous chunks
  unsigned char *d_window_history; // Device: last N bytes
  u32 window_history_size;         // Current filled size
  u32 window_history_capacity;     // Max capacity (32KB-128KB)

  // Hash chain persistence
  u32 *d_hash_table_state; // Persistent across chunks
  u32 *d_chain_table_state;

  // Offset tracking for proper distances
  u64 total_bytes_processed; // Cumulative across chunks

  // Frame state
  bool started_compression;
  bool finished_compression;
  u32 block_count;

  // Streaming xxHash state
  xxhash::XXH64_State *d_xxhash_state;

  // Long Distance Matching state
  ldm::LDMContext ldm_ctx;
  bool ldm_initialized;
};

// ==============================================================================
// RFC 8878 Frame Header Structure
// ==============================================================================
//
// Frame_Header = Magic_Number Frame_Header_Descriptor (Optional_Data_Block)*
// Magic_Number = 4 bytes = 0x28, 0xB5, 0x2F, 0xFD (= 0xFD2FB528 in
// little-endian) Frame_Header_Descriptor = 1 byte (FHDB) Optional_Data_Block =
// Window_Descriptor | Dictionary_ID | Content_Size | Checksum

// RFC 8878 Frame Header Structure
struct FrameHeaderDescriptor {
  u8 fhd;

  bool has_dictionary_id() const { return (fhd & 0x03) != 0; }

  bool has_content_size() const { return (fhd & 0xC0) != 0; }

  bool has_checksum() const {
    return (fhd & 0x04) != 0; // Bit 2
  }

  bool is_single_segment() const { return (fhd & 0x20) != 0; }

  u32 get_dictionary_id_size() const {
    u32 did = (fhd >> 0) & 0x03;
    if (did == 0)
      return 0;
    return (1 << (did - 1)); // 1->1, 2->2, 3->4
  }

  u32 get_content_size_bytes() const {
    u32 csf = (fhd >> 6) & 0x03;
    if (is_single_segment()) {
      if (csf == 0)
        return 1;
      return (1 << csf);
    } else {
      if (csf == 0)
        return 0;
      return (1 << csf);
    }
  }
};

struct ZstdFrameMetadata {
  u32 magic_number;
  u8 fhd; // Frame_Header_Descriptor

  // Optional fields
  u32 dictionary_id; // Optional
  u64 content_size;  // Optional

  // Computed
  u32 frame_header_size; // Total size of frame header
  u32 checksum_value;    // Content checksum (after all blocks)

  bool has_dict;
  bool has_content_size;
  bool has_checksum;
};

// Helper: Read little-endian u32
inline u32 read_u32_le(const unsigned char *data) {
  return ((u32)data[0]) | ((u32)data[1] << 8) | ((u32)data[2] << 16) |
         ((u32)data[3] << 24);
}

// Helper: Read little-endian u64
inline u64 read_u64_le(const unsigned char *data) {
  return ((u64)read_u32_le(data)) | (((u64)read_u32_le(data + 4)) << 32);
}

Status parse_zstd_frame_header(const unsigned char *compressed_data,
                               size_t compressed_size,
                               ZstdFrameMetadata *metadata) {
  if (!compressed_data || !metadata || compressed_size < 6) {
    return Status::ERROR_CORRUPT_DATA;
  }

  size_t offset = 0;

  // 1. Read magic number (4 bytes)
  metadata->magic_number = read_u32_le(compressed_data + offset);
  offset += 4;

  if (metadata->magic_number != ZSTD_MAGIC_NUMBER) {
    // Check for skippable frame
    if ((metadata->magic_number & 0xFFFFFFF0) == ZSTD_MAGIC_SKIPPABLE_START) {
      // This is a skippable frame - parse and skip it
      if (compressed_size < offset + 4) {
        return Status::ERROR_CORRUPT_DATA;
      }
      u32 frame_size = read_u32_le(compressed_data + offset);
      // Frame size doesn't include the 8-byte header
      offset += 4 + frame_size;

      // Check for next frame
      if (offset >= compressed_size) {
        return Status::ERROR_CORRUPT_DATA;
      }

      // Recursively parse the next frame
      return parse_zstd_frame_header(compressed_data + offset,
                                     compressed_size - offset, metadata);
    }
    return Status::ERROR_INVALID_MAGIC;
  }

  // 2. Read Frame Header Descriptor (1 byte)
  if (offset >= compressed_size) {
    return Status::ERROR_CORRUPT_DATA;
  }

  metadata->fhd = compressed_data[offset];
  offset += 1;

  FrameHeaderDescriptor fhd_parser;
  fhd_parser.fhd = metadata->fhd;

  metadata->has_dict = fhd_parser.has_dictionary_id();
  metadata->has_content_size = fhd_parser.has_content_size();
  metadata->has_checksum = fhd_parser.has_checksum();

  // 3. Parse optional Window_Descriptor (only if not single-segment)
  if (!fhd_parser.is_single_segment()) {
    if (offset >= compressed_size) {
      return Status::ERROR_CORRUPT_DATA;
    }
    // Window descriptor is 1 byte, extract window size for information
    offset += 1;
    // window_exponent = (window_desc >> 3) & 0x1F
    // window_size = (1 << (10 + window_exponent))
  }

  // 4. Parse Dictionary_ID (optional, variable size)
  metadata->dictionary_id = 0;
  if (metadata->has_dict) {
    u32 did_size = fhd_parser.get_dictionary_id_size();

    if (offset + did_size > compressed_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    switch (did_size) {
    case 1:
      metadata->dictionary_id = compressed_data[offset];
      break;
    case 2:
      metadata->dictionary_id = read_u32_le(compressed_data + offset) & 0xFFFF;
      break;
    case 4:
      metadata->dictionary_id = read_u32_le(compressed_data + offset);
      break;
    default:
      return Status::ERROR_CORRUPT_DATA;
    }
    offset += did_size;
  }

  // 5. Parse Content_Size (optional, variable size)
  metadata->content_size = 0;
  if (metadata->has_content_size) {
    u32 cs_size = fhd_parser.get_content_size_bytes();
    if (offset + cs_size > compressed_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    switch (cs_size) {
    case 1:
      metadata->content_size = compressed_data[offset];
      break;
    case 2:
      metadata->content_size = (read_u32_le(compressed_data + offset) & 0xFFFF);
      // RFC 8878 Section 3.1.1.1.2: When FCS_Field_Size is 2 and
      // Single_Segment_flag is set, the stored value has a +256 offset.
      // The writer subtracts 256 before storing, so we add it back.
      if (fhd_parser.is_single_segment()) {
        metadata->content_size += 256;
      }
      break;
    case 4:
      metadata->content_size = read_u32_le(compressed_data + offset);
      break;
    case 8:
      metadata->content_size = read_u64_le(compressed_data + offset);
      break;
    default:
      return Status::ERROR_CORRUPT_DATA;
    }
    offset += cs_size;
  }

  metadata->frame_header_size = (u32)offset;

  return Status::SUCCESS;
}

// New public function: Extract metadata from compressed data
// New public function: Extract metadata from compressed data
Status extract_metadata(const void *compressed_data, size_t compressed_size,
                        NvcompMetadata &metadata) {
  if (!compressed_data || compressed_size < 4) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  ZstdFrameMetadata internal_meta;
  Status status = parse_zstd_frame_header(
      static_cast<const unsigned char *>(compressed_data), compressed_size,
      &internal_meta);

  if (status != Status::SUCCESS)
    return status;

  // Map to NvcompMetadata
  metadata.format_version = 0x00010000; // 1.0.0
  metadata.compression_level = 3;       // Default (unknown)
  metadata.uncompressed_size = (u32)internal_meta.content_size;
  metadata.dictionary_id = internal_meta.dictionary_id;
  metadata.checksum_policy = internal_meta.has_checksum
                                 ? ChecksumPolicy::COMPUTE_AND_VERIFY
                                 : ChecksumPolicy::NO_COMPUTE_NO_VERIFY;

  // Estimate chunks if content size is known
  metadata.chunk_size = CUDA_ZSTD_BLOCKSIZE_MAX; // 128KB default
  if (internal_meta.has_content_size) {
    metadata.num_chunks =
        (metadata.uncompressed_size + metadata.chunk_size - 1) /
        metadata.chunk_size;
  } else {
    metadata.num_chunks = 0;
  }

  return Status::SUCCESS;
}

// ==============================================================================
// DEFAULT ZSTD MANAGER IMPLEMENTATION
// ==============================================================================

class DefaultZstdManager : public ZstdManager {
private:
  CompressionConfig config;
  CompressionStats stats;
  // Stream pooling implemented via StreamPool for efficient reuse.
  // Future enhancement: Add stream_pool member if needed.
  dictionary::Dictionary dict;
  bool has_dictionary;
  CompressionContext ctx;
  fse::FSEEncodeTable *d_cached_fse_tables; // Phase 2b: Cached tables for reuse
  bool ctx_initialized;
  u64 *d_checksum_buffer;
  std::unique_ptr<StreamPool> stream_pool_;
  mutable std::mutex api_mutex;

  struct PrevFseState {
    fse::FSEDecodeTable table{};
    bool has_table = false;
    u32 mode = 0;
    u8 rle_value = 0;
  };

  PrevFseState prev_ll_{};
  PrevFseState prev_of_{};
  PrevFseState prev_ml_{};

  static void free_fse_decode_table(fse::FSEDecodeTable &table) {
    delete[] table.newState;
    delete[] table.symbol;
    delete[] table.nbBits;
    delete[] table.nbAdditionalBits;
    delete[] table.baseValue;
    table = {};
  }

  static Status copy_fse_decode_table(const fse::FSEDecodeTable &src,
                                      fse::FSEDecodeTable &dst) {
    if (!src.newState || !src.symbol || !src.nbBits || !src.nbAdditionalBits ||
        !src.baseValue) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    if (dst.newState || dst.symbol || dst.nbBits || dst.nbAdditionalBits ||
        dst.baseValue) {
      free_fse_decode_table(dst);
    }

    dst.table_log = src.table_log;
    dst.table_size = src.table_size;
    dst.newState = new u16[src.table_size];
    dst.symbol = new u8[src.table_size];
    dst.nbBits = new u8[src.table_size];
    dst.nbAdditionalBits = new u8[src.table_size];
    dst.baseValue = new u32[src.table_size];

    std::copy(src.newState, src.newState + src.table_size, dst.newState);
    std::copy(src.symbol, src.symbol + src.table_size, dst.symbol);
    std::copy(src.nbBits, src.nbBits + src.table_size, dst.nbBits);
    std::copy(src.nbAdditionalBits, src.nbAdditionalBits + src.table_size,
              dst.nbAdditionalBits);
    std::copy(src.baseValue, src.baseValue + src.table_size, dst.baseValue);

    return Status::SUCCESS;
  }

public:
  DefaultZstdManager(int compression_level = 3)
      : config(CompressionConfig::from_level(compression_level)),
        has_dictionary(false), d_cached_fse_tables(nullptr),
        ctx_initialized(false) {
    //         // std::cerr << "DefaultZstdManager ctor start" << std::endl;
    // Initialize stream pool
    // Initialize stream pool
    stream_pool_ = std::make_unique<StreamPool>(1); // Try with 1 stream

    //         // std::cerr << "DefaultZstdManager ctor after
    //         get_global_stream_pool(1)" << std::endl;
    d_checksum_buffer = nullptr;
    config.min_match = 3;
    config.min_match = 3;
    config.strategy = Strategy::GREEDY;
    // Default LZ77 config (Moved from initialize_context to ensure
    // get_compress_temp_size is correct)
    config.window_log = 22; // 4MB window
    config.chain_log = 17;  // 128K entries
    config.hash_log = 18;   // 256K entries
    config.cpu_threshold =
        0; // Usage of GPU is enforced for debugging/correctness verification

    reset_stats();
    memset(&ctx, 0, sizeof(CompressionContext));
    // LAZY INITIALIZATION: Don't allocate GPU memory in constructor
    // initialize_context() will be called on first compress() at line 1065-1074
    // initialize_context();
    //         // std::cerr << "DefaultZstdManager ctor end" << std::endl;
  }

  void cleanup_context() {
    if (!ctx_initialized)
      return;

    free_tables(0); // Cleanup cached tables

    // NOTE: Synchronize device BEFORE destroying streams - REMOVED for

    // concurrency cudaDeviceSynchronize() blocks ALL threads, causing timeouts.
    // We strictly rely on stream_pool and assume user synchronizes their own
    // streams.

    // Stream pool cleaning is handled by unique_ptr logic or trivial
    // destruction if needed. ctx.streams is unused (legacy code), removing
    // loop.

    // Free workspace AFTER synchronization
    free_compression_workspace(ctx.workspace);

    // Free other resources...
    if (ctx.seq_ctx) {
      if (ctx.seq_ctx->d_literals_buffer)
        cudaFree(ctx.seq_ctx->d_literals_buffer);
      if (ctx.seq_ctx->d_literal_lengths)
        cudaFree(ctx.seq_ctx->d_literal_lengths);
      if (ctx.seq_ctx->d_match_lengths)
        cudaFree(ctx.seq_ctx->d_match_lengths);
      if (ctx.seq_ctx->d_offsets)
        cudaFree(ctx.seq_ctx->d_offsets);
      if (ctx.seq_ctx->d_num_sequences)
        cudaFree(ctx.seq_ctx->d_num_sequences);
      if (ctx.seq_ctx->d_sequences)
        cudaFree(ctx.seq_ctx->d_sequences);
      delete ctx.seq_ctx;
      ctx.seq_ctx = nullptr;
    }

    if (ctx.huff_ctx) {
      if (ctx.huff_ctx->codes)
        cudaFree(ctx.huff_ctx->codes);
      delete ctx.huff_ctx;
      ctx.huff_ctx = nullptr;
    }

    delete ctx.lit_fse_table;
    ctx.lit_fse_table = nullptr;
    delete ctx.ml_fse_table;
    ctx.ml_fse_table = nullptr;
    delete ctx.of_fse_table;
    ctx.of_fse_table = nullptr;

    if (ctx.lz77_ctx) {
      if (ctx.lz77_ctx->d_window) {
        cudaFree(ctx.lz77_ctx->d_window);
        ctx.lz77_ctx->d_window = nullptr;
      }

      // Free potential internal allocations from init_lz77_context
      if (ctx.lz77_ctx->hash_table.table) {
        cudaFree(ctx.lz77_ctx->hash_table.table);
        ctx.lz77_ctx->hash_table.table = nullptr;
      }
      if (ctx.lz77_ctx->chain_table.prev) {
        cudaFree(ctx.lz77_ctx->chain_table.prev);
        ctx.lz77_ctx->chain_table.prev = nullptr;
      }
      if (ctx.lz77_ctx->d_matches) {
        cudaFree(ctx.lz77_ctx->d_matches);
        ctx.lz77_ctx->d_matches = nullptr;
      }
      if (ctx.lz77_ctx->d_costs) {
        cudaFree(ctx.lz77_ctx->d_costs);
        ctx.lz77_ctx->d_costs = nullptr;
      }

      // Note: h_matches_dict_trainer is a host buffer, usually managed by
      // std::vector or new[] Assuming it's simple usage or not allocated on
      // GPU. Focus on d_window which is GPU.

      delete ctx.lz77_ctx;
      ctx.lz77_ctx = nullptr;
    }

    ctx_initialized = false;
  }

  virtual ~DefaultZstdManager() {
    // Cleanup any device-side checksum buffer allocated lazily
    if (d_checksum_buffer != nullptr) {
      cudaFree(d_checksum_buffer);
      d_checksum_buffer = nullptr;
    }
    cleanup_context();
    if (prev_ll_.has_table) {
      free_fse_decode_table(prev_ll_.table);
      prev_ll_.has_table = false;
    }
    if (prev_of_.has_table) {
      free_fse_decode_table(prev_of_.table);
      prev_of_.has_table = false;
    }
    if (prev_ml_.has_table) {
      free_fse_decode_table(prev_ml_.table);
      prev_ml_.has_table = false;
    }
  }

  // ==========================================================================
  // Workspace Queries
  // ==========================================================================
  size_t get_compress_temp_size(size_t input_size) const override {
    if (input_size == 0)
      return 0;

    size_t total = 0;

    // 0. Dictionary buffer and content struct
    if (has_dictionary) {
      total +=
          align_to_boundary(sizeof(DictionaryContent), GPU_MEMORY_ALIGNMENT);
      total += align_to_boundary(dict.raw_size, GPU_MEMORY_ALIGNMENT);
    }

    // NOTE: Input buffer - compress() partitions this from workspace if
    // input is on host. We must include it for worst-case.
    total += align_to_boundary(input_size, GPU_MEMORY_ALIGNMENT);

    // NOTE: Output buffer - compress() partitions this from workspace if
    // compressed_data is nullptr. Include estimated compressed size.
    size_t est_compressed =
        (input_size * 110) / 100 + 1024; // ~10% overhead + 1KB
    total += align_to_boundary(est_compressed, GPU_MEMORY_ALIGNMENT);
    u32 block_size = config.block_size;
    if (block_size == 0)
      block_size = get_optimal_block_size((u32)input_size, config.level);

    // Clamp block size to max supported by internal buffers (128KB)
    // Matches logic in compress()
    block_size = std::min(block_size, CUDA_ZSTD_BLOCKSIZE_MAX);

    if (block_size == 0)
      block_size = CUDA_ZSTD_BLOCKSIZE_MAX;

    size_t num_blocks = (input_size + block_size - 1) / block_size;
    if (num_blocks == 0)
      num_blocks = 1;

    // Per-block resources (Must match compress function allocations)
    // These resources are allocated from d_workspace in compress().

    // 1. Hash table (O(N) - Partitioned per block for parallel execution)

    // Reduce log for block-level parallelism (128KB blocks don't need 22 bits)
    // 18 bits = 256KB entries = 1MB per block
    u32 block_hash_log = std::min(config.hash_log, 18u);
    size_t hash_table_size = num_blocks * (1 << block_hash_log) * sizeof(u32);
    total += align_to_boundary(hash_table_size, GPU_MEMORY_ALIGNMENT);

    // 2. Chain table (O(1) - reused across blocks)
    // 2. Chain table (O(N) - Partitioned per block)
    u32 block_chain_log = std::min(config.chain_log, 18u);
    size_t chain_table_size = num_blocks * (1 << block_chain_log) * sizeof(u32);
    total += align_to_boundary(chain_table_size, GPU_MEMORY_ALIGNMENT);

    // Scaled with num_blocks for block-wise parallel addressing.
    size_t matches_bytes =
        num_blocks * ZSTD_BLOCKSIZE_MAX * 16; // sizeof(lz77::Match) ~16 bytes
    total += align_to_boundary(matches_bytes, GPU_MEMORY_ALIGNMENT);

    // 4. Costs buffer (O(N) - Partitioned per block)
    // Used by optimal parser, or potentially reused. allocate safely.
    size_t costs_bytes =
        num_blocks * (ZSTD_BLOCKSIZE_MAX + 1) * 8; // sizeof(ParseCost) ~8 bytes
    total += align_to_boundary(costs_bytes, GPU_MEMORY_ALIGNMENT);

    // 5. Reverse sequence buffers (Global Buffers)
    //    d_all_lit_lengths_reverse, d_all_match_lengths_reverse,
    //    d_all_offsets_reverse
    size_t reverse_seq_bytes =
        num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32) * 3;
    total += align_to_boundary(reverse_seq_bytes, GPU_MEMORY_ALIGNMENT);

    // 5b. On-buffer literals (d_all_literals_buffer) — allocated alongside
    //     reverse sequence buffers in compress() at the on_buffer section.
    //     Uses num_blocks * block_size bytes; upper-bound with
    //     ZSTD_BLOCKSIZE_MAX.
    size_t on_buffer_literals_bytes =
        (size_t)num_blocks * ZSTD_BLOCKSIZE_MAX;
    total += align_to_boundary(on_buffer_literals_bytes, GPU_MEMORY_ALIGNMENT);

    // 6. Forward sequence buffers
    size_t forward_seq_bytes = num_blocks * ZSTD_BLOCKSIZE_MAX * 16;
    total += align_to_boundary(forward_seq_bytes, GPU_MEMORY_ALIGNMENT);

    // 7. Literals buffer
    size_t literals_buffer_bytes = num_blocks * ZSTD_BLOCKSIZE_MAX;
    total += align_to_boundary(literals_buffer_bytes, GPU_MEMORY_ALIGNMENT);

    // 8. Block sums
    size_t block_sums_bytes = num_blocks * 3 * sizeof(u32) * 2;
    total += align_to_boundary(block_sums_bytes, GPU_MEMORY_ALIGNMENT);

    // 9. Checksum buffer
    total += align_to_boundary(sizeof(u64), GPU_MEMORY_ALIGNMENT);

    // 10. Huffman frequencies
    total += align_to_boundary(256 * sizeof(u32), GPU_MEMORY_ALIGNMENT);

    // 11. Huffman code_lengths (per-input-symbol)
    total += align_to_boundary(input_size * sizeof(u32), GPU_MEMORY_ALIGNMENT);

    // 12. Huffman bit_offsets (prefix sum result, per-input-symbol)
    total += align_to_boundary(input_size * sizeof(u32), GPU_MEMORY_ALIGNMENT);

    // 13. Sequence objects (all blocks)
    size_t sequence_objs_bytes =
        num_blocks * ZSTD_BLOCKSIZE_MAX * 16; // sizeof(sequence::Sequence) = 16
    total += align_to_boundary(sequence_objs_bytes, GPU_MEMORY_ALIGNMENT);

    // 14. Per-block metadata arrays (d_block_literals_sizes,
    //     d_block_num_sequences, d_block_has_dummy)
    total += align_to_boundary(3 * num_blocks * sizeof(u32), GPU_MEMORY_ALIGNMENT);

    // --- Per-Block Workspace (ws_base) ---
    // N * (OutputBuffer + FSE_Tables + Huff_Table + LZ77_Temp)
    // NOTE: Matches are allocated globally (above), NOT per-block.

    // 1. LZ77 Temp (decisions + offsets arrays)
    size_t lz77_temp_part = align_to_boundary(CUDA_ZSTD_BLOCKSIZE_MAX * 2 * sizeof(u32), GPU_MEMORY_ALIGNMENT);
    size_t per_block_base = lz77_temp_part;

    // 2. Output Buffer (Adaptive) - Must match compress()
    const size_t min_buffer_size = 256 * 1024; // 256KB min
    size_t adaptive_size = (size_t)(block_size * 4) + 8192; // 4x + 8KB (matches compress())
    size_t output_buffer_size = std::max(min_buffer_size, adaptive_size);
    per_block_base += align_to_boundary(output_buffer_size, GPU_MEMORY_ALIGNMENT);

    // 3. Metadata tables
    size_t fse_table_size_2 = 3 * sizeof(fse::FSEEncodeTable);
    size_t huff_size_2 = sizeof(huffman::HuffmanTable);
    per_block_base += align_to_boundary(fse_table_size_2, GPU_MEMORY_ALIGNMENT);
    per_block_base += align_to_boundary(huff_size_2, GPU_MEMORY_ALIGNMENT);

    // Add per-block size * num_blocks
    total += num_blocks * per_block_base;

    // Padding logic (Existing)
    size_t min_padding = 4 * 1024 * 1024;         // 4MB minimum
    size_t max_padding = 64 * 1024 * 1024;        // 64MB maximum
    size_t proportional_padding = input_size * 4; // 4x input
    size_t scaled_padding =
        std::max(min_padding, std::min(max_padding, proportional_padding));

    total += scaled_padding;

    // Only add padding once
    total += total / 4; // Increase to 25%

    // 10. Final round to reasonable boundary
    total = align_to_boundary(total, 1024 * 1024); // 1MB boundary

    return total;
  }

  // Similar for decompression
  size_t get_decompress_temp_size(size_t compressed_size) const override {
    if (compressed_size == 0)
      return 0;

    size_t total = 0;
    size_t alignment = 128;

    // 1. Checksum (u64)
    total += align_to_boundary(sizeof(u64), alignment);
    
    // 2. Header scratch (u32)
    total += align_to_boundary(sizeof(u32), alignment);
    
    // 3. Temp buffer (ZSTD_BLOCKSIZE_MAX)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX, alignment);
    
    // 4. Sequences (Sequence array)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX * sizeof(sequence::Sequence), alignment);
    
    // 5. Literal Lengths (u32 array)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);
    
    // 6. Match Lengths (u32 array)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);
    
    // 7. Offsets (u32 array)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);
    
    // 8. Rep Codes (3 * u32)
    total += align_to_boundary(3 * sizeof(u32), alignment);
    
    // 9. Literals Buffer (ZSTD_BLOCKSIZE_MAX)
    total += align_to_boundary(ZSTD_BLOCKSIZE_MAX, alignment);

    return total;
  }
  
  // Implementation of get_max_compressed_size
  virtual size_t get_max_compressed_size(size_t uncompressed_size) const override {
      return estimate_compressed_size(uncompressed_size, config.level);
  }

  // Implementation of preallocate_tables
  Status preallocate_tables(cudaStream_t stream = 0) override {
    if (d_cached_fse_tables)
      return Status::SUCCESS; // Already allocated

    // 1. Allocate 3 Tables
    cudaError_t err = cudaMallocAsync(&d_cached_fse_tables,
                                      3 * sizeof(fse::FSEEncodeTable), stream);
    if (err != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;

    // 2. Build Tables (Literals, Offsets, MatchLengths)
    auto build_table = [&](fse::TableType type, int idx) {
      u32 max_s, t_log;
      const u16 *h_norm = fse::get_predefined_norm(type, &max_s, &t_log);

      std::vector<u32> h_norm_u32(max_s + 1);
      for (u32 i = 0; i <= max_s; ++i)
        h_norm_u32[i] = h_norm[i];

      u32 *d_norm;
      cudaMallocAsync(&d_norm, (max_s + 1) * sizeof(u32), stream);
      cudaMemcpyAsync(d_norm, h_norm_u32.data(), (max_s + 1) * sizeof(u32),
                      cudaMemcpyHostToDevice, stream);

      fse::FSEEncodeTable h_desc;
      h_desc.max_symbol = max_s;
      h_desc.table_log = t_log;
      h_desc.table_size = 1 << t_log;

      cudaMallocAsync(
          &h_desc.d_symbol_table,
          (max_s + 1) * sizeof(fse::FSEEncodeTable::FSEEncodeSymbol), stream);
      cudaMallocAsync(&h_desc.d_next_state, h_desc.table_size * sizeof(u16),
                      stream);
      cudaMallocAsync(&h_desc.d_nbBits_table, h_desc.table_size * sizeof(u8),
                      stream);
      cudaMallocAsync(&h_desc.d_symbol_first_state, (max_s + 1) * sizeof(u16),
                      stream);
      cudaMallocAsync(&h_desc.d_state_to_symbol, h_desc.table_size * sizeof(u8),
                      stream);

      cudaMemcpyAsync(&d_cached_fse_tables[idx], &h_desc,
                      sizeof(fse::FSEEncodeTable), cudaMemcpyHostToDevice,
                      stream);

      // CRITICAL: Synchronize to ensure h_desc is copied before it goes out of
      // scope
      cudaStreamSynchronize(stream);

      fse::FSE_buildCTable_Device(
          d_norm, max_s, t_log, &d_cached_fse_tables[idx], nullptr, 0, stream);
      cudaFreeAsync(d_norm, stream);
    };

    build_table(fse::TableType::LITERALS, 0);
    build_table(fse::TableType::OFFSETS, 1);
    build_table(fse::TableType::MATCH_LENGTHS, 2);

    // CRITICAL: Sync to ensure all table builds complete before returning
    cudaStreamSynchronize(stream);

    return Status::SUCCESS;
  }



  Status free_tables(cudaStream_t stream = 0) override {
    if (d_cached_fse_tables) {
      cudaFreeAsync(d_cached_fse_tables, stream);
      d_cached_fse_tables = nullptr;
    }
    return Status::SUCCESS;
  }

  virtual Status configure(const CompressionConfig &new_config) override {
    std::lock_guard<std::mutex> lock(api_mutex);
    auto status = validate_config(new_config);
    if (status != Status::SUCCESS) {
#ifdef CUDA_ZSTD_DEBUG
      fprintf(stderr, "failed with status %d\n", (int)status);
#endif
      return status;
    }

    config = new_config;

    if (config.compression_mode == CompressionMode::LEVEL_BASED) {
      apply_level_parameters(config);
    }

    cleanup_context();
    return initialize_context();
  }

  virtual CompressionConfig get_config() const override { return config; }

  virtual Status set_compression_level(int level) override {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (!is_valid_compression_level(level)) {
      return Status::ERROR_INVALID_PARAMETER;
    }
    config.level = level;
    config.compression_mode = CompressionMode::LEVEL_BASED;
    apply_level_parameters(config);

    cleanup_context();
    return initialize_context();
  }

  virtual int get_compression_level() const override { return config.level; }

  // ============================================================================
  // compress() implementation
  // ============================================================================
  virtual Status compress(const void *uncompressed_data,
                          size_t uncompressed_size, void *compressed_data,
                          size_t *compressed_size, void *temp_workspace,
                          size_t temp_size, const void *dict_buffer,
                          size_t dict_size, cudaStream_t stream,
                          void *streaming_context = nullptr) override {
    std::lock_guard<std::mutex> lock(api_mutex);
    fflush(stdout);
    void *device_workspace = nullptr; // Scope for cleanup

    Status status = Status::SUCCESS;

    // === CRITICAL: Parameter Validation ===
    if (!uncompressed_data || !compressed_data || !compressed_size ||
        !temp_workspace) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    if (uncompressed_size == 0) {
      if (compressed_size)
        *compressed_size = 0;
      return Status::ERROR_INVALID_PARAMETER; // Standard ZSTD returns frame,
                                              // but test expects failure for 0
    }

    // Validate temp buffer size
    size_t required_temp = get_compress_temp_size(uncompressed_size);
    if (temp_size < required_temp) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // === CRITICAL: Block Size Validation ===
    // Block size must not exceed input size
    u32 effective_block_size = config.block_size;
    if (effective_block_size == 0) {
      // Use optimal block size based on input size (512KB for large files)
      effective_block_size =
          get_optimal_block_size((u32)uncompressed_size, config.level);
    }

    if (effective_block_size > uncompressed_size) {
      // Auto-adjust block size to input size
      effective_block_size = (u32)uncompressed_size;
    }

    // Create effective config with validated block size
    CompressionConfig effective_config = config;
    effective_config.block_size = effective_block_size;
    effective_config.cpu_threshold = config.cpu_threshold;

    if (!ctx_initialized) {
      auto status = initialize_context();
      if (status != Status::SUCCESS) {
        return status;
      }
    }

    if (!ctx.seq_ctx) {
      ctx.seq_ctx = new sequence::SequenceContext();
    }

    cudaGetLastError(); // Clear previous errors

    // ======================================================================
    // Hybrid Execution: Route small payloads (<1MB) to CPU to avoid overhead
    // ======================================================================
    // ======================================================================

    auto exec_path = ZstdBatchManager::select_execution_path(
        uncompressed_size, effective_config.cpu_threshold);

    if (exec_path == ZstdBatchManager::ExecutionPath::CPU) {

      // CPU Path: Use libzstd

      // 1. Allocate host buffers
      std::vector<unsigned char> h_input(uncompressed_size);
      std::vector<unsigned char> h_output(*compressed_size);

      // 2. Copy input from Device to Host
      // We assume uncompressed_data is on device, but let's be safe
      cudaPointerAttributes attrs;
      cudaError_t err = cudaPointerGetAttributes(&attrs, uncompressed_data);
      // Clear any error if not a device pointer (e.g.
      // cudaErrorInvalidValue on host ptrs)
      if (err != cudaSuccess)
        cudaGetLastError();

      if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaMemcpyAsync(h_input.data(), uncompressed_data,
                                   uncompressed_size, cudaMemcpyDeviceToHost,
                                   stream));
      } else {
        // Assume host or managed
        CUDA_CHECK(cudaMemcpyAsync(h_input.data(), uncompressed_data,
                                   uncompressed_size, cudaMemcpyDefault,
                                   stream));
      }

      // CRITICAL: Must wait for Async Copy to complete before CPU accesses
      // h_input
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // 3. Compress on CPU
      size_t cSize =
          ZSTD_compress(h_output.data(), h_output.size(), h_input.data(),
                        h_input.size(), effective_config.level);

      if (ZSTD_isError(cSize)) {
        return Status::ERROR_COMPRESSION;
      }

      // 4. Copy output from Host to Device
      err = cudaPointerGetAttributes(&attrs, compressed_data);
      if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaMemcpyAsync(compressed_data, h_output.data(), cSize,
                                   cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(compressed_data, h_output.data(), cSize,
                                   cudaMemcpyDefault, stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for output copy

      *compressed_size = cSize;
      return Status::SUCCESS;
    }

    // --- 1. Ensure temp_workspace is device memory ---
    cudaPointerAttributes temp_attrs;
    cudaError_t temp_attr_err =
        cudaPointerGetAttributes(&temp_attrs, temp_workspace);
    if (temp_attr_err != cudaSuccess)
      cudaGetLastError(); // Clear potential error (e.g. invalid
                          // value for host ptr)

    // Validate input/temp pointers

    if (temp_attr_err != cudaSuccess ||
        temp_attrs.type != cudaMemoryTypeDevice) {
      // Allocate device buffer if not already device memory
      cudaError_t alloc_err = cudaMalloc(&device_workspace, temp_size);
      if (alloc_err != cudaSuccess) {
      
        return Status::ERROR_CUDA_ERROR; // Early return OK here
                                         // (nothing allocated yet)
      }
      // Optionally copy host buffer to device if needed
      cudaMemcpy(device_workspace, temp_workspace, temp_size,
                 cudaMemcpyHostToDevice);
      temp_workspace = device_workspace;
    }

    // --- 1. Partition the temp_workspace ---
    unsigned char *workspace_ptr = static_cast<unsigned char *>(temp_workspace);
    size_t alignment = 128;

    DictionaryContent *d_dict_content = nullptr;
    unsigned char *d_dict_buffer = nullptr;
    if (has_dictionary) {
      // Validate dictionary before use
      if (!dict.raw_content || dict.raw_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
      }

      // Check dictionary size bounds
      if (dict.raw_size > dictionary::MAX_DICT_SIZE) {
        return Status::ERROR_INVALID_PARAMETER;
      }

      // Check workspace has enough space for dictionary
      size_t dict_workspace_needed =
          align_to_boundary(sizeof(DictionaryContent), alignment) +
          align_to_boundary(dict.raw_size, alignment);
      if ((size_t)(workspace_ptr -
                   static_cast<unsigned char *>(temp_workspace)) +
              dict_workspace_needed >
          temp_size) {
        return Status::ERROR_BUFFER_TOO_SMALL;
      }

      d_dict_content = reinterpret_cast<DictionaryContent *>(workspace_ptr);
      workspace_ptr =
          align_ptr(workspace_ptr + sizeof(DictionaryContent), alignment);

      d_dict_buffer = workspace_ptr;
      workspace_ptr = align_ptr(workspace_ptr + dict.raw_size, alignment);

      // Check if dictionary is already on device
      cudaPointerAttributes dict_attrs;
      cudaError_t dict_attr_err =
          cudaPointerGetAttributes(&dict_attrs, dict.raw_content);

      if (dict_attr_err != cudaSuccess)
        cudaGetLastError(); // Clear error

      if (dict_attr_err == cudaSuccess &&
          dict_attrs.type == cudaMemoryTypeDevice) {
        // Dictionary already on device, use device-to-device copy
        cudaError_t memcpy_err =
            cudaMemcpyAsync(d_dict_buffer, dict.raw_content, dict.raw_size,
                            cudaMemcpyDeviceToDevice, stream);
        if (memcpy_err != cudaSuccess) {
          return Status::ERROR_CUDA_ERROR;
        }
      } else {
        // Dictionary on host, use host-to-device copy
        cudaError_t memcpy_err =
            cudaMemcpyAsync(d_dict_buffer, dict.raw_content, dict.raw_size,
                            cudaMemcpyHostToDevice, stream);
        if (memcpy_err != cudaSuccess) {
          return Status::ERROR_CUDA_ERROR;
        }
      }

      // Synchronize to ensure dictionary upload is complete before use
      cudaError_t sync_err = cudaStreamSynchronize(stream);
      if (sync_err != cudaSuccess) {
        return Status::ERROR_CUDA_ERROR;
      }

      // Set up DictionaryContent structure
      DictionaryContent h_dict_content;
      h_dict_content.d_buffer = d_dict_buffer;
      h_dict_content.size = dict.raw_size;
      h_dict_content.dict_id = dict.header.dictionary_id;

      cudaError_t content_memcpy_err = cudaMemcpyAsync(
          d_dict_content, &h_dict_content, sizeof(DictionaryContent),
          cudaMemcpyHostToDevice, stream);
      if (content_memcpy_err != cudaSuccess) {
        return Status::ERROR_CUDA_ERROR;
      }
    }

    // Check if uncompressed_data is already on device BEFORE
    // allocating workspace for it

    cudaPointerAttributes input_attrs;
    cudaError_t input_attr_err =
        cudaPointerGetAttributes(&input_attrs, uncompressed_data);

    if (input_attr_err != cudaSuccess)
      cudaGetLastError(); // Clear potential error
    bool input_is_device = (input_attr_err == cudaSuccess &&
                            input_attrs.type == cudaMemoryTypeDevice);

    
    //
    unsigned char *d_input = nullptr;
    if (input_is_device) {
      // Input is already on device, use it directly
      d_input = const_cast<unsigned char *>(
          static_cast<const unsigned char *>(uncompressed_data));

      
      if (uncompressed_size >= 129) {
        unsigned char check[130];
        cudaMemcpy(check, d_input, 129, cudaMemcpyDeviceToHost);
      }

    } else {
      // Input is on host (or managed/unrecognized), allocate space in workspace
      d_input = workspace_ptr;
      workspace_ptr = align_ptr(workspace_ptr + uncompressed_size, alignment);

      //  Copy input data to the workspace buffer
      CUDA_CHECK(cudaMemcpyAsync(d_input, uncompressed_data, uncompressed_size,
                                 cudaMemcpyDefault, stream));
    }

    unsigned char *d_output;
    size_t d_output_max_size;

    if (compressed_data != nullptr) {
      d_output = static_cast<unsigned char *>(compressed_data);
      d_output_max_size = *compressed_size;
    } else {
      d_output = workspace_ptr;
      d_output_max_size =
          estimate_compressed_size(uncompressed_size, effective_config.level);
      workspace_ptr = align_ptr(workspace_ptr + d_output_max_size, alignment);
    }

    // ---  Setup CompressionWorkspace ---
    CompressionWorkspace call_workspace;
    call_workspace.d_fse_tables =
        (void *)d_cached_fse_tables; //  Assign cached tables

    unsigned char *workspace_start = workspace_ptr;

    u32 block_size;
    if (effective_config.block_size == 0 ||
        effective_config.block_size == 128 * 1024) {
      // Use optimal block size if default (0 or 128KB)
      block_size =
          get_optimal_block_size(uncompressed_size, effective_config.level);
    } else {
      // Use user-specified size
      block_size = effective_config.block_size;
    }

    //  Clamp block size to max supported by internal buffers (128KB)
    // Matches logic in get_compress_temp_size
    block_size = std::min(block_size, CUDA_ZSTD_BLOCKSIZE_MAX);
    if (block_size == 0)
      block_size = CUDA_ZSTD_BLOCKSIZE_MAX;

    // Update effective config to reflect clamped size
    effective_config.block_size = block_size;

    call_workspace.num_blocks =
        (uncompressed_size + block_size - 1) / block_size;
    if (call_workspace.num_blocks == 0)
      call_workspace.num_blocks = 1;

    //  Clamp hash/chain logs for block-based compression to avoid OOM
    // AND scale down for small inputs to avoid massive memset overhead.
    //  Clamp hash/chain logs for block-based compression
    // Use block_size instead of uncompressed_size to determine log
    u32 input_log = 0;
    if (block_size > 0) {
      size_t temp = block_size;
      while (temp >>= 1)
        input_log++;
      input_log = std::max(10u, input_log + 2);
    } else {
      input_log = 10;
    }

    effective_config.hash_log = std::min({config.hash_log, 18u, input_log});
    effective_config.chain_log = std::min({config.chain_log, 18u, input_log});

    // Partition hash/chain tables from workspace
    call_workspace.d_hash_table = reinterpret_cast<u32 *>(workspace_ptr);
    call_workspace.hash_table_size = (1 << effective_config.hash_log);
    // (OPTIMIZATION) Reuse Hash/Chain for O(1) memory
    size_t per_block_hash_bytes = call_workspace.hash_table_size * sizeof(u32);
    //  Allocated for ALL blocks for parallel execution
    size_t total_hash_bytes = per_block_hash_bytes * call_workspace.num_blocks;

    // Initialize hash table to -1 (0xFFFFFFFF)
    CUDA_CHECK(cudaMemsetAsync(call_workspace.d_hash_table, 0xFF,
                               total_hash_bytes, stream));
    // OPTIMIZATION: Removed sync, kernels will wait in stream
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    workspace_ptr = align_ptr(workspace_ptr + total_hash_bytes, alignment);

    call_workspace.d_chain_table = reinterpret_cast<u32 *>(workspace_ptr);
    call_workspace.chain_table_size = (1 << effective_config.chain_log);
    size_t per_block_chain_bytes =
        call_workspace.chain_table_size * sizeof(u32);
    //  Allocated for ALL blocks
    size_t total_chain_bytes =
        per_block_chain_bytes * call_workspace.num_blocks;

    // Initialize chain table to -1 (0xFFFFFFFF)
    CUDA_CHECK(cudaMemsetAsync(call_workspace.d_chain_table, 0xFF,
                               total_chain_bytes, stream));
    // OPTIMIZATION: Removed sync
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    workspace_ptr = align_ptr(workspace_ptr + total_chain_bytes, alignment);

    // Partition d_matches and d_costs from workspace
    call_workspace.d_matches = reinterpret_cast<void *>(workspace_ptr);
    // (OPTIMIZATION) Reuse buffers for O(1) memory
    call_workspace.max_matches = ZSTD_BLOCKSIZE_MAX;
    size_t matches_bytes =
        call_workspace.num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(lz77::Match);
    //  Initialize matches to 0 to avoid garbage if kernels are
    // skipped

    CUDA_CHECK(
        cudaMemsetAsync(call_workspace.d_matches, 0, matches_bytes, stream));
    workspace_ptr = align_ptr(workspace_ptr + matches_bytes, alignment);

    //  Assign d_lz77_temp to reuse d_matches memory (safe after LZ77)
    call_workspace.d_lz77_temp = (u32 *)call_workspace.d_matches;

    call_workspace.d_costs = reinterpret_cast<void *>(workspace_ptr);
    call_workspace.max_costs = ZSTD_BLOCKSIZE_MAX + 1;
    size_t costs_bytes = call_workspace.num_blocks * (ZSTD_BLOCKSIZE_MAX + 1) *
                         sizeof(lz77::ParseCost);

    CUDA_CHECK(cudaMemsetAsync(call_workspace.d_costs, 0, costs_bytes, stream));
    // OPTIMIZATION: Removed sync
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    workspace_ptr = align_ptr(workspace_ptr + costs_bytes, alignment);

    // Partition reverse sequence buffers for backtracking (MOVED to Phase 1
    // setup below)
    call_workspace.max_sequences = ZSTD_BLOCKSIZE_MAX;

    // Partitioned tables (FSE/Huffman) are inside per-block workspace
    // Offsets calculated later (lines ~1860) after sizes are known

    // Partition forward sequence buffers

    ctx.seq_ctx->d_literal_lengths = reinterpret_cast<u32 *>(workspace_ptr);
    size_t forward_lit_bytes =
        call_workspace.num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32);
    workspace_ptr = align_ptr(workspace_ptr + forward_lit_bytes, alignment);

    ctx.seq_ctx->d_match_lengths = reinterpret_cast<u32 *>(workspace_ptr);
    size_t forward_match_bytes =
        call_workspace.num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32);
    workspace_ptr = align_ptr(workspace_ptr + forward_match_bytes, alignment);

    ctx.seq_ctx->d_offsets = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + forward_match_bytes,
                              alignment); // reuse forward_match_bytes size

    ctx.seq_ctx->d_literals_buffer =
        reinterpret_cast<unsigned char *>(workspace_ptr);
    // Literals buffer size = block size * num_blocks
    size_t literals_bytes = call_workspace.num_blocks * ZSTD_BLOCKSIZE_MAX;
    workspace_ptr = align_ptr(workspace_ptr + literals_bytes, alignment);

    //  Assign d_bitstream to reuse literals buffer (literals done by then)
    call_workspace.d_bitstream = (void *)ctx.seq_ctx->d_literals_buffer;

    //  Partition per-block sums (3 slots per block)
    call_workspace.d_block_sums = reinterpret_cast<u32 *>(workspace_ptr);
    size_t block_sums_bytes = call_workspace.num_blocks * 3 * sizeof(u32);
    //  Initialize block sums to 0

    CUDA_CHECK(cudaMemsetAsync(call_workspace.d_block_sums, 0, block_sums_bytes,
                               stream));
    // OPTIMIZATION: Removed sync (serialized pipeline)
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    workspace_ptr = align_ptr(workspace_ptr + block_sums_bytes, alignment);

    call_workspace.d_scanned_block_sums =
        reinterpret_cast<u32 *>(workspace_ptr);
    size_t scanned_block_sums_bytes =
        call_workspace.num_blocks * 3 * sizeof(u32);
    workspace_ptr =
        align_ptr(workspace_ptr + scanned_block_sums_bytes, alignment);

    //  Partition checksum buffer
    size_t checksum_bytes = sizeof(u64);
    workspace_ptr = align_ptr(workspace_ptr + checksum_bytes, alignment);

    // Partition Huffman temporary buffers
    call_workspace.d_frequencies = reinterpret_cast<u32 *>(workspace_ptr);
    size_t frequencies_bytes = 256 * sizeof(u32); // MAX_HUFFMAN_SYMBOLS
    workspace_ptr = align_ptr(workspace_ptr + frequencies_bytes, alignment);

    call_workspace.d_code_lengths = reinterpret_cast<u32 *>(workspace_ptr);
    size_t code_lengths_bytes =
        uncompressed_size * sizeof(u32); // Per-input-symbol code lengths
    workspace_ptr = align_ptr(workspace_ptr + code_lengths_bytes, alignment);

    call_workspace.d_bit_offsets = reinterpret_cast<u32 *>(workspace_ptr);
    size_t bit_offsets_bytes =
        uncompressed_size * sizeof(u32); // Prefix sum result
    workspace_ptr = align_ptr(workspace_ptr + bit_offsets_bytes, alignment);

    // Set max_sequences capacity
    call_workspace.max_sequences = uncompressed_size; // Conservative estimate

    // Partition sequence storage from the workspace for compression
    ctx.seq_ctx->d_sequences =
        reinterpret_cast<sequence::Sequence *>(workspace_ptr);

    //  Also assign to call_workspace so block_ws inherits it
    call_workspace.d_sequences =
        reinterpret_cast<void *>(ctx.seq_ctx->d_sequences);

    //  Allocate sequences for ALL blocks (SoA layout simulation)
    // Safely allocated for num_blocks * ZSTD_BLOCKSIZE_MAX
    size_t sequence_objs_bytes = call_workspace.num_blocks *
                                 ZSTD_BLOCKSIZE_MAX *
                                 sizeof(sequence::Sequence);
    workspace_ptr = align_ptr(workspace_ptr + sequence_objs_bytes, alignment);

    // MOVED: call_workspace.d_workspace assignment deferred until after O(n)
    // allocations to prevent overlap.
    size_t intermediate_used =
        (unsigned char *)workspace_ptr - (unsigned char *)workspace_start;

    if (intermediate_used > temp_size) {
      if (device_workspace)
        cudaFree(device_workspace);
      
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // ...

    if (call_workspace.d_hash_table == nullptr) {
      
      return Status::ERROR_INVALID_PARAMETER;
      if (device_workspace)
        cudaFree(device_workspace);
    }
    if (call_workspace.d_chain_table == nullptr) {
      
      return Status::ERROR_INVALID_PARAMETER;
      if (device_workspace)
        cudaFree(device_workspace);
    }

    // Hash/chain tables already initialized during allocation

    // NOTE: d_costs is initialized by initialize_costs_kernel in
    // find_optimal_parse

    // Synchronize to catch any pending errors before starting
    // compression
    // REMOVED for CUDA Graph compatibility (and general performance)
    /*
    cudaError_t pre_compress_err = cudaDeviceSynchronize();
    if (pre_compress_err != cudaSuccess) {
      if (device_workspace)
        cudaFree(device_workspace);
      return Status::ERROR_CUDA_ERROR;
    }
    */
    // std::cerr << "Pre-compression sync: OK" << std::endl;

    // --- 2. Start Compression Pipeline ---

    // Copy input to workspace if it's on host (already determined
    // above)
    if (!input_is_device) {
      CUDA_CHECK(cudaMemcpyAsync(d_input, uncompressed_data, uncompressed_size,
                                 cudaMemcpyHostToDevice, stream));
    }

    u32 compressed_offset = 0;

    cudaError_t kernel_err = cudaGetLastError();

    // 3. Write Frame Header
    u32 header_size = 0;

    // 3. Write Frame Header
    // Status initialized above
    status = write_frame_header(
        d_output + compressed_offset, d_output_max_size - compressed_offset,
        &header_size, (u32)uncompressed_size,
        has_dictionary ? dict.raw_content : nullptr,
        (has_dictionary ? dict.raw_size : 0), effective_config, stream);

    if (status != Status::SUCCESS) {
      if (device_workspace)
        cudaFree(device_workspace);
      return status;
    }

    compressed_offset += header_size;

    // Block size already calculated and clamped at start of function
    // u32 num_blocks = ...; // Recalculation removed, using
    // call_workspace.num_blocks
    u32 num_blocks = call_workspace.num_blocks;

    // Calculate per-block workspace size (Must match
    // get_compress_temp_size)
    // Calculate per-block workspace size (Must match get_compress_temp_size)
    // 2 x u32 arrays (decisions, offsets) + potential extra for greedy
    size_t lz77_temp_size = CUDA_ZSTD_BLOCKSIZE_MAX * 2 * sizeof(u32);

    // Adaptive Output Buffer Sizing
    // Formula: Max(MinBuffer, BlockSize * ExpansionFactor + Overhead)
    // ExpansionFactor = 4 (Standard safe expansion)
    // Overhead = 8192 (Headers, alignment, safety margin)
    const size_t min_buffer_size = 256 * 1024; // 256KB min
    size_t adaptive_size = (size_t)(block_size * 4) + 8192;
    size_t output_buffer_size = std::max(min_buffer_size, adaptive_size);

    size_t fse_table_size = 3 * sizeof(fse::FSEEncodeTable);
    size_t huff_size = sizeof(huffman::HuffmanTable);

    size_t per_block_size = 0;
    per_block_size += align_to_boundary(lz77_temp_size, GPU_MEMORY_ALIGNMENT);
    per_block_size +=
        align_to_boundary(output_buffer_size,
                          GPU_MEMORY_ALIGNMENT); // Add separate buffer
    //  per_block_size should ONLY include resources that are
    // partitioned INSIDE d_workspace Hash, Chain, and Sequences are
    // allocated GLOBALLY (SoA) and reside BEFORE d_workspace so they do
    // not contribute to the stride. per_block_size +=
    // align_to_boundary(hash_table_size, GPU_MEMORY_ALIGNMENT);
    // per_block_size += align_to_boundary(chain_table_size,
    // GPU_MEMORY_ALIGNMENT); per_block_size
    // += align_to_boundary(seq_storage_size, GPU_MEMORY_ALIGNMENT);
    per_block_size += align_to_boundary(fse_table_size, GPU_MEMORY_ALIGNMENT);
    per_block_size += align_to_boundary(huff_size, GPU_MEMORY_ALIGNMENT);

    size_t total_needed = (size_t)num_blocks * per_block_size;

    // Check moved to after global allocations and total_size assignment

    // Vectors to store context for batch processing
    std::vector<sequence::SequenceContext> block_seq_ctxs(num_blocks);
    std::vector<unsigned char *> block_literals_ptrs(num_blocks);
    std::vector<u32> block_literals_sizes(num_blocks);
    std::vector<unsigned char *> block_outputs(num_blocks);
    std::vector<u32> block_num_sequences(num_blocks);
    std::vector<u32> block_compressed_sizes(num_blocks); // Restored

    // (TWO-PHASE) Per-block state for Phase 2 processing
    std::vector<const unsigned char *> block_inputs(num_blocks);
    std::vector<u32> current_block_sizes(num_blocks);
    std::vector<u32> h_has_dummy(num_blocks, 0);

    // =================================================================================
    //  Allocate O(n) LZ77 buffers to prevent race conditions in
    // parallel execution Shared O(1) buffers (d_hash_table, etc.) cause
    // atomic contention when 2000 blocks run in parallel.
    // =================================================================================
    // =================================================================================
    //  Use Pre-Allocated Workspace for LZ77 buffers
    // =================================================================================

    // Reuse pointers partitioned at start of function
    u32 *d_all_hash_tables = call_workspace.d_hash_table;
    u32 *d_all_chain_tables = call_workspace.d_chain_table;
    lz77::Match *d_all_matches = (lz77::Match *)call_workspace.d_matches;

    // Memsets were already done during partitioning (lines 1588, 1602, 1619)
    // No action needed here.

    // (OPTIMIZATION) Device array for async literal size reduction
    // This allows batch sync instead of per-block thrust::reduce sync
    // (OPTIMIZATION) Use workspace for small metadata arrays
    u32 *d_block_literals_sizes = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + num_blocks * sizeof(u32), alignment);

    cudaMemsetAsync(d_block_literals_sizes, 0, num_blocks * sizeof(u32),
                    stream);

    u32 *d_block_num_sequences = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + num_blocks * sizeof(u32), alignment);

    u32 *d_block_has_dummy = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + num_blocks * sizeof(u32), alignment);

    cudaMemsetAsync(d_block_num_sequences, 0, num_blocks * sizeof(u32), stream);
    cudaMemsetAsync(d_block_has_dummy, 0, num_blocks * sizeof(u32), stream);

    // (O(n) BATCH OPTIMIZATION) Per-block LZ77 result buffers
    size_t on_buffer_elements = (size_t)num_blocks * block_size;
    size_t on_buffer_bytes = on_buffer_elements * sizeof(u32);

    u32 *d_all_lit_lengths_reverse = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + on_buffer_bytes, alignment);

    u32 *d_all_match_lengths_reverse = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + on_buffer_bytes, alignment);

    u32 *d_all_offsets_reverse = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + on_buffer_bytes, alignment);

    //  Allocate literals buffer
    unsigned char *d_all_literals_buffer =
        reinterpret_cast<unsigned char *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + on_buffer_elements,
                              alignment); // Elements are bytes

    // Ensure we didn't overrun (simple check)
    size_t final_used =
        (unsigned char *)workspace_ptr - (unsigned char *)workspace_start;

    if (final_used > temp_size) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    CUDA_CHECK(
        cudaMemsetAsync(d_all_lit_lengths_reverse, 0, on_buffer_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_all_match_lengths_reverse, 0, on_buffer_bytes,
                               stream));
    //  Assign O(n) buffers to call_workspace so they are inherited by
    // blocks
    call_workspace.d_literal_lengths_reverse = d_all_lit_lengths_reverse;
    call_workspace.d_match_lengths_reverse = d_all_match_lengths_reverse;
    call_workspace.d_offsets_reverse = d_all_offsets_reverse;

    //  Set base pointer for block-partitioned workspace AFTER global O(n)
    // allocations
    call_workspace.d_workspace = workspace_ptr;
    call_workspace.total_size = temp_size - ((unsigned char *)workspace_ptr -
                                             (unsigned char *)temp_workspace);

    if (total_needed > call_workspace.total_size) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Futures for CPU post-processing (parallel backtracking)
    std::vector<std::future<Status>> futures;
    futures.reserve(num_blocks);

    // Create a single event to synchronize Memset completions with
    // async threads - REMOVED (Unused and causing crashes)

    // === PHASE 1: Parallel LZ77 & Sequence Generation ===

    // Timing instrumentation
    auto phase1_start = std::chrono::high_resolution_clock::now();

    // (STREAMING) Track offset for immediate block writes
    // Initialize with existing compressed_offset (Frame Header size)
    size_t streaming_compressed_offset = compressed_offset;
    size_t max_seq_bytes = ZSTD_BLOCKSIZE_MAX * sizeof(u32);
    u32 *d_lit_len_reuse = d_all_lit_lengths_reverse;
    u32 *d_match_len_reuse = d_all_match_lengths_reverse;
    u32 *d_off_reuse = d_all_offsets_reverse;
    //  Use locally allocated buffer
    unsigned char *d_lit_buf_reuse = d_all_literals_buffer;

    
    // event usage removed - causing invalid argument errors

    cudaError_t post_sync_err = cudaGetLastError();
    if (post_sync_err != cudaSuccess) {
      status = Status::ERROR_CUDA_ERROR;
      goto cleanup;
    }

    // (OPTIMIZATION) Reuse reusable temp buffers for Phase 1 (O(1)
    // memory) These are already allocated in the workspace (lines
    // 1370+) and assigned to ctx.seq_ctx We just point to them to avoid
    // malloc/free overhead and leaks.
    // Declarations moved to top of phase 1 for goto safety

    // CUDA_CHECK(cudaMalloc((void **)&d_lit_len_reuse, max_seq_bytes));
    // // REMOVED LEAK CUDA_CHECK(cudaMalloc((void
    // **)&d_match_len_reuse, max_seq_bytes)); // REMOVED LEAK
    // CUDA_CHECK(cudaMalloc((void
    // **)&d_off_reuse, max_seq_bytes)); // REMOVED LEAK
    // CUDA_CHECK(cudaMalloc((void **)&d_lit_buf_reuse,
    // ZSTD_BLOCKSIZE_MAX)); // REMOVED LEAK

    for (u32 block_idx = 0; block_idx < num_blocks; block_idx++) {
      cudaError_t loop_start_err = cudaGetLastError();
      if (loop_start_err != cudaSuccess) {
        status = Status::ERROR_CUDA_ERROR;
        goto cleanup;
      }

      u32 block_start = block_idx * block_size;
      u32 current_block_size_val =
          std::min(block_size, (u32)uncompressed_size - block_start);

      const unsigned char *block_input = d_input + block_start;

      // (TWO-PHASE) Store per-block state for Phase 2 processing
      block_inputs[block_idx] = block_input;
      current_block_sizes[block_idx] = current_block_size_val;

      // Setup per-block workspace
      //  Reuse workspace base for all blocks (O(1))
      size_t on_buffer_bytes = num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32);
      on_buffer_bytes = align_to_boundary(on_buffer_bytes, 256);
      size_t used_offset = 3 * on_buffer_bytes;
      unsigned char *ws_base = (unsigned char *)call_workspace.d_workspace +
                               used_offset + (block_idx * per_block_size);
      size_t ws_offset = 0;

      CompressionWorkspace block_ws;

      // 1. LZ77 Temp
      block_ws.d_lz77_temp = (u32 *)(ws_base + ws_offset);
      //  Initialize LZ77 decisions to 0 (Literals) to prevent
      // stale/garbage matches if the match finder doesn't write every
      // position compactly.
      cudaMemsetAsync(block_ws.d_lz77_temp, 0, lz77_temp_size, stream);
      ws_offset += align_to_boundary(lz77_temp_size, GPU_MEMORY_ALIGNMENT);

      // 2. Output Buffer (Part of per-block workspace)
      // Hash/Chain/Seq are GLOBAL (SoA) and do not consume workspace
      // offsets
      block_outputs[block_idx] = (unsigned char *)(ws_base + ws_offset);
      // Ensure we verify output buffer size logic matches allocation
      // (max * 2)
      ws_offset += align_to_boundary(output_buffer_size, GPU_MEMORY_ALIGNMENT);

      // Initialize d_matches and d_costs from call_workspace partitions
      // These will be used to set up sequence context (lines 1587-1604)
      // and later assigned to thread_block_ws in the parallel lambda
      // Use block_start (actual byte position) as offset, not block_idx
      // * block_size because the last block may be smaller!
      // 3. LZ77 Buffers (O(n) Partitioned)
      // Use locally allocated O(n) buffers instead of shared call_workspace
      u32 hash_stride = (1 << effective_config.hash_log);
      u32 chain_stride =
          (1 << effective_config.chain_log); // Assuming chain_log dictates size
      u32 matches_stride = block_size;

      block_ws.d_hash_table = d_all_hash_tables + (block_idx * hash_stride);
      block_ws.d_chain_table = d_all_chain_tables + (block_idx * chain_stride);

      // 5. Sequences (Global SoA) - Partitioned per block
      // Fix: Must offset sequence array to avoid race conditions!
      // block_ws.d_sequences = (sequence::Sequence *)call_workspace.d_sequences
      // + (block_idx * ZSTD_BLOCKSIZE_MAX); TRY: Force Array Output by setting
      // sequences to nullptr
      block_ws.d_sequences = nullptr;

      // Matches Buffer (O(n) Partitioned)
      block_ws.d_matches = d_all_matches + (block_idx * matches_stride);

      //  Assign O(n) Sequence Result Buffers (Previously missing!)
      // These are used by the pipeline to store found sequences.
      // Stride is block_size (capacity).
      block_ws.d_literal_lengths_reverse =
          d_all_lit_lengths_reverse + (block_idx * block_size);
      block_ws.d_match_lengths_reverse =
          d_all_match_lengths_reverse + (block_idx * block_size);
      block_ws.d_offsets_reverse =
          d_all_offsets_reverse + (block_idx * block_size);
      block_ws.max_sequences =
          block_size; // Capacity equals block size (worst case)

      // Costs Buffer - Not used by greedy pipeline?
      // greedy pipeline uses d_matches input -> sequences output.
      // If we used optimal parser, we'd need O(n) costs too.
      // Assuming greedy config for now (based on
      // lz77_parallel_greedy_pipeline call).
      block_ws.d_costs =
          (lz77::ParseCost *)
              call_workspace.d_costs; // Shared (unused in greedy)

      // 5. FSE Tables
      block_ws.d_fse_tables = (fse::FSEEncodeTable *)(ws_base + ws_offset);
      // CLEAR FSE TABLE
      cudaMemsetAsync(block_ws.d_fse_tables, 0, fse_table_size, stream);
      ws_offset += align_to_boundary(fse_table_size, GPU_MEMORY_ALIGNMENT);

      // 6. Huffman Table
      block_ws.d_huffman_table = (huffman::HuffmanTable *)(ws_base + ws_offset);
      // CLEAR HUFFMAN TABLE
      cudaMemsetAsync(block_ws.d_huffman_table, 0, huff_size, stream);
      ws_offset += align_to_boundary(huff_size, GPU_MEMORY_ALIGNMENT);

      // + (block_idx * call_workspace.hash_table_size);

      //  Reset Hash Table for reuse (O(1)) - PARTITIONED
      hash_stride = (1 << effective_config.hash_log);
      block_ws.d_hash_table =
          call_workspace.d_hash_table + (block_idx * hash_stride);
      size_t hash_reset_bytes = hash_stride * sizeof(u32);
      CUDA_CHECK(cudaMemsetAsync(block_ws.d_hash_table, 0xFF, hash_reset_bytes,
                                 stream));

      if (call_workspace.d_chain_table) {
        u32 chain_stride = (1 << effective_config.chain_log);
        block_ws.d_chain_table =
            call_workspace.d_chain_table + (block_idx * chain_stride);
        size_t chain_reset_bytes = chain_stride * sizeof(u32);
        CUDA_CHECK(cudaMemsetAsync(block_ws.d_chain_table, 0xFF,
                                   chain_reset_bytes, stream));
      }

      // Global buffers (shared/partitioned logically)
      // CRITICAL: Each block needs its own portion for parallel
      // processing Use block_idx * block_size as ELEMENT offset (not
      // byte offset!)

      cudaError_t ws_setup_err = cudaGetLastError();
      if (ws_setup_err != cudaSuccess) {
        // printf("[ERROR] compress: Error during Phase 1 Loop Setup
        // (Block %u):
        // "
        //        "%s\n",
        //        block_idx, cudaGetErrorString(ws_setup_err));
      }

      // Block sums (3 slots per block)
      block_ws.d_block_sums = call_workspace.d_block_sums + (block_idx * 3);
      block_ws.d_scanned_block_sums =
          call_workspace.d_scanned_block_sums + (block_idx * 3);

      // (O(n) BATCH) Use per-block partition of reverse buffers
      // Each block writes to its own slice: offset = block_idx * block_size
      block_ws.d_literal_lengths_reverse =
          d_all_lit_lengths_reverse + (block_idx * block_size);
      block_ws.d_match_lengths_reverse =
          d_all_match_lengths_reverse + (block_idx * block_size);
      block_ws.d_offsets_reverse =
          d_all_offsets_reverse + (block_idx * block_size);
      block_ws.max_sequences = current_block_size_val;

      // usage of seq_array_bytes removed
      // Construct per-block SequenceContext
      sequence::SequenceContext local_seq_ctx;

      // (OPTIMIZATION) Reuse pre-allocated buffers WITH OFFSET
      // Fully isolate per-block buffers to allow parallel/pipeline execution
      local_seq_ctx.d_literal_lengths =
          d_lit_len_reuse + (block_idx * ZSTD_BLOCKSIZE_MAX);
      local_seq_ctx.d_match_lengths =
          d_match_len_reuse + (block_idx * ZSTD_BLOCKSIZE_MAX);
      local_seq_ctx.d_offsets = d_off_reuse + (block_idx * ZSTD_BLOCKSIZE_MAX);
      local_seq_ctx.d_literals_buffer =
          d_lit_buf_reuse + (block_idx * ZSTD_BLOCKSIZE_MAX);

      // Initialize to zero for safety (reuse requires clear)
      // (OPTIMIZATION) Use Async Memset and scale size to block limits
      size_t clear_size = current_block_size_val * sizeof(u32);
      // Add margin for SIMD writes
      clear_size = std::min(clear_size + 64, max_seq_bytes);

      CUDA_CHECK(cudaMemsetAsync(local_seq_ctx.d_literal_lengths, 0, clear_size,
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(local_seq_ctx.d_match_lengths, 0, clear_size,
                                 stream));
      CUDA_CHECK(
          cudaMemsetAsync(local_seq_ctx.d_offsets, 0, clear_size, stream));

      // Clear literals buffer (only needed if writing partial bytes?
      // actually we overwrite it) Skipping literals buffer clear as we
      // write only valid count.
      // CUDA_CHECK(cudaMemsetAsync(local_seq_ctx.d_literals_buffer, 0,
      // current_block_size_val, stream));

      //  Assign per-block sequences from Global Buffer WITH OFFSET
      local_seq_ctx.d_sequences = reinterpret_cast<sequence::Sequence *>(
          reinterpret_cast<unsigned char *>(call_workspace.d_sequences) +
          block_idx * ZSTD_BLOCKSIZE_MAX * sizeof(sequence::Sequence));

      // Ensure strict ordering: Wait for previous block's usage to
      // finish? Since we use the same stream, operations are serialized
      // automatically. write_block (async) -> Memset (async next loop).
      // Correct.
      local_seq_ctx.d_num_sequences =
          block_ws.d_block_sums; // Reuse block sums slot 0
      local_seq_ctx.num_sequences = 0;
      local_seq_ctx.num_literals = 0;

      block_seq_ctxs[block_idx] = local_seq_ctx;

      // (REMOVED) Per-block event creation to avoid resource leak
      // cudaEvent_t start_event; ...

      // Launch async task for this block
      //  Execute synchronously to prevent race condition on O(1)
      // buffers Since we reuse the SAME d_matches/d_hash_table for all
      // blocks, we CANNOT run blocks in parallel (unless we have N sets
      // of buffers). For O(1) memory, strict serialization is required.
      {
        // Use the main stream (or a dedicated single stream)
        cudaStream_t block_stream = stream;
        // No need to create/destroy stream every block for sync
        // execution

        // Wait for input data copy (on main stream) to complete
        // (Implicit if we use the same stream)

        // Initialize thread_block_ws with block_ws which already has
        // correct O(n) pointers
        CompressionWorkspace thread_block_ws = block_ws;

        //  Set Table Sizes correctly (hash_log is exponential, size is
        // linear)
        thread_block_ws.hash_table_size = (1 << effective_config.hash_log);
        thread_block_ws.chain_table_size = (1 << effective_config.chain_log);

        // Previous assignments of d_hash_table/d_chain_table/d_matches to
        // call_workspace removed. We MUST use the O(n) pointers already in
        // block_ws.

        //  Assign per-block reverse sequence buffers
        // Correct O(n) buffers already assigned in block_ws above (lines
        // 1996-2001) thread_block_ws was initialized with block_ws, so it
        // has correct pointers. DO NOT overwrite with call_workspace
        // pointers here!

        // REMOVED DUPLICATE ASSIGNMENTS - block_seq_ctxs already set
        // correctly in Phase 1! These duplicate assignments were using
        // WRONG pointer arithmetic: ctx.seq_ctx->d_literal_lengths +
        // block_offset_idx * sizeof(u32) was adding BYTE offset to
        // u32*, causing the 256KB bug!

        // block_seq_ctxs[block_idx] is already correctly initialized
        // earlier (line ~1498) with proper recycled buffer pointers.
        // Don't overwrite it!

        // Run LZ77 (Async)
        // Construct LZ77Config from manager config
        cuda_zstd::lz77::LZ77Config lz77_config;
        lz77_config.window_log =
            config.window_log; // Window log isn't scaled down
        lz77_config.hash_log = effective_config.hash_log;   // USE EFFECTIVE
        lz77_config.chain_log = effective_config.chain_log; // USE EFFECTIVE
        // (OPTIMIZATION) Force deeper search for better ratios
        lz77_config.min_match = config.min_match;
        // (REVERT) Use standard nice_length to prevent overflow/edge
        // cases
        // (OPTIMIZATION) Use configuration-driven parameters
        lz77_config.nice_length = effective_config.target_length > 0
                                      ? effective_config.target_length
                                      : 256;
        lz77_config.good_length = lz77_config.nice_length;
        lz77_config.search_depth = (1 << effective_config.search_log);
        // Run V2 Pipeline

        /*
         * Initialization handled by init_hash_table_kernel inside
         * find_matches_parallel
         */

        // LZ77 kernel will be launched below
        // Pass 1: Find Matches
        StreamingContext *s_ctx = static_cast<StreamingContext *>(streaming_context);
        u32 *d_hash_p = s_ctx ? s_ctx->d_hash_table_state : nullptr;
        u32 *d_chain_p = s_ctx ? s_ctx->d_chain_table_state : nullptr;

        status = static_cast<Status>(cuda_zstd::lz77::find_matches_parallel(
            block_input, current_block_size_val,
            reinterpret_cast<cuda_zstd::CompressionWorkspace *>(
                &thread_block_ws),
            lz77_config, block_stream, d_hash_p, d_chain_p));

        // Integrated Long Distance Matching (LDM)
        if (s_ctx && s_ctx->ldm_initialized) {
            ldm::ldm_process_block(s_ctx->ldm_ctx, block_input, current_block_size_val, 
                                 (lz77::Match*)thread_block_ws.d_matches, 
                                 (u32)s_ctx->total_bytes_processed, block_stream);
            s_ctx->total_bytes_processed += current_block_size_val;
        }

        // GRANULAR ERROR CHECK: Sync stream first to catch async errors
        // (OPTIMIZATION) Removed per-block sync to allow pipelining
        // cudaStreamSynchronize(block_stream);
        cudaError_t find_err = cudaGetLastError();
        if (find_err != cudaSuccess) {
          status = Status::ERROR_CUDA_ERROR;
          goto cleanup;
        }

        if (status != Status::SUCCESS) {
          // cudaStreamDestroy(block_stream); // WRONG: Do not destroy user
          // stream
          goto cleanup;
        }

        // Pass 2+3: ASYNC PARALLEL GREEDY LZ77 PIPELINE
        // Uses GPU parallel greedy parsing - writes results to device
        // memory This eliminates per-block D2H sync inside the pipeline

        status = lz77::lz77_parallel_greedy_pipeline_async(
            current_block_size_val, thread_block_ws, lz77_config,
            &d_block_num_sequences[block_idx], &d_block_has_dummy[block_idx],
            block_stream);

        cudaError_t pipeline_err = cudaGetLastError();
        if (pipeline_err != cudaSuccess) {
          status = Status::ERROR_CUDA_ERROR;
          goto cleanup;
        }

        if (status != Status::SUCCESS) {
          // cudaStreamDestroy(block_stream); // WRONG
          goto cleanup;
        }

        
        u32 ph1_num_seq = 0;
        cudaMemcpyAsync(&ph1_num_seq, &d_block_num_sequences[block_idx],
                        sizeof(u32), cudaMemcpyDeviceToHost, block_stream);
        cudaStreamSynchronize(block_stream);

        // (TWO-PHASE OPTIMIZATION) Phase 1 complete for this block
        // NO SYNC HERE - all blocks run LZ77 in parallel
        // Phase 2 processing deferred to batch loop after all blocks
        // complete

      } // End of Phase 1 for this block
    } // End of Phase 1 Loop

    // =========================================================================
    // BATCH SYNC POINT - Wait for all LZ77 operations to complete
    // =========================================================================
    //  Use cudaDeviceSynchronize to ensure ALL GPU operations finish
    // cudaStreamSynchronize may not sync operations on other
    // streams/contexts
    cudaDeviceSynchronize();

    // Batch copy all seq counts from device to host
    cudaMemcpy(block_num_sequences.data(), d_block_num_sequences,
               num_blocks * sizeof(u32), cudaMemcpyDeviceToHost);

    // Copy has_dummy flags
    for (u32 i = 0; i < num_blocks; i++) {
      u32 tmp;
      cudaMemcpy(&tmp, &d_block_has_dummy[i], sizeof(u32),
                 cudaMemcpyDeviceToHost);
      h_has_dummy[i] = tmp;
    }

    // =========================================================================
    // PHASE 2: Process all blocks - sequence copies, Thrust reduce,
    // encoding
    // =========================================================================
    for (u32 block_idx = 0; block_idx < num_blocks; block_idx++) {
      // Re-compute per-block state from stored values
      const unsigned char *block_input = block_inputs[block_idx];
      u32 current_block_size_val = current_block_sizes[block_idx];
      bool has_dummy = h_has_dummy[block_idx];
      // fflush(stdout); // stderr is unbuffered

      // Re-setup workspace pointers for this block
      size_t on_buffer_bytes = num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32);
      on_buffer_bytes = align_to_boundary(on_buffer_bytes, 256);
      size_t used_offset = 3 * on_buffer_bytes;
      unsigned char *ws_base = (unsigned char *)call_workspace.d_workspace +
                               used_offset + (block_idx * per_block_size);

      CompressionWorkspace block_ws;
      block_ws.d_lz77_temp = (u32 *)ws_base;
      block_ws.d_literal_lengths_reverse =
          d_all_lit_lengths_reverse + (block_idx * block_size);
      block_ws.d_match_lengths_reverse =
          d_all_match_lengths_reverse + (block_idx * block_size);
      block_ws.d_offsets_reverse =
          d_all_offsets_reverse + (block_idx * block_size);

      //  Use copy_count for reduction to include trailing literals
      u32 num_seq = block_num_sequences[block_idx];
      u32 copy_count = num_seq + (has_dummy ? 1 : 0);

      if (current_block_size_val > 0) {
      }
      if (copy_count > 0) {
        // (PHASE 2a) Async Reduction using custom kernel (outputs to device
        // memory)
        // Ensure we sum ALL literal runs (sequences + trailing)
        launch_async_sum_reduce(block_ws.d_literal_lengths_reverse, copy_count,
                                &d_block_literals_sizes[block_idx], stream);

      } else {
        //  If no sequences, the entire block is literals
        // (PHASE 2a) Update device array required for batch copy later
        // Use kernel to set value asynchronously (avoids Stack->Device sync)
        k_record_size<<<1, 1, 0, stream>>>(&d_block_literals_sizes[block_idx],
                                           current_block_size_val);

        // Copy input directly to literals buffer
        CUDA_CHECK(cudaMemcpyAsync(block_seq_ctxs[block_idx].d_literals_buffer,
                                   block_input, current_block_size_val,
                                   cudaMemcpyDeviceToDevice, stream));
        if (current_block_size_val <= 10) {
        }
      }

      // Calculate output ptr for Phase 2b (used to init writer later)
      size_t aligned_lz77 =
          align_to_boundary(lz77_temp_size, GPU_MEMORY_ALIGNMENT);
      block_outputs[block_idx] = ws_base + aligned_lz77;

    } // End of PHASE 2a Loop

    // =========================================================================
    // PHASE 2b PREPARATION - Batch Sync & Copy Sizes
    // =========================================================================
    cudaStreamSynchronize(stream);

    //  Batch Copy of literals sizes (One API call instead of N)
    cudaMemcpyAsync(block_literals_sizes.data(), d_block_literals_sizes,
                    num_blocks * sizeof(u32), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Wait for sizes to be on host

    // =========================================================================
    // PHASE 2b: ENCODING Loop
    // =========================================================================
    for (u32 block_idx = 0; block_idx < num_blocks; block_idx++) {
      // Re-read state
      u32 current_block_size_val = current_block_sizes[block_idx];
      const unsigned char *block_input = block_inputs[block_idx];

      BlockBufferWriter writer(block_outputs[block_idx], output_buffer_size);

      bool has_dummy = h_has_dummy[block_idx];
      u32 num_seq = block_num_sequences[block_idx];
      u32 copy_count = num_seq + (has_dummy ? 1 : 0);

      if (copy_count > 0) {
        // Restore sequences from O(n) buffers to shared O(1) buffers
        // (OPTIMIZATION) ZERO-COPY: Point context directly to O(n) buffers
        // instead of copying back to O(1) buffers.
        block_seq_ctxs[block_idx].d_literal_lengths =
            d_all_lit_lengths_reverse + (block_idx * block_size);
        block_seq_ctxs[block_idx].d_match_lengths =
            d_all_match_lengths_reverse + (block_idx * block_size);
        block_seq_ctxs[block_idx].d_offsets =
            d_all_offsets_reverse + (block_idx * block_size);

        // Fix: Use the populated "reverse" buffers (which actually contain
        // forward sequences from greedy) instead of the empty block_seq_ctx
        // buffers.
        u32 *d_lit_len_ptr =
            d_all_lit_lengths_reverse + (block_idx * block_size);
        u32 *d_match_len_ptr =
            d_all_match_lengths_reverse + (block_idx * block_size);

        // Extract literals using valid sequence data (including dummy)
        u32 *d_extracted_size;
        CUDA_CHECK(cudaMallocAsync(&d_extracted_size, sizeof(u32), stream));

        launch_copy_literals(block_input, current_block_size_val, d_lit_len_ptr,
                             d_match_len_ptr, num_seq,
                             block_seq_ctxs[block_idx].d_literals_buffer,
                             stream, block_idx, d_extracted_size);

        if (num_seq == 0) {
          // No sequences (literals only)
          CUDA_CHECK(cudaMemcpyAsync(block_seq_ctxs[block_idx].d_literals_buffer,
                                     block_input, current_block_size_val,
                                     cudaMemcpyDeviceToDevice, stream));
          block_literals_sizes[block_idx] = current_block_size_val;
        } else {
           u32 h_extracted_size = 0;
           CUDA_CHECK(cudaMemcpyAsync(&h_extracted_size, d_extracted_size, sizeof(u32),
                                      cudaMemcpyDeviceToHost, stream));
           CUDA_CHECK(cudaStreamSynchronize(stream));
           block_literals_sizes[block_idx] = h_extracted_size;
        }
        CUDA_CHECK(cudaFreeAsync(d_extracted_size, stream));
      }

      // Compress Literals
      block_seq_ctxs[block_idx].num_literals = block_literals_sizes[block_idx];

      
      if (block_literals_sizes[block_idx] > 131072) { // 128KB
        status = Status::ERROR_CORRUPT_DATA;
        goto cleanup;
      }

      // Status initialized to SUCCESS

      status =
          compress_literals(block_seq_ctxs[block_idx].d_literals_buffer,
                            block_literals_sizes[block_idx], writer, stream);

      // (OPTIMIZATION) Removed per-block sync
      // cudaStreamSynchronize(stream);
      cudaError_t lit_err = cudaDeviceSynchronize();
      if (lit_err != cudaSuccess) {
        status = Status::ERROR_CUDA_ERROR;
        goto cleanup;
      }

      if (status != Status::SUCCESS) {
        goto cleanup;
      }

      // Build Sequences (Async)
      u32 num_sequences = block_num_sequences[block_idx];
      if (num_sequences > 0) {
        const u32 seq_threads = 512;
        const u32 seq_blocks = (num_sequences + seq_threads - 1) / seq_threads;
        status =
            sequence::build_sequences(block_seq_ctxs[block_idx], num_sequences,
                                      seq_blocks, seq_threads, stream);

        cudaError_t seq_err = cudaDeviceSynchronize();
        if (seq_err != cudaSuccess) {
          status = Status::ERROR_CUDA_ERROR;
          goto cleanup;
        }

        if (status != Status::SUCCESS)
          goto cleanup;
      }

      // Compress Sequences
      //  writer.align4(); // Ensure
      
      // Re-construct needed workspace part for sequences
      CompressionWorkspace seq_ws;
      // Use the same base as phase 2a (it's safe to reuse now)
      size_t on_buffer_bytes = num_blocks * ZSTD_BLOCKSIZE_MAX * sizeof(u32);
      on_buffer_bytes = align_to_boundary(on_buffer_bytes, 256);
      size_t used_offset = 3 * on_buffer_bytes;
      unsigned char *ws_base = (unsigned char *)call_workspace.d_workspace +
                               used_offset + (block_idx * per_block_size);
      
      // Partition ws_base for d_matches and d_lz77_temp
      // Max possible sequences per block (128KB) is around 43K.
      // d_matches needs 6 bytes per seq. d_lz77_temp needs 16 bytes per seq.
      // We assume worst case num_sequences = ZSTD_BLOCKSIZE_MAX.
      size_t matches_size = ZSTD_BLOCKSIZE_MAX * 6; // 6 bytes per seq
      matches_size = align_to_boundary(matches_size, 256);
      
      seq_ws.d_matches = (void*)ws_base;
      seq_ws.d_lz77_temp = (u32 *)(ws_base + matches_size);
      // NOTE: We do NOT use call_workspace.d_matches because it aliases d_match_lengths (input)
      // and would be overwritten by output codes!

      status = compress_sequences(&block_seq_ctxs[block_idx],
                                  block_num_sequences[block_idx], writer,
                                  stream, &seq_ws);

      cudaError_t comp_seq_err = cudaDeviceSynchronize();
      if (comp_seq_err != cudaSuccess) {
        status = Status::ERROR_CUDA_ERROR;
        goto cleanup;
      }

      if (status != Status::SUCCESS) {
        goto cleanup;
      }


      block_compressed_sizes[block_idx] = writer.get_offset();

      // (STREAMING OPTIMIZATION) Write block immediately instead of
      // batching This enables O(1) memory and requires no separate
      // block storage
      bool is_last_block = (block_idx == num_blocks - 1);

      status = write_block(d_output,          // output (Global buffer)
                           d_output_max_size, // max_size (Global buffer size)
                           block_outputs[block_idx],          // compressed_data
                           d_input + block_idx * block_size,  // original_data
                           block_compressed_sizes[block_idx], // compressed_size
                           current_block_size_val,            // original_size
                           is_last_block,                     // is_last
                           &streaming_compressed_offset, // Update offset inline
                           stream                        // stream
      );

      if (status != Status::SUCCESS) {
        goto cleanup;
      }

      /* continue; */ // End of block logic

      // CRITICAL  Synchronize after writing block/before next
      // block's LZ77 effectively overwrites the SHARED O(1) buffers
      // (d_matches, etc.) Block 0 Phase 2 (Here) reads d_matches. Block
      // 1 Phase 1 (Next Loop) writes d_matches. Must ensure Block 0 is
      // done reading before Block 1 writes.
      // cudaStreamSynchronize(stream); // REMOVED: Stream serialization
      // handles this.

    } // End of Merged Loop

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
      u64 *d_checksum_result =
          (u64 *)((unsigned char *)call_workspace.d_workspace +
                  call_workspace.total_size - sizeof(u64));
      xxhash::compute_xxhash64(d_input, uncompressed_size, 0, d_checksum_result,
                               stream);

      // Copy to output
      CUDA_CHECK(cudaMemcpyAsync(d_output + streaming_compressed_offset,
                                 d_checksum_result, 4, cudaMemcpyDeviceToDevice,
                                 stream));
      streaming_compressed_offset += 4;

      cudaError_t chk_sync_err = cudaSuccess;
      // cudaStreamSynchronize(stream); // REMOVED for Graph
      if (chk_sync_err != cudaSuccess) {
        status = Status::ERROR_CUDA_ERROR;
        goto cleanup;
      }
    }

    *compressed_size = streaming_compressed_offset;

    // Final synchronization to ensure all async operations complete
    // before the caller accesses the output buffer
    // Final synchronization to ensure all async operations complete
    // before the caller accesses the output buffer
    stats.bytes_produced += *compressed_size;
    stats.blocks_processed += num_blocks;

  cleanup:
    // Final synchronization to ensure all async operations complete
    cudaStreamSynchronize(stream);

    //  Clear borrowed pointers from ctx.seq_ctx so cleanup_context doesn't
    // try to free them
    if (ctx.seq_ctx) {
      ctx.seq_ctx->d_literals_buffer = nullptr;
      ctx.seq_ctx->d_literal_lengths = nullptr;
      ctx.seq_ctx->d_match_lengths = nullptr;
      ctx.seq_ctx->d_offsets = nullptr;
      ctx.seq_ctx->d_num_sequences = nullptr;
      ctx.seq_ctx->d_sequences = nullptr;
    }

    // Arrays are now part of workspace, no free needed

    if (device_workspace)
      cudaFree(device_workspace);

    return status;
  }

  // //
  // ==========================================================================
  // Helper: Read Block Header
  // ==========================================================================
  static Status read_block_header(const unsigned char *d_input,
                                  size_t input_size, void *d_scratch,
                                  cudaStream_t stream, bool *is_last,
                                  bool *is_compressed, bool *is_rle,
                                  u32 *block_size, u32 *header_size) {
    if (input_size < 3)
      return Status::ERROR_BUFFER_TOO_SMALL;

    unsigned char h_buf[4];
    cudaError_t err =
        cudaMemcpyAsync(h_buf, d_input, 3, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
      return Status::ERROR_CUDA_ERROR;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
      return Status::ERROR_CUDA_ERROR;

    u32 val = h_buf[0] | (h_buf[1] << 8) | (h_buf[2] << 16);

    *is_last = (val & 1);
    u32 type = (val >> 1) & 3;
    // RFC 8878 Block Size decoding: Extract 21 bits from positions 3-23
    // val bits 0-23: [Last:0][Type:1-2][Size:3-7][Size:8-15][Size:16-20]
    // Byte 0 bits 3-7 = Size bits 0-4
    // Byte 1 bits 0-7 = Size bits 5-12
    // Byte 2 bits 0-7 = Size bits 13-20
    u32 size =
        ((val >> 3) & 0x1F) |         // 5 bits from byte 0
        (((val >> 8) & 0xFF) << 5) |  // 8 bits from byte 1 (shifted by 5)
        (((val >> 16) & 0xFF) << 13); // 8 bits from byte 2 (shifted by 13)

    *block_size = size;
    *header_size = 3;

    if (type == 0) { // Raw
      *is_compressed = false;
      *is_rle = false;
    } else if (type == 1) { // RLE
      *is_compressed = false;
      *is_rle = true;
    } else if (type == 2) { // Compressed
      *is_compressed = true;
      *is_rle = false;
    } else {
      return Status::ERROR_CORRUPT_DATA;
    }
    return Status::SUCCESS;
  }

  // ==========================================================================
  // decompress() implementation - RFC 8878 COMPLIANT
  // ==========================================================================

  virtual Status decompress(const void *compressed_data, size_t compressed_size,
                            void *uncompressed_data, size_t *uncompressed_size,
                            void *temp_workspace, size_t temp_size,
                            cudaStream_t stream = 0) override {
    std::lock_guard<std::mutex> lock(api_mutex);
    // compressed_size=%zu, ptr=%p\n", compressed_size, compressed_data);

    // === Parameter Validation ===
    if (!compressed_data || !uncompressed_data || !uncompressed_size ||
        !temp_workspace || compressed_size < 4) {
      //             parameters\n");
      return Status::ERROR_INVALID_PARAMETER;
    }

    size_t required_size = get_decompress_temp_size(compressed_size);
    // required=%zu\n", temp_size, required_size);
    if (temp_size < required_size) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // === Initialize Context if Needed ===
    if (!ctx_initialized) {
      // initialize_context()\n");
      auto status = initialize_context();
      if (status != Status::SUCCESS) {
        //                 initialize_context failed with status %d\n",
        //                 (int)status);

        return status;
      }
      // initialize_context success\n");
      ctx_initialized = true;
    }

    //  Assign temp_workspace to ctx.d_temp_buffer
    ctx.d_temp_buffer = (unsigned char *)temp_workspace;

    // === Handle Skippable Frames (RFC 8878) ===
    // Zstd may have skippable frames at the beginning
    const unsigned char *h_compressed_data_ptr =
        static_cast<const unsigned char *>(compressed_data);
    size_t h_compressed_size_remaining = compressed_size;
    u32 data_offset = 0;

    // frame check loop\n");

    // Skip all skippable frames to find the real Zstd frame
    while (h_compressed_size_remaining >= 8) {
      u32 magic;
      //  compressed_data is a DEVICE pointer, must use cudaMemcpy
      CUDA_CHECK(cudaMemcpy(&magic, h_compressed_data_ptr + data_offset,
                            sizeof(u32), cudaMemcpyDeviceToHost));

      // at offset %u: 0x%X\n", data_offset, magic);

      // Check if this is the Zstd magic number
      if (magic == ZSTD_MAGIC_NUMBER) {
        // ZSTD magic number\n");
        break;
      }

      // Check if this is a skippable frame
      if ((magic & 0xFFFFFFF0) == ZSTD_MAGIC_SKIPPABLE_START) {
        // Read the frame size (next 4 bytes)
        u32 frame_size;
        CUDA_CHECK(cudaMemcpy(&frame_size,
                              h_compressed_data_ptr + data_offset + 4,
                              sizeof(u32), cudaMemcpyDeviceToHost));

        // Total frame size = 8 byte header + frame_size
        u32 total_frame_size = 8 + frame_size;

        // Move past this skippable frame
        data_offset += total_frame_size;
        h_compressed_size_remaining -= total_frame_size;
      } else {
        // Invalid magic number
        return Status::ERROR_INVALID_MAGIC;
      }
    }

    if (h_compressed_size_remaining < 4) {
      return Status::ERROR_CORRUPT_DATA;
    }
    //
    // === Partition the temp_workspace ===
    unsigned char *workspace_ptr = static_cast<unsigned char *>(temp_workspace);
    size_t alignment = 128;

    // Allocate device input buffer
    //  Use input directly. d_input must point to the START of the Frame.
    const unsigned char *d_input = h_compressed_data_ptr;
    //
    // PROBE REMOVED (Fixed crash on small inputs)

    // ...

    // Allocate checksum verification buffer
    u64 *d_checksum = reinterpret_cast<u64 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + sizeof(u64), alignment);

    // ...

    // Allocate scratch for header reading (aligned u32)
    u32 *d_header_scratch = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + sizeof(u32), alignment);

    // Allocate persistent buffers from workspace
    ctx.d_temp_buffer = workspace_ptr;
    workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX, alignment);

    ctx.seq_ctx->d_sequences =
        reinterpret_cast<sequence::Sequence *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX *
                                                  sizeof(sequence::Sequence),
                              alignment);

    //  Partition raw component arrays (Must match get_decompress_temp_size
    // order)
    ctx.seq_ctx->d_literal_lengths = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);

    ctx.seq_ctx->d_match_lengths = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);

    ctx.seq_ctx->d_offsets = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr =
        align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX * sizeof(u32), alignment);

    // Initialize sequence arrays to 0 to prevent garbage data from previous
    // workspace uses
    const size_t seq_array_size = ZSTD_BLOCKSIZE_MAX * sizeof(u32);
    CUDA_CHECK(cudaMemsetAsync(ctx.seq_ctx->d_literal_lengths, 0, seq_array_size,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.seq_ctx->d_match_lengths, 0, seq_array_size,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.seq_ctx->d_offsets, 0, seq_array_size,
                               stream));

    // NEW: Persistent rep-offsets for multi-block support
    u32 *d_rep_codes = reinterpret_cast<u32 *>(workspace_ptr);
    workspace_ptr = align_ptr(workspace_ptr + 3 * sizeof(u32), alignment);

    // Initialize rep-offsets to Zstd defaults
    static const u32 h_rep_codes[3] = {1, 4, 8};
    CUDA_CHECK(cudaMemcpyAsync(d_rep_codes, h_rep_codes, 3 * sizeof(u32),
                               cudaMemcpyHostToDevice, stream));

    // Allocate literals buffer ( Was missing!)
    ctx.seq_ctx->d_literals_buffer = workspace_ptr;
    workspace_ptr = align_ptr(workspace_ptr + ZSTD_BLOCKSIZE_MAX, alignment);

    unsigned char *d_output = static_cast<unsigned char *>(uncompressed_data);
    size_t d_output_max_size = *uncompressed_size;

    // === Parse Frame Header (RFC 8878) ===
    u32 header_size_val = 0;
    u32 frame_content_size = 0;
    bool is_single_segment = false;
    bool has_checksum = false;

    auto status = parse_frame_header(d_input + data_offset, h_compressed_size_remaining,
                                     &header_size_val, &frame_content_size,
                                     &is_single_segment, &has_checksum);
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[DECOMPRESS-DBG] parse_frame_header: status=%d, header_size=%u, content_size=%u, single_seg=%d, checksum=%d, data_offset=%u, remaining=%zu\n",
            (int)status, header_size_val, frame_content_size, (int)is_single_segment, (int)has_checksum, data_offset, h_compressed_size_remaining);
#endif
    if (status != Status::SUCCESS) {
      return status;
    }

    if (frame_content_size > 0 && frame_content_size > d_output_max_size) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Update read_offset to point past frame header
    u32 read_offset = data_offset + header_size_val;
    u32 write_offset = 0;

    // Debug output removed for production

    // Unified Block Loop (Handles both Single Segment and Multi-Block frames)
    while (read_offset < h_compressed_size_remaining) {
      // Check if only checksum remains (8 bytes)
      if (has_checksum &&
          read_offset + 4 ==
              h_compressed_size_remaining) { // For single segment, checksum is
                                             // 4 bytes
        break;
      }
      if (config.checksum ==
          ChecksumPolicy::COMPUTE_AND_VERIFY) { // For multi-block, checksum is
                                                // 4 bytes (ZSTD Frame Checksum)
        if (read_offset + 4 == h_compressed_size_remaining)
          break;
      }

      // Debug output removed for production

      // Debug output removed for production

      bool blk_is_last = false;
      u32 blk_size = 0;
      bool blk_is_compressed = false;
      u32 blk_header_size = 0;
      bool blk_is_rle = false;

      Status status = Status::SUCCESS;
    if (is_single_segment) {
        // Single Segment: Still has a block header!
        // The single_segment flag only means no window descriptor in FRAME
        // header. Block structure is the same.  Always read the block
        // header properly.
        status = read_block_header(
            d_input + read_offset, h_compressed_size_remaining - read_offset,
            d_header_scratch, stream, &blk_is_last, &blk_is_compressed,
            &blk_is_rle, &blk_size, &blk_header_size);
      } else {
        status = read_block_header(
            d_input + read_offset, h_compressed_size_remaining - read_offset,
            d_header_scratch, stream, &blk_is_last, &blk_is_compressed,
            &blk_is_rle, &blk_size, &blk_header_size);
      }

      if (status != Status::SUCCESS) {
#ifdef CUDA_ZSTD_DEBUG
        fprintf(stderr, "[DECOMPRESS-DBG] read_block_header FAILED: status=%d at read_offset=%u\n", (int)status, read_offset);
#endif
        return status;
      }

#ifdef CUDA_ZSTD_DEBUG
      // Print Raw Header for debugging
      {
        unsigned char bh_raw[3];
        cudaMemcpy(bh_raw, d_input + read_offset, 3, cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DECOMPRESS-DBG] Block: raw=[%02X %02X %02X], is_last=%d, compressed=%d, rle=%d, size=%u, hdr_size=%u\n",
                bh_raw[0], bh_raw[1], bh_raw[2], (int)blk_is_last, (int)blk_is_compressed, (int)blk_is_rle, blk_size, blk_header_size);
      }
#endif

      read_offset += blk_header_size;
      // Debug output removed for production

      // === Process Block ===
      if (blk_is_compressed) {
        u32 decompressed_size = 0;
#ifdef CUDA_ZSTD_DEBUG
        fprintf(stderr, "[DECOMPRESS-DBG] Decompressing block: input_offset=%u, blk_size=%u, write_offset=%u\n", read_offset, blk_size, write_offset);
#endif
        status =
            decompress_block(d_input + read_offset, blk_size,
                             d_output + write_offset, &decompressed_size,
                             d_output, d_output_max_size, stream, d_rep_codes);
        if (status != Status::SUCCESS) {
#ifdef CUDA_ZSTD_DEBUG
          fprintf(stderr, "[DECOMPRESS-DBG] decompress_block FAILED: status=%d\n", (int)status);
#endif
          return status;
        }
#ifdef CUDA_ZSTD_DEBUG
        fprintf(stderr, "[DECOMPRESS-DBG] Block decompressed: %u bytes\n", decompressed_size);
#endif

        write_offset += decompressed_size;
        read_offset += blk_size;
      } else if (blk_is_rle) {
        // Validate output bounds for RLE expansion
        if (write_offset + blk_size > d_output_max_size) {
          return Status::ERROR_BUFFER_TOO_SMALL;
        }
        // Handle empty RLE block (blk_size = 0)
        if (blk_size > 0) {
          unsigned char rle_byte;
          CUDA_CHECK(cudaMemcpy(&rle_byte, d_input + read_offset, 1,
                                cudaMemcpyDeviceToHost));
          const u32 threads = 256;
          const u32 blocks = (blk_size + threads - 1) / threads;
          // Ensure at least 1 block for kernel launch
          if (blocks > 0) {
            expand_rle_kernel<<<blocks, threads, 0, stream>>>(
                d_output + write_offset, blk_size, rle_byte);
          }
        }
        write_offset += blk_size;
        read_offset += 1;
      } else { // RAW
        // Validate input bounds before reading
        if (read_offset + blk_size > h_compressed_size_remaining) {
          return Status::ERROR_CORRUPT_DATA;
        }
        // Validate output bounds before writing
        if (write_offset + blk_size > d_output_max_size) {
          return Status::ERROR_BUFFER_TOO_SMALL;
        }
        // Handle empty Raw block (blk_size = 0) - skip memcpy
        if (blk_size > 0) {
          CUDA_CHECK(cudaMemcpyAsync(d_output + write_offset,
                                     d_input + read_offset, blk_size,
                                     cudaMemcpyDeviceToDevice, stream));
        }
        write_offset += blk_size;
        read_offset += blk_size;
      }

      // Sync after block
      cudaError_t blk_sync_err = cudaStreamSynchronize(stream);
      if (blk_sync_err != cudaSuccess)
        return Status::ERROR_CUDA_ERROR;

      if (blk_is_last)
        break;
    }

    // === Verify Checksum (if present) ===
    if (config.checksum == ChecksumPolicy::COMPUTE_AND_VERIFY) {
      // Check if there's a checksum at the end (4 bytes)
      if (read_offset + 4 <= h_compressed_size_remaining) {
        u32 stored_checksum;

        // Copy checksum from device to host
        CUDA_CHECK(cudaMemcpyAsync(&stored_checksum, d_input + read_offset,
                                   sizeof(u32), cudaMemcpyDeviceToHost,
                                   stream));

        // Compute checksum of decompressed data
        xxhash::compute_xxhash64(d_output, write_offset, 0, d_checksum, stream);

        u64 computed_checksum;
        CUDA_CHECK(cudaMemcpyAsync(&computed_checksum, d_checksum, sizeof(u64),
                                   cudaMemcpyDeviceToHost, stream));

        // Wait for all GPU operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Debug output removed for production

        // Compare checksums (Lower 32 bits)
        if (stored_checksum != (u32)computed_checksum) {
          return Status::ERROR_CHECKSUM_FAILED;
        }
      } else {
        // Debug output removed for production
      }
    }

    // === Set output size ===
    *uncompressed_size = write_offset;

    // === Update statistics ===
    stats.bytes_decompressed += write_offset;

    //  Clear borrowed pointers from ctx to prevent
    // double-free in cleanup_context
    ctx.d_temp_buffer = nullptr;
    if (ctx.seq_ctx) {
      ctx.seq_ctx->d_literals_buffer = nullptr;
      ctx.seq_ctx->d_literal_lengths = nullptr;
      ctx.seq_ctx->d_match_lengths = nullptr;
      ctx.seq_ctx->d_offsets = nullptr;
      ctx.seq_ctx->d_num_sequences = nullptr;
      ctx.seq_ctx->d_sequences = nullptr;
    }

    return Status::SUCCESS;
  }

  // ==========================================================================
  // Dictionary support
  // ==========================================================================
  virtual Status
  set_dictionary(const dictionary::Dictionary &new_dict) override {
    // Ensure context is initialized before setting dictionary
    if (!ctx_initialized) {
      Status init_status = initialize_context();
      if (init_status != Status::SUCCESS) {
        return init_status;
      }
    }

    // Validate dictionary parameters
    if (!new_dict.raw_content) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Validate dictionary size
    if (new_dict.raw_size == 0) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Check dictionary size bounds
    if (new_dict.raw_size < dictionary::MIN_DICT_SIZE ||
        new_dict.raw_size > dictionary::MAX_DICT_SIZE) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // CRITICAL  Manually copy fields to avoid double-free issue
    // The Dictionary copy assignment does deep copy with malloc,
    // but we don't want to own the memory - just reference it
    dict.header = new_dict.header;
    dict.raw_size = new_dict.raw_size;

    // Shallow copy the pointer - we DON'T own this memory
    // The caller (test) owns it and will free it
    dict.raw_content = new_dict.raw_content;

    has_dictionary = true;

    // Load the FSE entropy tables from the dictionary
    Status load_status = load_dictionary_tables(new_dict);
    if (load_status != Status::SUCCESS) {
      // Log warning but don't fail - some dictionaries may not have tables
      // and we can still use the raw content for matches
      // Reset has_dictionary if critical error
      if (load_status == Status::ERROR_OUT_OF_MEMORY) {
        has_dictionary = false;
        dict.raw_content = nullptr;
        dict.raw_size = 0;
        return load_status;
      }
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Load FSE entropy tables from a trained dictionary
   *
   * This function parses the dictionary's raw content to extract pre-computed
   * FSE tables for literals, match lengths, and offsets. These tables are used
   * during compression to achieve better compression ratios.
   *
   * Dictionary format (per RFC 8878):
   * - Bytes 0-3: Magic number (0xEC30A437)
   * - Bytes 4-7: Dictionary ID
   * - Following: Entropy tables (Huffman for literals, FSE for ML/Offsets)
   * - Remainder: Content portion for string matching
   *
   * @param dict The dictionary to load tables from
   * @return Status::SUCCESS if tables loaded, error code otherwise
   */
  Status load_dictionary_tables(const dictionary::Dictionary &dict) {
    // Validate input parameters
    if (!dict.raw_content) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Check minimum dictionary size
    if (dict.raw_size < dictionary::MIN_DICT_SIZE) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Check maximum dictionary size to prevent overflow
    if (dict.raw_size > dictionary::MAX_DICT_SIZE) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    const unsigned char *ptr = dict.raw_content;
    size_t remaining = dict.raw_size;

    // Verify magic number (first 4 bytes)
    if (remaining < 8) {
      // Dictionary too small to contain header
      // Not an error - may be raw content only
      return Status::SUCCESS;
    }

    // Bounds-check before reading magic number
    if (!ptr) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    u32 magic = 0;
    // Safe copy instead of reinterpret_cast to avoid alignment issues
    memcpy(&magic, ptr, sizeof(u32));

    if (magic != dictionary::DICT_MAGIC_NUMBER) {
      // Not a trained dictionary - may be raw content only
      // This is not an error, just means no entropy tables available
      return Status::SUCCESS;
    }

    // Bounds check before advancing
    if (remaining < 4) {
      return Status::ERROR_CORRUPT_DATA;
    }
    ptr += 4;
    remaining -= 4;

    // Read dictionary ID (next 4 bytes)
    if (remaining < 4) {
      return Status::ERROR_CORRUPT_DATA;
    }
    // u32 dict_id = 0;
    // memcpy(&dict_id, ptr, sizeof(u32));
    ptr += 4;
    remaining -= 4;

    // The entropy tables section follows
    // Format: Huffman literals table, then FSE tables for ML and Offsets

    // Parse entropy tables header sizes from dict.header if available
    if (dict.header.entropy_tables_size > 0) {
      // Validate entropy tables size against remaining buffer
      if (dict.header.entropy_tables_size > remaining) {
        return Status::ERROR_CORRUPT_DATA;
      }

      // Dictionary has valid entropy tables section
      // The tables are parsed lazily during compression/decompression
      // when they are actually needed, using offsets from the header
    }

    return Status::SUCCESS;
  }

  virtual Status
  get_dictionary(dictionary::Dictionary &dict_out) const override {
    if (!has_dictionary)
      return Status::ERROR_INVALID_PARAMETER;
    dict_out = dict;
    return Status::SUCCESS;
  }

  virtual Status clear_dictionary() override {
    has_dictionary = false;
    return Status::SUCCESS;
  }

  // ==========================================================================
  // Statistics
  // ==========================================================================
  virtual const CompressionStats &get_stats() const override { return stats; }

  virtual void reset_stats() override {
    memset(&stats, 0, sizeof(CompressionStats));
  }

private:
  // ==========================================================================
  // Context management
  // ==========================================================================
  Status initialize_context() {
    // Locked by api_mutex (recursive_mutex or mutex) from caller
    // (configure/compress) Removed static mutex to allow concurrent
    // initialization of separate managers.
    //         // std::cerr << "initialize_context() entered (guarded)" <<
    //         std::endl;

    // Check for pre-existing errors to avoid confusing debugging
    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess) {
    }

    // Initialize LZ77 context
    if (!ctx.lz77_ctx) {
      ctx.lz77_ctx = new lz77::LZ77Context();
      // Default LZ77 config (Moved to constructor)
      // config.window_log = 22;
      // config.chain_log = 17;
      // config.hash_log = 18;
      // config.min_match = 3; // Already set in ctor
      lz77::LZ77Config lz77_config;
      lz77_config.hash_log = config.hash_log;
      lz77_config.chain_log = config.chain_log; // Use config.chain_log
      lz77_config.search_depth = 8;             // Standard search depth
      lz77_config.min_match = config.min_match;
      lz77::init_lz77_context(*ctx.lz77_ctx, lz77_config, ZSTD_BLOCKSIZE_MAX);
    }

    // Initialize sequence context
    if (!ctx.seq_ctx) {
      ctx.seq_ctx = new sequence::SequenceContext();

      // Granular checks
      CUDA_CHECK(
          cudaMalloc(&ctx.seq_ctx->d_literals_buffer, ZSTD_BLOCKSIZE_MAX));
      {
        cudaError_t e = cudaGetLastError();
      }

      CUDA_CHECK(cudaMalloc(&ctx.seq_ctx->d_literal_lengths,
                            ZSTD_BLOCKSIZE_MAX * sizeof(u32)));
      {
        cudaError_t e = cudaGetLastError();
      }

      CUDA_CHECK(cudaMalloc(&ctx.seq_ctx->d_match_lengths,
                            ZSTD_BLOCKSIZE_MAX * sizeof(u32)));
      {
        cudaError_t e = cudaGetLastError();
      }

      CUDA_CHECK(cudaMalloc(&ctx.seq_ctx->d_offsets,
                            ZSTD_BLOCKSIZE_MAX * sizeof(u32)));
      {
        cudaError_t e = cudaGetLastError();
      }

      CUDA_CHECK(cudaMalloc(&ctx.seq_ctx->d_num_sequences, sizeof(u32)));
      {
        cudaError_t e = cudaGetLastError();
      }

      CUDA_CHECK(cudaMalloc(&ctx.seq_ctx->d_sequences,
                            ZSTD_BLOCKSIZE_MAX * sizeof(sequence::Sequence)));
      {
        cudaError_t e = cudaGetLastError();
      }
    }

    // Initialize FSE tables
    if (!ctx.lit_fse_table) {
      ctx.lit_fse_table = new fse::FSEEncodeTable();
    }
    if (!ctx.ml_fse_table) {
      ctx.ml_fse_table = new fse::FSEEncodeTable();
    }
    if (!ctx.of_fse_table) {
      ctx.of_fse_table = new fse::FSEEncodeTable();
    }

    // Initialize Huffman context
    if (!ctx.huff_ctx) {
      ctx.huff_ctx = new huffman::HuffmanTable();
      CUDA_CHECK(
          cudaMalloc(&ctx.huff_ctx->codes, 256 * sizeof(huffman::HuffmanCode)));
      {
        cudaError_t e = cudaGetLastError();
      }
      //             // std::cerr << "initialize_context: initialized
      //             huff_ctx"
      //             << std::endl;
    }

    ctx.total_matches = 0;
    ctx.total_literals = 0;
    ctx.total_sequences = 0;

    ctx_initialized = true;

    // Final check
    cudaError_t post_err = cudaGetLastError();
    if (post_err != cudaSuccess) {
      return Status::ERROR_CUDA_ERROR;
    }

    //         // std::cerr << "initialize_context: complete" << std::endl;
    return Status::SUCCESS;
  }

  // ==========================================================================
  // Frame operations
  // ==========================================================================
  Status write_frame_header(unsigned char *output, size_t max_size,
                            u32 *header_size, u32 content_size,
                            const void *dict_buffer, size_t dict_size,
                            const CompressionConfig &config,
                            cudaStream_t stream) {
    if (max_size < FRAME_HEADER_SIZE_MIN) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Use a safe fixed size buffer to avoid potential stack overflow if
    // FRAME_HEADER_SIZE_MAX is small
    unsigned char h_header[128];
    u32 offset = 0;

    // 1. Write magic number
    u32 magic = ZSTD_MAGIC_NUMBER;
    memcpy(h_header + offset, &magic, 4);
    offset += 4;

    // 2. Frame Header Descriptor
    unsigned char fhd = 0;

    // Dictionary ID
    u32 dict_id = 0;
    if (dict_buffer && dict_size > 0) {
      fhd |= 0x01;     // 1-byte dict ID for now (Bits 0-1 = 01)
      dict_id = xxhash::xxhash_32_cpu(
          static_cast<const unsigned char *>(dict_buffer), dict_size, 0);
    }

    // Set checksum bit if enabled
    if (config.checksum != ChecksumPolicy::NO_COMPUTE_NO_VERIFY) {
      fhd |= 0x04; // Content checksum bit
    }

    // Single Segment mode (Bit 5 = 0x20)
    // For small files < block_size, we can use Single Segment mode.
    // This reduces header size by 1 byte (omits Window Descriptor).
    bool single_segment = (content_size <= config.block_size);
    if (single_segment) {
      fhd |= 0x20;
    }

    // Determine bits 7-6 (FCS_Flag)
    // Table 3: SS=1 -> 0:1B, 1:2B, 2:4B, 3:8B
    // Table 2: SS=0 -> 0:0B, 1:2B, 2:4B, 3:8B
    u32 fcs_flag = 0;
    u32 fcs_field_size = 0;

    if (single_segment) {
      if (content_size < 256) {
        fcs_flag = 0;
        fcs_field_size = 1;
      } else if (content_size < 65536 + 256) {
        fcs_flag = 1;
        fcs_field_size = 2;
      } else {
        fcs_flag = 2;
        fcs_field_size = 4;
      }
    } else {
      if (content_size == 0) {
        fcs_flag = 0;
        fcs_field_size = 0;
      } else if (content_size < 65536) {
        fcs_flag = 1;
        fcs_field_size = 2;
      } else {
        fcs_flag = 2;
        fcs_field_size = 4;
      }
    }
    fhd |= (fcs_flag << 6);

    h_header[offset++] = fhd;

    // 2. Window Descriptor (ONLY if !Single Segment)
    if (!single_segment) {
      unsigned char wd = (config.window_log - 10) << 3;
      h_header[offset++] = wd;
    }

    // 3. Dictionary ID
    if (dict_buffer && dict_size > 0) {
      h_header[offset++] = (unsigned char)dict_id;
    }

    // 4. Content Size
    if (fcs_field_size == 1) {
      h_header[offset++] = (unsigned char)content_size;
    } else if (fcs_field_size == 2) {
      u32 val = single_segment ? (content_size - 256) : content_size;
      h_header[offset++] = (unsigned char)(val & 0xFF);
      h_header[offset++] = (unsigned char)((val >> 8) & 0xFF);
    } else if (fcs_field_size == 4) {
      memcpy(h_header + offset, &content_size, 4);
      offset += 4;
    }

    // Copy to device
    // bytes to device ptr %p\n", offset, output);
    // Copy to device (synchronous since h_header is on stack)
    CUDA_CHECK(cudaMemcpy(output, h_header, offset, cudaMemcpyHostToDevice));

    *header_size = offset;
    return Status::SUCCESS;
  }

  Status parse_frame_header(
      const unsigned char *input, // Device pointer to compressed data
      u32 input_size,
      u32 *header_size,        // Output: total header size (host)
      u32 *content_size,       // Output: decompressed size if present (host)
      bool *is_single_segment, // Output: single segment flag
      bool *has_checksum       // Output: checksum flag
  ) {
    // input_size=%u\n", input_size);
    if (input_size < 5) {
      //             fprintf(stderr, "[ERROR] parse_frame_header: input too
      //             small (%u < 5)\n", input_size);
      return Status::ERROR_CORRUPT_DATA;
    }

    // Copy frame header to host for parsing
    // Standard Zstd Header is max 18 bytes:
    // Magic(4) + FHD(1) + WD(1) + DictID(4) + FCS(8)
    unsigned char h_header[18];
    CUDA_CHECK(cudaMemcpy(h_header, input, std::min(18u, input_size),
                          cudaMemcpyDeviceToHost));
    
    //        h_header[1], h_header[2], h_header[3], h_header[4]);

    // RFC 8878 Literals Section Header:
    // copied, first bytes: %02X %02X %02X %02X %02X\n",
    // //                 h_header[0], h_header[1], h_header[2], h_header[3],
    // h_header[4]);

    u32 offset = 4; // Skip magic number (already validated)

    // === Parse Frame Header Descriptor (1 byte) ===
    unsigned char fhd = h_header[offset++];
    // offset now=%u\n", fhd, offset);

    bool single_segment = (fhd >> 5) & 0x01;
    bool has_dict_id = (fhd & 0x03) != 0;
    // Note: (fhd & 0xC0) indicates content size field presence in header
    bool checksum_flag = (fhd >> 2) & 0x01;

    if (is_single_segment)
      *is_single_segment = single_segment;

    if (has_checksum)
      *has_checksum = checksum_flag;

    // === Parse Window Descriptor (if not single segment) ===
    if (!single_segment) {
      if (offset >= input_size) {
        return Status::ERROR_CORRUPT_DATA;
      }

      unsigned char wd = h_header[offset++];
      u32 window_log = 10 + (wd >> 3);

      if (window_log >= CUDA_ZSTD_WINDOWLOG_MIN &&
          window_log <= CUDA_ZSTD_WINDOWLOG_MAX) {
        config.window_log = window_log;
      } else {
        return Status::ERROR_CORRUPT_DATA;
      }
    }

    // === Parse Dictionary ID (if present) ===
    u32 dict_id_size = 0;
    u32 dictionary_id = 0;

    if (has_dict_id) {
      u32 did_flag = (fhd & 0x03);
      if (did_flag == 1) {
        dict_id_size = 1;
        dictionary_id = h_header[offset];
      } else if (did_flag == 2) {
        dict_id_size = 2;
        dictionary_id =
            (u32)h_header[offset] | ((u32)h_header[offset + 1] << 8);
      } else if (did_flag == 3) {
        dict_id_size = 4;
        memcpy(&dictionary_id, h_header + offset, 4);
      }
      offset += dict_id_size;
    }

    // === Parse Content Size (if present) ===
    u32 h_content_size = 0;
    u32 csf = (fhd >> 6) & 0x03;

    if (single_segment) {
      if (csf == 0) {
        h_content_size = h_header[offset];
        offset += 1;
      } else if (csf == 1) {
        u16 size_val;
        memcpy(&size_val, h_header + offset, 2);
        h_content_size = (u32)size_val + 256;
        offset += 2;
      } else if (csf == 2) {
        memcpy(&h_content_size, h_header + offset, 4);
        offset += 4;
      } else if (csf == 3) {
        u64 size_val;
        memcpy(&size_val, h_header + offset, 8);
        h_content_size = (u32)size_val;
        offset += 8;
      }
    } else {
      if (csf == 1) {
        u16 size_val;
        memcpy(&size_val, h_header + offset, 2);
        h_content_size = size_val;
        offset += 2;
      } else if (csf == 2) {
        memcpy(&h_content_size, h_header + offset, 4);
        offset += 4;
      } else if (csf == 3) {
        u64 size_val;
        memcpy(&size_val, h_header + offset, 8);
        h_content_size = (u32)size_val;
        offset += 8;
      }
    }

    *header_size = offset;
    *content_size = h_content_size;

    return Status::SUCCESS;
  }

  Status
  write_block(unsigned char *output, size_t max_size,
              const unsigned char *compressed_data, // Changed from block_data
              const unsigned char *original_data,   // NEW: Need original data
                                                    // for raw block fallback
              u32 compressed_size, u32 original_size, bool is_last,
              size_t *compressed_offset, cudaStream_t stream) {
    bool use_compressed = (compressed_size < original_size);
    if (compressed_size <= 8 && original_size > 100) {
      // Debug logic removed
    }

    // Determine block type and size
    u32 block_size = use_compressed ? compressed_size : original_size;
    u32 block_type = use_compressed ? 2 : 0; // 2 = Compressed, 0 = Raw

    // RFC 8878 Block Header Encoding (3 bytes, little-endian):
    // Bits 0:    Last_Block (1 bit)
    // Bits 1-2:  Block_Type (2 bits: 0=Raw, 1=RLE, 2=Compressed)
    // Bits 3-23: Block_Size (21 bits)
    //
    // Byte layout (little-endian):
    // Byte 0: [Last_Block:0][Block_Type:1-2][Block_Size:3-7] (5 bits)
    // Byte 1: [Block_Size:8-15] (8 bits)
    // Byte 2: [Block_Size:16-23] (8 bits)
    unsigned char header[3];
    u32 last_bit = is_last ? 1 : 0;
    u32 type_bits = (block_type & 0x03) << 1; // Type occupies bits 1-2

    // Extract 5 lowest bits of size and place at bits 3-7
    header[0] =
        (unsigned char)(((block_size & 0x1F) << 3) | type_bits | last_bit);
    // Next 8 bits of size (bits 5-12)
    header[1] = (unsigned char)((block_size >> 5) & 0xFF);
    // Final 8 bits of size (bits 13-20)
    header[2] = (unsigned char)((block_size >> 13) & 0xFF);

    // Write header
    CUDA_CHECK(cudaMemcpy(output + *compressed_offset, header, 3,
                          cudaMemcpyHostToDevice));
    *compressed_offset += 3;

    // Write block content
    if (use_compressed) {
      if (compressed_data && compressed_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(output + *compressed_offset, compressed_data,
                                   compressed_size, cudaMemcpyDeviceToDevice,
                                   stream));
      }
    } else {
      if (original_data && original_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(output + *compressed_offset, original_data,
                                   original_size, cudaMemcpyDeviceToDevice,
                                   stream));
      }
    }
    *compressed_offset += block_size;

    return Status::SUCCESS;
  }

  // ==========================================================================
  // Decompression Helpers
  // ==========================================================================

  Status decompress_block(const unsigned char *input, u32 input_size,
                          unsigned char *output,
                          u32 *output_size, // Host pointer for output
                          const unsigned char *output_base, u32 output_max_size,
                          cudaStream_t stream, u32 *d_rep_codes) {
    if (!input || !output || !output_size) {
      return Status::ERROR_INVALID_PARAMETER;
    }
    //        input_size, output_max_size);
    // fflush(stdout);

    // Clear any previous error state to avoid false positives in validation
    (void)cudaGetLastError();

    // Use temp buffer for literals
    unsigned char *d_decompressed_literals = ctx.d_temp_buffer;

    
    u32 literals_header_size = 0;
    u32 literals_compressed_size = 0;
    u32 literals_decompressed_size = 0;

    auto status = decompress_literals(
        input, input_size, d_decompressed_literals, &literals_header_size,
        &literals_compressed_size, &literals_decompressed_size, stream);
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[BLOCK-DBG] decompress_literals: status=%d, hdr_size=%u, comp_size=%u, decomp_size=%u\n",
            (int)status, literals_header_size, literals_compressed_size, literals_decompressed_size);
#endif
    if (status != Status::SUCCESS) {
      return status;
    }

    // Sync after literals
    cudaError_t lit_sync_err = cudaStreamSynchronize(stream);
    if (lit_sync_err != cudaSuccess) {
      //             printf("[ERROR] decompress_literals failed: %s\n",
      //             cudaGetErrorString(lit_sync_err));
      //       return Status::ERROR_CUDA_ERROR;
    }

    //        literals_header_size, literals_compressed_size,
    //        literals_decompressed_size);

    

    // === Decompress Sequences ===
    // {
    //   cudaError_t e = cudaGetLastError();
    //   if (e)
    //         // }
    
    u32 sequences_offset = literals_header_size + literals_compressed_size;
    
    if (sequences_offset == input_size) {
      *output_size = literals_decompressed_size;
      return Status::SUCCESS;
    }
    if (sequences_offset > input_size) {
#ifdef CUDA_ZSTD_DEBUG
      fprintf(stderr, "[BLOCK-DBG] sequences_offset (%u) > input_size (%u) -> CORRUPT_DATA\n", sequences_offset, input_size);
#endif
      return Status::ERROR_CORRUPT_DATA;
    }

#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[BLOCK-DBG] Decoding sequences at offset=%u, remaining=%u\n", sequences_offset, input_size - sequences_offset);
#endif
    status = decompress_sequences(input + sequences_offset,
                                  input_size - sequences_offset, ctx.seq_ctx,
                                  literals_decompressed_size, stream);
#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[BLOCK-DBG] decompress_sequences: status=%d, num_sequences=%u\n", (int)status, ctx.seq_ctx->num_sequences);
#endif

    if (status != Status::SUCCESS) {
      return status;
    }

    // Sync after sequences
    cudaError_t seq_sync_err = cudaStreamSynchronize(stream);
    if (seq_sync_err != cudaSuccess) {
      return Status::ERROR_CUDA_ERROR;
    }

    // === Build Sequence Structs ===
    // decompress_sequences populates the
    // component arrays (d_literal_lengths,
    // d_offsets, d_match_lengths) We need to
    // build the Sequence structs from these
    // arrays before calling execute_sequences
    if (ctx.seq_ctx->num_sequences > 0) {
      const u32 threads = 256;
      const u32 blocks = (ctx.seq_ctx->num_sequences + threads - 1) / threads;

      status = sequence::build_sequences(
          *ctx.seq_ctx, ctx.seq_ctx->num_sequences, blocks, threads, stream);
      if (status != Status::SUCCESS) {
        return status;
      }

      cudaError_t build_sync_err = cudaStreamSynchronize(stream);
      if (build_sync_err != cudaSuccess) {
        return Status::ERROR_CUDA_ERROR;
      }
    }

    // === Execute Sequences ===
    u32 *d_output_size;
    CUDA_CHECK(cudaMalloc(&d_output_size, sizeof(u32)));
    CUDA_CHECK(cudaMemsetAsync(d_output_size, 0, sizeof(u32), stream));

    status = sequence::execute_sequences(
        d_decompressed_literals, literals_decompressed_size,
        ctx.seq_ctx->d_sequences, ctx.seq_ctx->num_sequences, output,
        d_output_size, ctx.seq_ctx->is_raw_offsets, // Pass correct tier flag
        stream, output_base, output_max_size, d_rep_codes);

    // Sync after execute
    cudaError_t exec_sync_err = cudaStreamSynchronize(stream);
    if (exec_sync_err != cudaSuccess) {
      return Status::ERROR_CUDA_ERROR;
    }

    // Copy result size from device
    CUDA_CHECK(cudaMemcpyAsync(output_size, d_output_size, sizeof(u32),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_output_size);

    return status;
  }

  Status compress_literals(
      const unsigned char *literals, u32 num_literals,
      BlockBufferWriter &writer, // Use Writer instead of raw pointers
      cudaStream_t stream) {

    // ZSTD Literals Section
    // Elements: [Literals Header] [Literals Content]

    
     if (num_literals > 0 && literals == nullptr) {
       return Status::ERROR_INVALID_PARAMETER;
     }

    // Case 1: No literals (Still need a header saying so, usually)
    // Actually, empty block doesn't seem to have literals section?
    // ZSTD Spec: "A Block has a generic structure... Block_Header
    // [Literals_Section] [Sequences_Section]" If Block_Type is
    // Compressed_Block (typically), it MUST have Literals Section. If
    // num_literals is 0, we verify how to encode "empty literals". Raw
    // Literals Block (formatted): Header byte: (Block_Type=00) |
    // (Size_Format=00) | (Size << 3) If Size=0, Header = 0x00.

    

    unsigned char header[3];
    u32 header_len = 0;

    // Use Raw Literals Block (Block_Type=00) for simplicity and stability
    // for now. Compressed Literals (Huffman) can be added later as
    // optimization.

    if (num_literals < 32) {
      // Format 0 (00): 1 byte header.
      // Bits 0-1: Type (00=Raw), Bits 2-4: SF (00), Bits 3-7: Size
      // Actually Table 11: SF=0 -> b2=0, b1-0=Type. b7-3=Size.
      header[0] = (unsigned char)((num_literals << 3) | 0x00); 
      header_len = 1;
    } else if (num_literals <= 4095) {
      // Format 1 (01): 2 bytes.
      // SF=01 (2 bytes): regenerated_size = (h[0]>>4) + (h[1]<<4)
      // h[0] bits 4-7 = bits 0-3 of size. h[1] bits 0-7 = bits 4-11 of size.
      header[0] = (unsigned char)(((num_literals & 0x0F) << 4) | 0x04);
      header[1] = (unsigned char)((num_literals >> 4) & 0xFF);
      header_len = 2;
    } else if (num_literals <= 1048575) {
      // Format 2 (10): 3 bytes.
      // SF=10 (3 bytes): regenerated_size = (h[0]>>4) + (h[1]<<4) + (h[2]<<12)
      header[0] = (unsigned char)(((num_literals & 0x0F) << 4) | 0x08);
      header[1] = (unsigned char)((num_literals >> 4) & 0xFF);
      header[2] = (unsigned char)((num_literals >> 12) & 0xFF);
      header_len = 3;
    } else { // num_literals <= 268435455
      // Format 3 (11): 4 bytes.
      // SF=11 (4 bytes): regenerated_size = (h[0]>>4) + (h[1]<<4) + (h[2]<<12) + (h[3]<<20)
      header[0] = (unsigned char)(((num_literals & 0x0F) << 4) | 0x0C);
      header[1] = (unsigned char)((num_literals >> 4) & 0xFF);
      header[2] = (unsigned char)((num_literals >> 12) & 0xFF);
      header[3] = (unsigned char)((num_literals >> 20) & 0xFF);
      header_len = 4;
    }

    
#ifdef CUDA_ZSTD_DEBUG
    printf("[COMPRESS] Writing Raw literals header: num_literals=%u, header_len=%u, header=[",
           num_literals, header_len);
    for (u32 i = 0; i < header_len; i++) {
      printf("%02X ", header[i]);
    }
    printf("], compressed_size=%u (same as num_literals for Raw)\n", num_literals);
#endif

    if (!writer.write_bytes(header, header_len, stream)) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    if (num_literals > 0) {
      if (!writer.write_bytes(literals, num_literals, stream, true)) {
        return Status::ERROR_BUFFER_TOO_SMALL;
      }
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Tier 1: Encode sequences using predefined ZSTD FSE tables
   * Fastest option, uses standard tables without frequency analysis.
   *
   * Format:
   * [mode_byte=0x01][ll_fse_stream][ml_fse_stream][offset_fse_stream]
   */
  Status encode_sequences_with_predefined_fse(
      const sequence::SequenceContext *seq_ctx, u32 num_sequences,
      unsigned char *output, u32 *output_size, CompressionWorkspace *workspace,
      cudaStream_t stream) {
    if (num_sequences == 0) {
      if (output_size)
        *output_size = 0;
      return Status::SUCCESS;
    }

    // === GPU IMPLEMENTATION ===
    namespace fse = cuda_zstd::fse;

    // 1. Prepare Header (Number of Sequences)
    // 1-3 bytes.
    u8 header_buffer[4];
    u32 header_len = 0;

    if (num_sequences < 128) {
      header_buffer[header_len++] = (unsigned char)num_sequences;
    } else if (num_sequences < 32512) {
      header_buffer[header_len++] = (unsigned char)((num_sequences >> 8) + 128);
      header_buffer[header_len++] = (unsigned char)(num_sequences & 0xFF);
    } else {
      header_buffer[header_len++] = (unsigned char)255;
      header_buffer[header_len++] = (unsigned char)((num_sequences - 0x7F00) & 0xFF);
      header_buffer[header_len++] = (unsigned char)((num_sequences - 0x7F00) >> 8);
    }

    // Mode Byte: Predefined (0,0,0) -> 0x00 (LL=0, OF=0, ML=0)
    header_buffer[header_len++] = 0x00;

    // Copy Header to Output (Device)
    cudaMemcpyAsync(output, header_buffer, header_len, cudaMemcpyHostToDevice,
                    stream);

    // 2. Build Tables on GPU (Phase 2a: Rebuild every time for
    // correctness, Phase 2b: Use Cache) Allocate 3 Tables
    fse::FSEEncodeTable *d_tables;
    bool using_cache = (d_cached_fse_tables != nullptr);

    if (using_cache) {
      d_tables = d_cached_fse_tables;
    } else {
      cudaMallocAsync(&d_tables, 3 * sizeof(fse::FSEEncodeTable), stream);

      // Helper to build one table
      auto build_table = [&](fse::TableType type, int idx) {
        u32 max_s, t_log;
        const u16 *h_norm = fse::get_predefined_norm(type, &max_s, &t_log);

        std::vector<u32> h_norm_u32(max_s + 1);
        for (u32 i = 0; i <= max_s; ++i)
          h_norm_u32[i] = h_norm[i];

        u32 *d_norm;
        cudaMallocAsync(&d_norm, (max_s + 1) * sizeof(u32), stream);
        cudaMemcpyAsync(d_norm, h_norm_u32.data(), (max_s + 1) * sizeof(u32),
                        cudaMemcpyHostToDevice, stream);

        fse::FSEEncodeTable h_desc;
        h_desc.max_symbol = max_s;
        h_desc.table_log = t_log;
        u32 table_size = 1 << t_log;
        h_desc.table_size = table_size;

        cudaMallocAsync(
            &h_desc.d_symbol_table,
            (max_s + 1) * sizeof(fse::FSEEncodeTable::FSEEncodeSymbol), stream);
        cudaMallocAsync(&h_desc.d_next_state, table_size * sizeof(u16), stream);
        cudaMallocAsync(&h_desc.d_nbBits_table, table_size * sizeof(u8),
                        stream);
        cudaMallocAsync(&h_desc.d_symbol_first_state, (max_s + 1) * sizeof(u16),
                        stream);
        cudaMallocAsync(&h_desc.d_state_to_symbol, table_size * sizeof(u8),
                        stream);

        // CRITICAL: Zero-initialize all table memory before kernel runs
        cudaMemsetAsync(h_desc.d_symbol_table, 0, 
                        (max_s + 1) * sizeof(fse::FSEEncodeTable::FSEEncodeSymbol), stream);
        cudaMemsetAsync(h_desc.d_symbol_first_state, 0, 
                        (max_s + 1) * sizeof(u16), stream);

        cudaMemcpyAsync(&d_tables[idx], &h_desc, sizeof(fse::FSEEncodeTable),
                        cudaMemcpyHostToDevice, stream);

        // CRITICAL: Synchronize to ensure h_desc is copied before it goes
        // out of scope
        cudaStreamSynchronize(stream);

        fse::FSE_buildCTable_Device(d_norm, max_s, t_log, &d_tables[idx],
                                    nullptr, 0, stream);

        // Note: d_norm leaked for Phase 2a prototype
      };

      build_table(fse::TableType::LITERALS, 0);
      build_table(fse::TableType::OFFSETS, 1);
      build_table(fse::TableType::MATCH_LENGTHS, 2);
    } // End else (!using_cache)

    // 3. Launch Encoding
    size_t *d_pos;
    cudaMallocAsync(&d_pos, sizeof(size_t), stream);

    size_t capacity = num_sequences * 8 + 512;
    unsigned char *d_bitstream;
    cudaMallocAsync(&d_bitstream, capacity, stream);
    cudaMemsetAsync(d_bitstream, 0, capacity, stream);

    // Use interleaved FSE encoder matching decode_sequences_interleaved
    Status launchStatus = fse::launch_fse_encoding_kernel(
        seq_ctx->d_ll_codes, seq_ctx->d_ll_extras, seq_ctx->d_ll_num_bits,
        seq_ctx->d_of_codes, seq_ctx->d_of_extras, seq_ctx->d_of_num_bits,
        seq_ctx->d_ml_codes, seq_ctx->d_ml_extras, seq_ctx->d_ml_num_bits,
        num_sequences, d_bitstream, d_pos, capacity, d_tables, stream);

    if (launchStatus != Status::SUCCESS)
      return launchStatus;

    // 4. Append Bitstream to Output
    size_t h_pos_val;
    cudaMemcpyAsync(&h_pos_val, d_pos, sizeof(size_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // Verify output size fits
    if (output_size && header_len + h_pos_val > *output_size) {
      // Buffer too small?
      // Wait, encode_sequences_fse usually writes directly?
      // But here we write to d_bitstream temp buffer.
      // We copy to output.
    }

    cudaMemcpyAsync(output + header_len, d_bitstream, h_pos_val,
                    cudaMemcpyDeviceToDevice, stream);

    if (output_size)
      *output_size = header_len + h_pos_val;

    if (!using_cache) {
      if (!using_cache) {
        cudaFreeAsync(d_tables, stream);
      }
    }
    cudaFreeAsync(d_pos, stream);
    cudaFreeAsync(d_bitstream, stream);

    return Status::SUCCESS;
  }

  // NOTE: Compress Sequences Logic Updated Below

#if 0
    // === HOST FALLBACK IMPLEMENTATION ===
    // 1. Copy Sequences to Host
    u32 *h_offsets = new u32[num_sequences];
    u32 *h_literal_lengths = new u32[num_sequences];
    u32 *h_match_lengths = new u32[num_sequences];

    cudaMemcpy(h_offsets, seq_ctx->d_offsets, num_sequences * sizeof(u32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_literal_lengths, seq_ctx->d_literal_lengths,
               num_sequences * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_match_lengths, seq_ctx->d_match_lengths,
               num_sequences * sizeof(u32), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    // 2. Build CTables (LL, ML, OF)
    using namespace fse; // For FSEEncodeTable
    FSEEncodeTable ct_ll, ct_ml, ct_of;
    // Allocate Host-side mirrors (Note: FSE_buildCTable_Host allocates temp
    // host memory but EXPECTS device pointers in table struct... Wait, my
    // FSE_buildCTable_Host implementation COPIES to device pointers. This is a
    // mismatch for a pure Host fallback! I need FSE_buildCTable_Host to give me
    // HOST tables if I want to encode on HOST.

    // CORRECTION: FSE_buildCTable_Host as implemented attempts to cudaMemcpy to
    // d_symbol_table. Ideally I should have a version that keeps it on Host.
    // For now, I will manually build the tables HERE to avoid refactoring
    // fse.cu again and risking breakage. Or I can modify FSE_buildCTable_Host
    // to accept host pointers if device pointers are null? No, I'll implement a
    // `build_ctable_host_only` helper LOCALLY here.

    // Structs for Host Encoding
    struct HostFSECTable {
      u16 *nextState;
      FSEEncodeTable::FSEEncodeSymbol *symbolTT;
      u32 tableLog;
    };

    auto build_host_ctable = [&](TableType type, u32 max_sym, u32 default_log,
                                 HostFSECTable &table) {
      u32 max_s = 0;
      u32 t_log = 0;
      const u16 *norm = get_predefined_norm(type, &max_s, &t_log);
      // Note: predefined norms use default logs.
      t_log = default_log; // Force default for predefined

      u32 table_size = 1 << t_log;
      table.tableLog = t_log;
      table.nextState = new u16[table_size];
      table.symbolTT = new FSEEncodeTable::FSEEncodeSymbol[max_sym + 1];

      // --- COPY PASTE LOGIC from FSE_buildCTable_Host adjusted for HOST
      // pointers ---
      std::vector<u8> tableU8(table_size);
      u32 *cumul = new u32[max_sym + 2];
      cumul[0] = 0;
      u32 step = (table_size >> 1) + (table_size >> 3) + 3;
      u32 mask = table_size - 1;
      u32 position = 0;
      for (u32 s = 0; s <= max_sym; s++) {
        int n = norm[s];
        if (n == 0)
          continue;
        for (int i = 0; i < n; i++) {
          tableU8[position] = (u8)s;
          position = (position + step) & mask;
        }
      }
      u32 total = 0;
      for (u32 s = 0; s <= max_sym; s++) {
        u32 proba = norm[s];
        cumul[s] = total;
        if (proba == 0) {
          table.symbolTT[s].deltaNbBits = (u32)((-1) << 16);
          table.symbolTT[s].deltaFindState = 0;
          continue;
        }
        u32 minBitsOut = t_log - (31 - __builtin_clz(proba));
        u32 minStatePlus = proba << minBitsOut;
        table.symbolTT[s].deltaNbBits = (minBitsOut << 16) - minStatePlus;
        table.symbolTT[s].deltaFindState = total - 1;
        total += proba;
      }
      cumul[max_sym + 1] = total;
      for (u32 u = 0; u < table_size; u++) {
        u32 s = tableU8[u];
        table.nextState[cumul[s]++] = (u16)(table_size + u);
      }
      delete[] cumul;
    };

    HostFSECTable ctLL, ctML, ctOF;
    build_host_ctable(TableType::LITERALS, 35, 6, ctLL); // Max sym 35, log 6
    build_host_ctable(TableType::MATCH_LENGTHS, 52, 6,
                      ctML);                            // Max sym 52, log 6
    build_host_ctable(TableType::OFFSETS, 28, 5, ctOF); // Max sym 28, log 5

    // 3. Encode Loop
    // Host Bitstream
    std::vector<u8> bitStream;
    bitStream.reserve(num_sequences * 4); // Estimation
    u64 bitContainer = 0;
    u32 bitPos = 0;

    // Buffer for bit operations to support Reverse Writing (RFC 8878)
    struct BitOp {
      u64 val;
      u32 nbBits;
    };
    std::vector<BitOp> bitStack;
    bitStack.reserve(num_sequences * 4); // Approx size

    auto addBits = [&](u64 val, u32 nbBits) {
      if (nbBits == 0)
        return;
      bitStack.push_back({val, nbBits});
    };

    // States
    u16 stateLL = 0;
    u16 stateML = 0;
    u16 stateOF = 0;

    // Helper: Encode Symbol
    auto host_fse_encode_symbol = [&](HostFSECTable &ct, u32 sym, u16 &state) {
      FSEEncodeTable::FSEEncodeSymbol *symTT = ct.symbolTT;
      u16 *nextAlloc = ct.nextState;

      // FSE Encoding Logic (RFC 8878)
      // nbBits = (state + deltaNbBits) >> 16
      u32 nbBits = (state + symTT[sym].deltaNbBits) >> 16;

      // Write bits
      // Note: Value is (state & mask).
      addBits(state & ((1 << nbBits) - 1), nbBits);

      // Update State
      // sub = state >> nbBits
      // state = nextStateTable[ sub + deltaFindState ]
      state = nextAlloc[(state >> nbBits) + symTT[sym].deltaFindState];
    };

    // Helper: Initialize State (First Sequence / Seq 0)
    auto host_fse_init_state = [&](HostFSECTable &ct, u32 sym) -> u16 {
      // Init state such that it is valid for "sym".
      // Using the first valid slot in nextState table.
      return ct.nextState[ct.symbolTT[sym].deltaFindState + 1];
    };

    if (num_sequences > 0) {
      // Init states from Seq 0 (Start of Chain)
      stateLL = host_fse_init_state(ctLL, h_literal_lengths[0]);
      stateML = host_fse_init_state(ctML, h_match_lengths[0]);
      stateOF = host_fse_init_state(ctOF, h_offsets[0]);

      // Note: We DO NOT write headers here. Headers are the FINAL state.
      // We encode sequences 1 to N-1, updating the state.
      // The final state corresponds to the start of reading (End of stream).

      for (u32 i = 1; i < num_sequences; i++) {
        // Encode Seq[i] (Forward encoding builds up state chain)
        host_fse_encode_symbol(ctLL, h_literal_lengths[i], stateLL);
        host_fse_encode_symbol(ctML, h_match_lengths[i], stateML);
        host_fse_encode_symbol(ctOF, h_offsets[i], stateOF);
      }

      // Write Final States to Bitstream (Header)
      // Decoder reads these FIRST (from End of stream).
      // Order: LL, OF, ML (Matches Decoder Read Order: ML, OF, LL ? per RFC)
      // RFC: "Initial states order is LL, OF, ML" (read first).
      // Since we flush LIFO (Reverse), the LAST pushed is FIRST written to physical End.
      // Wait. Reader reads Backward.
      // Physical: [Seq Bits] ... [Header Bits] [Sentinel]
      // Reader @ End: Reads Header Bits.
      // Header Order checks:
      // Code 4047: read(ll), read(of), read(ml).
      // So LL is at Highest Address (First Read).
      // So LL should be written LAST physically.
      // Reverse Flush writes Stack Top -> Bottom.
      // Stack Top is Written LAST.
      // So Stack Top should be LL.
      // So Push ML, OF, LL.
      // Headers are written in flushBits to ensure they are at the End.
    }

    // Flush Bits Implementation (Reverse)
    auto flushBits = [&]() {
      
      
      // Local bit buffer for packing
      u64 localContainer = bitContainer; // preserve existing? (usually 0)
      u32 localPos = bitPos;

      // Helper to pack into local buffer
      auto pack = [&](u64 val, u32 nbBits) {
        if (nbBits == 0) return;
        localContainer |= (val & ((1ULL << nbBits) - 1)) << localPos;
        localPos += nbBits;
        while (localPos >= 8) {
            bitStream.push_back((u8)(localContainer));
            localContainer >>= 8;
            localPos -= 8;
        }
      };

      // 1. Iterate Stack in REVERSE (LIFO) for Body Bits
      // This puts the LAST pushed item (Seq N-1) at the Low Address of the New Segment
      // Wait. Stack Top = Seq N-1.
      // rbegin = Seq N-1.
      // Pack Seq N-1 (Low).
      // Pack Seq N-2 (Higher).
      // ...
      // Stream: [Seq N-1] [Seq N-2] ...
      // Reader @ End: Reads Init.
      // Moves back. Reads Seq N-2?
      // NO. Reader decrement bit_pos.
      // So Next Read comes from LOWER address.
      // So Reader reads [Init], then [item BELOW Init].
      // Item Below Init is [Seq N-2].
      // Decoder Loop Step N-1 needs [Seq N-1] bits!
      // So Below Init must be [Seq N-1].
      // So Stream: ... [Seq N-1] [Init].
      // So [Seq N-1] is packed JUST BEFORE [Init].
      // So [Seq N-1] must be packed LAST of the body.
      // Stack Top is [Seq N-1].
      // So we must pack Stack Top LAST.
      // So we must iterate Stack Forward (begin to end)!
      // Stack: [Seq 0 ... Seq N-1].
      // Pack Seq 0 (Low).
      // ...
      // Pack Seq N-1 (High - Just before Init).
      // Pack Init (Highest).
      // Stream: [Seq 0 ... Seq N-1 Init].
      // Reader: Reads Init.
      // Reads Seq N-1.
      // ...
      // Reads Seq 0.
      // Matches Decoder Loop!
      
      // So: Iterate Forward!
      for (const auto& op : bitStack) {
          pack(op.val, op.nbBits);
      }

      // 2. Pack Headers (Init States)
      // Must be at End.
      // Decoder reads LL, OF, ML.
      // Read LL (First) -> Must be Highest.
      // Pack ML (Low).
      // Pack OF.
      // Pack LL (High).
      
#ifdef CUDA_ZSTD_DEBUG
      printf("[FSE_PACK] InitStates: LL=%u (%u bits), OF=%u (%u bits), ML=%u (%u bits)\n",
             stateLL, ctLL.tableLog, stateOF, ctOF.tableLog, stateML, ctML.tableLog);
#endif

      pack(stateML, ctML.tableLog);
      pack(stateOF, ctOF.tableLog);
      pack(stateLL, ctLL.tableLog);
      
      // Flush remaining bits
      if (localPos > 0) {
          bitStream.push_back((u8)(localContainer));
      }

      // 3. Sentinel Bit
      pack(1, 1);

      // 4. Flush remaining
      while (localPos > 0) {
        bitStream.push_back((u8)(localContainer));
        localContainer >>= 8;
        if (localPos >= 8) localPos -= 8; else localPos = 0;
      }
      
      // Update member vars
      bitContainer = localContainer;
      bitPos = localPos;
    };

    flushBits();

    // 4. Construct Final Output
    // Format: [NumSeq][ModeByte][BitStream]
    // Zstd Block Format:
    // [Block Header (Managed by caller? No, caller calls us for Sequences
    // Section)] Sequences Section: 1-3 bytes: Number of Sequences 1 byte:
    // Symbol Compression Modes Bitstream.

    // Write to d_output
    // Need a Host Buffer first to assemble.
    std::vector<u8> header;

    // Num Sequences
    if (num_sequences < 128) {
      header.push_back((u8)num_sequences);
    } else if (num_sequences < 0x7F00) {
      header.push_back((u8)((num_sequences >> 8) + 128));
      header.push_back((u8)(num_sequences & 0xFF));
    } else {
      header.push_back((u8)255);
      header.push_back((u8)((num_sequences - 0x7F00) & 0xFF));
      header.push_back((u8)((num_sequences - 0x7F00) >> 8));
    }
    // Fixed: Previous logic was loose. Stick to <128, <Longer.
    // Simplifying: Just handle < 128 or error for now? No, assume standard
    // encoding. If num_sequences >= 128, use 2-byte form: (byte0 >= 128). byte0
    // = ((num_sequences >> 8) | 0x80) ? No. Value = ((byte0 - 128) << 8) +
    // byte1. So byte0 = 128 + (num_sequences >> 8). byte1 = num_sequences &
    // 0xFF.

    // Mode Byte
    // All Predefined (Mode 2? or 1?). RFC: Predefined=1.
    // LL=1, OF=1, ML=1.
    // Byte = (1<<6) | (1<<4) | (1<<2) = 0x54.
    header.push_back(0x54);

    // Total Size
    u32 totalSize = header.size() + bitStream.size();
    if (output_size)
      *output_size = totalSize;
    
    if (bitStream.size() > 0) {
    }

    // Copy Header + Bitstream to Device
    cudaMemcpyAsync(output, header.data(), header.size(),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(output + header.size(), bitStream.data(), bitStream.size(),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    // 5. Cleanup
    delete[] h_offsets;
    delete[] h_literal_lengths;
    delete[] h_match_lengths;

    auto free_ct = [&](HostFSECTable &t) {
      delete[] t.nextState;
      delete[] t.symbolTT;
    };
    free_ct(ctLL);
    free_ct(ctML);
    free_ct(ctOF);

    return Status::SUCCESS;
  }
    // RFC 8878:
    // "The first state is initialized by the first symbol to encode."
    // state = nextStateTable[symbol + deltaFindState]? No.
    //
    // State initialization:
    // state = CTable.nextState[ symbol_base + symbol ];  (Roughly)
    // Actually, for the very first symbol encoded (which is the LAST symbol in
    // the block), the state is just the Cumulative distribution index?
    //
    // Let's check Zstd source or spec.
    // Spec: "The last symbol encoded (first one decoded) ... is encoded with
    // state value." The decoder reads the initial state directly. The encoder
    // thus must produce that state.
    //
    // Zstd Encoder Loop:
    // 1. Init: state = CTable.nextState[ symbol ]; // Wait, this depends on
    // which occurrence of symbol? Actually, usually it's just `state =
    // table.nextState[symbol_start + symbol]`. But normalized counts > 1 means
    // multiple states per symbol.
    //
    // Correct Init:
    // state = nextState[ cumul[symbol] ]; // The very first slot for that
    // symbol? No, any valid state for that symbol works for the *start*.
    // Usually we pick the first one?
    // Zstd reference `FSE_initCState`:
    // state = header_table[symbol].nextState; // This implies a table lookup
    // for init. In our `nextState` table, we have mapped (cumul[s]++) ->
    // (table_size + u). So `nextState` contains the state values.
    //
    // For the FIRST symbol (last in stream), we just need ANY valid state `s`
    // such that `symbol(s) == symbol`. Our CTable maps `index -> next_state`.
    //
    // Let's assume we pick a specific state for initialization.
    // If we look at `FSE_encodeSymbol`:
    // It takes `state`, emits bits, transitions to `next_state`.
    //
    // Issue: We process sequences in REVERSE (N-1 down to 0).
    // The state is updated at each step.
    // The Final State (after processing Seq 0) is written to the bitstream as
    // the "Initial State" for the decoder.
    //
    // So, we need to initialize the state "before" the loop?
    // Wait, the decoder starts with `state = readBits(tableLog)`.
    // This `state` corresponds to the state *after* encoding the last symbol
    // (Seq 0). So yes, we encode Seq 0, produce a state, and that state is
    // written?
    //
    // NO!
    // FSE is a state machine.
    // Encoder:
    // Init state.
    // Loop (Symbol S):
    //   Emit bits derived from (State, S).
    //   Update State = NextState(State, S).
    //
    // If we process in Reverse (Decoder: forward read):
    // Decoder: Read State. Loop: Emit S(State). Update State.
    // Encoder: ???
    //
    // Zstd Encoder operates in REVERSE of Decoder direction.
    // Decoder reads bits -> Symbols.
    // Encoder takes Symbols -> bits.
    //
    // To encode:
    // We start with the LAST symbol (Seq N-1)?
    // Decoder reads Seq 0, then Seq 1...
    // Input stream: [Seq0] [Seq1] ...
    //
    // Decoder reads from END of bitstream backwards?
    // "The FSE bitstream is read in reverse direction"
    // So the bitstream is a stack.
    // We push symbols onto the stack.
    // Last Symbol Encoded = First Symbol Decoded.
    // If Decoder reads Seq 0 first, then Seq 0 must be LAST Encoded.
    //
    // Correct Order:
    // Encode Seq N-1 -> State changes -> Bits emitted.
    // ...
    // Encode Seq 0 -> State changes -> Bits emitted.
    // Final State is written as bitstream header.
    //
    // So we iterate: i = num_sequences - 1 down to 0. (Wait, if Seq 0 is last
    // encoded, we iterate sequences Forward 0..N-1?)
    //
    // Let's check `fse.cu` decoder:
    // `decode_sequences_interleaved`:
    // `bit_position = input_size * 8;` (Starts at end)
    // `state = read_bits(..., tableLog);` (Initial state)
    // Loop `i = num_sequences - 1 down to 0`:
    //    Decoder produces Seq i.
    //
    // So Decoder produces Seq N-1 FIRST?
    // `h_output[i] = symbol;` -> i counts down.
    // So Seq N-1 is at the END of the output array.
    // Seq 0 is at the START.
    //
    // If loop is `i = N-1 down to 0`, first iteration handles i=N-1.
    // So the Initial State (start of bitstream read) corresponds to Seq N-1.
    //
    // Therefore, Encoder must finish with Seq N-1.
    // So Encoder iterates 0 to N-1?
    //
    // Correct. Encoder processes Seq 0, then 1... then N-1.
    // The resulting state after Seq N-1 is written to the stream.
    //
    // But how do we Initialize Encoder State?
    // We don't "Encode Seq 0" using a previous state.
    // The very first symbol (Seq 0) determines the state?
    //
    // No. "The last symbol encoded is encoded with state value".
    // If we encode 0..N-1, the last one is N-1.
    //
    // Basic FSE Encoding:
    // State = CTable.nextState[ symbol_0 ]; // Pick a state for first symbol.
    // Loop i = 1 to N-1:
    //    Symbol s = seq[i];
    //    Encode(s, State):
    //       nbBits = ...
    //       Emit(nbBits)
    //       State = Next(State, s)
    //
    // This produces a state chain.
    //
    // Let's verify Sequence vs Interleaving.
    // We have 3 streams: LL, ML, OF.
    // They are interleaved.
    // "Decoder Read Order: OF State, ML State, LL State, LL Extra, ML Extra, OF
    // Extra"
    //
    // Decoder Loop (i=N-1 to 0):
    // 1. Decode OF (uses stateOF) -> update stateOF
    // 2. Decode ML (uses stateML) -> update stateML
    // 3. Decode LL (uses stateLL) -> update stateLL
    // ...
    //
    // So for Sequence i=N-1:
    // We need stateOF, stateML, stateLL ready.
    // These states come from the "Initial State" read from stream.
    //
    // So the Encoder must produce stateOF, stateML, stateLL corresponding to
    // Seq N-1. This is the Result of encoding sequences 0..N-1.
    //
    // So Encoder Loop:
    // Init stateLL, stateML, stateOF from Seq[0].
    // Loop i = 1 to N-1:
    //   Encode LL[i], ML[i], OF[i].
    //
    // Wait, interleaving is tricky.
    // Decoder reads: OF bits? ML bits?
    // Decoder reads bits for NEXT state.
    //
    // Encoder:
    // We emit bits for the PREVIOUS state transition?
    // Logic: `nbBits = (state + deltaNbBits) >> 16`.
    // These bits allow the decoder (which has `state` (our `nextState`)) to
    // distinguish which `previousState` (decoder's `nextState`) we came from.
    //
    // So if Decoder goes N-1 -> 0.
    // Encoder goes 0 -> N-1.
    //
    // Init:
    // stateLL = TableLL.nextState[ cumul[ LL[0] ] ]; // Just one valid state?
    // stateML = ...
    // stateOF = ...
    //
    // Loop i = 1 to N-1:
    //    (Need to handle Interleaving: OF, ML, LL)
    //    BUT Decoder reads OF, ML, LL.
    //    Encoder writes LL, ML, OF? (Reverse stack)
    //    No, bitstream is a stack.
    //    If Decoder pops OF, then ML, then LL...
    //    Encoder must Push LL, then ML, then OF.
    //
    //    So inside the loop (i = 0 to N-1? or 1 to N-1?):
    //    We check the loop range carefully.
    //    If we init with i=0.
    //    We loop i=1 to N-1.
    //    At step i: we encode Seq[i].
    //    Push LL[i] bits. Push ML[i] bits. Push OF[i] bits.
    //
    //    Let's check `encodeSymbol` logic.
    //    It modifies `state` and outputs bits.
    //    `host_fse_encode_symbol(bitStream, table, symbol, state)`
    //
    //    Do we treat init specially?
    //    Yes, the first symbol (0) doesn't emit bits? It ESTABLISHES the
    //    baseline state. So for i=0, we just set `state`. For i=1..N-1, we
    //    perform transitions.
    //
    //    Wait, does Seq 0 emit bits?
    //    If we have 1 sequence.
    //    Init state from Seq 0.
    //    Write state to stream.
    //    Decoder reads state.
    //    Decodes Seq 0.
    //    Loop 0 to 0. Done.
    //    Correct. Seq 0 emits NO bits during the loop.
    //
    //    So Encoder Loop is `i = 1 to num_sequences - 1`.
    //    Inside loop:
    //       Encode Seq[i].
    //       Order: Push LL, ML, OF (To reverse Decoder's OF, ML, LL).
    //       Actually, check Decoder:
    //       `if (decode_of) ... stateOF = ...`
    //       `if (decode_ml) ... stateML = ...`
    //       `if (decode_ll) ... stateLL = ...`
    //       Then `Read Extra Bits`.
    //
    //       Decoder reads bitstream "Backwards" from end?
    //       `read_bits_from_buffer` reads from `bit_position`. `bit_position -=
    //       num_bits`. So it consumes from high to low. So "Pushing" bits means
    //       appending to the stream (forward write), but logic assumes stack?
    //       YES.
    //       So if Decoder reads OF, then ML, then LL.
    //       And reads "Top" of stack.
    //       Then Encoder must write LL (bottom), then ML, then OF (top).
    //
    //    Wait. `write_bits_verified` writes to `bit_position`. `bit_position +=
    //    num_bits`. So we write Forward. Decoder reads Backward. This is a
    //    Stack. First Written = Last Read. Last Written = First Read.
    //
    //    Decoder reads: OF, ML, LL.
    //    So Encoder writes: LL, ML, OF.
    //
    // Okay, that's the order.
    //
    // One catch: `deltaNbBits` gives the number of bits to emit for the *state
    // transition*.
    //
    // Correct.
    //
    // Implementation Detail:
    // `state` needs to be initialized correctly.
    // `state = nextState[ cumul[symbol] ]`.
    //
    // Extra Bits:
    // `get_lit_len_extra_bits`, etc.
    //
    // This looks complete.
    //
    // Check `num_sequences == 0` handled.
    // Check `num_sequences == 1`. Loop doesn't run. State written. Correct.

    // Write Header:
    // Flush bitstream (align).
    // Write 3 bytes (final states) or packed?
    // Zstd Spec: "Compressed FSE streams... The bitstream starts with the FSE
    // state values." "Each state is stored using tableLog bits." Packed? "The 3
    // states are packed... LL (tableLog), OF (tableLog), ML (tableLog)." Wait,
    // check spec. Actually, `decode_sequences_interleaved` reads: `bit_position
    // -= table_log;` `state = read_bits...` It does this for EACH stream if
    // they are separate? But here they are INTERLEAVED.
    //
    // Ah, `decode_sequences_interleaved` in `fse.cu` (Host Code) isn't the
    // whole story. Wait, `decode_sequences_interleaved` IS the kernel/host
    // decoder. Lines 3988: `if (bit_pos < ll_log + ml_log + of_log)` `stateLL =
    // read_bits...` `stateML = read_bits...` `stateOF = read_bits...` (Wait,
    // `bit_pos` handling there) It reads LL, then ML, then OF? Let's re-read
    // the decoder I fixed/viewed earlier (Line 3996 in fse.cu).
    //
    // 3996: `if (decode_ll) ... stateLL = ...`
    // 4012: `if (decode_of) ... stateOF = ...`
    // 4029: `if (decode_ml) ... stateML = ...`
    //
    // The order in source seems to be LL, OF, ML.
    // BUT they read from `bit_pos`.
    // `bit_pos` is decremented.
    //
    // So the Last one read (Lowest bit address) is the First one written?
    // No.
    // `bit_pos` initialized to `input_size * 8`.
    // Decrement -> High bits first.
    //
    // If Source order is:
    // 1. LL (Reads from Top)
    // 2. OF (Reads from New Top)
    // 3. ML (Reads from New New Top)
    //
    // Then Encoder must write: ML, OF, LL.
    //
    // Wait, `bit_position` logic:
    // `stateLL = read_bits(..., bit_pos -= ll_log, ...)`
    //
    // So LL is at the VERY END of the stream.
    // OF is before LL.
    // ML is before OF.
    //
    // Encoder (Forward Write):
    // Write ML Final State.
    // Write OF Final State.
    // Write LL Final State.
    //
    // Then Body.
    //
    // Wait, the Body loop in Decoder also reads backwards.
    // So Encoder Body (Forward Write) must push bits such that Decoder reads
    // them in correct order.
    //
    // Decoder Loop (N-1 to 0):
    // 1. OF State (Top)
    // ...
    //
    // So Encoder Body (0 to N-1):
    // Push OF State bits? No.
    // If Decoder reads OF State FIRST (Highest Address),
    // Encoder must write OF State LAST (Highest Address).
    //
    // So in Encoder Loop (i=1 to N-1):
    // (We are generating bits for transition FROM i-1 TO i? No, from i TO i-1?)
    //
    // Encode Step:
    // Decoder: State' = Table[State] + bits. (State grows/changes)
    // Encoder: State = Table[State'] >> bits ??
    //
    // Zstd "Encoding is reverse of decoding".
    //
    // Let's just follow the logic:
    // Decoder reads OF, ML, LL.
    // So Encoder writes LL, ML, OF. (So OF is at top).
    //
    // Okay.

    // Final check on `addBits` helper.
    // It appends to `bitStream`.
    // Make sure `bitStream` is `vector<unsigned char>`.

    // Copy to `output`.
    // `output[0] = 0x01;` // Header Mode? (Predefined)
    // Actually `encode_sequences_with_predefined_fse` header?
    // "Format:
    // [mode_byte=0x01][ll_fse_stream][ml_fse_stream][offset_fse_stream]" This
    // comment in `manager.cu` suggests 3 separate streams? But
    // `decode_sequences_interleaved` handles 3 interleaved streams.
    //
    // If they are separate streams, they are concatenated.
    // If interleaved, they are one stream.
    //
    // Standard Zstd Compressed Block:
    // - Literals Section
    // - Sequences Section
    //   - Sequences Header (Num Sequences, Modes)
    //   - FSE Tables (if needed)
    //   - Bitstream (Interleaved)
    //
    // My function `encode_sequences_with_predefined_fse` is called to produce
    // THE SEQUENCES SECTION? Or just the bitstream?
    //
    // "Format: [mode_byte=0x01]..."
    // This signature suggests it produces the WHOLE sequences section,
    // including header/modes.
    //
    // `encode_sequences_raw` writes `[num_sequences_header][fse_modes=0xFF]...`
    //
    // So yes.
    // 1. Write Num Sequences (1-3 bytes).
    // 2. Write Mode Byte (0x54 for all predefined? or 0x9B?).
    //       LL=Predef(1), OF=Predef(1), ML=Predef(1). -> 0x55 (01 01 01 01?)
    //       Bits 6-7: LL. 4-5: OF. 2-3: ML.
    //       1<<6 | 1<<4 | 1<<2 = 64+16+4 = 84 = 0x54.
    //
    //    3. Write Bitstream.
    //       Since we are Predefined, no tables.
    //       Just the Interleaved Stream.
    //
    //       Wait, Zstd spec says if Predefined, no tables.
    //
    //    So output = NumSeq + ModeByte + Bitstream.

    // I will simply implement the bitstream encoding and wrap it.

    // 5. Cleanup: delete host arrays.

    // Ready to write.
#endif

  /**
   * @brief Tier 4 fallback: Encode sequences without compression (raw u32
   * format) Stores full u32 values without truncation.
   *
   * Format:
   * [num_sequences_header][fse_modes=0xFF][ll_u32_data][of_u32_data][ml_u32_data]
   * fse_modes=0xFF signals custom raw u32 mode (not standard ZSTD)
   */

  Status encode_sequences_raw(const sequence::SequenceContext *seq_ctx,
                              u32 num_sequences, BlockBufferWriter &writer,
                              cudaStream_t stream) {
    // Early return if no sequences
    if (num_sequences == 0) {
      return Status::SUCCESS;
    }

    // Step 1: Copy sequences to host for validation
    u32 *h_offsets = new u32[num_sequences];
    u32 *h_literal_lengths = new u32[num_sequences];
    u32 *h_match_lengths = new u32[num_sequences];

    CUDA_CHECK(cudaMemcpy(h_offsets, seq_ctx->d_offsets,
                          num_sequences * sizeof(u32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_literal_lengths, seq_ctx->d_literal_lengths,
                          num_sequences * sizeof(u32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_match_lengths, seq_ctx->d_match_lengths,
                          num_sequences * sizeof(u32), cudaMemcpyDeviceToHost));

    // Step 2: Validation (Removed - d_matches is now initialized)
    // Step 2: Validation
    u32 valid_count = 0;
    u32 offset_zero_count = 0;
    for (u32 i = 0; i < num_sequences; i++) {
      if (!(h_match_lengths[i] > 0 && h_offsets[i] == 0)) {
        valid_count++;
      } else {
        offset_zero_count++;
        if (offset_zero_count <= 5) {
        }
      }
    }

    // Step 3: If ALL sequences are invalid, skip sequences section
    // entirely
    if (valid_count == 0) {
      // section\n");
      delete[] h_offsets;
      delete[] h_literal_lengths;
      delete[] h_match_lengths;
      return Status::SUCCESS; // Omit sequences section per ZSTD spec
    }

    // Step 4: Write header with FILTERED count
    unsigned char header_buf[3];
    u32 header_len = 0;
    if (valid_count < 128) {
      header_buf[0] = (unsigned char)valid_count;
      header_len = 1;
    } else if (valid_count < 0x7F00) {
      header_buf[0] = (unsigned char)((valid_count >> 8) + 0x80);
      header_buf[1] = (unsigned char)(valid_count & 0xFF);
      header_len = 2;
    } else {
      header_buf[0] = (unsigned char)0xFF;
      header_buf[1] = (unsigned char)((valid_count - 0x7F00) & 0xFF);
      header_buf[2] = (unsigned char)((valid_count - 0x7F00) >> 8);
      header_len = 3;
    }

    if (!writer.write_bytes(header_buf, header_len, stream)) {
      delete[] h_offsets;
      delete[] h_literal_lengths;
      delete[] h_match_lengths;
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Step 5: Write FSE mode byte (0xFF = Tier 4 raw u32)
    if (!writer.write_byte(0xFF, stream)) {
      delete[] h_offsets;
      delete[] h_literal_lengths;
      delete[] h_match_lengths;
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // Step 6: Write sequence data (filtered or unfiltered)
    if (valid_count < num_sequences) {
      // Some sequences are invalid - filter them out
      // valid\n",
      //        num_sequences - valid_count, valid_count);

      // Create filtered arrays
      u32 *h_valid_ll = new u32[valid_count];
      u32 *h_valid_of = new u32[valid_count];
      u32 *h_valid_ml = new u32[valid_count];

      u32 valid_idx = 0;
      for (u32 i = 0; i < num_sequences; i++) {
        // Keep sequences with (match_length == 0) OR (offset != 0)
        if (!(h_match_lengths[i] > 0 && h_offsets[i] == 0)) {
          h_valid_ll[valid_idx] = h_literal_lengths[i];
          h_valid_of[valid_idx] = h_offsets[i];
          h_valid_ml[valid_idx] = h_match_lengths[i];
          valid_idx++;
        }
      }

      // Allocate device memory and copy filtered data
      u32 *d_valid_ll, *d_valid_of, *d_valid_ml;
      CUDA_CHECK(cudaMalloc(&d_valid_ll, valid_count * sizeof(u32)));
      CUDA_CHECK(cudaMalloc(&d_valid_of, valid_count * sizeof(u32)));
      CUDA_CHECK(cudaMalloc(&d_valid_ml, valid_count * sizeof(u32)));

      CUDA_CHECK(cudaMemcpy(d_valid_ll, h_valid_ll, valid_count * sizeof(u32),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_valid_of, h_valid_of, valid_count * sizeof(u32),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_valid_ml, h_valid_ml, valid_count * sizeof(u32),
                            cudaMemcpyHostToDevice));

      delete[] h_valid_ll;
      delete[] h_valid_of;
      delete[] h_valid_ml;
      delete[] h_offsets;
      delete[] h_literal_lengths;
      delete[] h_match_lengths;

      // Write filtered arrays (LL, OF, ML order)
      u32 array_size = valid_count * sizeof(u32);
      if (!writer.write_bytes(d_valid_ll, array_size, stream, true)) {
        cudaFree(d_valid_ll);
        cudaFree(d_valid_of);
        cudaFree(d_valid_ml);
        return Status::ERROR_BUFFER_TOO_SMALL;
      }
      if (!writer.write_bytes(d_valid_of, array_size, stream, true)) {
        cudaFree(d_valid_ll);
        cudaFree(d_valid_of);
        cudaFree(d_valid_ml);
        return Status::ERROR_BUFFER_TOO_SMALL;
      }
      if (!writer.write_bytes(d_valid_ml, array_size, stream, true)) {
        cudaFree(d_valid_ll);
        cudaFree(d_valid_of);
        cudaFree(d_valid_ml);
        return Status::ERROR_BUFFER_TOO_SMALL;
      }

      cudaFree(d_valid_ll);
      cudaFree(d_valid_of);
      cudaFree(d_valid_ml);

      return Status::SUCCESS;
    }
    // Always write from corrected host arrays (validation may have
    // modified them) Allocate device memory and copy corrected data
    u32 *d_corrected_ll, *d_corrected_of, *d_corrected_ml;
    CUDA_CHECK(cudaMalloc(&d_corrected_ll, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_corrected_of, num_sequences * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_corrected_ml, num_sequences * sizeof(u32)));

    CUDA_CHECK(cudaMemcpy(d_corrected_ll, h_literal_lengths,
                          num_sequences * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_corrected_of, h_offsets,
                          num_sequences * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_corrected_ml, h_match_lengths,
                          num_sequences * sizeof(u32), cudaMemcpyHostToDevice));

    delete[] h_offsets;
    delete[] h_literal_lengths;
    delete[] h_match_lengths;

    u32 array_size = num_sequences * sizeof(u32);
    if (!writer.write_bytes(d_corrected_ll, array_size, stream, true)) {
      cudaFree(d_corrected_ll);
      cudaFree(d_corrected_of);
      cudaFree(d_corrected_ml);
      return Status::ERROR_BUFFER_TOO_SMALL;
    }
    if (!writer.write_bytes(d_corrected_of, array_size, stream, true)) {
      cudaFree(d_corrected_ll);
      cudaFree(d_corrected_of);
      cudaFree(d_corrected_ml);
      return Status::ERROR_BUFFER_TOO_SMALL;
    }
    if (!writer.write_bytes(d_corrected_ml, array_size, stream, true)) {
      cudaFree(d_corrected_ll);
      cudaFree(d_corrected_of);
      cudaFree(d_corrected_ml);
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    cudaFree(d_corrected_ll);
    cudaFree(d_corrected_of);
    cudaFree(d_corrected_ml);

    return Status::SUCCESS;
  }

  Status compress_sequences(const sequence::SequenceContext *seq_ctx,
                            u32 num_sequences, BlockBufferWriter &writer,
                            cudaStream_t stream,
                            CompressionWorkspace *workspace) {
    if (num_sequences == 0) {
      return Status::SUCCESS;
    }

    // Try Predefined Mode if Workspace is available
    if (workspace && workspace->d_lz77_temp) {
      // 1. Setup Buffers in Temp Workspace
      // Partition d_lz77_temp (assumed large enough for 4*N u32s -> 16*N
      // bytes) We need 3*N bytes for codes, plus aligned space for
      // incompatible flag.

      // 1. Setup Buffers in Temp Workspace
      // Use d_matches (void*) for u8 buffers (Codes, Bits) -> 6 * N bytes
      u8 *d_match_base = (u8 *)workspace->d_matches;
      u8 *d_ll_codes = d_match_base;
      u8 *d_ml_codes = d_ll_codes + num_sequences;
      u8 *d_of_codes = d_ml_codes + num_sequences;
      u8 *d_ll_bits = d_of_codes + num_sequences;
      u8 *d_ml_bits = d_ll_bits + num_sequences;
      u8 *d_of_bits = d_ml_bits + num_sequences;

      // Use d_lz77_temp (u32*) for u32 buffers (Extras) -> 3 * N u32s
      u32 *d_temp_u32 = workspace->d_lz77_temp;
      u32 *d_ll_extras = d_temp_u32;
      u32 *d_ml_extras = d_ll_extras + num_sequences;
      u32 *d_of_extras = d_ml_extras + num_sequences;
      u32 *d_incompatible = d_of_extras + num_sequences;

      cudaMemsetAsync(d_incompatible, 0, sizeof(u32), stream);

      // 2. Launch Conversion Kernel
      u32 threads = 512;
      u32 blocks = (num_sequences + threads - 1) / threads;
      convert_sequences_to_fse_codes_kernel<<<blocks, threads, 0, stream>>>(
          seq_ctx->d_literal_lengths, seq_ctx->d_match_lengths,
          seq_ctx->d_offsets, num_sequences, d_ll_codes, d_ll_extras, d_ll_bits,
          d_ml_codes, d_ml_extras, d_ml_bits, d_of_codes, d_of_extras,
          d_of_bits, d_incompatible);

      // 3. Check Flag
      u32 h_incompatible = 0;
      cudaMemcpyAsync(&h_incompatible, d_incompatible, sizeof(u32),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream); // Sync required to decide branch

#ifdef CUDA_ZSTD_DEBUG
      printf("[compress_sequences] num_sequences=%u, h_incompatible=%u\n",
             num_sequences, h_incompatible);
#endif

      // WORKAROUND: Predefined Mode Encoder (k_fse_encode) fixed.
      if (h_incompatible == 0) {
        // Compatible! Use Predefined Mode.
        sequence::SequenceContext code_ctx = *seq_ctx;
        code_ctx.d_ll_codes = d_ll_codes;
        code_ctx.d_ml_codes = d_ml_codes;
        code_ctx.d_of_codes = d_of_codes;

        code_ctx.d_ll_extras = d_ll_extras;
        code_ctx.d_ml_extras = d_ml_extras;
        code_ctx.d_of_extras = d_of_extras;

        code_ctx.d_ll_num_bits = d_ll_bits;
        code_ctx.d_ml_num_bits = d_ml_bits;
        code_ctx.d_of_num_bits = d_of_bits;

        // Compute size available in writer? Writer doesn't expose
        // capacity easily here? encode_sequences_with_predefined_fse
        // allocates its own temp buffer for bitstream. We write directly
        // to writer's buffer.
        u32 output_size = 0;
        Status status = encode_sequences_with_predefined_fse(
            &code_ctx, num_sequences, writer.get_current_ptr(), &output_size,
            workspace, stream);

        if (status == Status::SUCCESS) {
#ifdef CUDA_ZSTD_DEBUG
          printf("[compress_sequences] Predefined FSE succeeded, output_size=%u\n", output_size);
#endif
          writer.advance(output_size);
          return Status::SUCCESS;
        }
#ifdef CUDA_ZSTD_DEBUG
        printf("[compress_sequences] Predefined FSE FAILED with status=%d, falling back to raw\n", (int)status);
#endif
        // If failed (e.g. buffer too small), Fallback to Raw?
        // Or just return error. Usually Predefined is smaller than Raw.
        // But if it failed, maybe Raw works? Let's try Raw as fallback or
        // return error.
        if (status != Status::ERROR_BUFFER_TOO_SMALL) {
          return status;
        }
        // If buffer too small, Raw might also fail, but let's try.
      } else {
#ifdef CUDA_ZSTD_DEBUG
        printf("[compress_sequences] Incompatible with predefined tables, falling back to raw\n");
#endif
      }

      // Fallback: Encode Raw
#ifdef CUDA_ZSTD_DEBUG
      printf("[compress_sequences] Using encode_sequences_raw (NON-STANDARD 0xFF mode)\n");
#endif
      return encode_sequences_raw(seq_ctx, num_sequences, writer, stream);
    }
    return encode_sequences_raw(seq_ctx, num_sequences, writer, stream);
  }

  /* TIER 2 & 3 - Not yet implemented
  // Tier 2: Custom FSE with larger table
  // Tier 3: Huffman encoding
  */

  Status decompress_literals(const unsigned char *input, u32 input_size,
                             unsigned char *output, u32 *h_header_size,
                             u32 *h_compressed_size, u32 *h_decompressed_size,
                             cudaStream_t stream) {
    unsigned char h_header[5] = {0};
    if (input_size == 0)
      return Status::ERROR_CORRUPT_DATA;
    CUDA_CHECK(cudaMemcpy(h_header, input, std::min(5u, input_size),
                          cudaMemcpyDeviceToHost));
    

    auto read_le16 = [&](const unsigned char *src) -> u32 {
      return (u32)src[0] | ((u32)src[1] << 8);
    };
    auto read_le24 = [&](const unsigned char *src) -> u32 {
      return (u32)src[0] | ((u32)src[1] << 8) | ((u32)src[2] << 16);
    };
    auto read_le32 = [&](const unsigned char *src) -> u32 {
      return (u32)src[0] | ((u32)src[1] << 8) | ((u32)src[2] << 16) |
             ((u32)src[3] << 24);
    };

    u32 literals_type = h_header[0] & 0x03;
    u32 size_format = (h_header[0] >> 2) & 0x03;
    

    // RFC 8878 Literals Section Header:
    // Bits 0-1: Block Type
    // Bits 2-3: Size Format
    // Bits 4-7: Size (part of)

    if (literals_type == 0 || literals_type == 1) {
      u32 lhl_code = size_format;
      if (lhl_code == 0) {
        *h_header_size = 1;
        *h_decompressed_size = h_header[0] >> 3;
      } else if (lhl_code == 1) {
        if (input_size < 2)
          return Status::ERROR_CORRUPT_DATA;
        *h_header_size = 2;
        *h_decompressed_size = (h_header[0] >> 4) | ((u32)h_header[1] << 4);
      } else if (lhl_code == 2) {
        if (input_size < 3)
          return Status::ERROR_CORRUPT_DATA;
        *h_header_size = 3;
        *h_decompressed_size = (h_header[0] >> 4) | ((u32)h_header[1] << 4) | ((u32)h_header[2] << 12);
      } else { // lhl_code == 3
        if (input_size < 4)
          return Status::ERROR_CORRUPT_DATA;
        *h_header_size = 4;
        *h_decompressed_size = (h_header[0] >> 4) | ((u32)h_header[1] << 4) | ((u32)h_header[2] << 12) | ((u32)h_header[3] << 20);
      }
      *h_compressed_size =
          (literals_type == 0) ? *h_decompressed_size : static_cast<u32>(1);

      if (*h_header_size + *h_compressed_size > input_size) {
        return Status::ERROR_CORRUPT_DATA;
      }

      if (literals_type == 0) { // Raw
        if (*h_compressed_size > 0) {
          CUDA_CHECK(cudaMemcpyAsync(output, input + *h_header_size,
                                     *h_compressed_size, cudaMemcpyDefault,
                                     stream));
        }
        return Status::SUCCESS;
      }

      // RLE
      unsigned char rle_value = h_header[*h_header_size];
      const u32 threads = 256;
      const u32 blocks = (*h_decompressed_size + threads - 1) / threads;
      if (blocks > 0) {
        expand_rle_kernel<<<blocks, threads, 0, stream>>>(
            output, *h_decompressed_size, rle_value);
      }
      return Status::SUCCESS;
    }

    // Compressed (type 2) or Treeless (type 3)
    u32 lhc = read_le32(h_header);
    if (size_format == 0 || size_format == 1) {
      *h_header_size = 3;
      *h_decompressed_size = (lhc >> 4) & 0x3FF;
      *h_compressed_size = (lhc >> 14) & 0x3FF;
    } else if (size_format == 2) {
      if (input_size < 4)
        return Status::ERROR_CORRUPT_DATA;
      *h_header_size = 4;
      *h_decompressed_size = (lhc >> 4) & 0x3FFF;
      *h_compressed_size = (lhc >> 18) & 0x3FFF;
    } else { // size_format == 3
      if (input_size < 5)
        return Status::ERROR_CORRUPT_DATA;
      *h_header_size = 5;
      *h_decompressed_size = (lhc >> 4) & 0x3FFFF;
      *h_compressed_size = ((lhc >> 22) & 0x3FF) | ((u32)h_header[4] << 10);
    }

    if (*h_header_size + *h_compressed_size > input_size)
      return Status::ERROR_CORRUPT_DATA;

    

      const unsigned char *d_data_start = input + *h_header_size;

    // RFC 8878 Section 3.1.1.3:
    // - Type 2: Compressed_Literals_Block (Huffman with embedded tree)
    // - Type 3: Treeless_Literals_Block (Huffman reusing previous tree)
    // Both use Huffman encoding, NOT FSE.

    // Use RFC 8878-compliant Huffman decoder for standard Zstandard
    // format size_format >= 1 means 4-stream format
    bool four_streams = (size_format != 0);
    
    size_t h_huff_output_size = 0;
    return huffman::decode_huffman_rfc8878(
        d_data_start, *h_compressed_size, output, &h_huff_output_size,
        *h_decompressed_size, four_streams, stream);
  }

  Status decompress_sequences(const unsigned char *input, u32 input_size,
                              sequence::SequenceContext *seq_ctx,
                              u32 total_literal_count, cudaStream_t stream) {
    
    if (input_size < 1) {
      seq_ctx->num_sequences = 0;
      return Status::SUCCESS;
    }

    if (seq_ctx == nullptr) {
      return Status::ERROR_INVALID_PARAMETER;
    }

    // Read header (max 3 bytes for num_sequences + 1 byte for modes)
    unsigned char h_header[16];
    CUDA_CHECK(cudaMemcpy(h_header, input, std::min(16u, input_size),
                          cudaMemcpyDeviceToHost));

    u32 num_sequences = 0;
    u32 offset = 0;

    if (h_header[0] < 128) {
      num_sequences = h_header[0];
      offset = 1;
    } else if (h_header[0] < 255) {
      num_sequences = ((u32)(h_header[0] - 128) << 8) + h_header[1];
      offset = 2;
    } else {
      // 3-byte form: nSeq = byte1 + (byte2 << 8) + 0x7F00
      num_sequences = (u32)h_header[1] + ((u32)h_header[2] << 8) + 0x7F00;
      offset = 3;
    }

    
    seq_ctx->num_sequences = num_sequences;
    if (num_sequences == 0) {
      return Status::SUCCESS;
    }

    if (offset >= input_size) {
       return Status::ERROR_CORRUPT_DATA;
    }

    unsigned char fse_modes = h_header[offset];
    offset += 1;

#ifdef CUDA_ZSTD_DEBUG
    fprintf(stderr, "[SEQ-DBG] fse_modes=0x%02X, num_sequences=%u, offset=%u, input_size=%u\n",
            fse_modes, num_sequences, offset, input_size);
    fprintf(stderr, "[SEQ-DBG] header bytes: ");
    for (u32 i = 0; i < std::min(16u, input_size); i++) {
      fprintf(stderr, "%02X ", h_header[i]);
    }
    fprintf(stderr, "\n");
#endif

    // Check for custom raw u32 mode (fse_modes=0xFF)
    if (fse_modes == 0xFF) {
      // Custom raw u32 mode: data is stored as full u32 arrays
      u32 array_size = num_sequences * sizeof(u32);

      // Bounds check: seq_ctx buffers are allocated for
      // ZSTD_BLOCKSIZE_MAX elements
      if (num_sequences > ZSTD_BLOCKSIZE_MAX) {
        
        return Status::ERROR_BUFFER_TOO_SMALL;
      }

      if (offset + array_size * 3 > input_size) {
        return Status::ERROR_CORRUPT_DATA;
      }

      // Copy literal_lengths
      CUDA_CHECK(cudaMemcpyAsync(seq_ctx->d_literal_lengths, input + offset,
                                 array_size, cudaMemcpyDefault, stream));
      offset += array_size;

      // Copy offsets
      CUDA_CHECK(cudaMemcpyAsync(seq_ctx->d_offsets, input + offset, array_size,
                                 cudaMemcpyDefault, stream));
      offset += array_size;

      // Copy match_lengths
      CUDA_CHECK(cudaMemcpyAsync(seq_ctx->d_match_lengths, input + offset,
                                 array_size, cudaMemcpyDefault, stream));
      offset += array_size;

      // IMPORTANT: Set tier flag for raw offsets
      seq_ctx->is_raw_offsets = true;

      return Status::SUCCESS;
    }

    // Standard ZSTD modes (0-3) - offsets are FSE-encoded with +3 bias
    seq_ctx->is_raw_offsets = false;
    u32 ll_mode = (fse_modes >> 6) & 0x03;
    u32 of_mode = (fse_modes >> 4) & 0x03;
    u32 ml_mode = (fse_modes >> 2) & 0x03;
    const u32 orig_ll_mode = ll_mode;
    const u32 orig_of_mode = of_mode;
    const u32 orig_ml_mode = ml_mode;

    // Kernel config for RLE
    const u32 threads = 256;
    const u32 blocks = (num_sequences + threads - 1) / threads;

    //     //     // for (int i = 0; i < 8 && i < input_size; i++)
    //   fprintf(stderr, "%02x ", h_header[i]);
    // fprintf(stderr, "\n");

    //     // if (input_size >= 8) {
    //       //   for (int i = input_size - 8; i < input_size; i++) {
    //     // Safe access via device read? No, h_header only has 16
    //     bytes.
    //     // We need to read from 'input' (device).
    //     unsigned char b;
    //     CUDA_CHECK(cudaMemcpy(&b, input + i, 1,
    //     cudaMemcpyDeviceToHost)); fprintf(stderr, "%02x ", b);
    //   }
    //   fprintf(stderr, "\n");
    // }
    fse::FSEDecodeTable ll_table_obj = {};
    fse::FSEDecodeTable of_table_obj = {};
    fse::FSEDecodeTable ml_table_obj = {};
    fse::FSEDecodeTable *p_ll_table = nullptr;
    fse::FSEDecodeTable *p_of_table = nullptr;
    fse::FSEDecodeTable *p_ml_table = nullptr;
    u32 max_symbol = 0, table_log = 0;  // For mode 0 predefined tables

    // Helper to process headers logic (Skipping/Parsing)
    auto decode_rle_value = [&](u32 *out_value) -> Status {
      if (offset + 1 > input_size)
        return Status::ERROR_CORRUPT_DATA;
      unsigned char val;
      CUDA_CHECK(cudaMemcpy(&val, input + offset, 1, cudaMemcpyDeviceToHost));
      offset++;
      *out_value = val;
      return Status::SUCCESS;
    };

    u32 ll_rle_value = 0;
    u32 of_rle_value = 0;
    u32 ml_rle_value = 0;
    bool used_prev_tables = false;
    bool use_prev_ll = false;
    bool use_prev_of = false;
    bool use_prev_ml = false;

    // Handle repeat mode using previous tables (RFC 8878)
    if (ll_mode == 3) {
      if (!prev_ll_.has_table) {
        return Status::ERROR_NOT_INITIALIZED;
      }
      ll_mode = prev_ll_.mode;
      if (ll_mode == 1) {
        ll_rle_value = prev_ll_.rle_value;
      } else {
        p_ll_table = &prev_ll_.table;
        use_prev_ll = true;
      }
      used_prev_tables = true;
    }
    if (of_mode == 3) {
      if (!prev_of_.has_table) {
        return Status::ERROR_NOT_INITIALIZED;
      }
      of_mode = prev_of_.mode;
      if (of_mode == 1) {
        of_rle_value = prev_of_.rle_value;
      } else {
        p_of_table = &prev_of_.table;
        use_prev_of = true;
      }
      used_prev_tables = true;
    }
    if (ml_mode == 3) {
      if (!prev_ml_.has_table) {
        return Status::ERROR_NOT_INITIALIZED;
      }
      ml_mode = prev_ml_.mode;
      if (ml_mode == 1) {
        ml_rle_value = prev_ml_.rle_value;
      } else {
        p_ml_table = &prev_ml_.table;
        use_prev_ml = true;
      }
      used_prev_tables = true;
    }

    // 1. Literal Lengths
    if (ll_mode == 0) { // Predefined
      if (use_prev_ll) {
        p_ll_table = &prev_ll_.table;
      } else {
        const u16 *norm = fse::get_predefined_norm(fse::TableType::LITERALS, &max_symbol, &table_log);
        ll_table_obj.table_log = table_log;
        ll_table_obj.table_size = 1u << table_log;
        ll_table_obj.newState = new u16[1u << table_log];
        ll_table_obj.symbol = new u8[1u << table_log];
        ll_table_obj.nbBits = new u8[1u << table_log];
        ll_table_obj.nbAdditionalBits = new u8[1u << table_log];
        ll_table_obj.baseValue = new u32[1u << table_log];
        Status st = fse::FSE_buildDTable_Host(norm, max_symbol, 1u << table_log, ll_table_obj);
        if (st != Status::SUCCESS) return st;
        p_ll_table = &ll_table_obj;
      }
    } else if (ll_mode == 1) {
      if (!use_prev_ll) {
        Status st = decode_rle_value(&ll_rle_value);
        if (st != Status::SUCCESS)
          return st;
      }
    } else if (ll_mode == 2) { // Compressed (Table)
      if (!use_prev_ll) {
      std::vector<u16> normalized_counts;
      u32 max_symbol, table_log, bytes_read;

      // Fix: Copy FSE header to host before parsing
      unsigned char h_header_buf[512];
      u32 copy_size = std::min((u32)sizeof(h_header_buf), input_size - offset);
      CUDA_CHECK(cudaMemcpy(h_header_buf, input + offset, copy_size,
                            cudaMemcpyDeviceToHost));

      Status st =
          fse::read_fse_header(h_header_buf, copy_size, normalized_counts,
                               &max_symbol, &table_log, &bytes_read);
      if (st != Status::SUCCESS)
        return st;
      offset += bytes_read;

      // Pre-allocate arrays for FSE_buildDTable_Host
      ll_table_obj.table_log = table_log;
      ll_table_obj.table_size = 1u << table_log;
       ll_table_obj.newState = new u16[1u << table_log];
       ll_table_obj.symbol = new u8[1u << table_log];
       ll_table_obj.nbBits = new u8[1u << table_log];
       ll_table_obj.nbAdditionalBits = new u8[1u << table_log];
       ll_table_obj.baseValue = new u32[1u << table_log];

      st = fse::FSE_buildDTable_Host(normalized_counts.data(), max_symbol,
                                     1u << table_log, ll_table_obj);
      if (st != Status::SUCCESS)
        return st;
      p_ll_table = &ll_table_obj;
      }
    }

    // 2. Offsets
    if (of_mode == 1) { // RLE
      if (!use_prev_of) {
        Status st = decode_rle_value(&of_rle_value);
        if (st != Status::SUCCESS)
          return st;
      }
    } else if (of_mode == 0) { // Predefined
      if (use_prev_of) {
        p_of_table = &prev_of_.table;
      } else {
        const u16 *norm = fse::get_predefined_norm(fse::TableType::OFFSETS, &max_symbol, &table_log);
        of_table_obj.table_log = table_log;
        of_table_obj.table_size = 1u << table_log;
        of_table_obj.newState = new u16[1u << table_log];
        of_table_obj.symbol = new u8[1u << table_log];
        of_table_obj.nbBits = new u8[1u << table_log];
        of_table_obj.nbAdditionalBits = new u8[1u << table_log];
        of_table_obj.baseValue = new u32[1u << table_log];
        Status st = fse::FSE_buildDTable_Host(norm, max_symbol, 1u << table_log, of_table_obj);
        if (st != Status::SUCCESS) return st;
        p_of_table = &of_table_obj;
      }
    } else if (of_mode == 2) { // Compressed
      if (!use_prev_of) {
      std::vector<u16> normalized_counts;
      u32 max_symbol, table_log, bytes_read;

      // Fix: Copy FSE header to host before parsing
      unsigned char h_header_buf[512];
      u32 copy_size = std::min((u32)sizeof(h_header_buf), input_size - offset);
      CUDA_CHECK(cudaMemcpy(h_header_buf, input + offset, copy_size,
                            cudaMemcpyDeviceToHost));

      Status st =
          fse::read_fse_header(h_header_buf, copy_size, normalized_counts,
                               &max_symbol, &table_log, &bytes_read);
      if (st != Status::SUCCESS)
        return st;
      offset += bytes_read;

      // Pre-allocate arrays for FSE_buildDTable_Host
      of_table_obj.table_log = table_log;
      of_table_obj.table_size = 1u << table_log;
      of_table_obj.newState = new u16[1u << table_log];
      of_table_obj.symbol = new u8[1u << table_log];
      of_table_obj.nbBits = new u8[1u << table_log];
      of_table_obj.nbAdditionalBits = new u8[1u << table_log];
      of_table_obj.baseValue = new u32[1u << table_log];

      st = fse::FSE_buildDTable_Host(normalized_counts.data(), max_symbol,
                                     1u << table_log, of_table_obj);
      if (st != Status::SUCCESS)
        return st;
      p_of_table = &of_table_obj;
      }
    }

    // 3. Match Lengths
    if (ml_mode == 1) { // RLE
      if (!use_prev_ml) {
        Status st = decode_rle_value(&ml_rle_value);
        if (st != Status::SUCCESS)
          return st;
      }
    } else if (ml_mode == 0) { // Predefined
      if (use_prev_ml) {
        p_ml_table = &prev_ml_.table;
      } else {
        const u16 *norm = fse::get_predefined_norm(fse::TableType::MATCH_LENGTHS, &max_symbol, &table_log);
        ml_table_obj.table_log = table_log;
        ml_table_obj.table_size = 1u << table_log;
        ml_table_obj.newState = new u16[1u << table_log];
        ml_table_obj.symbol = new u8[1u << table_log];
        ml_table_obj.nbBits = new u8[1u << table_log];
        ml_table_obj.nbAdditionalBits = new u8[1u << table_log];
        ml_table_obj.baseValue = new u32[1u << table_log];
        Status st = fse::FSE_buildDTable_Host(norm, max_symbol, 1u << table_log, ml_table_obj);
        if (st != Status::SUCCESS) return st;
        p_ml_table = &ml_table_obj;
      }
    } else if (ml_mode == 2) { // Compressed
      if (!use_prev_ml) {
      std::vector<u16> normalized_counts;
      u32 max_symbol, table_log, bytes_read;

      // Fix: Copy FSE header to host before parsing
      unsigned char h_header_buf[512];
      u32 copy_size = std::min((u32)sizeof(h_header_buf), input_size - offset);
      CUDA_CHECK(cudaMemcpy(h_header_buf, input + offset, copy_size,
                            cudaMemcpyDeviceToHost));

      Status st =
          fse::read_fse_header(h_header_buf, copy_size, normalized_counts,
                               &max_symbol, &table_log, &bytes_read);
      if (st != Status::SUCCESS)
        return st;
      offset += bytes_read;

      // Pre-allocate arrays for FSE_buildDTable_Host
      ml_table_obj.table_log = table_log;
      ml_table_obj.table_size = 1u << table_log;
      ml_table_obj.newState = new u16[1u << table_log];
      ml_table_obj.symbol = new u8[1u << table_log];
      ml_table_obj.nbBits = new u8[1u << table_log];
      ml_table_obj.nbAdditionalBits = new u8[1u << table_log];
      ml_table_obj.baseValue = new u32[1u << table_log];

      st = fse::FSE_buildDTable_Host(normalized_counts.data(), max_symbol,
                                     1u << table_log, ml_table_obj);
      if (st != Status::SUCCESS)
        return st;
      p_ml_table = &ml_table_obj;
      }
    }

    // Bounds check
    if (num_sequences > ZSTD_BLOCKSIZE_MAX) {
      return Status::ERROR_BUFFER_TOO_SMALL;
    }

    // 4. Decode FSE Streams (Interleaved)
    // Pass pointer to remaining bitstream (after tables/RLE bytes)
    // Predefined (Mode 0) also handled here (no header, just stream)

    // Check if we need FSE decoding (if any mode is 0 or 2)
    // Actually, decode_sequences_interleaved handles flags internally
    // or via modes. If Mode is RLE (1), decode_sequences_interleaved
    // skips it.

    if (ml_mode == 0 && p_ml_table) {
      // Predefined verification logic removed
    }

    Status status = fse::decode_sequences_interleaved(
        input + offset, input_size - offset, num_sequences,
        seq_ctx->d_literal_lengths, seq_ctx->d_offsets,
        seq_ctx->d_match_lengths, orig_ll_mode, orig_of_mode, orig_ml_mode,
        p_ll_table, p_of_table, p_ml_table, total_literal_count, stream);

    if (!used_prev_tables) {
      if (ll_mode == 1) {
        prev_ll_.rle_value = static_cast<u8>(ll_rle_value);
        prev_ll_.mode = ll_mode;
        prev_ll_.has_table = true;
      } else if (ll_mode == 0 || ll_mode == 2) {
        Status st = copy_fse_decode_table(ll_table_obj, prev_ll_.table);
        if (st == Status::SUCCESS) {
          prev_ll_.rle_value = 0;
          prev_ll_.mode = ll_mode;
          prev_ll_.has_table = true;
        }
      }

      if (of_mode == 1) {
        prev_of_.rle_value = static_cast<u8>(of_rle_value);
        prev_of_.mode = of_mode;
        prev_of_.has_table = true;
      } else if (of_mode == 0 || of_mode == 2) {
        Status st = copy_fse_decode_table(of_table_obj, prev_of_.table);
        if (st == Status::SUCCESS) {
          prev_of_.rle_value = 0;
          prev_of_.mode = of_mode;
          prev_of_.has_table = true;
        }
      }

      if (ml_mode == 1) {
        prev_ml_.rle_value = static_cast<u8>(ml_rle_value);
        prev_ml_.mode = ml_mode;
        prev_ml_.has_table = true;
      } else if (ml_mode == 0 || ml_mode == 2) {
        Status st = copy_fse_decode_table(ml_table_obj, prev_ml_.table);
        if (st == Status::SUCCESS) {
          prev_ml_.rle_value = 0;
          prev_ml_.mode = ml_mode;
          prev_ml_.has_table = true;
        }
      }
    }

    if (!used_prev_tables) {
      if (p_ll_table) {
        if (p_ll_table->newState) delete[] p_ll_table->newState;
        if (p_ll_table->symbol) delete[] p_ll_table->symbol;
        if (p_ll_table->nbBits) delete[] p_ll_table->nbBits;
        if (p_ll_table->nbAdditionalBits) delete[] p_ll_table->nbAdditionalBits;
        if (p_ll_table->baseValue) delete[] p_ll_table->baseValue;
      }
      if (p_of_table) {
        if (p_of_table->newState) delete[] p_of_table->newState;
        if (p_of_table->symbol) delete[] p_of_table->symbol;
        if (p_of_table->nbBits) delete[] p_of_table->nbBits;
        if (p_of_table->nbAdditionalBits) delete[] p_of_table->nbAdditionalBits;
        if (p_of_table->baseValue) delete[] p_of_table->baseValue;
      }
      if (p_ml_table) {
        if (p_ml_table->newState) delete[] p_ml_table->newState;
        if (p_ml_table->symbol) delete[] p_ml_table->symbol;
        if (p_ml_table->nbBits) delete[] p_ml_table->nbBits;
        if (p_ml_table->nbAdditionalBits) delete[] p_ml_table->nbAdditionalBits;
        if (p_ml_table->baseValue) delete[] p_ml_table->baseValue;
      }
    }

    return status;
  }

  
};

// ==============================================================================
// BATCH MANAGER IMPLEMENTATION
// ==============================================================================

class ZstdBatchManager::Impl {
public:
  std::unique_ptr<ZstdManager> manager;
  CompressionStats batch_stats;

  // ---  Stream pool for parallel batching ---
  std::vector<cudaStream_t> streams;
  int num_streams;
  // --- (END NEW) ---

  explicit Impl(const CompressionConfig &config) {
    manager = create_manager(config);

    // ---  Create stream pool ---
    num_streams = 8; // Default pool size
    for (int i = 0; i < num_streams; ++i) {
      cudaStream_t s;
      cudaError_t err = cudaStreamCreate(&s);
      if (err != cudaSuccess) {
        // Clean up already created streams
        for (auto created_s : streams) {
          cudaStreamDestroy(created_s);
        }
        streams.clear();
        // Don't throw for now, just log. The Manager might still work
        // with defaults or fail later cleanly.
        break;
      }
      streams.push_back(s);
    }
    // --- (END NEW) ---
  }

  // ---  Destructor to clean up streams ---
  ~Impl() {
    for (auto s : streams) {
      cudaStreamDestroy(s);
    }
  }
  // --- (END NEW) ---

  void reset_stats() {
    memset(&batch_stats, 0, sizeof(CompressionStats));
    manager->reset_stats();
  }
};

ZstdBatchManager::ZstdBatchManager() {
  pimpl_ = std::make_unique<Impl>(
      CompressionConfig::from_level(CUDA_ZSTD_DEFAULT_CLEVEL));
}

ZstdBatchManager::ZstdBatchManager(const CompressionConfig &config) {
  pimpl_ = std::make_unique<Impl>(config);
}

ZstdBatchManager::~ZstdBatchManager() {
  //  Sync ONLY our own streams before destruction, NOT the entire
  // device. cudaDeviceSynchronize() causes race conditions in
  // multi-manager scenarios because it blocks ALL GPU operations across
  // ALL threads/managers.
  if (pimpl_) {
    for (auto s : pimpl_->streams) {
      if (s) {
        cudaStreamSynchronize(s);
      }
    }
  }
  // Clear any sticky CUDA errors from this compression session
  (void)cudaGetLastError();
}

Status ZstdBatchManager::configure(const CompressionConfig &config) {
  return pimpl_->manager->configure(config);
}

CompressionConfig ZstdBatchManager::get_config() const {
  return pimpl_->manager->get_config();
}

Status ZstdBatchManager::set_compression_level(int level) {
  return pimpl_->manager->set_compression_level(level);
}

int ZstdBatchManager::get_compression_level() const {
  return pimpl_->manager->get_compression_level();
}

Status ZstdBatchManager::set_dictionary(const dictionary::Dictionary &dict) {
  return pimpl_->manager->set_dictionary(dict);
}

Status ZstdBatchManager::get_dictionary(dictionary::Dictionary &dict) const {
  return pimpl_->manager->get_dictionary(dict);
}

Status ZstdBatchManager::clear_dictionary() {
  return pimpl_->manager->clear_dictionary();
}

const CompressionStats &ZstdBatchManager::get_stats() const {
  return pimpl_->batch_stats;
}

void ZstdBatchManager::reset_stats() { pimpl_->reset_stats(); }

size_t
ZstdBatchManager::get_compress_temp_size(size_t uncompressed_size) const {
  return pimpl_->manager->get_compress_temp_size(uncompressed_size);
}

size_t
ZstdBatchManager::get_decompress_temp_size(size_t compressed_size) const {
  return pimpl_->manager->get_decompress_temp_size(compressed_size);
}

size_t
ZstdBatchManager::get_max_compressed_size(size_t uncompressed_size) const {
  return pimpl_->manager->get_max_compressed_size(uncompressed_size);
}

size_t ZstdBatchManager::get_batch_compress_temp_size(
    const std::vector<size_t> &uncompressed_sizes) const {
  // --- (MODIFIED) ---
  // The total workspace is the sum of max workspace per item,
  // as we partition it.
  size_t total_workspace = 0;
  for (const auto &size : uncompressed_sizes) {
    total_workspace += pimpl_->manager->get_compress_temp_size(size);
  }
  return total_workspace;
  // --- (END MODIFIED) ---
}

size_t ZstdBatchManager::get_batch_decompress_temp_size(
    const std::vector<size_t> &compressed_sizes) const {
  // --- (MODIFIED) ---
  size_t max_item_temp_size = 0;
  for (const auto &size : compressed_sizes) {
    max_item_temp_size = std::max(
        max_item_temp_size, pimpl_->manager->get_decompress_temp_size(size));
  }
  // Align to 128 bytes for safety
  max_item_temp_size = (max_item_temp_size + 127) & ~127;
  return max_item_temp_size * compressed_sizes.size();
  // --- (END MODIFIED) ---
}

Status ZstdBatchManager::compress(const void *uncompressed_data,
                                  size_t uncompressed_size,
                                  void *compressed_data,
                                  size_t *compressed_size, void *temp_workspace,
                                  size_t temp_size, const void *dict_buffer,
                                  size_t dict_size, cudaStream_t stream,
                                  void *streaming_context) {
  return pimpl_->manager->compress(
      uncompressed_data, uncompressed_size, compressed_data, compressed_size,
      temp_workspace, temp_size, dict_buffer, dict_size, stream, streaming_context);
}

Status ZstdBatchManager::decompress(const void *compressed_data,
                                    size_t compressed_size,
                                    void *uncompressed_data,
                                    size_t *uncompressed_size,
                                    void *temp_workspace, size_t temp_size,
                                    cudaStream_t stream) {
  return pimpl_->manager->decompress(compressed_data, compressed_size,
                                     uncompressed_data, uncompressed_size,
                                     temp_workspace, temp_size, stream);
}

Status ZstdBatchManager::compress_batch(const std::vector<BatchItem> &items,
                                        void *temp_workspace, size_t temp_size,
                                        cudaStream_t stream) {
  // ---  OPTIMIZED PIPELINE IMPLEMENTATION ---
  pimpl_->manager->reset_stats();
  bool all_success = true;

  if (items.empty())
    return Status::SUCCESS;

  std::vector<size_t> uncompressed_sizes;
  for (const auto &item : items) {
    uncompressed_sizes.push_back(item.input_size);
  }
  size_t required_size = get_batch_compress_temp_size(uncompressed_sizes);
  if (temp_size < required_size) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  // Create events for pipeline synchronization
  std::vector<cudaEvent_t> transfer_complete_events(items.size());
  std::vector<cudaEvent_t> compute_complete_events(items.size());

  for (size_t i = 0; i < items.size(); ++i) {
    cudaEventCreate(&transfer_complete_events[i]);
    cudaEventCreate(&compute_complete_events[i]);
  }

  //  PIPELINED EXECUTION: Overlap H2D, Compute, D2H
  for (size_t i = 0; i < items.size(); ++i) {
    auto &item = const_cast<std::vector<BatchItem> &>(items)[i];

    // Select streams for 3-stage pipeline
    cudaStream_t h2d_stream = pimpl_->streams[0];     // H2D transfer
    cudaStream_t compute_stream = pimpl_->streams[1]; // Kernel execution
    cudaStream_t d2h_stream = pimpl_->streams[2];     // D2H transfer

    // Partition the workspace
    size_t item_workspace_size =
        pimpl_->manager->get_compress_temp_size(item.input_size);
    unsigned char *item_workspace =
        static_cast<unsigned char *>(temp_workspace) + i * item_workspace_size;

    // Pipeline stage 1: H2D transfer (async, overlapped with previous
    // compute) Note: Input already on device in most cases, but this
    // shows the pattern

    // Pipeline stage 2: Compression (overlapped with H2D of next block)
    item.status = pimpl_->manager->compress(
        item.input_ptr, item.input_size, item.output_ptr, &item.output_size,
        item_workspace, item_workspace_size, nullptr, 0,
        compute_stream, // Use dedicated compute stream
        nullptr         // No streaming context for batch
    );

    if (item.status != Status::SUCCESS) {
      all_success = false;
    }

    // Record event when compression completes
    cudaEventRecord(compute_complete_events[i], compute_stream);

    if (item.status != Status::SUCCESS) {
      all_success = false;
    }
  }

  //  Only synchronize at the end - all streams execute in parallel
  for (auto s : pimpl_->streams) {
    cudaStreamSynchronize(s);
  }

  // Cleanup events
  for (size_t i = 0; i < items.size(); ++i) {
    cudaEventDestroy(transfer_complete_events[i]);
    cudaEventDestroy(compute_complete_events[i]);
  }

  pimpl_->batch_stats = pimpl_->manager->get_stats();

  return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
  // --- (END NEW PIPELINE) ---
}

Status ZstdBatchManager::decompress_batch(const std::vector<BatchItem> &items,
                                          void *temp_workspace,
                                          size_t temp_size,
                                          cudaStream_t stream) {
  // --- (START REPLACEMENT) ---
  pimpl_->manager->reset_stats();
  bool all_success = true;
  int stream_idx = 0;

  if (items.empty())
    return Status::SUCCESS;

  // Calculate the size of a single item's workspace
  size_t max_item_temp_size = 0;
  for (const auto &item : items) {
    max_item_temp_size =
        std::max(max_item_temp_size,
                 pimpl_->manager->get_decompress_temp_size(item.input_size));
  }
  max_item_temp_size = (max_item_temp_size + 127) & ~127; // Align

  if (temp_size < max_item_temp_size * items.size()) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  for (size_t i = 0; i < items.size(); ++i) {
    auto &item = const_cast<std::vector<BatchItem> &>(items)[i];

    cudaStream_t item_stream =
        pimpl_->streams[stream_idx % pimpl_->num_streams];
    stream_idx++;

    unsigned char *item_workspace =
        static_cast<unsigned char *>(temp_workspace) + i * max_item_temp_size;

    item.status = pimpl_->manager->decompress(
        item.input_ptr, item.input_size, item.output_ptr, &item.output_size,
        item_workspace, max_item_temp_size, item_stream);
    if (item.status != Status::SUCCESS) {
      all_success = false;
    }
  }

  // Wait for all pooled streams to finish
  for (auto s : pimpl_->streams) {
    cudaStreamSynchronize(s);
  }

  pimpl_->batch_stats = pimpl_->manager->get_stats();

  return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
  // --- (END REPLACEMENT) ---
}

// ==============================================================================
// INFERENCE-READY API IMPLEMENTATION (Holographic/JIT Inference)
// ==============================================================================
// These methods enable zero-malloc decompression for LLM inference by
// reusing pre-allocated "Zipper Buffers" that rotate during layer-wise
// processing.
// ==============================================================================

Status ZstdBatchManager::decompress_to_preallocated(
    const void *compressed_data, size_t compressed_size,
    void *preallocated_output, size_t output_capacity,
    size_t *actual_output_size, void *temp_workspace, size_t temp_size,
    cudaStream_t stream) {

  if (!compressed_data || !preallocated_output || !actual_output_size) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (output_capacity == 0) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  // Use the single-item decompress path with the provided output buffer
  // The key difference: we DON'T allocate, we use the caller's buffer
  size_t decompressed_size = output_capacity;

  Status result = pimpl_->manager->decompress(
      compressed_data, compressed_size, preallocated_output, &decompressed_size,
      temp_workspace, temp_size, stream);

  if (result == Status::SUCCESS) {
    *actual_output_size = decompressed_size;
  }

  return result;
}

Status ZstdBatchManager::decompress_batch_preallocated(
    std::vector<BatchItem> &items, void *temp_workspace, size_t temp_size,
    cudaStream_t stream) {

  pimpl_->manager->reset_stats();
  bool all_success = true;
  int stream_idx = 0;

  if (items.empty())
    return Status::SUCCESS;

  // Calculate the size of a single item's workspace
  size_t max_item_temp_size = 0;
  for (const auto &item : items) {
    // Validate that output_ptr is pre-allocated (non-null)
    if (item.output_ptr == nullptr) {
      return Status::ERROR_INVALID_PARAMETER;
    }
    max_item_temp_size =
        std::max(max_item_temp_size,
                 pimpl_->manager->get_decompress_temp_size(item.input_size));
  }
  max_item_temp_size = (max_item_temp_size + 127) & ~127; // Align

  if (temp_size < max_item_temp_size * items.size()) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  for (size_t i = 0; i < items.size(); ++i) {
    auto &item = items[i];

    cudaStream_t item_stream =
        pimpl_->streams[stream_idx % pimpl_->num_streams];
    stream_idx++;

    unsigned char *item_workspace =
        static_cast<unsigned char *>(temp_workspace) + i * max_item_temp_size;

    // Decompress directly into the pre-allocated output buffer
    item.status = pimpl_->manager->decompress(
        item.input_ptr, item.input_size, item.output_ptr,
        &item.output_size, // output_ptr is pre-allocated
        item_workspace, max_item_temp_size, item_stream);

    if (item.status != Status::SUCCESS) {
      all_success = false;
    }
  }

  // Wait for all pooled streams to finish
  for (auto s : pimpl_->streams) {
    cudaStreamSynchronize(s);
  }

  pimpl_->batch_stats = pimpl_->manager->get_stats();

  return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
}

Status ZstdBatchManager::decompress_async_no_sync(
    const void *compressed_data, size_t compressed_size,
    void *preallocated_output, size_t output_capacity,
    size_t *d_actual_size, // Device pointer for async size write
    void *temp_workspace, size_t temp_size, cudaStream_t stream) {

  if (!compressed_data || !preallocated_output) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (output_capacity == 0) {
    return Status::ERROR_BUFFER_TOO_SMALL;
  }

  // For async operation, we need a host-side size variable first
  size_t decompressed_size = output_capacity;

  // Launch decompression on the provided stream
  Status result = pimpl_->manager->decompress(
      compressed_data, compressed_size, preallocated_output, &decompressed_size,
      temp_workspace, temp_size, stream);

  // If caller provided a device pointer for size, write it
  // asynchronously
  if (result == Status::SUCCESS && d_actual_size != nullptr) {
    cudaMemcpyAsync(d_actual_size, &decompressed_size, sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    // CRITICAL: Synchronize to ensure decompressed_size is copied
    // before it goes out of scope. In a future version, we should use
    // pinned memory for true async.
    cudaStreamSynchronize(stream);
  }

  // NOTE: We do NOT synchronize here - caller is responsible for sync
  // This enables pipelining: decompress layer N+1 while computing layer
  // N

  return result;
}

// ==============================================================================
// INFERENCE UTILITY METHODS
// ==============================================================================

size_t
ZstdBatchManager::get_inference_workspace_size(size_t max_compressed_size,
                                               size_t max_output_size) const {

  // Workspace needs to accommodate decompression temp buffers
  size_t decompress_temp =
      pimpl_->manager->get_decompress_temp_size(max_compressed_size);

  // Add some alignment padding
  decompress_temp = (decompress_temp + 255) & ~255;

  return decompress_temp;
}

Status ZstdBatchManager::allocate_inference_workspace(
    size_t max_compressed_size, size_t max_output_size, void **workspace_ptr,
    size_t *workspace_size) {

  if (!workspace_ptr || !workspace_size) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  size_t required_size =
      get_inference_workspace_size(max_compressed_size, max_output_size);

  cudaError_t err = cudaMalloc(workspace_ptr, required_size);
  if (err != cudaSuccess) {
    *workspace_ptr = nullptr;
    *workspace_size = 0;
    return Status::ERROR_ALLOCATION_FAILED;
  }

  *workspace_size = required_size;
  return Status::SUCCESS;
}

Status ZstdBatchManager::free_inference_workspace(void *workspace_ptr) {
  if (workspace_ptr) {
    cudaError_t err = cudaFree(workspace_ptr);
    if (err != cudaSuccess) {
      return Status::ERROR_GENERIC;
    }
  }
  return Status::SUCCESS;
}

// ==============================================================================
// STREAMING MANAGER IMPLEMENTATION
// ==============================================================================

class ZstdStreamingManager::Impl {
public:
  CompressionConfig config;
  dictionary::Dictionary dict;
  bool has_dictionary;
  cudaStream_t stream;
  bool owns_stream;

  std::unique_ptr<ZstdManager> manager;
  StreamingContext streaming_ctx;

  void *d_workspace;
  size_t workspace_size;

  bool comp_initialized;
  bool decomp_initialized;
  bool frame_header_parsed;

  // History / Streaming state
  size_t history_capacity;     // Size of d_window_history
  size_t current_history_size; // Bytes currently in history

  explicit Impl(const CompressionConfig &cfg)
      : config(cfg), has_dictionary(false), stream(nullptr), owns_stream(false),
        d_workspace(nullptr), workspace_size(0), comp_initialized(false),
        decomp_initialized(false), frame_header_parsed(false),
        history_capacity(0), current_history_size(0) {
    manager = create_manager(config);
  }

  ~Impl() {
    cleanup_streaming_context();
    if (d_workspace) {
      cudaFree(d_workspace);
    }
  }

  void cleanup_streaming_context() {
    if (streaming_ctx.d_window_history) {
      cudaFree(streaming_ctx.d_window_history);
      streaming_ctx.d_window_history = nullptr;
    }
    if (streaming_ctx.d_hash_table_state) {
      cudaFree(streaming_ctx.d_hash_table_state);
      streaming_ctx.d_hash_table_state = nullptr;
    }
    if (streaming_ctx.d_chain_table_state) {
      cudaFree(streaming_ctx.d_chain_table_state);
      streaming_ctx.d_chain_table_state = nullptr;
    }
    if (streaming_ctx.d_xxhash_state) {
      cudaFree(streaming_ctx.d_xxhash_state);
      streaming_ctx.d_xxhash_state = nullptr;
    }
    if (streaming_ctx.ldm_initialized) {
      ldm::ldm_cleanup_context(streaming_ctx.ldm_ctx);
      streaming_ctx.ldm_initialized = false;
    }
  }

  Status alloc_workspace(cudaStream_t s, size_t max_input_size = 0) {
    stream = s;
    size_t chunk_size =
        (max_input_size > 0) ? max_input_size : ZSTD_BLOCKSIZE_MAX;
    size_t comp_size = manager->get_compress_temp_size(chunk_size);
    size_t decomp_size = manager->get_decompress_temp_size(chunk_size * 2);
    size_t required_size = std::max(comp_size, decomp_size);

    // Only allocate if not already allocated or size is insufficient
    if (d_workspace && workspace_size >= required_size) {
      return Status::SUCCESS; // Reuse existing workspace
    }

    // Free old workspace if exists
    if (d_workspace) {
      cudaFree(d_workspace);
      d_workspace = nullptr;
    }

    workspace_size = required_size;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    return Status::SUCCESS;
  }
};

ZstdStreamingManager::ZstdStreamingManager() {
  pimpl_ = std::make_unique<Impl>(
      CompressionConfig::from_level(CUDA_ZSTD_DEFAULT_CLEVEL));
}

ZstdStreamingManager::ZstdStreamingManager(const CompressionConfig &config) {
  pimpl_ = std::make_unique<Impl>(config);
}

ZstdStreamingManager::~ZstdStreamingManager() = default;

Status ZstdStreamingManager::set_config(const CompressionConfig &config) {
  if (pimpl_->comp_initialized || pimpl_->decomp_initialized) {
    return Status::ERROR_GENERIC;
  }
  pimpl_->config = config;
  return pimpl_->manager->configure(config);
}

CompressionConfig ZstdStreamingManager::get_config() const {
  return pimpl_->config;
}

Status
ZstdStreamingManager::set_dictionary(const dictionary::Dictionary &dict) {
  pimpl_->dict = dict;
  pimpl_->has_dictionary = true;
  return pimpl_->manager->set_dictionary(dict);
}

Status ZstdStreamingManager::init_compression(cudaStream_t stream,
                                              size_t max_chunk_size) {
  if (pimpl_->comp_initialized)
    return Status::SUCCESS;

  Status s = pimpl_->alloc_workspace(stream, max_chunk_size);
  if (s != Status::SUCCESS)
    return s;

  // Pre-allocate FSE tables for high throughput
  s = pimpl_->manager->preallocate_tables(stream);
  if (s != Status::SUCCESS)
    return s;

  pimpl_->comp_initialized = true;
  return Status::SUCCESS;
}

Status ZstdStreamingManager::init_compression_with_history(cudaStream_t stream,
                                                           size_t max_chunk_size) {
  if (pimpl_->comp_initialized)
    return Status::SUCCESS;

  Status s = pimpl_->alloc_workspace(stream, max_chunk_size);
  if (s != Status::SUCCESS)
    return s;

  // Allocate history buffer
  size_t window_size = 1ULL << pimpl_->config.window_log;
  pimpl_->history_capacity = window_size;

  if (pimpl_->streaming_ctx.d_window_history) {
    cudaFree(pimpl_->streaming_ctx.d_window_history);
    pimpl_->streaming_ctx.d_window_history = nullptr;
  }

  if (cudaMalloc(&pimpl_->streaming_ctx.d_window_history, window_size) !=
      cudaSuccess) {
    return Status::ERROR_OUT_OF_MEMORY;
  }

  // Allocate persistent hash/chain state for true streaming
  size_t hash_size = (1ULL << pimpl_->config.hash_log) * sizeof(u32);
  size_t chain_size = (1ULL << pimpl_->config.chain_log) * sizeof(u32);

  if (cudaMalloc(&pimpl_->streaming_ctx.d_hash_table_state, hash_size) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;
  if (cudaMalloc(&pimpl_->streaming_ctx.d_chain_table_state, chain_size) != cudaSuccess)
    return Status::ERROR_OUT_OF_MEMORY;

  // Clear state
  cudaMemsetAsync(pimpl_->streaming_ctx.d_window_history, 0, window_size, stream);
  cudaMemsetAsync(pimpl_->streaming_ctx.d_hash_table_state, 0xFF, hash_size, stream);
  cudaMemsetAsync(pimpl_->streaming_ctx.d_chain_table_state, 0xFF, chain_size, stream);

    pimpl_->current_history_size = 0;
    pimpl_->comp_initialized = true;

    // Initialize LDM if enabled
    if (pimpl_->config.enable_ldm) {
        ldm::ldm_init_context(pimpl_->streaming_ctx.ldm_ctx, pimpl_->config.ldm_hash_log);
        pimpl_->streaming_ctx.ldm_initialized = true;
    } else {
        pimpl_->streaming_ctx.ldm_initialized = false;
    }

    // Pre-allocate FSE tables
    s = pimpl_->manager->preallocate_tables(stream);
  return s;
}

Status ZstdStreamingManager::init_decompression(cudaStream_t stream) {
  if (pimpl_->decomp_initialized)
    reset();

  auto status = pimpl_->alloc_workspace(stream);
  if (status != Status::SUCCESS)
    return status;

  pimpl_->decomp_initialized = true;
  return Status::SUCCESS;
}

Status ZstdStreamingManager::reset() {
  pimpl_->cleanup_streaming_context();
  if (pimpl_->d_workspace) {
    cudaFree(pimpl_->d_workspace);
    pimpl_->d_workspace = nullptr;
  }
  pimpl_->workspace_size = 0;
  pimpl_->comp_initialized = false;
  pimpl_->decomp_initialized = false;
  return Status::SUCCESS;
}

Status ZstdStreamingManager::flush(cudaStream_t stream) {
  return Status::SUCCESS;
}

size_t ZstdStreamingManager::get_temp_size() const {
  size_t comp_size =
      pimpl_->manager->get_compress_temp_size(ZSTD_BLOCKSIZE_MAX);
  size_t decomp_size =
      pimpl_->manager->get_decompress_temp_size(ZSTD_BLOCKSIZE_MAX * 2);
  return std::max(comp_size, decomp_size);
}

bool ZstdStreamingManager::is_compression_initialized() const {
  return pimpl_->comp_initialized;
}

bool ZstdStreamingManager::is_decompression_initialized() const {
  return pimpl_->decomp_initialized;
}

Status ZstdStreamingManager::compress_chunk(const void *input,
                                            size_t input_size, void *output,
                                            size_t *output_size,
                                            bool is_last_chunk,
                                            cudaStream_t stream) {
  if (!input || !output || !output_size) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Initialize output size to 0 in case of early return
  *output_size = 0;

  // Calculate required workspace size for this input
  size_t required_temp = pimpl_->manager->get_compress_temp_size(input_size);

  // Allocate or reallocate workspace if needed
  if (!pimpl_->d_workspace || pimpl_->workspace_size < required_temp) {
    if (pimpl_->d_workspace) {
      cudaFree(pimpl_->d_workspace);
      pimpl_->d_workspace = nullptr;
    }
    pimpl_->workspace_size = required_temp;
    if (cudaMalloc(&pimpl_->d_workspace, pimpl_->workspace_size) !=
        cudaSuccess) {
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }
  pimpl_->comp_initialized = true;

  // Delegate to the underlying manager's compress function
  // Each chunk is compressed as a complete ZSTD frame
  // BUG compressed_size MUST be initialized to output buffer
  // capacity
  size_t compressed_size = pimpl_->manager->get_max_compressed_size(input_size);
  Status status = pimpl_->manager->compress(
      input, input_size, output, &compressed_size, pimpl_->d_workspace,
      pimpl_->workspace_size, nullptr, 0, // No dictionary for now
      stream, nullptr);

  if (status != Status::SUCCESS) {
    return status;
  }

  *output_size = compressed_size;

  // If this is the last chunk, we could flush any internal state
  if (is_last_chunk) {
    // Currently no additional action needed - each chunk is a complete
    // frame
  }

  return Status::SUCCESS;
}

Status ZstdStreamingManager::compress_chunk_with_history(
    const void *input, size_t input_size, void *output, size_t *output_size,
    bool is_last_chunk, cudaStream_t stream) {

  if (!input || !output || !output_size) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Initialize output size to 0
  *output_size = 0;

  // Auto-initialize if needed
  if (!pimpl_->comp_initialized) {
    Status s = init_compression_with_history(stream, input_size);
    if (s != Status::SUCCESS)
      return s;
  }

  // Ensure history buffer exists
  if (!pimpl_->streaming_ctx.d_window_history) {
    return Status::ERROR_NOT_INITIALIZED;
  }

  // Workspace management
  size_t required_temp = pimpl_->manager->get_compress_temp_size(input_size);

  // Allocate or reallocate workspace if needed
  if (!pimpl_->d_workspace || pimpl_->workspace_size < required_temp) {
    if (pimpl_->d_workspace) {
      cudaFree(pimpl_->d_workspace);
      pimpl_->d_workspace = nullptr;
    }
    pimpl_->workspace_size = required_temp;
    if (cudaMalloc(&pimpl_->d_workspace, pimpl_->workspace_size) !=
        cudaSuccess) {
      return Status::ERROR_OUT_OF_MEMORY;
    }
  }

  // Compress chunk using history and persistent state
  size_t compressed_size = pimpl_->manager->get_max_compressed_size(input_size);
  Status status = pimpl_->manager->compress(
      input, input_size, output, &compressed_size, pimpl_->d_workspace,
      pimpl_->workspace_size, pimpl_->streaming_ctx.d_window_history,
      pimpl_->current_history_size, stream, &pimpl_->streaming_ctx);

  if (status != Status::SUCCESS) {
    return status;
  }

  *output_size = compressed_size;

  // Update History (Sliding Window)
  if (input_size >= pimpl_->history_capacity) {
    // Input is larger than history capacity: keep only the last chunk
    const char *src =
        (const char *)input + input_size - pimpl_->history_capacity;
    cudaMemcpyAsync(pimpl_->streaming_ctx.d_window_history, src,
                    pimpl_->history_capacity, cudaMemcpyDeviceToDevice, stream);
    pimpl_->current_history_size = pimpl_->history_capacity;
  } else {
    size_t remaining = pimpl_->history_capacity - pimpl_->current_history_size;

    if (input_size <= remaining) {
      // Append to history
      char *dst = (char *)pimpl_->streaming_ctx.d_window_history +
                  pimpl_->current_history_size;
      cudaMemcpyAsync(dst, input, input_size, cudaMemcpyDeviceToDevice, stream);
      pimpl_->current_history_size += input_size;
    } else {
      // Shift and append
      size_t keep = pimpl_->history_capacity - input_size;
      char *history_base = (char *)pimpl_->streaming_ctx.d_window_history;
      char *src_move = history_base + (pimpl_->current_history_size - keep);

      // Use workspace as scratch for safe shift (no overlap)
      cudaMemcpyAsync(pimpl_->d_workspace, src_move, keep,
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(history_base, pimpl_->d_workspace, keep,
                      cudaMemcpyDeviceToDevice, stream);

      // Append new data
      char *dst_append = history_base + keep;
      cudaMemcpyAsync(dst_append, input, input_size, cudaMemcpyDeviceToDevice,
                      stream);

      pimpl_->current_history_size = pimpl_->history_capacity;
    }
  }

  return Status::SUCCESS;
}

Status ZstdStreamingManager::decompress_chunk(const void *input,
                                              size_t input_size, void *output,
                                              size_t *output_size,
                                              bool *is_last_chunk,
                                              cudaStream_t stream) {
  if (!input || !output || !output_size || !is_last_chunk) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Initialize outputs
  // *output_size = 0; // BUG Don't clear output size, it contains
  // capacity
  *is_last_chunk = true; // Each chunk is currently a complete frame

  // Ensure decompression is initialized
  if (!pimpl_->decomp_initialized) {
    auto status = init_decompression(stream);
    if (status != Status::SUCCESS) {
      return status;
    }
  }

  // Delegate to the underlying manager's decompress function
  size_t decompressed_size = *output_size;
  Status status = pimpl_->manager->decompress(
      input, input_size, output, &decompressed_size, pimpl_->d_workspace,
      pimpl_->workspace_size, stream);

  if (status != Status::SUCCESS) {
    return status;
  }

  *output_size = decompressed_size;

  return Status::SUCCESS;
}

// ==============================================================================
// FACTORY FUNCTIONS
// ==============================================================================

// ==============================================================================
// SMART PATH SELECTOR IMPLEMENTATION
// ==============================================================================

ZstdManager::ExecutionPath
ZstdManager::select_execution_path(size_t size, int cpu_threshold) {
  // Use GPU path for all data to ensure RFC 8878 consistency
  // and maximum performance. The GPU implementation is now robust
  // for all sizes.
  return ExecutionPath::GPU_BATCH;
}

std::unique_ptr<ZstdManager> create_manager(const CompressionConfig &config) {
  auto manager = std::make_unique<DefaultZstdManager>();
  manager->configure(config);
  return manager;
}

std::unique_ptr<ZstdManager> create_manager(int compression_level) {
  auto config = CompressionConfig::from_level(compression_level);
  config.cpu_threshold = 0; // FORCE GPU
  return create_manager(config);
}

std::unique_ptr<ZstdBatchManager> create_batch_manager(int compression_level) {
  return std::make_unique<ZstdBatchManager>(
      CompressionConfig::from_level(compression_level));
}

std::unique_ptr<ZstdStreamingManager>
create_streaming_manager(int compression_level) {
  return std::make_unique<ZstdStreamingManager>(
      CompressionConfig::from_level(compression_level));
}

} // namespace cuda_zstd
