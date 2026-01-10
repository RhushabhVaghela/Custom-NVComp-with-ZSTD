// ==============================================================================
// cuda_zstd_dictionary.h - Dictionary Training and Management
// ==============================================================================

#ifndef CUDA_ZSTD_DICTIONARY_H_
#define CUDA_ZSTD_DICTIONARY_H_

#include "cuda_zstd_fse.h"
#include "cuda_zstd_huffman.h"
#include "cuda_zstd_types.h"


#ifdef __cplusplus
#include <cstdlib>
#include <cstring>
#include <vector>

#endif

#ifdef __cplusplus
namespace cuda_zstd {
namespace dictionary {

// ==============================================================================
// Constants
// ==============================================================================

constexpr u32 DICT_MAGIC_NUMBER = 0xEC30A437;
constexpr u32 MIN_DICT_SIZE = 256;
constexpr u32 MAX_DICT_SIZE = 128 * 1024; // 128 KB

// ==============================================================================
// Dictionary Training Parameters
// ==============================================================================

struct DictionaryTrainingParams {
  u32 optimization_level = 0; // 0 = default, higher = more optimization
  bool use_gpu = true;        // Use GPU acceleration if available
  u32 max_threads = 256;      // Max CUDA threads per block
  u32 reserved[5] = {0};      // Future expansion
};

// Legacy parameter structure (for backward compatibility)
struct CoverParams {
  u32 k = 0;          // Segment size
  u32 d = 0;          // Dmer size
  u32 steps = 0;      // Number of steps
  u32 splitPoint = 0; // Split point percentage
  double accel = 0.0; // Acceleration factor
};

// ==============================================================================
// Dictionary Structure
// ==============================================================================

struct DictionaryHeader {
  u32 magic_number;
  u32 dictionary_id;
  u32 entropy_tables_size;
  u32 offsets_size;
  u32 match_lengths_size;
  u32 literal_lengths_size;
  u32 huffman_table_size;
  u32 raw_content_size;
};

struct DictionaryContent {
  unsigned char *d_buffer;
  u32 size;
  u32 dict_id;
};

// Complete Dictionary struct for backward compatibility
struct Dictionary {
  DictionaryHeader header;
  unsigned char *raw_content;
  u32 raw_size;

  Dictionary() : raw_content(nullptr), raw_size(0) {
    header.magic_number = DICT_MAGIC_NUMBER;
    header.dictionary_id = 0;
    header.entropy_tables_size = 0;
    header.offsets_size = 0;
    header.match_lengths_size = 0;
    header.literal_lengths_size = 0;
    header.huffman_table_size = 0;
    header.raw_content_size = 0;
  }

  // CRITICAL FIX: Do NOT free memory in destructor!
  // Memory is managed externally via
  // allocate_dictionary_gpu/free_dictionary_gpu. Shallow copies from
  // set_dictionary would cause double-free segfault.
  ~Dictionary() {
    // Intentionally do nothing - caller is responsible for memory management
    // via DictionaryManager::free_dictionary_gpu()
  }

  // Copy constructor with deep copy
  Dictionary(const Dictionary &other) : raw_content(nullptr), raw_size(0) {
    header = other.header;
    raw_size = other.raw_size;
    if (other.raw_content && raw_size > 0) {
      raw_content = (unsigned char *)malloc(raw_size);
      if (raw_content) {
        memcpy(raw_content, other.raw_content, raw_size);
      }
    }
  }

  // Copy assignment with deep copy
  Dictionary &operator=(const Dictionary &other) {
    if (this != &other) {
      // Free existing content
      if (raw_content) {
        free(raw_content);
        raw_content = nullptr;
      }

      // Copy header and size
      header = other.header;
      raw_size = other.raw_size;

      // Deep copy content
      if (other.raw_content && raw_size > 0) {
        raw_content = (unsigned char *)malloc(raw_size);
        if (raw_content) {
          memcpy(raw_content, other.raw_content, raw_size);
        }
      }
    }
    return *this;
  }
};

// ==============================================================================
// Main API Functions (New Implementation)
// ==============================================================================

/**
 * @brief Train a dictionary from multiple samples (GPU-accelerated)
 *
 * @param samples Array of sample data pointers (host memory)
 * @param sample_sizes Array of sample sizes
 * @param dict_buffer Output buffer for trained dictionary (host memory)
 * @param dict_size Size of dictionary to train
 * @param params Training parameters (optional)
 * @param stream CUDA stream for async operations
 * @return Status code
 */
Status train_dictionary(const std::vector<const void *> &samples,
                        const std::vector<size_t> &sample_sizes,
                        void *dict_buffer, size_t dict_size,
                        const DictionaryTrainingParams *params = nullptr,
                        cudaStream_t stream = 0);

/**
 * @brief Create dictionary from concatenated samples buffer
 *
 * @param samples_buffer Concatenated samples (host memory)
 * @param sample_offsets Array of byte offsets for each sample
 * @param num_samples Number of samples
 * @param dict_buffer Output dictionary buffer
 * @param dict_size Dictionary size
 * @param params Training parameters (optional)
 * @param stream CUDA stream
 * @return Status code
 */
Status create_dictionary_from_samples(
    const void *samples_buffer, const size_t *sample_offsets,
    size_t num_samples, void *dict_buffer, size_t dict_size,
    const DictionaryTrainingParams *params = nullptr, cudaStream_t stream = 0);

/**
 * @brief Get optimal dictionary size for given data size
 *
 * @param total_data_size Total size of data to compress
 * @return Recommended dictionary size (clamped to valid range)
 */
u32 get_optimal_dict_size(size_t total_data_size);

/**
 * @brief Validate dictionary size
 */
bool is_valid_dictionary_size(size_t size);

// ==============================================================================
// Backward Compatibility API (for existing test code)
// ==============================================================================

namespace compat {

/**
 * @brief Backward-compatible dictionary trainer wrapper
 */
class DictionaryTrainerWrapper {
public:
  static Status train_dictionary(const std::vector<const void *> &samples,
                                 const std::vector<size_t> &sample_sizes,
                                 Dictionary &dict_out, size_t dict_size,
                                 const CoverParams *params = nullptr,
                                 cudaStream_t stream = 0) {
    // Allocate dictionary buffer
    dict_out.raw_content = (unsigned char *)malloc(dict_size);
    if (!dict_out.raw_content) {
      return Status::ERROR_OUT_OF_MEMORY;
    }
    dict_out.raw_size = dict_size;

    // Convert params
    DictionaryTrainingParams train_params;
    if (params) {
      train_params.optimization_level = params->k;
    }

    // Call new API
    Status status = ::cuda_zstd::dictionary::train_dictionary(
        samples, sample_sizes, dict_out.raw_content, dict_size, &train_params,
        stream);

    if (status == Status::SUCCESS) {
      // Compute dictionary ID (simple hash)
      dict_out.header.dictionary_id = 0;
      for (size_t i = 0; i < dict_size && i < 256; i++) {
        dict_out.header.dictionary_id =
            (dict_out.header.dictionary_id * 31) + dict_out.raw_content[i];
      }
    } else {
      free(dict_out.raw_content);
      dict_out.raw_content = nullptr;
      dict_out.raw_size = 0;
    }

    return status;
  }
};

/**
 * @brief Backward-compatible dictionary manager wrapper
 */
class DictionaryManagerWrapper {
public:
  static Status allocate_dictionary_gpu(Dictionary &dict, size_t size,
                                        cudaStream_t stream = 0) {
    (void)stream; // Suppress unused parameter warning

    dict.raw_content = (unsigned char *)malloc(size);
    if (!dict.raw_content) {
      return Status::ERROR_OUT_OF_MEMORY;
    }
    dict.raw_size = size;
    return Status::SUCCESS;
  }

  static Status free_dictionary_gpu(Dictionary &dict, cudaStream_t stream = 0) {
    (void)stream; // Suppress unused parameter warning

    if (dict.raw_content) {
      free(dict.raw_content);
      dict.raw_content = nullptr;
      dict.raw_size = 0;
    }
    dict.header.dictionary_id = 0;
    return Status::SUCCESS;
  }

  static Status load_dictionary(const void *dict_buffer, size_t dict_size,
                                Dictionary &dict_out) {
    dict_out.raw_content = (unsigned char *)malloc(dict_size);
    if (!dict_out.raw_content) {
      return Status::ERROR_OUT_OF_MEMORY;
    }

    memcpy(dict_out.raw_content, dict_buffer, dict_size);
    dict_out.raw_size = dict_size;

    // Compute dictionary ID
    dict_out.header.dictionary_id = 0;
    for (size_t i = 0; i < dict_size && i < 256; i++) {
      dict_out.header.dictionary_id =
          (dict_out.header.dictionary_id * 31) + dict_out.raw_content[i];
    }

    return Status::SUCCESS;
  }
};

} // namespace compat

// Alias old names to new wrappers for backward compatibility
using DictionaryTrainer = compat::DictionaryTrainerWrapper;
using DictionaryManager = compat::DictionaryManagerWrapper;

} // namespace dictionary
} // namespace cuda_zstd

#endif // __cplusplus

#endif // CUDA_ZSTD_DICTIONARY_H_
