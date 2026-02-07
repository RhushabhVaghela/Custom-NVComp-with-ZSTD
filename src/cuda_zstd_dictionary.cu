// ==============================================================================
// cuda_zstd_dictionary.cu - COMPLETE Dictionary Training Implementation
// ==============================================================================

#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_xxhash.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>


namespace cuda_zstd {
namespace dictionary {

// ==============================================================================
// Training Parameters
// ==============================================================================

constexpr u32 MIN_SAMPLES_FOR_TRAINING = 1;
constexpr u32 DEFAULT_SAMPLE_SIZE = 8 * 1024; // 8 KB per sample

// ==============================================================================
// CUDA Kernels for Dictionary Training
// ==============================================================================

/**
 * @brief Count byte frequencies across all samples
 */
__global__ void count_byte_frequencies_kernel(const unsigned char *d_data,
                                              u32 data_size,
                                              u32 *d_frequencies // 256 counters
) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < data_size; i += stride) {
    unsigned char byte_val = d_data[i];
    atomicAdd(&d_frequencies[byte_val], 1);
  }
}

/**
 * @brief Extract frequent patterns (n-grams) from samples
 */
__global__ void
extract_ngrams_kernel(const unsigned char *d_data, u32 data_size,
                      u32 ngram_length,
                      u64 *d_ngram_hashes, // Output: hash of each ngram
                      u32 *d_ngram_counts, // Output: frequency of each ngram
                      u32 max_ngrams) {
  u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = idx; i < data_size - ngram_length + 1; i += stride) {
    // Compute simple hash of ngram
    u64 hash = 0;
    for (u32 j = 0; j < ngram_length; j++) {
      hash = hash * 31 + d_data[i + j];
    }

    // Find slot in hash table (simple linear probing)
    u32 slot = hash % max_ngrams;
    for (u32 probe = 0; probe < 100; probe++) {
      u32 test_slot = (slot + probe) % max_ngrams;
      u64 existing = atomicCAS((unsigned long long *)&d_ngram_hashes[test_slot],
                               0ULL, hash);

      if (existing == 0 || existing == hash) {
        atomicAdd(&d_ngram_counts[test_slot], 1);
        break;
      }
    }
  }
}

/**
 * @brief Select top-k most frequent patterns for dictionary
 */
__global__ void select_top_patterns_kernel(
    const u32 *d_ngram_counts,
    u32 *d_ngram_indices, // Output: indices of top-k patterns
    u32 num_ngrams, u32 k) {
  
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= k) return;

  // Optimized parallel selection: each thread finds its k-th element
  // This is still a bit simplified but functional for dictionary training.
  u32 best_idx = 0;
  u32 best_count = 0;
  
  // We use a simple pass to find top-k (since k is small, usually 100-1000)
  // For production, we would use a proper parallel sort + select.
  // Given dictionary training is offline, this is acceptable for now.
  
  for (u32 i = 0; i < num_ngrams; i++) {
      u32 count = d_ngram_counts[i];
      if (count > best_count) {
          // Check if already selected (poor man's top-k)
          bool already = false;
          for(u32 j=0; j<tid; j++) {
              if (d_ngram_indices[j] == i) { already = true; break; }
          }
          if (!already) {
              best_count = count;
              best_idx = i;
          }
      }
  }
  d_ngram_indices[tid] = best_idx;
}

// ==============================================================================
// Host-side Training Functions
// ==============================================================================

/**
 * @brief Simple frequency-based dictionary builder (CPU version)
 * This is a fallback when GPU training is not optimal for small datasets
 */
Status
build_frequency_dict_cpu(const std::vector<const unsigned char *> &samples,
                         const std::vector<size_t> &sample_sizes,
                         unsigned char *dict_buffer, size_t dict_size) {
  //     std::cout << "[DictTraining] Using CPU frequency-based training" <<
  //     std::endl;

  // Count byte frequencies
  std::vector<u32> frequencies(256, 0);

  for (size_t s = 0; s < samples.size(); s++) {
    for (size_t i = 0; i < sample_sizes[s]; i++) {
      frequencies[samples[s][i]]++;
    }
  }

  // Build dictionary with most frequent bytes
  std::vector<std::pair<u32, unsigned char>> freq_pairs;
  for (u32 i = 0; i < 256; i++) {
    if (frequencies[i] > 0) {
      freq_pairs.push_back({frequencies[i], (unsigned char)i});
    }
  }

  // Sort by frequency (descending)
  std::sort(freq_pairs.begin(), freq_pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Fill dictionary
  size_t dict_pos = 0;
  for (const auto &[freq, byte_val] : freq_pairs) {
    if (dict_pos >= dict_size)
      break;

    // Add repeated bytes based on frequency
    u32 repeat_count = std::min((u32)(freq / 100), (u32)(dict_size - dict_pos));
    for (u32 r = 0; r < repeat_count && dict_pos < dict_size; r++) {
      dict_buffer[dict_pos++] = byte_val;
    }
  }

  // Fill remaining with most common sequences
  while (dict_pos < dict_size) {
    dict_buffer[dict_pos++] = freq_pairs[0].second;
  }

  //     std::cout << "[DictTraining] Built frequency dict: " << dict_size << "
  //     bytes" << std::endl;

  return Status::SUCCESS;
}

/**
 * @brief GPU-accelerated dictionary training
 */
Status train_dictionary_gpu(const std::vector<const unsigned char *> &h_samples,
                            const std::vector<size_t> &sample_sizes,
                            unsigned char *h_dict_buffer, size_t dict_size,
                            cudaStream_t stream) {
  // Validate inputs
  if (!h_dict_buffer || dict_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (h_samples.empty() || sample_sizes.empty()) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Concatenate all samples into a single device buffer
  size_t total_size = 0;
  for (auto size : sample_sizes) {
    total_size += size;
  }

  if (total_size == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Use RAII pattern for GPU memory management
  struct GPUMemoryGuard {
    void *ptr;
    GPUMemoryGuard() : ptr(nullptr) {}
    ~GPUMemoryGuard() {
      if (ptr)
        cudaFree(ptr);
    }
    cudaError_t alloc(size_t size) {
      cudaError_t err = cudaMalloc(&ptr, size);
      return err;
    }
    void *get() { return ptr; }
    void release() { ptr = nullptr; }
  };

  GPUMemoryGuard d_all_samples_guard;
  GPUMemoryGuard d_frequencies_guard;
  GPUMemoryGuard d_ngram_hashes_guard;
  GPUMemoryGuard d_ngram_counts_guard;

  // Allocate and copy sample data
  cudaError_t err = d_all_samples_guard.alloc(total_size);
  if (err != cudaSuccess) {
    return Status::ERROR_OUT_OF_MEMORY;
  }
  unsigned char *d_all_samples =
      static_cast<unsigned char *>(d_all_samples_guard.get());

  size_t offset = 0;
  for (size_t i = 0; i < h_samples.size(); i++) {
    if (!h_samples[i] || sample_sizes[i] == 0)
      continue;
    err = cudaMemcpyAsync(d_all_samples + offset, h_samples[i], sample_sizes[i],
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      return Status::ERROR_CUDA_ERROR;
    }
    offset += sample_sizes[i];
  }

  // Allocate frequency counters
  err = d_frequencies_guard.alloc(256 * sizeof(u32));
  if (err != cudaSuccess) {
    return Status::ERROR_OUT_OF_MEMORY;
  }
  u32 *d_frequencies = static_cast<u32 *>(d_frequencies_guard.get());

  err = cudaMemsetAsync(d_frequencies, 0, 256 * sizeof(u32), stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Count byte frequencies
  const u32 threads = 256;
  const u32 blocks =
      std::min((u32)((total_size + threads - 1) / threads), 1024u);

  count_byte_frequencies_kernel<<<blocks, threads, 0, stream>>>(
      d_all_samples, total_size, d_frequencies);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Copy frequencies back to host
  std::vector<u32> h_frequencies(256);
  err = cudaMemcpyAsync(h_frequencies.data(), d_frequencies, 256 * sizeof(u32),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Build dictionary from frequencies (on CPU for simplicity)
  std::vector<std::pair<u32, unsigned char>> freq_pairs;
  for (u32 i = 0; i < 256; i++) {
    if (h_frequencies[i] > 0) {
      freq_pairs.push_back({h_frequencies[i], (unsigned char)i});
    }
  }

  std::sort(freq_pairs.begin(), freq_pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Extract frequent n-grams (3-grams, 4-grams)
  const u32 max_ngrams = 10000;

  err = d_ngram_hashes_guard.alloc(max_ngrams * sizeof(u64));
  if (err != cudaSuccess) {
    return Status::ERROR_OUT_OF_MEMORY;
  }
  u64 *d_ngram_hashes = static_cast<u64 *>(d_ngram_hashes_guard.get());

  err = d_ngram_counts_guard.alloc(max_ngrams * sizeof(u32));
  if (err != cudaSuccess) {
    return Status::ERROR_OUT_OF_MEMORY;
  }
  u32 *d_ngram_counts = static_cast<u32 *>(d_ngram_counts_guard.get());

  err = cudaMemsetAsync(d_ngram_hashes, 0, max_ngrams * sizeof(u64), stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  err = cudaMemsetAsync(d_ngram_counts, 0, max_ngrams * sizeof(u32), stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Extract 4-grams
  extract_ngrams_kernel<<<blocks, threads, 0, stream>>>(
      d_all_samples, total_size, 4, d_ngram_hashes, d_ngram_counts, max_ngrams);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Copy n-gram data back
  std::vector<u64> h_ngram_hashes(max_ngrams);
  std::vector<u32> h_ngram_counts(max_ngrams);

  err = cudaMemcpy(h_ngram_hashes.data(), d_ngram_hashes,
                   max_ngrams * sizeof(u64), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  err = cudaMemcpy(h_ngram_counts.data(), d_ngram_counts,
                   max_ngrams * sizeof(u32), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status::ERROR_CUDA_ERROR;
  }

  // Build dictionary: frequency bytes + top n-grams
  size_t dict_pos = 0;

  // Phase 1: Add most frequent individual bytes (first 25% of dict)
  size_t byte_section = dict_size / 4;
  for (const auto &[freq, byte_val] : freq_pairs) {
    if (dict_pos >= byte_section)
      break;

    u32 repeat = std::min((u32)(freq / 1000), (u32)(byte_section - dict_pos));
    if (repeat == 0)
      repeat = 1;

    for (u32 r = 0; r < repeat && dict_pos < byte_section; r++) {
      h_dict_buffer[dict_pos++] = byte_val;
    }
  }

  // Phase 2: Add frequent n-grams (remaining 75%)
  std::vector<std::pair<u32, u32>> ngram_freq_idx;
  for (u32 i = 0; i < max_ngrams; i++) {
    if (h_ngram_counts[i] > 10) { // Threshold: appears at least 10 times
      ngram_freq_idx.push_back({h_ngram_counts[i], i});
    }
  }

  std::sort(ngram_freq_idx.begin(), ngram_freq_idx.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Reconstruct n-grams from samples and add to dictionary
  for (const auto &[freq, idx] : ngram_freq_idx) {
    if (dict_pos + 4 > dict_size)
      break;

    u64 target_hash = h_ngram_hashes[idx];

    // Find this n-gram in the samples
    for (size_t s = 0; s < h_samples.size() && dict_pos + 4 <= dict_size; s++) {
      for (size_t i = 0; i + 4 <= sample_sizes[s]; i++) {
        u64 hash = 0;
        for (u32 j = 0; j < 4; j++) {
          hash = hash * 31 + h_samples[s][i + j];
        }

        if (hash == target_hash) {
          // Found it! Add to dictionary
          for (u32 j = 0; j < 4 && dict_pos < dict_size; j++) {
            h_dict_buffer[dict_pos++] = h_samples[s][i + j];
          }
          goto next_ngram;
        }
      }
    }
  next_ngram:;
  }

  // Fill any remaining space with most frequent bytes
  while (dict_pos < dict_size) {
    if (!freq_pairs.empty()) {
      h_dict_buffer[dict_pos++] = freq_pairs[0].second;
    } else {
      h_dict_buffer[dict_pos++] = 0;
    }
  }

  // Memory will be automatically freed by guards

  return Status::SUCCESS;
}

// ==============================================================================
// Public API Implementation
// ==============================================================================

Status train_dictionary(const std::vector<const void *> &samples,
                        const std::vector<size_t> &sample_sizes,
                        void *dict_buffer, size_t dict_size,
                        const DictionaryTrainingParams *params,
                        cudaStream_t stream) {
  // Validate inputs
  if (samples.empty() || sample_sizes.empty() || !dict_buffer) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (samples.size() != sample_sizes.size()) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (samples.size() < MIN_SAMPLES_FOR_TRAINING) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  if (dict_size < MIN_DICT_SIZE || dict_size > MAX_DICT_SIZE) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  //     std::cout << "[DictTraining] Training dictionary with " <<
  //     samples.size()
  //               << " samples, target size: " << dict_size << " bytes" <<
  //               std::endl;

  // Convert to byte pointers
  std::vector<const unsigned char *> byte_samples;
  for (const auto *sample : samples) {
    byte_samples.push_back(static_cast<const unsigned char *>(sample));
  }

  unsigned char *h_dict = static_cast<unsigned char *>(dict_buffer);

  // Choose training method based on dataset size
  size_t total_size = 0;
  for (auto size : sample_sizes) {
    total_size += size;
  }

  Status status;

  if (total_size < 100 * 1024) { // < 100 KB: use CPU
    status =
        build_frequency_dict_cpu(byte_samples, sample_sizes, h_dict, dict_size);
  } else { // >= 100 KB: use GPU
    status = train_dictionary_gpu(byte_samples, sample_sizes, h_dict, dict_size,
                                  stream);
  }

  return status;
}

Status create_dictionary_from_samples(const void *samples_buffer,
                                      const size_t *sample_offsets,
                                      size_t num_samples, void *dict_buffer,
                                      size_t dict_size,
                                      const DictionaryTrainingParams *params,
                                      cudaStream_t stream) {
  if (!samples_buffer || !sample_offsets || !dict_buffer || num_samples == 0) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Convert offsets to individual samples
  std::vector<const void *> samples;
  std::vector<size_t> sample_sizes;

  const unsigned char *base =
      static_cast<const unsigned char *>(samples_buffer);

  for (size_t i = 0; i < num_samples; i++) {
    size_t start = sample_offsets[i];
    size_t end = (i + 1 < num_samples) ? sample_offsets[i + 1]
                                       : start + DEFAULT_SAMPLE_SIZE;

    samples.push_back(base + start);
    sample_sizes.push_back(end - start);
  }

  return train_dictionary(samples, sample_sizes, dict_buffer, dict_size, params,
                          stream);
}

u32 get_optimal_dict_size(size_t total_data_size) {
  // Heuristic: dictionary should be ~1% of data size, clamped to limits
  size_t suggested = total_data_size / 100;

  if (suggested < MIN_DICT_SIZE)
    return MIN_DICT_SIZE;
  if (suggested > MAX_DICT_SIZE)
    return MAX_DICT_SIZE;

  // Round to nearest KB
  return ((u32)suggested + 1023) / 1024 * 1024;
}

bool is_valid_dictionary_size(size_t size) {
  return size >= MIN_DICT_SIZE && size <= MAX_DICT_SIZE;
}

} // namespace dictionary
} // namespace cuda_zstd
