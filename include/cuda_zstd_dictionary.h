// ============================================================================
// cuda_zstd_dictionary.h - Dictionary Training (COVER Algorithm)
// ============================================================================

#ifndef CUDA_ZSTD_DICTIONARY_H
#define CUDA_ZSTD_DICTIONARY_H

#include "cuda_zstd_types.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_huffman.h"
#include <vector>
#include <cstring>

namespace cuda_zstd {
namespace dictionary {

constexpr u32 DICT_MAGIC_NUMBER = 0xEC30A437;

struct DictionaryHeader {
    u32 magic_number;
    u32 dictionary_id;
    u32 raw_content_size;
    u32 entropy_tables_size;
    DictionaryHeader() : magic_number(DICT_MAGIC_NUMBER), dictionary_id(0), raw_content_size(0), entropy_tables_size(0) {}
};

struct Dictionary {
    DictionaryHeader header;
    byte_t* raw_content;
    size_t raw_size;
    fse::FSEEncodeTable* ll_fse_table;
    fse::FSEEncodeTable* of_fse_table;
    fse::FSEEncodeTable* ml_fse_table;
    huffman::HuffmanCode* huffman_table;
    u32 ll_table_log;
    u32 of_table_log;
    u32 ml_table_log;
    u32 huf_table_log;
    Dictionary() : raw_content(nullptr), raw_size(0), ll_fse_table(nullptr), of_fse_table(nullptr), ml_fse_table(nullptr), huffman_table(nullptr), ll_table_log(0), of_table_log(0), ml_table_log(0), huf_table_log(0) {}
};

struct CoverParams {
    u32 k, d, steps;
    double split_point;
    bool optimize_for_size;
    CoverParams() : k(1024), d(8), steps(32), split_point(0.75), optimize_for_size(false) {}
};

struct DictSegment {
    u32 offset, length, frequency;
    double score;
    __device__ __host__ DictSegment() : offset(0), length(0), frequency(0), score(0.0) {}
    __device__ __host__ DictSegment(u32 o, u32 l, u32 f, double s) : offset(o), length(l), frequency(f), score(s) {}
    __device__ __host__ bool operator>(const DictSegment& other) const { return score > other.score; }
};

struct Dmer {
    u64 hash;
    u32 position;
    __host__ __device__ Dmer() : hash(0), position(0) {}
    __host__ __device__ bool operator<(const Dmer& other) const { return hash < other.hash; }
};

// ============================================================================
// Dictionary Training Host Functions
// ============================================================================

class DictionaryTrainer {
public:
    /**
     * @brief Trains a dictionary using a COVER variant (Fast or Exact).
     * This now serves as the consolidated training function.
     */
    static Status train_dictionary(
        const std::vector<const byte_t*>& samples,
        const std::vector<size_t>& sample_sizes,
        Dictionary& output_dict,
        size_t max_dict_size,
        const CoverParams& params,
        cudaStream_t stream = 0);

    /**
     * @brief Finds optimal parameters (k, d) using a validation set.
     */
    static Status train_dictionary_optimized(
        const std::vector<const byte_t*>& samples,
        const std::vector<size_t>& sample_sizes,
        Dictionary& output_dict,
        size_t max_dict_size,
        cudaStream_t stream = 0);

    /**
     * @brief Validates a dictionary against a single flat validation buffer.
     */
    static double validate_dictionary(
        const Dictionary& dict,
        const byte_t* validation_data, // Host pointer
        size_t validation_size,
        cudaStream_t stream);

    /**
     * @brief Validates a dictionary against a set of validation samples.
     */
    static double validate_dictionary(
        const Dictionary& dict,
        const std::vector<const byte_t*>& val_samples,
        const std::vector<size_t>& val_sizes,
        cudaStream_t stream);
};

// ============================================================================
// Dictionary I/O & Management
// ============================================================================

class DictionaryIO {
public:
    static Status save_dictionary(
        const Dictionary& dict,
        const char* filename
    );
    
    static Status load_dictionary(
        Dictionary& dict,
        const char* filename,
        cudaStream_t stream = 0
    );
    
    static Status serialize_dictionary(
        const Dictionary& dict,
        byte_t* buffer,
        size_t buffer_size,
        size_t* bytes_written
    );
    
    static Status deserialize_dictionary(
        Dictionary& dict,
        const byte_t* buffer,
        size_t buffer_size,
        cudaStream_t stream = 0
    );
    
    static u32 compute_dictionary_id(
        const byte_t* dict_content,
        size_t content_size
    );
};

class DictionaryManager {
public:
    static Status allocate_dictionary_gpu(
        Dictionary& dict,
        size_t max_raw_size,
        cudaStream_t stream = 0) {
        // Allocate GPU memory for dictionary
        dict.raw_size = max_raw_size;
        CUDA_CHECK(cudaMallocAsync(&dict.raw_content, max_raw_size, stream));
        CUDA_CHECK(cudaMemsetAsync(dict.raw_content, 0, max_raw_size, stream));
        
        // Initialize header
        dict.header.magic_number = DICT_MAGIC_NUMBER;
        dict.header.raw_content_size = max_raw_size;
        dict.header.dictionary_id = 0;
        dict.header.entropy_tables_size = 0;
        
        return Status::SUCCESS;
    }
    
    static Status free_dictionary_gpu(Dictionary& dict, 
        cudaStream_t stream = 0
    );
    
    static Status copy_dictionary_to_gpu(
        Dictionary& gpu_dict,
        const Dictionary& cpu_dict,
        cudaStream_t stream = 0
    );
    
    static Status copy_dictionary_to_cpu(
        Dictionary& cpu_dict,
        const Dictionary& gpu_dict,
        cudaStream_t stream = 0
    );
};

// ============================================================================
// Entropy Table Builder
// ============================================================================

Status build_entropy_tables(
    Dictionary& dict,
    const byte_t* training_data, // device pointer (optional)
    size_t data_size,
    cudaStream_t stream
);

// Create prefix dictionary (no training, just use raw data)
Status create_prefix_dictionary(
    Dictionary& dict,
    const byte_t* prefix_data,
    size_t prefix_size,
    cudaStream_t stream = 0
);

} // namespace dictionary
} // namespace cuda_zstd

#endif // CUDA_ZSTD_DICTIONARY_H