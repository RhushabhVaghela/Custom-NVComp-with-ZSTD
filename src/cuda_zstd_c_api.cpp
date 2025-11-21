// ============================================================================
// cuda_zstd_c_api.cpp - C-API Implementation
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_nvcomp.h" // For status_to_nvcomp_error

// Define opaque structs
struct cuda_zstd_manager_t {
    std::unique_ptr<cuda_zstd::ZstdManager> manager;
};

struct cuda_zstd_dict_t {
    std::unique_ptr<cuda_zstd::dictionary::Dictionary> dict;
};

extern "C" {

// ============================================================================
// Manager Lifecycle
// ============================================================================

cuda_zstd_manager_t* cuda_zstd_create_manager(int compression_level) {
    try {
        auto m = new cuda_zstd_manager_t;
        m->manager = cuda_zstd::create_manager(compression_level);
        return m;
    } catch (...) {
        return nullptr;
    }
}

void cuda_zstd_destroy_manager(cuda_zstd_manager_t* manager) {
    if (manager) {
        delete manager;
    }
}

// ============================================================================
// Compression/Decompression
// ============================================================================

int cuda_zstd_compress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    if (!manager || !manager->manager) {
        return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(cuda_zstd::Status::ERROR_INVALID_PARAMETER);
    }
    
    auto status = manager->manager->compress(
        src, src_size,
        dst, dst_size,
        workspace, workspace_size,
        nullptr, 0,
        stream
    );
    return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(status);
}

int cuda_zstd_decompress(
    cuda_zstd_manager_t* manager,
    const void* src,
    size_t src_size,
    void* dst,
    size_t* dst_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    if (!manager || !manager->manager) {
        return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(cuda_zstd::Status::ERROR_INVALID_PARAMETER);
    }

    auto status = manager->manager->decompress(
        src, src_size,
        dst, dst_size,
        workspace, workspace_size,
        stream
    );
    return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(status);
}

// ============================================================================
// Workspace Queries
// ============================================================================

size_t cuda_zstd_get_compress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t src_size
) {
    if (!manager || !manager->manager) return 0;
    return manager->manager->get_compress_temp_size(src_size);
}

size_t cuda_zstd_get_decompress_workspace_size(
    cuda_zstd_manager_t* manager,
    size_t compressed_size
) {
    if (!manager || !manager->manager) return 0;
    return manager->manager->get_decompress_temp_size(compressed_size);
}

// ============================================================================
// Dictionary Training (Full Implementation)
// ============================================================================

/**
 * @brief Trains a dictionary using the COVER algorithm on provided samples.
 *
 * @param samples Array of pointers to sample buffers (host memory).
 * @param sample_sizes Array of sizes for each sample.
 * @param num_samples Number of samples.
 * @param dict_size Target dictionary size in bytes.
 * @return cuda_zstd_dict_t* Pointer to trained dictionary handle, or NULL on error.
 *
 * Error Codes:
 *   - NULL: Invalid parameters or training failure.
 *   - Dictionary handle: Success.
 */
cuda_zstd_dict_t* cuda_zstd_train_dictionary(
    const void** samples,
    const size_t* sample_sizes,
    size_t num_samples,
    size_t dict_size
) {
    if (!samples || !sample_sizes || num_samples == 0 || dict_size == 0) {
        return nullptr;
    }

    try {
        // Prepare sample vectors for trainer (cast to const void*)
        std::vector<const void*> sample_vec;
        std::vector<size_t> size_vec;
        sample_vec.reserve(num_samples);
        size_vec.reserve(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            if (!samples[i] || sample_sizes[i] == 0) {
                return nullptr;
            }
            sample_vec.push_back(reinterpret_cast<const void*>(samples[i]));
            size_vec.push_back(sample_sizes[i]);
        }

        // Create dictionary object
        auto dict_handle = new cuda_zstd_dict_t;
        dict_handle->dict = std::make_unique<cuda_zstd::dictionary::Dictionary>();

        // Use default COVER parameters (can be extended)
        cuda_zstd::dictionary::CoverParams params;

        // Train dictionary (on default stream) - pass nullptr for optional params
        cuda_zstd::Status status = cuda_zstd::dictionary::DictionaryTrainer::train_dictionary(
            sample_vec,
            size_vec,
            *dict_handle->dict,
            dict_size,
            nullptr,  // Pass nullptr instead of params to avoid type mismatch
            0 // cudaStream_t
        );
        if (status != cuda_zstd::Status::SUCCESS) {
            delete dict_handle;
            return nullptr;
        }

        return dict_handle;
    } catch (...) {
        return nullptr;
    }
}

void cuda_zstd_destroy_dictionary(cuda_zstd_dict_t* dict) {
    if (dict) {
        // This will automatically free the GPU memory held by the dict
        delete dict;
    }
}

int cuda_zstd_set_dictionary(
    cuda_zstd_manager_t* manager,
    cuda_zstd_dict_t* dict
) {
    if (!manager || !manager->manager || !dict || !dict->dict) {
        return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(cuda_zstd::Status::ERROR_INVALID_PARAMETER);
    }
    auto status = manager->manager->set_dictionary(*dict->dict);
    return cuda_zstd::nvcomp_v5::status_to_nvcomp_error(status);
}

// ============================================================================
// Error Handling
// ============================================================================

const char* cuda_zstd_get_error_string(int error_code) {
    return cuda_zstd::status_to_string(
        cuda_zstd::nvcomp_v5::nvcomp_error_to_status(error_code)
    );
}

int cuda_zstd_is_error(int code) {
    return (code != 0);
}

} // extern "C"
