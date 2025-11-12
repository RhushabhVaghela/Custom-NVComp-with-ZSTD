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

cuda_zstd_dict_t* cuda_zstd_train_dictionary(
    const void** samples,
    const size_t* sample_sizes,
    size_t num_samples,
    size_t dict_size
) {
    // --- START REPLACEMENT ---
    if (!samples || !sample_sizes || num_samples == 0 || dict_size == 0) {
        return nullptr;
    }

    try {
        // 1. Convert C-style arrays to C++ vectors
        std::vector<const cuda_zstd::byte_t*> cpp_samples(num_samples);
        std::vector<size_t> cpp_sample_sizes(num_samples);
        
        for (size_t i = 0; i < num_samples; ++i) {
            cpp_samples[i] = static_cast<const cuda_zstd::byte_t*>(samples[i]);
            cpp_sample_sizes[i] = sample_sizes[i];
        }

        // 2. Create the opaque C struct and the C++ Dictionary
        cuda_zstd_dict_t* c_dict = new cuda_zstd_dict_t;
        c_dict->dict = std::make_unique<cuda_zstd::dictionary::Dictionary>();

        // 3. Allocate GPU memory for the dictionary
        auto status = cuda_zstd::dictionary::DictionaryManager::allocate_dictionary_gpu(
            *(c_dict->dict), dict_size, 0
        );
        if (status != cuda_zstd::Status::SUCCESS) {
            delete c_dict;
            return nullptr;
        }

        // 4. Call the C++ trainer
        cuda_zstd::dictionary::CoverParams params; // Use default params
        status = cuda_zstd::dictionary::DictionaryTrainer::train_dictionary(
            cpp_samples,
            cpp_sample_sizes,
            *(c_dict->dict),
            dict_size,
            params,
            0
        );
        
        // Wait for training to finish
        cudaStreamSynchronize(0);

        if (status != cuda_zstd::Status::SUCCESS || c_dict->dict->raw_size == 0) {
            cuda_zstd::dictionary::DictionaryManager::free_dictionary_gpu(*(c_dict->dict), 0);
            delete c_dict;
            return nullptr;
        }
        
        // 5. Build entropy tables (required for a usable dictionary)
        // We'll train on the dictionary content itself as a simple default
        status = cuda_zstd::dictionary::build_entropy_tables(
            *(c_dict->dict),
            c_dict->dict->raw_content,
            c_dict->dict->raw_size,
            0
        );
        cudaStreamSynchronize(0);
        
        if (status != cuda_zstd::Status::SUCCESS) {
            cuda_zstd::dictionary::DictionaryManager::free_dictionary_gpu(*(c_dict->dict), 0);
            delete c_dict;
            return nullptr;
        }

        // 6. Return the opaque handle
        return c_dict;
        
    } catch (...) {
        return nullptr;
    }
    // --- END REPLACEMENT ---
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