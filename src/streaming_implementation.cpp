/**
 * streaming_implementation.cpp - Proper Streaming Manager Implementation
 * 
 * This file contains the proper implementation of the Streaming Manager
 * with window history support for better compression ratios across chunks.
 */

#include "cuda_zstd_manager.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_lz77.h"
#include <cuda_runtime.h>
#include <cstring>

namespace cuda_zstd {

// Window size constants
constexpr u32 MAX_WINDOW_SIZE = 128 * 1024; // 128KB max window
constexpr u32 MIN_MATCH_LENGTH = 3;

// Streaming compression with window history
Status ZstdStreamingManager::compress_chunk_with_history(
    const void *input,
    size_t input_size, 
    void *output,
    size_t *output_size,
    bool is_last_chunk,
    cudaStream_t stream) {
    
    if (!input || !output || !output_size) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // Initialize output
    *output_size = 0;
    
    // Calculate window size from config
    u32 window_size = (1u << pimpl_->config.window_log);
    if (window_size > MAX_WINDOW_SIZE) {
        window_size = MAX_WINDOW_SIZE;
    }
    
    // Total size for LZ77: history + current input
    size_t combined_size = pimpl_->streaming_ctx.window_history_size + input_size;
    
    // Allocate combined buffer if needed
    void *d_combined = nullptr;
    if (cudaMalloc(&d_combined, combined_size) != cudaSuccess) {
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Copy history + input into combined buffer
    if (pimpl_->streaming_ctx.window_history_size > 0) {
        cudaMemcpyAsync(d_combined, 
                       pimpl_->streaming_ctx.d_window_history,
                       pimpl_->streaming_ctx.window_history_size,
                       cudaMemcpyDeviceToDevice, stream);
    }
    cudaMemcpyAsync(static_cast<char*>(d_combined) + pimpl_->streaming_ctx.window_history_size,
                   input, input_size, cudaMemcpyHostToDevice, stream);
    
    cudaStreamSynchronize(stream);
    
    // Allocate output buffer
    size_t max_compressed = pimpl_->manager->get_max_compressed_size(combined_size);
    void *d_compressed = nullptr;
    if (cudaMalloc(&d_compressed, max_compressed) != cudaSuccess) {
        cudaFree(d_combined);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Compress combined data (history + current chunk)
    size_t compressed_size = max_compressed;
    Status status = pimpl_->manager->compress(
        d_combined, combined_size,
        d_compressed, &compressed_size,
        pimpl_->d_workspace, pimpl_->workspace_size,
        pimpl_->has_dictionary ? pimpl_->dict.raw_content : nullptr,
        pimpl_->has_dictionary ? pimpl_->dict.raw_size : 0,
        stream
    );
    
    if (status != Status::SUCCESS) {
        cudaFree(d_combined);
        cudaFree(d_compressed);
        return status;
    }
    
    // Copy result to output
    cudaMemcpyAsync(output, d_compressed, compressed_size, 
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    *output_size = compressed_size;
    
    // Update history: save last window_size bytes for next chunk
    u32 new_history_size = (input_size < window_size) ? input_size : window_size;
    if (new_history_size > 0) {
        // Reallocate history buffer if needed
        if (new_history_size > pimpl_->streaming_ctx.window_history_capacity) {
            if (pimpl_->streaming_ctx.d_window_history) {
                cudaFree(pimpl_->streaming_ctx.d_window_history);
            }
            pimpl_->streaming_ctx.window_history_capacity = new_history_size;
            cudaMalloc(&pimpl_->streaming_ctx.d_window_history, 
                      pimpl_->streaming_ctx.window_history_capacity);
        }
        
        // Copy last N bytes of current input to history
        const void *history_src = static_cast<const char*>(input) + (input_size - new_history_size);
        cudaMemcpyAsync(pimpl_->streaming_ctx.d_window_history,
                       history_src,
                       new_history_size,
                       cudaMemcpyHostToDevice, stream);
        pimpl_->streaming_ctx.window_history_size = new_history_size;
        
        cudaStreamSynchronize(stream);
    }
    
    // Update total bytes processed
    pimpl_->streaming_ctx.total_bytes_processed += input_size;
    pimpl_->streaming_ctx.block_count++;
    
    // Cleanup temporary buffers
    cudaFree(d_combined);
    cudaFree(d_compressed);
    
    return Status::SUCCESS;
}

// Initialize streaming compression with history buffers
Status ZstdStreamingManager::init_compression_with_history(
    cudaStream_t stream, 
    size_t max_chunk_size) {
    
    if (pimpl_->comp_initialized) {
        return Status::SUCCESS;
    }
    
    // Allocate basic workspace
    Status s = pimpl_->alloc_workspace(stream, max_chunk_size);
    if (s != Status::SUCCESS) {
        return s;
    }
    
    // Initialize window history
    u32 window_size = (1u << pimpl_->config.window_log);
    if (window_size > MAX_WINDOW_SIZE) {
        window_size = MAX_WINDOW_SIZE;
    }
    
    pimpl_->streaming_ctx.window_history_capacity = window_size;
    pimpl_->streaming_ctx.window_history_size = 0;
    
    if (cudaMalloc(&pimpl_->streaming_ctx.d_window_history, 
                   pimpl_->streaming_ctx.window_history_capacity) != cudaSuccess) {
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize hash table state if needed (for LDM - not implemented yet)
    pimpl_->streaming_ctx.d_hash_table_state = nullptr;
    pimpl_->streaming_ctx.d_chain_table_state = nullptr;
    
    // Reset state
    pimpl_->streaming_ctx.total_bytes_processed = 0;
    pimpl_->streaming_ctx.block_count = 0;
    pimpl_->streaming_ctx.started_compression = true;
    pimpl_->streaming_ctx.finished_compression = false;
    
    // Pre-allocate FSE tables
    s = pimpl_->manager->preallocate_tables(stream);
    if (s != Status::SUCCESS) {
        return s;
    }
    
    pimpl_->comp_initialized = true;
    return Status::SUCCESS;
}

// Reset streaming state
Status ZstdStreamingManager::reset_streaming() {
    pimpl_->cleanup_streaming_context();
    
    // Reset state variables
    pimpl_->streaming_ctx.window_history_size = 0;
    pimpl_->streaming_ctx.window_history_capacity = 0;
    pimpl_->streaming_ctx.total_bytes_processed = 0;
    pimpl_->streaming_ctx.block_count = 0;
    pimpl_->streaming_ctx.started_compression = false;
    pimpl_->streaming_ctx.finished_compression = false;
    
    if (pimpl_->d_workspace) {
        cudaFree(pimpl_->d_workspace);
        pimpl_->d_workspace = nullptr;
    }
    pimpl_->workspace_size = 0;
    pimpl_->comp_initialized = false;
    pimpl_->decomp_initialized = false;
    
    return Status::SUCCESS;
}

// Flush any remaining state
Status ZstdStreamingManager::flush_streaming(cudaStream_t stream) {
    // Mark compression as finished
    pimpl_->streaming_ctx.finished_compression = true;
    
    // Synchronize to ensure all async operations complete
    cudaStreamSynchronize(stream);
    
    return Status::SUCCESS;
}

} // namespace cuda_zstd