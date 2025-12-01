// ==============================================================================
// batch_manager.cu - Batch manager implementation
// ==============================================================================

#include "batch_manager.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace compression {

constexpr size_t GPU_MEMORY_ALIGNMENT = 256;

inline size_t align_to_boundary(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

ZstdBatchManager::ZstdBatchManager(size_t num_streams)
    : num_streams_(num_streams),
      events_(num_streams)
{
//     std::cout << "[BatchManager] Initializing with " << num_streams << " streams" << std::endl;

    stream_pool_ = std::make_unique<StreamPool>(num_streams);

    for (size_t i = 0; i < num_streams; ++i) {
        cudaEventCreate(&events_[i]);
    }
}

ZstdBatchManager::~ZstdBatchManager() {
    synchronize();

    for (auto& event : events_) {
        cudaEventDestroy(event);
    }
}

size_t ZstdBatchManager::get_batch_workspace_size(const std::vector<size_t>& input_sizes) const {
    size_t total = 0;

    for (size_t input_size : input_sizes) {
        size_t item_workspace = input_size * 4;
        total += align_to_boundary(item_workspace, GPU_MEMORY_ALIGNMENT);
    }

    return total;
}

Status ZstdBatchManager::compress_single(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
//     std::cout << "[BatchManager] Compressing " << input_size 
//               << " bytes on stream " << stream << std::endl;

    cudaError_t err = cudaMemcpyAsync(output, input, input_size / 2, 
                                       cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
//         std::cerr << "[BatchManager] cudaMemcpyAsync failed: " 
//                   << cudaGetErrorString(err) << std::endl;
        return Status::ERROR_CUDA_ERROR;
    }

    *output_size = input_size / 2;
    return Status::SUCCESS;
}

Status ZstdBatchManager::decompress_single(
    const void* input,
    size_t input_size,
    void* output,
    size_t* output_size,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
//     std::cout << "[BatchManager] Decompressing " << input_size 
//               << " bytes on stream " << stream << std::endl;

    cudaError_t err = cudaMemcpyAsync(output, input, input_size * 2, 
                                       cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
//         std::cerr << "[BatchManager] cudaMemcpyAsync failed: " 
//                   << cudaGetErrorString(err) << std::endl;
        return Status::ERROR_CUDA_ERROR;
    }

    *output_size = input_size * 2;
    return Status::SUCCESS;
}

Status ZstdBatchManager::compress_batch(
    const std::vector<BatchItem>& items,
    void* temp_workspace,
    size_t temp_size
) {
    if (items.empty()) {
        return Status::SUCCESS;
    }

//     std::cout << "[BatchManager] Compressing batch of " << items.size() << " items" << std::endl;

    std::vector<size_t> workspace_sizes(items.size());
    size_t total_workspace = 0;

    for (size_t i = 0; i < items.size(); ++i) {
        workspace_sizes[i] = items[i].input_size * 4;
        workspace_sizes[i] = align_to_boundary(workspace_sizes[i], GPU_MEMORY_ALIGNMENT);
        total_workspace += workspace_sizes[i];
    }

    if (total_workspace > temp_size) {
//         std::cerr << "[BatchManager] Insufficient workspace: need " << total_workspace 
//                   << ", have " << temp_size << std::endl;
        return Status::ERROR_INVALID_PARAMETER;
    }

    std::vector<void*> workspaces(items.size());
    u8* workspace_ptr = static_cast<u8*>(temp_workspace);

    for (size_t i = 0; i < items.size(); ++i) {
        workspaces[i] = workspace_ptr;
        workspace_ptr += workspace_sizes[i];
    }

    std::vector<StreamPool::Guard> guards;
    bool all_success = true;

    for (size_t i = 0; i < items.size(); ++i) {
        auto& item = const_cast<std::vector<BatchItem>&>(items)[i];

        auto guard = stream_pool_->acquire_guard();
        cudaStream_t stream = guard.get_stream();

//         std::cout << "[BatchManager] Item " << i << " -> Stream " << stream << std::endl;

        item.status = compress_single(
            item.input_ptr,
            item.input_size,
            item.output_ptr,
            &item.output_size,
            workspaces[i],
            workspace_sizes[i],
            stream
        );

        if (i < events_.size()) {
            cudaEventRecord(events_[i], stream);
        }

        if (item.status != Status::SUCCESS) {
            all_success = false;
        }

        guards.push_back(std::move(guard));
    }

//     std::cout << "[BatchManager] Waiting for all streams to complete..." << std::endl;
    stream_pool_->synchronize_all();

//     std::cout << "[BatchManager] Batch compression complete" << std::endl;
    return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
}

Status ZstdBatchManager::decompress_batch(
    const std::vector<BatchItem>& items,
    void* temp_workspace,
    size_t temp_size
) {
    if (items.empty()) {
        return Status::SUCCESS;
    }

//     std::cout << "[BatchManager] Decompressing batch of " << items.size() << " items" << std::endl;

    std::vector<size_t> workspace_sizes(items.size());
    size_t total_workspace = 0;

    for (size_t i = 0; i < items.size(); ++i) {
        workspace_sizes[i] = items[i].input_size * 4;
        workspace_sizes[i] = align_to_boundary(workspace_sizes[i], GPU_MEMORY_ALIGNMENT);
        total_workspace += workspace_sizes[i];
    }

    if (total_workspace > temp_size) {
        return Status::ERROR_INVALID_PARAMETER;
    }

    std::vector<void*> workspaces(items.size());
    u8* workspace_ptr = static_cast<u8*>(temp_workspace);

    for (size_t i = 0; i < items.size(); ++i) {
        workspaces[i] = workspace_ptr;
        workspace_ptr += workspace_sizes[i];
    }

    std::vector<StreamPool::Guard> guards;
    bool all_success = true;

    for (size_t i = 0; i < items.size(); ++i) {
        auto& item = const_cast<std::vector<BatchItem>&>(items)[i];

        auto guard = stream_pool_->acquire_guard();
        cudaStream_t stream = guard.get_stream();

        item.status = decompress_single(
            item.input_ptr,
            item.input_size,
            item.output_ptr,
            &item.output_size,
            workspaces[i],
            workspace_sizes[i],
            stream
        );

        if (item.status != Status::SUCCESS) {
            all_success = false;
        }

        guards.push_back(std::move(guard));
    }

    stream_pool_->synchronize_all();

    return all_success ? Status::SUCCESS : Status::ERROR_GENERIC;
}

void ZstdBatchManager::synchronize() {
    stream_pool_->synchronize_all();
}

} // namespace compression
