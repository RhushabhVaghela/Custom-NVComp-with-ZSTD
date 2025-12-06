// ==============================================================================
// batch_manager.h - Batch compression with stream pooling
// ==============================================================================

#ifndef BATCH_MANAGER_H
#define BATCH_MANAGER_H

#include "stream_pool.h"
#include "workspace_manager.h"
#include <vector>
#include <memory>

namespace cuda_zstd {

class ZstdBatchManager {
public:
    explicit ZstdBatchManager(size_t num_streams = 8);
    ~ZstdBatchManager();

    Status compress_batch(
        const std::vector<BatchItem>& items,
        void* temp_workspace,
        size_t temp_size
    );

    Status decompress_batch(
        const std::vector<BatchItem>& items,
        void* temp_workspace,
        size_t temp_size
    );

    size_t get_batch_workspace_size(const std::vector<size_t>& input_sizes) const;

    void synchronize();

private:
    std::unique_ptr<StreamPool> stream_pool_;
    std::vector<cudaEvent_t> events_;
    size_t num_streams_;

    Status compress_single(
        const void* input,
        size_t input_size,
        void* output,
        size_t* output_size,
        void* workspace,
        size_t workspace_size,
        cudaStream_t stream
    );

    Status decompress_single(
        const void* input,
        size_t input_size,
        void* output,
        size_t* output_size,
        void* workspace,
        size_t workspace_size,
        cudaStream_t stream
    );
};

} // namespace cuda_zstd

#endif // BATCH_MANAGER_H
