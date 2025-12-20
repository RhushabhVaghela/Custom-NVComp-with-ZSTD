#pragma once

#include "cuda_zstd_manager.h"
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <vector>


namespace cuda_zstd {

struct RingBufferSlot {
  // Device Pointers
  void *d_input = nullptr;
  void *d_output = nullptr;
  void *d_workspace = nullptr;

  // Host Pinned Pointers (for async transfer)
  void *h_input = nullptr;
  void *h_output = nullptr;

  size_t input_capacity = 0;
  size_t output_capacity = 0;
  size_t workspace_capacity = 0;

  // Utilization
  size_t current_input_size = 0;
  size_t current_output_size = 0;

  // Synchronization
  cudaEvent_t event_uploaded = nullptr;
  cudaEvent_t event_compressed = nullptr;
  cudaEvent_t event_downloaded = nullptr; // Marks when slot is free (D2H done)
};

class PipelinedBatchManager {
public:
  explicit PipelinedBatchManager(const CompressionConfig &config,
                                 size_t batch_size_bytes = 64 * 1024 * 1024,
                                 int num_slots = 3);
  ~PipelinedBatchManager();

  PipelinedBatchManager(const PipelinedBatchManager &) = delete;
  PipelinedBatchManager &operator=(const PipelinedBatchManager &) = delete;

  /**
   * @brief Pipeline compression.
   * @param input_callback User function to fill 'h_input' with data. Returns
   * false on EOF. Callback args: (void* buffer_to_fill, size_t max_size) ->
   * returns actual_size in 2nd arg
   * @param output_callback User function to consume 'h_output'.
   */
  Status compress_stream_pipeline(
      std::function<bool(void *h_input, size_t max_len, size_t *out_len)>
          input_callback,
      std::function<void(const void *h_output, size_t size)> output_callback);

private:
  std::unique_ptr<ZstdManager> manager_;
  CompressionConfig config_;
  size_t batch_size_;
  int num_slots_;

  std::vector<RingBufferSlot> ring_buffer_;
  std::vector<cudaStream_t> streams_; // [0]=H2D, [1]=Compute, [2]=D2H

  Status init_resources();
  void cleanup_resources();
};

} // namespace cuda_zstd
