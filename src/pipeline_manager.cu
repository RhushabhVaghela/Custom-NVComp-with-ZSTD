#include "pipeline_manager.hpp"
#include "cuda_zstd_safe_alloc.h"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace cuda_zstd {

// --- Helper: Thread-Safe Queue ---
template <typename T> class ThreadSafeQueue {
public:
  void push(T val) {
    std::lock_guard<std::mutex> lock(m_);
    q_.push(val);
    cv_.notify_one();
  }

  bool pop(T &val) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [this] { return !q_.empty() || done_; });
    if (q_.empty() && done_)
      return false;
    val = q_.front();
    q_.pop();
    return true;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lock(m_);
    done_ = true;
    cv_.notify_all();
  }

private:
  std::queue<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
  bool done_ = false;
};

// --- PipelinedBatchManager Implementation ---

PipelinedBatchManager::PipelinedBatchManager(const CompressionConfig &config,
                                             size_t batch_size_bytes,
                                             int num_slots)
    : config_(config), batch_size_(batch_size_bytes), num_slots_(num_slots) {
  manager_ = create_manager(config);
  ring_buffer_.resize(num_slots);
  streams_.resize(3); // [0]=H2D, [1]=Compute, [2]=D2H
  init_resources();
}

PipelinedBatchManager::~PipelinedBatchManager() { cleanup_resources(); }

Status PipelinedBatchManager::init_resources() {
  // 1. Create Streams (Non-blocking to allow concurrent host threads)
  for (int i = 0; i < 3; ++i) {
    cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);
  }

  // 2. Determine sizes
  // Note: workspace size here likely assumes single block.
  // Ideally we'd use a robust check, but for O(1) memory, the workspace is
  // fixed. We allocate enough for the worst case ZSTD block size.
  size_t output_bound = manager_->get_max_compressed_size(batch_size_);
  size_t workspace_size = manager_->get_compress_temp_size(batch_size_);

  // 3. Allocate Ring Buffer Slots
  for (int i = 0; i < num_slots_; ++i) {
    if (cuda_zstd::safe_cuda_malloc(&ring_buffer_[i].d_input, batch_size_) != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;
    if (cuda_zstd::safe_cuda_malloc(&ring_buffer_[i].d_output, output_bound) != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;
    if (cuda_zstd::safe_cuda_malloc(&ring_buffer_[i].d_workspace, workspace_size) != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;

    // Host Pinned Allocation for maximum bandwidth
    if (cuda_zstd::safe_cuda_malloc_host(&ring_buffer_[i].h_input, batch_size_) != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;
    if (cuda_zstd::safe_cuda_malloc_host(&ring_buffer_[i].h_output, output_bound) != cudaSuccess)
      return Status::ERROR_OUT_OF_MEMORY;

    ring_buffer_[i].input_capacity = batch_size_;
    ring_buffer_[i].output_capacity = output_bound;
    ring_buffer_[i].workspace_capacity = workspace_size;

    cudaEventCreate(&ring_buffer_[i].event_uploaded);
    cudaEventCreate(&ring_buffer_[i].event_compressed);
    cudaEventCreate(&ring_buffer_[i].event_downloaded);

    // Initial state: "Downloaded" event is recorded so H2D can start
    // immediately
    cudaEventRecord(ring_buffer_[i].event_downloaded, streams_[2]);
  }

  return Status::SUCCESS;
}

void PipelinedBatchManager::cleanup_resources() {
  for (int i = 0; i < 3; ++i) {
    if (streams_[i])
      cudaStreamDestroy(streams_[i]);
  }
  for (auto &slot : ring_buffer_) {
    cudaFree(slot.d_input);
    cudaFree(slot.d_output);
    cudaFree(slot.d_workspace);
    cudaFreeHost(slot.h_input);
    cudaFreeHost(slot.h_output);

    cudaEventDestroy(slot.event_uploaded);
    cudaEventDestroy(slot.event_compressed);
    cudaEventDestroy(slot.event_downloaded);
  }
}

Status PipelinedBatchManager::compress_stream_pipeline(
    std::function<bool(void *h_input, size_t max_len, size_t *out_len)>
        input_callback,
    std::function<void(const void *h_output, size_t size)> output_callback) {
  if (!manager_)
    return Status::ERROR_NOT_INITIALIZED;

  ThreadSafeQueue<int> q_compute;
  ThreadSafeQueue<int> q_d2h;
  std::atomic<bool> pipeline_error{false};
  Status final_status = Status::SUCCESS;

  // --- 2. Compute Thread ---
  std::thread thread_compute([&]() {
    int slot_idx;
    while (q_compute.pop(slot_idx)) {
      if (pipeline_error)
        break;
      auto &slot = ring_buffer_[slot_idx];

      // Wait for Upload to finish (on GPU)
      cudaStreamWaitEvent(streams_[1], slot.event_uploaded, 0);

      // Run Compression (Blocking on CPU)
      size_t comp_size = slot.output_capacity;
      Status s = manager_->compress(
          slot.d_input, slot.current_input_size, slot.d_output, &comp_size,
          slot.d_workspace, slot.workspace_capacity, nullptr, 0, streams_[1]);

      if (s != Status::SUCCESS) {
        pipeline_error = true;
        final_status = s;
      }
      slot.current_output_size = comp_size;

      // Mark Compute Done
      cudaEventRecord(slot.event_compressed, streams_[1]);

      // Notify D2H
      q_d2h.push(slot_idx);
    }
    q_d2h.shutdown();
  });

  // --- 3. D2H Thread ---
  std::thread thread_d2h([&]() {
    int slot_idx;
    while (q_d2h.pop(slot_idx)) {
      if (pipeline_error)
        break;
      auto &slot = ring_buffer_[slot_idx];

      // Wait for Compute to finish
      cudaStreamWaitEvent(streams_[2], slot.event_compressed, 0);

      // Copy Output D2H
      if (slot.current_output_size > 0) {
        cudaMemcpyAsync(slot.h_output, slot.d_output, slot.current_output_size,
                        cudaMemcpyDeviceToHost, streams_[2]);
      }

      // Sync Stream 2
      cudaStreamSynchronize(streams_[2]);

      // Mark as Downloaded
      cudaEventRecord(slot.event_downloaded, streams_[2]);

      // Callback to User
      if (slot.current_output_size > 0) {
        output_callback(slot.h_output, slot.current_output_size);
      }
    }
  });

  // --- 1. H2D (Main Thread Driver) ---
  size_t batch_idx = 0;
  while (!pipeline_error) {
    int slot_idx = batch_idx % num_slots_;
    auto &slot = ring_buffer_[slot_idx];

    // Wait for slot to be free
    cudaEventSynchronize(slot.event_downloaded);

    // Fill Input Buffer
    size_t bytes_read = 0;
    bool has_more =
        input_callback(slot.h_input, slot.input_capacity, &bytes_read);
    slot.current_input_size = bytes_read;

    if (bytes_read == 0 && !has_more) {
      break; // EOF
    }

    // Upload H2D
    cudaMemcpyAsync(slot.d_input, slot.h_input, bytes_read,
                    cudaMemcpyHostToDevice, streams_[0]);
    cudaEventRecord(slot.event_uploaded, streams_[0]);

    // Push to Compute
    q_compute.push(slot_idx);

    batch_idx++;
    if (!has_more)
      break;
  }

  // Wrap up
  q_compute.shutdown();
  thread_compute.join();
  thread_d2h.join();

  return final_status;
}

} // namespace cuda_zstd
