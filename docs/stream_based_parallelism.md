# Stateless Stream-Based Parallelism for NVComp-Zstandard

## 1. Overview
The current `ZstdBatchManager` is a robust, stateful object that manages its own CUDA streams, events, and temporary memory. While excellent for standard use cases, this stateful model imposes overhead in high-frequency server environments (e.g., thousands of independent requests per second) where creating and destroying a manager per request is expensive, and managing a pool of managers introduces complexity (global mutexes, lifecycle management).

This document outlines the design for a **Stateless Architecture**, enabling true lightweight parallelism where the manager becomes a pure function factory, and the caller manages execution resources.

## 2. Core Concept
**"The Manager is a Library, not an Object."**

In this model, `ZstdManager` (or `ZstdDeviceCompressor`) holds **no mutable state** related to a specific compression job. It contains only read-only configuration (like Huffman tables if pre-computed).

### Key Constraints
1.  **No Internal Mutexes**: All functions must be thread-safe by default.
2.  **Caller-Owned Resources**: The caller provides the `cudaStream_t` and the temporary workspace memory.
3.  **Zero Allocation**: Critical path `compress()` calls must *never* call `cudaMalloc` or `cudaFree`.

## 3. Architecture Comparison

### Current (Stateful)
```cpp
// 1. Heavy Initialization (cudaStreamCreate, cudaMalloc)
ZstdBatchManager manager(config); 

// 2. Serialization (Internal Mutex)
// 3. Implicit Synchronization
manager.compress(input, size, output, ...); 

// 4. Heavy Destruction (cudaDeviceSynchronize, cudaFree)
```

### Proposed (Stateless)
```cpp
// 1. One-time Setup (Global Constant)
static ZstdStatelessCompressor compressor; 

// 2. Per-Request (Cheap)
void handle_request(cudaStream_t stream, void* workspace) {
    // No mutex, no allocation, purely asynchronous submission
    compressor.compress_async(stream, workspace, input, size, output);
}
```

## 4. Implementation Plan

### Phase 1: Stateless Kernel Wrappers
Refactor `cuda_zstd_manager.cu` to expose kernel launch logic as static functions that accept stream handles.

*   `launch_fse_encoder(...)`
*   `launch_lz77_matcher(...)`
*   `launch_huffman_encoder(...)`

### Phase 2: Workspace Management API
The Stateless Manager requires the caller to provide workspace. We need a standalone API to calculate requirements.

```cpp
size_t required_bytes = ZstdStateless::get_workspace_size(size_t max_chunk_size, int compression_level);
```

### Phase 3: The `compress_async` Interface
```cpp
Status compress_async(
    void* d_workspace,      // Pre-allocated scratchpad
    size_t workspace_size,
    const void* d_input,
    size_t input_size,
    void* d_output,
    size_t output_size,
    cudaStream_t stream     // Caller's stream
);
```

### Phase 4: Migration Strategy for `NvcompV5BatchManager`
The `NvcompV5BatchManager` currently wraps the stateful `DefaultZstdManager`. It can be updated to:
1.  Maintain a pool of `cudaStream_t` and workspaces.
2.  Dispatch tasks to the stateless static methods using these pooled resources.

## 5. Benefits
1.  **Driver-Level Parallelism**: By removing internal locks, thousands of threads can submit kernel launches simultaneously to different streams.
2.  **Predictable Latency**: Removing `cudaMalloc` from the hot path eliminates OS involvement and driver stalls.
3.  **Simplification**: Removes complex destructor cleanup logic (which caused the recent concurrency bugs).

## 6. Trade-offs
*   **Caller Responsibility**: Users must manage streams and memory. (Can be mitigated by providing a `ConnectionPool` utility).
*   **API Change**: Moving from an object-oriented API to a functional one.

## Future Architecture (Long-Term)
For high-frequency server environments requiring true lightweight parallelism (1000s of requests/sec or high concurrency context switching):

Stream-Based Parallelism (Stateless Architecture)

Status: Documented in 
docs/stream_based_parallelism.md
Goal: Remove the overhead of creating/destroying 
ZstdBatchManager
 per thread.
Concept: One single shared 
ZstdManager
 instance.
Key Changes:
Remove internal api_mutex (make methods stateless).
Caller provides cudaStream_t and pre-allocated 
workspace
 to 
compress()
.
Manager becomes a factory for kernels, not a state holder.
Benefits:
Zero per-request initialization cost.
Minimal memory overhead (pooled workspaces).
True driver-level parallelism without object lifecycle management.
