// ============================================================================
// cuda_error_checking.h - Comprehensive CUDA Error Checking Utilities
// ============================================================================

#ifndef CUDA_ERROR_CHECKING_H
#define CUDA_ERROR_CHECKING_H

#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "cuda_zstd_safe_alloc.h"

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

// Temporarily override library CUDA_CHECK for tests if it was defined.
#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif
// Check CUDA errors with detailed reporting (test variant) — returns false on failure
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  Function: " << #call << std::endl; \
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

// Check CUDA errors with custom message
#define CUDA_CHECK_MSG(call, msg) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  Function: " << #call << std::endl; \
            std::cerr << "  Message: " << msg << std::endl; \
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

    // Variant that returns a specified value on failure (useful in tests when
    // the surrounding function returns non-bool types like Status or int).
    #define CUDA_CHECK_RET(call, ret) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::cerr << "  Function: " << #call << std::endl; \
                std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
                std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
                return (ret); \
            } \
        } while(0)

    #define CUDA_CHECK_MSG_RET(call, msg, ret) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::cerr << "  Function: " << #call << std::endl; \
                std::cerr << "  Message: " << msg << std::endl; \
                std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
                std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
                return (ret); \
            } \
        } while(0)

// Check CUDA errors but don't return (for void functions)
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  Function: " << #call << std::endl; \
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
            std::abort(); \
        } \
    } while(0)

// Check CUDA errors with cleanup
#define CUDA_CHECK_CLEANUP(call, cleanup) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  Function: " << #call << std::endl; \
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl; \
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl; \
            cleanup; \
            return false; \
        } \
    } while(0)

// Check CUDA device synchronization
#define CUDA_SYNC_CHECK() CUDA_CHECK(cudaDeviceSynchronize())

// Check CUDA device synchronization with message
#define CUDA_SYNC_CHECK_MSG(msg) CUDA_CHECK_MSG(cudaDeviceSynchronize(), msg)

// Check CUDA device synchronization with cleanup
#define CUDA_SYNC_CHECK_CLEANUP(cleanup) CUDA_CHECK_CLEANUP(cudaDeviceSynchronize(), cleanup)

// ============================================================================
// Memory Management Utilities
// ============================================================================

// Safe CUDA malloc with error checking (bool-returning test helper).
// Named test_safe_cuda_malloc to avoid ambiguity with cuda_zstd::safe_cuda_malloc
// in test files that use 'using namespace cuda_zstd'.
static inline bool test_safe_cuda_malloc(void** ptr, size_t size) {
    CUDA_CHECK(cuda_zstd::safe_cuda_malloc(ptr, size));
    if (*ptr == nullptr) {
        std::cerr << "CUDA MALLOC returned null pointer at " << __FILE__ << ":" << __LINE__ << std::endl;
        return false;
    }
    return true;
}

// Convenience overload for typed pointers
template <typename T>
static inline bool test_safe_cuda_malloc(T** ptr, size_t size) {
    return test_safe_cuda_malloc(reinterpret_cast<void**>(ptr), size);
}

// Safe CUDA free (doesn't fail if pointer is null)
void safe_cuda_free(void* ptr) {
    if (ptr != nullptr) {
        cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            std::cerr << "WARNING: CUDA FREE failed at " << __FILE__ << ":" << __LINE__ << std::endl;
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl;
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl;
        }
    }
}

// Safe CUDA memcpy with error checking
bool safe_cuda_memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
    return true;
}

// Safe CUDA memset with error checking
bool safe_cuda_memset(void* devPtr, int value, size_t count) {
    CUDA_CHECK(cudaMemset(devPtr, value, count));
    return true;
}

// ============================================================================
// Stream Management Utilities
// ============================================================================

// Safe CUDA stream creation
bool safe_cuda_stream_create(cudaStream_t* stream) {
    CUDA_CHECK(cudaStreamCreate(stream));
    return true;
}

// Safe CUDA stream destruction
void safe_cuda_stream_destroy(cudaStream_t stream) {
    if (stream != 0) {
        cudaError_t error = cudaStreamDestroy(stream);
        if (error != cudaSuccess) {
            std::cerr << "WARNING: CUDA STREAM DESTROY failed at " << __FILE__ << ":" << __LINE__ << std::endl;
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl;
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl;
        }
    }
}

// Safe CUDA stream synchronization
bool safe_cuda_stream_synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

// ============================================================================
// Event Management Utilities
// ============================================================================

// Safe CUDA event creation
bool safe_cuda_event_create(cudaEvent_t* event) {
    CUDA_CHECK(cudaEventCreate(event));
    return true;
}

// Safe CUDA event destruction
void safe_cuda_event_destroy(cudaEvent_t event) {
    if (event != 0) {
        cudaError_t error = cudaEventDestroy(event);
        if (error != cudaSuccess) {
            std::cerr << "WARNING: CUDA EVENT DESTROY failed at " << __FILE__ << ":" << __LINE__ << std::endl;
            std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl;
            std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl;
        }
    }
}

// Safe CUDA event recording
bool safe_cuda_event_record(cudaEvent_t event, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(event, stream));
    return true;
}

// Safe CUDA event synchronization
bool safe_cuda_event_synchronize(cudaEvent_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
    return true;
}

// ============================================================================
// Device Management Utilities
// ============================================================================

// Check if CUDA device is available
bool check_cuda_device(int device_id = 0) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return false;
    }
    
    if (device_id >= device_count) {
        std::cerr << "Device " << device_id << " not found. Available devices: 0-" << (device_count - 1) << std::endl;
        return false;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    std::cout << "Using CUDA device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

// --------------------------------------------------------------------------
// Lightweight device-presence helper for tests. Returns true if a CUDA
// device is available and can be set; otherwise returns false without
// printing or exiting. Tests should use this to decide to skip device
// specific checks on CPU-only environments.
// --------------------------------------------------------------------------
inline bool has_cuda_device() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        printf("has_cuda_device: cudaGetDeviceCount failed with error %d: %s\n", (int)error, cudaGetErrorString(error));
        return false;
    }
    return device_count > 0;
}

// --------------------------------------------------------------------------
// Test macros for skipping when no device is present; call the variant that
// matches the surrounding function's return value. These do not change test
// logic — they only avoid failing when no GPU is available.
// --------------------------------------------------------------------------
#define LOG_SKIP(name, msg) std::cout << "  [SKIP] " << name << ": " << msg << std::endl
#define SKIP_IF_NO_CUDA() do { if (!has_cuda_device()) { LOG_SKIP(__func__, "No CUDA device available"); return true; } } while(0)
// Use this when the current function returns an int (e.g., main); pass 0 or 1.
#define SKIP_IF_NO_CUDA_RET(ret) do { if (!has_cuda_device()) { LOG_SKIP(__func__, "No CUDA device available"); return (ret); } } while(0)

// Get memory information
bool get_memory_info(size_t& free_mem, size_t& total_mem) {
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return true;
}

// Print memory information
bool print_memory_info(const std::string& label = "") {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    std::cout << "Memory " << (label.empty() ? "" : "(" + label + ") ") 
              << "- Free: " << free_mem / (1024 * 1024) << " MB, "
              << "Total: " << total_mem / (1024 * 1024) << " MB" << std::endl;
    return true;
}

// ============================================================================
// Error Reporting Utilities
// ============================================================================

// Print detailed CUDA error information
void print_cuda_error(cudaError_t error, const char* function, const char* file, int line) {
    std::cerr << "CUDA ERROR DETECTED:" << std::endl;
    std::cerr << "  Location: " << file << ":" << line << std::endl;
    std::cerr << "  Function: " << function << std::endl;
    std::cerr << "  Error Code: " << static_cast<int>(error) << std::endl;
    std::cerr << "  Error Name: " << cudaGetErrorName(error) << std::endl;
    std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl;
}

// Print runtime error information
void print_runtime_error(cudaError_t error, const char* operation, const char* context = "") {
    std::cerr << "CUDA RUNTIME ERROR:" << std::endl;
    std::cerr << "  Operation: " << operation << std::endl;
    if (context && strlen(context) > 0) {
        std::cerr << "  Context: " << context << std::endl;
    }
    std::cerr << "  Error: " << cudaGetErrorName(error) << std::endl;
    std::cerr << "  Description: " << cudaGetErrorString(error) << std::endl;
}

#endif // CUDA_ERROR_CHECKING_H