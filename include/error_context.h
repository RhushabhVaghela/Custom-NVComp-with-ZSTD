// ==============================================================================
// error_context.h - Enhanced error handling with context
// ==============================================================================

#ifndef ERROR_CONTEXT_H
#define ERROR_CONTEXT_H

#include "common_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

namespace compression {

// Error context structure
struct ErrorContext {
    Status status = Status::SUCCESS;
    const char* file = nullptr;
    int line = 0;
    const char* function = nullptr;
    const char* message = nullptr;
    cudaError_t cuda_error = cudaSuccess;

    ErrorContext() = default;

    ErrorContext(Status s, const char* f, int l, const char* fn, const char* msg = nullptr)
        : status(s), file(f), line(l), function(fn), message(msg) {}
};

// Global error handling
namespace error_handling {
    extern std::mutex error_mutex;
    extern ErrorContext last_error;

    inline void set_last_error(const ErrorContext& ctx) {
        std::lock_guard<std::mutex> lock(error_mutex);
        last_error = ctx;
    }

    inline ErrorContext get_last_error() {
        std::lock_guard<std::mutex> lock(error_mutex);
        return last_error;
    }

    inline void clear_last_error() {
        std::lock_guard<std::mutex> lock(error_mutex);
        last_error = ErrorContext();
    }

    inline void log_error(const ErrorContext& ctx) {
        set_last_error(ctx);

        std::cerr << "[ERROR] " << status_to_string(ctx.status);
        if (ctx.file) {
            std::cerr << " at " << ctx.file << ":" << ctx.line;
        }
        if (ctx.function) {
            std::cerr << " in " << ctx.function << "()";
        }
        if (ctx.message) {
            std::cerr << ": " << ctx.message;
        }
        if (ctx.cuda_error != cudaSuccess) {
            std::cerr << " (CUDA: " << cudaGetErrorString(ctx.cuda_error) << ")";
        }
        std::cerr << std::endl;
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t __err = (call); \
    if (__err != cudaSuccess) { \
        ErrorContext __ctx(Status::ERROR_CUDA_ERROR, __FILE__, __LINE__, __FUNCTION__); \
        __ctx.cuda_error = __err; \
        error_handling::log_error(__ctx); \
        return Status::ERROR_CUDA_ERROR; \
    } \
} while(0)

// Status checking macro
#define CHECK_STATUS(status) do { \
    if ((status) != Status::SUCCESS) { \
        ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__); \
        error_handling::log_error(__ctx); \
        return (status); \
    } \
} while(0)

// Status checking with message
#define CHECK_STATUS_MSG(status, msg) do { \
    if ((status) != Status::SUCCESS) { \
        ErrorContext __ctx((status), __FILE__, __LINE__, __FUNCTION__, (msg)); \
        error_handling::log_error(__ctx); \
        return (status); \
    } \
} while(0)

// Validation macros
#define VALIDATE_NOT_NULL(ptr, name) do { \
    if (!(ptr)) { \
        ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__, __FUNCTION__, \
                          name " is null"); \
        error_handling::log_error(__ctx); \
        return Status::ERROR_INVALID_PARAMETER; \
    } \
} while(0)

#define VALIDATE_RANGE(val, min, max, name) do { \
    if ((val) < (min) || (val) > (max)) { \
        ErrorContext __ctx(Status::ERROR_INVALID_PARAMETER, __FILE__, __LINE__, __FUNCTION__, \
                          name " out of range"); \
        error_handling::log_error(__ctx); \
        return Status::ERROR_INVALID_PARAMETER; \
    } \
} while(0)

} // namespace compression

#endif // ERROR_CONTEXT_H
