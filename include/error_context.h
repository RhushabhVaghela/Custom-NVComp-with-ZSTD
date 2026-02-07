// ==============================================================================
// error_context.h - Enhanced error handling with context
// ==============================================================================

#ifndef ERROR_CONTEXT_H
#define ERROR_CONTEXT_H

#include "cuda_zstd_types.h"
#include <iostream>
#include <mutex>

namespace cuda_zstd {

// ErrorContext is defined in cuda_zstd_types.h

// Global error handling
namespace error_handling {
    // CRITICAL FIX: Use accessor function instead of extern to enable lazy initialization
    std::mutex& get_error_mutex();
    extern ErrorContext last_error;

    inline void set_last_error(const ErrorContext& ctx) {
        std::lock_guard<std::mutex> lock(get_error_mutex());
        last_error = ctx;
    }

    inline ErrorContext get_last_error() {
        std::lock_guard<std::mutex> lock(get_error_mutex());
        return last_error;
    }

    inline void clear_last_error() {
        std::lock_guard<std::mutex> lock(get_error_mutex());
        last_error = ErrorContext();
    }

    inline void log_error(const ErrorContext& ctx) {
        set_last_error(ctx);

#ifdef CUDA_ZSTD_DEBUG
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
#endif
    }
}

} // namespace cuda_zstd

#endif // ERROR_CONTEXT_H
