// ==============================================================================
// error_context.cpp - Error context implementation (matches error_context.h)
// ==============================================================================

#include "error_context.h"
#include <mutex>

namespace cuda_zstd {
namespace error_handling {

// CRITICAL FIX: Use pointer with lazy initialization to avoid static init heap corruption
static std::mutex* error_mutex_ptr = nullptr;

std::mutex& get_error_mutex() {
    if (!error_mutex_ptr) {
        error_mutex_ptr = new std::mutex();
    }
    return *error_mutex_ptr;
}

// Define the last_error extern
ErrorContext last_error;

} // namespace error_handling
} // namespace cuda_zstd