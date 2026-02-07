// ==============================================================================
// error_context.cpp - Error context implementation (matches error_context.h)
// ==============================================================================

#include "error_context.h"
#include <mutex>

namespace cuda_zstd {
namespace error_handling {

static std::mutex* error_mutex_ptr = nullptr;
static std::once_flag error_mutex_flag;

std::mutex& get_error_mutex() {
    std::call_once(error_mutex_flag, []() {
        error_mutex_ptr = new std::mutex();
    });
    return *error_mutex_ptr;
}

// Define the last_error extern
ErrorContext last_error;

} // namespace error_handling
} // namespace cuda_zstd