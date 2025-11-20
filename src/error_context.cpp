// ==============================================================================
// error_context.cpp - Error context implementation (matches error_context.h)
// ==============================================================================

#include "error_context.h"
#include <mutex>

namespace compression {
namespace error_handling {

// Define the extern variables declared in the header
std::mutex error_mutex;
ErrorContext last_error;

} // namespace error_handling
} // namespace compression