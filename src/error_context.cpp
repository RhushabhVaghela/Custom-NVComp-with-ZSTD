// ==============================================================================
// error_context.cpp - Error context implementation
// ==============================================================================

#include "error_context.h"

namespace compression {
namespace error_handling {

std::mutex error_mutex;
ErrorContext last_error;

} // namespace error_handling
} // namespace compression
