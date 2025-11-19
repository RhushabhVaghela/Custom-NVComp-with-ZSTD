#pragma once

#include <string>

namespace cuda_zstd {
namespace util {

// Capture a stack trace string. Returns an empty string on platforms that
// don't support obtaining a backtrace.
std::string capture_stacktrace(int max_frames = 32);

// Debug free: prints a stack trace before freeing a host pointer; useful
// for locating host-heap corruption / double-free bugs during debug runs.
void debug_free(void* ptr);

// Debug allocation: record stacktrace and return pointer. Use instead of
// direct malloc for host fallback allocations while debugging.
void* debug_alloc(size_t size);

} // namespace util
} // namespace cuda_zstd
