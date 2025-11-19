#include "cuda_zstd_stacktrace.h"
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")
#else
#include <execinfo.h>
#include <iostream>
#endif

namespace cuda_zstd {
namespace util {

std::string capture_stacktrace(int max_frames) {
#ifndef _WIN32
    if (max_frames <= 0) return std::string();
    void **buffer = (void**)malloc(sizeof(void*) * max_frames);
    if (!buffer) return std::string();
    int frames = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, frames);
    std::ostringstream oss;
    for (int i = 0; i < frames; ++i) {
        oss << symbols[i] << "\n";
    }
    free(symbols);
    free(buffer);
    return oss.str();
#else
    // Windows minimal fallback: Capture addresses only; symbol lookup would
    // require more elaborate initialization and DLLs.
    void* *stack = (void**)alloca(sizeof(void*) * max_frames);
    USHORT frames = CaptureStackBackTrace(0, static_cast<ULONG>(max_frames), stack, nullptr);
    std::ostringstream oss;
    for (USHORT i = 0; i < frames; ++i) {
        oss << stack[i] << "\n";
    }
    return oss.str();
#endif
}

void debug_free(void* ptr) {
    if (!ptr) return;
    std::cerr << "debug_free: freeing host ptr=" << ptr << "\n";
    std::string st = cuda_zstd::util::capture_stacktrace(32);
    std::cerr << "debug_free: stacktrace:\n" << st << "\n";
    free(ptr);
}

void* debug_alloc(size_t size) {
    void* p = malloc(size);
    std::cerr << "debug_alloc: allocated host ptr=" << p << " size=" << size << "\n";
    std::string st = cuda_zstd::util::capture_stacktrace(32);
    std::cerr << "debug_alloc: stacktrace:\n" << st << "\n";
    return p;
}

} // namespace util
} // namespace cuda_zstd
