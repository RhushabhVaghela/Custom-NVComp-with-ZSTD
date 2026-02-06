#include "error_context.h"
#include <iostream>

namespace cuda_zstd {

void log_error(const ErrorContext &ctx) {
#ifdef CUDA_ZSTD_DEBUG
  std::cerr << "[CUDA_ZSTD_ERROR] " << ctx.message << " (File: " << ctx.file
            << ":" << ctx.line << ")" << std::endl;
#endif
  // In release builds, errors are reported via Status return codes only.
  // Callers should check the return value of the API that triggered this error.
  (void)ctx; // suppress unused-parameter warning in release
}

} // namespace cuda_zstd
