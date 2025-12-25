#include "error_context.h"
#include <iostream>

namespace cuda_zstd {

void log_error(const ErrorContext &ctx) {
  std::cerr << "[CUDA_ZSTD_ERROR] " << ctx.message << " (File: " << ctx.file
            << ":" << ctx.line << ")" << std::endl;
}

} // namespace cuda_zstd
