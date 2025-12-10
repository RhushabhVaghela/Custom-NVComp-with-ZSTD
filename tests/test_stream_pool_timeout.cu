#include "cuda_error_checking.h"
#include "cuda_zstd_stream_pool.h"
#include <iostream>
#include <thread>

using namespace cuda_zstd;

int main() {
  SKIP_IF_NO_CUDA_RET(0);
  // Configure a pool with a single stream for the test
  // Local pool for testing
  StreamPool local_pool(1);
  StreamPool *pool = &local_pool;

  std::cout << "Pool size: " << pool->size() << std::endl;

  // Acquire the only stream using the blocking API and automatically
  // release it at end of this scope.
  {
    auto g1 = pool->acquire();
    if (!g1.get_stream()) {
      std::cerr << "Failed to acquire first stream" << std::endl;
      return 1;
    }

    // Attempt to acquire another stream with a short timeout; should fail.
    auto g2 = pool->acquire_for(100); // 100ms
    if (g2.has_value()) {
      std::cerr << "Timed acquire unexpectedly succeeded" << std::endl;
      return 1;
    }

    std::cout << "Timed acquire properly returned empty optional -> timeout"
              << std::endl;
  }

  return 0;
}
