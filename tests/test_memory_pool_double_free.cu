#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include <iostream>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    // Ensure some prewarm to create pools
    MemoryPoolManager &g = get_global_pool();
    g.prewarm(8ULL * 1024 * 1024);

    size_t query_size = 1024 * 1024; // 1MB
    void *ptr = g.allocate(query_size, 0);
    if (!ptr) {
        std::cerr << "test_memory_pool_double_free: initial allocation failed" << std::endl;
        return 1;
    }

    // First deallocation should succeed
    cuda_zstd::Status s1 = g.deallocate(ptr);
    if (s1 != cuda_zstd::Status::SUCCESS) {
        std::cerr << "test_memory_pool_double_free: first deallocate failed status=" << (int)s1 << std::endl;
        return 1;
    }

    // Second deallocation should be a double free error
    cuda_zstd::Status s2 = g.deallocate(ptr);
    if (s2 != cuda_zstd::Status::ERROR_INVALID_PARAMETER) {
        std::cerr << "test_memory_pool_double_free: second deallocate expected ERROR_INVALID_PARAMETER but was=" << (int)s2 << std::endl;
        return 1;
    }

    std::cout << "test_memory_pool_double_free PASSED" << std::endl;
    return 0;
}
