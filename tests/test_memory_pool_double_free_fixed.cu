// ============================================================================
// test_memory_pool_double_free_fixed.cu - Simplified Double-Free Detection Test
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include <iostream>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    try {
        // Create a simple memory pool manager with minimal complexity
        MemoryPoolManager pool(false); // disable defrag for simpler testing

        // Prewarm with a minimal amount to create pools
        pool.prewarm(1ULL * 1024 * 1024); // Only 1MB instead of 8MB

        size_t query_size = 64 * 1024; // 64KB instead of 1MB (faster allocation)
        void* ptr = pool.allocate(query_size, 0);
        
        if (!ptr) {
            std::cerr << "test_memory_pool_double_free: initial allocation failed" << std::endl;
            return 1;
        }

        std::cout << "Initial allocation successful: ptr=" << ptr << std::endl;

        // First deallocation should succeed
        cuda_zstd::Status s1 = pool.deallocate(ptr);
        if (s1 != cuda_zstd::Status::SUCCESS) {
            std::cerr << "test_memory_pool_double_free: first deallocate failed status=" << (int)s1 << std::endl;
            return 1;
        }

        std::cout << "First deallocation successful" << std::endl;

        // Second deallocation should be a double free error
        cuda_zstd::Status s2 = pool.deallocate(ptr);
        if (s2 != cuda_zstd::Status::ERROR_INVALID_PARAMETER) {
            std::cerr << "test_memory_pool_double_free: second deallocate expected ERROR_INVALID_PARAMETER but was=" << (int)s2 << std::endl;
            return 1;
        }

        std::cout << "Second deallocation correctly detected as double-free" << std::endl;
        std::cout << "test_memory_pool_double_free PASSED" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "test_memory_pool_double_free: Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "test_memory_pool_double_free: Unknown exception" << std::endl;
        return 1;
    }
}
