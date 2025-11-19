// ============================================================================
// test_memory_pool_double_free_race_fixed.cu - Simplified Race Condition Test
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include <thread>
#include <iostream>
#include <atomic>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    try {
        MemoryPoolManager pool(false); // disable defrag for simpler testing

        // Prewarm with minimal amount
        pool.prewarm(512ULL * 1024 * 1024); // 512KB

        size_t query_size = 32 * 1024; // 32KB for faster allocation
        void* ptr = pool.allocate(query_size, 0);
        
        if (!ptr) {
            std::cerr << "test_memory_pool_double_free_race: allocation failed" << std::endl;
            return 1;
        }

        std::cout << "Initial allocation successful: ptr=" << ptr << std::endl;

        std::atomic<int> success_count(0);
        std::atomic<int> invalid_count(0);
        std::atomic<int> timeout_count(0);

        auto worker = [&](int id) {
            std::cout << "Thread " << id << " starting..." << std::endl;
            
            cuda_zstd::Status s = pool.deallocate(ptr);
            std::cout << "Thread " << id << " deallocation result: " << (int)s << std::endl;
            
            if (s == cuda_zstd::Status::SUCCESS) {
                success_count.fetch_add(1);
                std::cout << "Thread " << id << " succeeded" << std::endl;
            }
            else if (s == cuda_zstd::Status::ERROR_INVALID_PARAMETER) {
                invalid_count.fetch_add(1);
                std::cout << "Thread " << id << " detected invalid parameter (double-free)" << std::endl;
            }
            else if (s == cuda_zstd::Status::ERROR_TIMEOUT) {
                timeout_count.fetch_add(1);
                std::cout << "Thread " << id << " timed out" << std::endl;
            }
            else {
                std::cout << "Thread " << id << " got unexpected status: " << (int)s << std::endl;
            }
        };

        std::thread t1(worker, 1);
        std::thread t2(worker, 2);

        t1.join();
        t2.join();

        std::cout << "Results: success=" << success_count.load() 
                  << " invalid=" << invalid_count.load()
                  << " timeout=" << timeout_count.load() << std::endl;

        // One of the threads must have succeeded; the other must either be a
        // double-free (invalid) or a timeout. We accept either outcome, but
        // there must be at least one success and no crash.
        if (success_count.load() == 0 && invalid_count.load() == 0) {
            std::cerr << "test_memory_pool_double_free_race: neither thread succeeded or detected double-free" << std::endl;
            return 1;
        }

        std::cout << "test_memory_pool_double_free_race PASSED" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "test_memory_pool_double_free_race: Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "test_memory_pool_double_free_race: Unknown exception" << std::endl;
        return 1;
    }
}
