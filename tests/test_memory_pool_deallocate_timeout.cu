#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include <thread>
#include <iostream>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    // Make locks short so we can exercise timed fallback
    setenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS", "100", 1);

    MemoryPoolManager &g = get_global_pool();
    g.prewarm(8ULL * 1024 * 1024);

    // allocate to create mapping
    size_t query_size = 1024 * 1024; // 1MB
    void* ptr = g.allocate(query_size, 0);
    if (!ptr) {
        std::cerr << "Failed to allocate for test setup" << std::endl;
        return 1;
    }

    int pool_idx_to_lock = -1;
    for (int i = 0; i < MemoryPoolManager::NUM_POOL_SIZES; ++i) {
        if (query_size <= MemoryPoolManager::POOL_SIZES[i]) {
            pool_idx_to_lock = i;
            break;
        }
    }

    std::vector<std::unique_lock<std::timed_mutex>> held_locks;
    std::thread locker([&]() {
        bool ok = g.lock_pools_ordered({pool_idx_to_lock}, 0, held_locks);
        if (!ok) {
            std::cerr << "test_memory_pool_deallocate_timeout: locker failed to acquire lock" << std::endl;
            return;
        }
        // Hold lock long enough for the other thread to get timed out
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // Release locks explicitly before exiting thread so the main thread
        // can attempt another deallocation (test expects lock to be released
        // after join()). Clearing the vector destroys unique_lock objects
        // which unlock the mutexes.
        held_locks.clear();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    cuda_zstd::Status s = g.deallocate(ptr);
    // Because the locker is holding the lock and we configured a short
    // timeout this should return ERROR_TIMEOUT.
    if (s != cuda_zstd::Status::ERROR_TIMEOUT) {
        std::cerr << "test_memory_pool_deallocate_timeout: expected ERROR_TIMEOUT, got=" << (int)s << std::endl;
        g.deallocate(ptr);
        locker.join();
        return 1;
    }

    // Now that the lock has been released, this should succeed
    locker.join();
    cuda_zstd::Status s2 = g.deallocate(ptr);
    if (s2 != cuda_zstd::Status::SUCCESS) {
        std::cerr << "test_memory_pool_deallocate_timeout: second deallocate failed with status=" << (int)s2 << std::endl;
        return 1;
    }

    std::cout << "MemoryPool deallocate timeout test PASSED" << std::endl;
    return 0;
}
