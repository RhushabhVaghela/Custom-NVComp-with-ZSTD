#include "cuda_error_checking.h"
#include "cuda_zstd_memory_pool.h"
#include <thread>
#include <iostream>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    // Force the pool to prewarm and hold locks via heavier allocations
    setenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS", "100", 1);

    MemoryPoolManager &g = get_global_pool();

    // Instead of relying on prewarm timing, explicitly acquire the pool lock
    // for the 1MB pool and hold it so allocate times out.
    std::vector<std::unique_lock<std::timed_mutex>> held_locks;
    // Compute the pool index for the allocation size to be deterministic.
    size_t query_size = 1024 * 1024; // 1MB
    int pool_idx_to_lock = -1;
    for (int i = 0; i < MemoryPoolManager::NUM_POOL_SIZES; ++i) {
        if (query_size <= MemoryPoolManager::POOL_SIZES[i]) {
            pool_idx_to_lock = i;
            break;
        }
    }

    if (pool_idx_to_lock < 0) {
        std::cerr << "Unable to determine pool index for query_size=" << query_size << "\n";
        return 1;
    }

    std::thread locker([&]() {
        bool ok = g.lock_pools_ordered({pool_idx_to_lock}, 0, held_locks);
        if (!ok) {
            std::cerr << "Failed to acquire initial pool lock in locker thread\n";
            return;
        }
        // Hold the lock for a while to force timeout from the main thread
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });

    // Give the locker thread time to lock
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Attempt an allocation for a size that maps to a pool that is locked.
    // Because we set short pool lock timeout, this should fail fast and
    // return nullptr.
    // query_size already defined above
    void *ptr = g.allocate(query_size, 0);

    if (ptr != nullptr) {
        std::cerr << "TEST FAILED: pool allocation succeeded, but expected lock timeout\n";
        g.deallocate(ptr);
        locker.join();
        return 1;
    }

    locker.join();

    std::cout << "MemoryPool lock-timeout test PASSED\n";
    return 0;
}
