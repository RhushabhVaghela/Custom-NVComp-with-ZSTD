#include "cuda_error_checking.h"
#include "cuda_zstd_memory_pool.h"
#include <thread>
#include <iostream>
#include <vector>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);

    MemoryPoolManager &g = get_global_pool();

    // Ensure some prewarm to create pools
    setenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS", "1000", 1);
    g.prewarm(8ULL * 1024 * 1024);

    bool a_ok = false, b_ok = false;

    auto thread_a = std::thread([&]() {
        std::vector<std::unique_lock<std::timed_mutex>> locks;
        bool ok = g.lock_pools_ordered({1, 2}, 2000, locks);
        if (ok) {
            a_ok = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    auto thread_b = std::thread([&]() {
        std::vector<std::unique_lock<std::timed_mutex>> locks;
        bool ok = g.lock_pools_ordered({2, 1}, 2000, locks);
        if (ok) {
            b_ok = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    thread_a.join();
    thread_b.join();

    if (!a_ok || !b_ok) {
        std::cerr << "Lock ordering test FAILED (a_ok=" << a_ok << ", b_ok=" << b_ok << ")\n";
        return 1;
    }

    std::cout << "Memory pool lock ordering test PASSED\n";
    return 0;
}
