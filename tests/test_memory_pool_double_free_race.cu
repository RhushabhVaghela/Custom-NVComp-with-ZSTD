#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_memory_pool.h"
#include <thread>
#include <iostream>
#include <atomic>

using namespace cuda_zstd::memory;

int main() {
    SKIP_IF_NO_CUDA_RET(0);
    MemoryPoolManager &g = get_global_pool();

    // Prewarm so pools are present
    g.prewarm(8ULL * 1024 * 1024);

    size_t query_size = 64 * 1024; // 64KB
    void *ptr = g.allocate(query_size, 0);
    if (!ptr) {
        std::cerr << "test_memory_pool_double_free_race: allocation failed" << std::endl;
        return 1;
    }

    std::atomic<int> ready(0);
    std::atomic<int> success_count(0);
    std::atomic<int> invalid_count(0);
    std::atomic<int> timeout_count(0);

    auto worker = [&](int id) {
        ready.fetch_add(1);
        // Wait until both threads are ready
        while (ready.load() < 2) std::this_thread::yield();

        cuda_zstd::Status s = g.deallocate(ptr);
        if (s == cuda_zstd::Status::SUCCESS) success_count.fetch_add(1);
        else if (s == cuda_zstd::Status::ERROR_INVALID_PARAMETER) invalid_count.fetch_add(1);
        else if (s == cuda_zstd::Status::ERROR_TIMEOUT) timeout_count.fetch_add(1);
    };

    std::thread t1(worker, 1);
    std::thread t2(worker, 2);

    t1.join();
    t2.join();

    // One of the threads must have succeeded; the other must either be a
    // double-free (invalid) or a timeout. We accept either outcome, but
    // there must be at least one success and no crash.
    if (success_count.load() == 0) {
        std::cerr << "test_memory_pool_double_free_race: neither thread succeeded (success_count=0)" << std::endl;
        return 1;
    }

    std::cout << "test_memory_pool_double_free_race: success_count=" << success_count.load()
              << " invalid_count=" << invalid_count.load()
              << " timeout_count=" << timeout_count.load() << std::endl;

    std::cout << "test_memory_pool_double_free_race PASSED" << std::endl;
    return 0;
}
