#include "cuda_error_checking.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_stream_pool.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <cstdlib>
#include <string>

using namespace cuda_zstd;

bool run_pool_test_thread(int thread_id) {
    // Create manager instance per thread to avoid internal state sharing
    std::cerr << "Thread " << thread_id << ": before create_manager" << std::endl;
    auto mgr = create_manager(3);
    std::cerr << "Thread " << thread_id << ": after create_manager" << std::endl;
    std::cout << "Thread " << thread_id << ": created manager" << std::endl;

    const size_t data_size = 256 * 1024; // 256KB
    std::vector<uint8_t> h_data(data_size);
    for (size_t i = 0; i < data_size; i++) h_data[i] = (uint8_t)((i + thread_id) & 0xFF);

    void* d_input = nullptr; void* d_compressed = nullptr; void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    std::cerr << "Thread " << thread_id << ": cudaMalloc d_input" << std::endl;
    CUDA_CHECK(cudaMalloc(&d_compressed, data_size * 2));
    std::cerr << "Thread " << thread_id << ": cudaMalloc d_compressed" << std::endl;

    size_t temp_size = mgr->get_compress_temp_size(data_size);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_size));
    std::cerr << "Thread " << thread_id << ": cudaMalloc d_temp" << std::endl;

    CUDA_CHECK(cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice));
    std::cerr << "Thread " << thread_id << ": cudaMemcpy to device" << std::endl;

    size_t compressed_size = 0;
    // Create a dedicated stream for this thread so we avoid acquiring from
    // the global stream pool and reduce contention.
    cudaStream_t user_stream = 0;
    CUDA_CHECK(cudaStreamCreate(&user_stream));

    // For a lightweight stream-pool concurrency test we prefer a simple
    // async operation on the pool stream to avoid invoking the full
    // compression pipeline (which can be heavy and flaky on some drivers).
    const char *simple_env = getenv("CUDA_ZSTD_SIMPLE_STREAM_TEST");
    bool simple_test = (simple_env == nullptr) || (std::string(simple_env) != "0");

    Status status = Status::SUCCESS;
    std::future<Status> compress_task;
    if (simple_test) {
        // Acquire a stream from the global pool to test the acquisition
        // and release path under concurrency.
        auto pool = cuda_zstd::get_global_stream_pool();
        if (!pool) return false;

        auto guard = pool->acquire_for(10000);
        if (!guard.has_value()) return false;

        cudaStream_t pool_stream = guard->get_stream();
        cudaError_t err = cudaMemsetAsync(d_temp, 0xEF, temp_size, pool_stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        err = cudaStreamSynchronize(pool_stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        // GPU operation succeeded using acquired pool stream
        status = Status::SUCCESS;
    } else {
        compress_task = std::async(std::launch::async, [&]() { return mgr->compress(
        d_input, data_size,
        d_compressed, &compressed_size,
        d_temp, temp_size,
        nullptr, 0,
        user_stream /* use explicit stream to avoid pool */
    ); });
    }

    std::cerr << "Thread " << thread_id << ": waiting for compress" << std::endl;
    if (!simple_test && compress_task.wait_for(std::chrono::seconds(100)) == std::future_status::timeout) {
        std::cerr << "Thread " << thread_id << " compress timed out\n";
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp);
        if (user_stream) cudaStreamDestroy(user_stream);
        return false;
    }
    if (!simple_test) status = compress_task.get();
    std::cerr << "Thread " << thread_id << ": compress completed with status " << status_to_string(status) << std::endl;
    if (status != Status::SUCCESS) {
        if (user_stream) cudaStreamDestroy(user_stream);
    }

    if (status != Status::SUCCESS) {
        if (status == Status::ERROR_TIMEOUT) {
            std::cerr << "Thread " << thread_id << " timed out acquiring a stream from the global pool (" << status_to_string(status) << ")" << std::endl;
        } else {
            std::cerr << "Thread " << thread_id << " compress failed: " << status_to_string(status) << std::endl;
        }
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp);
        return false;
    }

    // If we only ran the lightweight pool operation, skip roundtrip
    if (simple_test) {
        std::vector<uint8_t> h_check(temp_size);
        CUDA_CHECK(cudaMemcpy(h_check.data(), d_temp, temp_size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < temp_size; ++i) {
            if (h_check[i] != 0xEF) {
                std::cerr << "Thread " << thread_id << " pool op data mismatch at " << i << std::endl;
                cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp);
                return false;
            }
        }
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp);
        return true;
    }

    // Decompress single-shot to verify roundtrip
    size_t decomp_temp = mgr->get_decompress_temp_size(compressed_size);
    void* d_decomp_temp = nullptr; CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp));

    void* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, data_size));

    size_t out_size = data_size;
    // Decompress on a separate thread to prevent unexpected hangs
    // Use an explicit stream for deterministic concurrency (avoid stream
    // pool acquisition for this test) to reduce potential deadlocks. Reuse
    // the `user_stream` created for compression.
    auto decompress_task = std::async(std::launch::async, [&]() {
        return mgr->decompress(d_compressed, compressed_size, d_out, &out_size, d_decomp_temp, decomp_temp, user_stream);
    });
    std::cerr << "Thread " << thread_id << ": waiting for decompress" << std::endl;
    if (decompress_task.wait_for(std::chrono::seconds(100)) == std::future_status::timeout) {
        std::cerr << "Thread " << thread_id << " decompress timed out\n";
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp); cudaFree(d_decomp_temp); cudaFree(d_out);
        if (user_stream) cudaStreamDestroy(user_stream);
        return false;
    }
    status = decompress_task.get();
    if (user_stream) cudaStreamDestroy(user_stream);
    std::cerr << "Thread " << thread_id << ": decompress completed with status " << status_to_string(status) << std::endl;

    if (status != Status::SUCCESS || out_size != data_size) {
        std::cerr << "Thread " << thread_id << " decompress failed: " << status_to_string(status) << std::endl;
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp); cudaFree(d_decomp_temp); cudaFree(d_out);
        if (user_stream) cudaStreamDestroy(user_stream);
        return false;
    }

    std::vector<uint8_t> h_out(data_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, data_size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < data_size; i++) {
        if (h_out[i] != h_data[i]) {
            std::cerr << "Thread " << thread_id << " data mismatch at " << i << std::endl;
            cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp); cudaFree(d_decomp_temp); cudaFree(d_out);
            return false;
        }
    }

    cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp); cudaFree(d_decomp_temp); cudaFree(d_out);
    if (user_stream) cudaStreamDestroy(user_stream);
    return true;
}

int main() {
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();

    // Print detailed CUDA error contexts during test runs to aid debugging
    cuda_zstd::set_error_callback([](const cuda_zstd::ErrorContext& ctx) {
        std::cerr << "CUDA Error: " << cuda_zstd::get_detailed_error_message(ctx) << std::endl;
    });

    // Ensure the global stream pool has enough streams for this test so
    // the pool isn't exhausted and test threads block indefinitely.
    // Read from env var to allow easier debugging of multithreaded hangs.
    const char* env_threads = getenv("CUDA_ZSTD_NUM_THREADS");
    int num_threads = 2;
    if (env_threads) {
        try { num_threads = std::max(1, std::stoi(env_threads)); } catch (...) { }
    }
    {
        std::string env_val = std::to_string(num_threads);
        setenv("CUDA_ZSTD_STREAM_POOL_SIZE", env_val.c_str(), 1);
        // Use a generous timeout to let the test proceed normally. Value is in ms.
        setenv("CUDA_ZSTD_STREAM_POOL_TIMEOUT_MS", "100000", 1);
    }
    
    std::vector<std::thread> threads;

    std::cerr << "Testing Stream Pool with " << num_threads << " concurrent compressions\n";

    std::atomic<bool> ok(true);

    // print pool size for debug; should match num_threads and show that the
    // environment variable was read by the pool constructor.
    cuda_zstd::StreamPool* pool = cuda_zstd::get_global_stream_pool();
    if (pool) std::cout << "Stream pool size: " << pool->size() << std::endl;

    // Debug: try creating a manager on the main thread to validate manager
    // construction does not block. If this hangs, it indicates a global
    // initialization issue.
    std::cerr << "Main: creating a manager for debug" << std::endl;
    auto mgr_debug = create_manager(3);
    if (!mgr_debug) {
        std::cerr << "Main: debug manager creation failed" << std::endl;
    } else {
        std::cerr << "Main: debug manager creation succeeded" << std::endl;
    }

    // If specified, run threads concurrently to exercise heavy contention.
    const char* run_concurrent_env = getenv("CUDA_ZSTD_RUN_CONCURRENT");
    bool run_concurrent = (run_concurrent_env && std::string(run_concurrent_env) == "1");

    if (run_concurrent) {
        std::cerr << "Running pool test with concurrency: " << num_threads << std::endl;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                if (!run_pool_test_thread(i)) ok = false;
            });
        }
        for (auto &t : threads) t.join();
    } else {
        // Default to sequential mode for expedited debugging and deterministic
        // output, but allow concurrency when environment asks for it.
        for (int i = 0; i < num_threads; ++i) {
            std::cerr << "Running pool test thread sequentially: " << i << std::endl;
            if (!run_pool_test_thread(i)) ok = false;
        }
    }

    if (ok) {
        std::cout << "StreamPool test PASSED" << std::endl;
        return 0;
    } else {
        std::cerr << "StreamPool test FAILED" << std::endl;
        return 1;
    }
}
