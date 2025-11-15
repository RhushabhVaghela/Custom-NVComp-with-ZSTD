#include "cuda_error_checking.h"
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

using namespace cuda_zstd;

bool run_pool_test_thread(int thread_id) {
    // Create manager instance per thread to avoid internal state sharing
    auto mgr = create_manager(3);

    const size_t data_size = 1024 * 1024; // 1MB
    std::vector<uint8_t> h_data(data_size);
    for (size_t i = 0; i < data_size; i++) h_data[i] = (uint8_t)((i + thread_id) & 0xFF);

    void* d_input = nullptr; void* d_compressed = nullptr; void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_compressed, data_size * 2));

    size_t temp_size = mgr->get_compress_temp_size(data_size);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice));

    size_t compressed_size = 0;
    Status status = mgr->compress(
        d_input, data_size,
        d_compressed, &compressed_size,
        d_temp, temp_size,
        nullptr, 0,
        0 /* stream == 0 triggers StreamPool acquisition */
    );

    if (status != Status::SUCCESS) {
        std::cerr << "Thread " << thread_id << " compress failed: " << status_to_string(status) << std::endl;
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp);
        return false;
    }

    // Decompress single-shot to verify roundtrip
    size_t decomp_temp = mgr->get_decompress_temp_size(compressed_size);
    void* d_decomp_temp = nullptr; CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp));

    void* d_out = nullptr; CUDA_CHECK(cudaMalloc(&d_out, data_size));

    size_t out_size = data_size;
    status = mgr->decompress(d_compressed, compressed_size, d_out, &out_size, d_decomp_temp, decomp_temp, 0);

    if (status != Status::SUCCESS || out_size != data_size) {
        std::cerr << "Thread " << thread_id << " decompress failed: " << status_to_string(status) << std::endl;
        cudaFree(d_input); cudaFree(d_compressed); cudaFree(d_temp); cudaFree(d_decomp_temp); cudaFree(d_out);
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
    return true;
}

int main() {
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();

    const int num_threads = 4;
    std::vector<std::thread> threads;

    std::cout << "Testing Stream Pool with " << num_threads << " concurrent compressions\n";

    std::atomic<bool> ok(true);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, &ok](){
            if (!run_pool_test_thread(i)) ok = false;
        });
    }

    for (auto &t : threads) t.join();

    if (ok) {
        std::cout << "StreamPool test PASSED" << std::endl;
        return 0;
    } else {
        std::cerr << "StreamPool test FAILED" << std::endl;
        return 1;
    }
}
