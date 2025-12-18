// FIXED Benchmark - cudaDeviceReset() bug removed!
#include "cuda_zstd_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace cuda_zstd;

namespace formulas {
    u32 sqrt_k400(size_t s) { return (u32)(std::sqrt((double)s) * 400.0); }
    u32 logarithmic(size_t s) { return (u32)(512.0*1024.0 * std::pow(s/(1024.0*1024.0), 0.25)); }
    u32 linear_128(size_t s) { return (u32)(s / 128); }
    u32 cuberoot_k150(size_t s) { return (u32)(std::cbrt((double)s) * 150.0); }
    u32 piecewise(size_t s) { return s<10*1024*1024 ? 2*1024*1024 : (s<100*1024*1024 ? 4*1024*1024 : 8*1024*1024); }
    u32 hybrid(size_t s) {
        u32 ideal = (u32)(std::sqrt((double)s) * 400.0);
        size_t blocks = std::clamp(s/ideal, (size_t)64, (size_t)256);
        return (u32)(1 << (u32)std::ceil(std::log2(s/blocks)));
    }
}

struct Result { std::string formula; size_t input_mb; u32 block_kb; double time_ms; double throughput_mbps; };

bool test(const char* name, size_t input_size, u32 block_size, Result& r) {
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) h_input[i] = i % 256;
    
    CompressionConfig config = CompressionConfig::from_level(3);
    config.block_size = std::min(block_size, (u32)input_size);
    ZstdBatchManager mgr(config);
    
    void *d_in, *d_out, *d_tmp;
    if (cudaMalloc(&d_in, input_size) != cudaSuccess) return false;
    if (cudaMemcpy(d_in, h_input.data(), input_size, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d_in); return false; }
    
    size_t max_out = mgr.get_max_compressed_size(input_size);
    size_t tmp_size = mgr.get_compress_temp_size(input_size);
    
    if (cudaMalloc(&d_out, max_out) != cudaSuccess) { cudaFree(d_in); return false; }
    if (cudaMalloc(&d_tmp, tmp_size) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return false; }
    
    size_t out_size = max_out;
    if (mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0) != Status::SUCCESS) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
        return false;
    }
    cudaDeviceSynchronize();
    
    double total = 0;
    for (int i = 0; i < 3; i++) {
        out_size = max_out;
        auto start = std::chrono::high_resolution_clock::now();
        mgr.compress(d_in, input_size, d_out, &out_size, d_tmp, tmp_size, nullptr, 0, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
    
    r = {name, input_size/(1024*1024), block_size/1024, total/3, (input_size/(1024.0*1024.0))/(total/3/1000.0)};
    
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    // NO cudaDeviceReset() - That was the bug!
    return true;
}

int main() {
    std::cout << "==============================================\n"
              << "  FIXED Benchmark (cudaDeviceReset removed)\n"
              << "==============================================\n\n";
    
    struct {const char* n; u32 (*f)(size_t);} formulas[] = {
        {"Sqrt_K400", formulas::sqrt_k400},
        {"Logarithmic", formulas::logarithmic},
        {"Linear_128", formulas::linear_128},
        {"CubeRoot_K150", formulas::cuberoot_k150},
        {"Piecewise", formulas::piecewise},
        {"Hybrid", formulas::hybrid}
    };
    
    size_t sizes[] = {10*1024*1024, 25*1024*1024, 50*1024*1024, 100*1024*1024};
    std::vector<Result> results;
    
    int n=0, total=24;
    for (auto& f : formulas) {
        for (auto sz : sizes) {
            std::cout << "[" << ++n << "/" << total << "] " << f.n << " @ " << (sz/(1024*1024)) 
                      << "MB (block=" << (f.f(sz)/1024) << "KB)... " << std::flush;
            Result r;
            if (test(f.n, sz, f.f(sz), r)) {
                results.push_back(r);
                std::cout << "OK (" << std::fixed << std::setprecision(2) << r.throughput_mbps << " MB/s)\n";
            } else {
                std::cout << "FAILED\n";
            }
        }
    }
    
    std::cout << "\n==============================================\n"
              << "Results: " << results.size() << "/" << total << " passed\n"
              << "==============================================\n\n"
              << "Formula,InputMB,BlockKB,TimeMS,ThroughputMBPS\n";
    for (auto& r : results)
        std::cout << r.formula << "," << r.input_mb << "," << r.block_kb << "," 
                  << std::fixed << std::setprecision(3) << r.time_ms << "," 
                  << std::fixed << std::setprecision(2) << r.throughput_mbps << "\n";
    return 0;
}
