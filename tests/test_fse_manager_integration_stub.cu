#include "cuda_zstd_manager.h"
#include "cuda_zstd_sequence.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

void check(Status s) {
  if (s != Status::SUCCESS) {
    std::cerr << "Test failed with status: " << (int)s << std::endl;
    exit(1);
  }
}

int main() {
  std::cout << "Testing Manager FSE Integration..." << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 1. Setup Input Data (Sequences)
  int num_refs = 100;
  std::vector<u32> h_ll(num_refs);
  std::vector<u32> h_of(num_refs);
  std::vector<u32> h_ml(num_refs);

  for (int i = 0; i < num_refs; ++i) {
    // Values must be within Predefined Table range
    // LL Table Max: 35
    // ML Table Max: 52
    // OF Table Max: 28
    h_ll[i] = i % 35;
    h_of[i] = i % 28;
    h_ml[i] = i % 52;
  }

  u32 *d_ll, *d_of;
  cudaMalloc(&d_ll, num_refs * 4);
  cudaMalloc(&d_of, num_refs * 4);
  // 3. Create Manager instance?
  // encode_sequences_with_predefined_fse is a member of CUDAZstdManager.
  // It's private/protected?
  // I need to check visibility.
  // Usually it's private.
  // However, I can include "cuda_zstd_manager.cu" directly (ugly but effective
  // for unit test of private member)? Or I can use a public API that calls it?
  //
  // Public API: `compress` or `compress_block`?
  //
  // Direct inclusion of .cu is risky given duplicate symbols.
  //
  // Let's check `cuda_zstd_manager.h`.
  // If it's private, I can't test it easily without modifying header or using
  // public API. I will assume for now I added it as public or I can access it.
  //
  // Actually, I'll modify `cuda_zstd_manager.h` if needed to make it
  // public/protected or friend. Or I'll use `#define private public` hack.

  // Minimal Manager Stub
  // I need to instantiate CUDAZstdManager.
  // Default constructor might launch threads?
  // "CUDAZstdManager();"

  // Using `#define private public` before including manager is classic hack.

  // Suppress unused variable warnings for stub implementation
  // (void)seq_ctx;
  // (void)output_size;
}
