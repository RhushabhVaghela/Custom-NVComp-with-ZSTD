// ============================================================================
// benchmark_memory_alloc.cu
//
// Measures overhead of cudaMalloc/cudaFree during compression loop.
// Helps justify the need for a Memory Pool (Phase 2).
//
// Compatible with Asus Zephyrus G16 (32GB RAM / 16GB VRAM)
// - Max allocation size: 32MB (well within hardware limits)
// - Max iterations: 100 (low memory pressure)
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_safe_alloc.h"
#include <chrono>
#include <iostream>
#include <vector>

void benchmark_allocations(size_t size, int iterations) {
  std::cout << "Benchmarking Alloc/Free: Size=" << size
            << ", Iterations=" << iterations << "\n";

  float total_ms = 0;

  for (int i = 0; i < iterations; i++) {
    void *ptr = nullptr;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cuda_zstd::safe_cuda_malloc(&ptr, size);
    cudaFree(ptr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  float avg_ms = total_ms / iterations;
  std::cout << "  Average Alloc+Free Time: " << avg_ms << " ms\n";
  std::cout << "  Total Time: " << total_ms << " ms\n\n";
}

int main() {
  // Test small allocations (metadata buffers)
  benchmark_allocations(1024, 100);

  // Test medium allocations (1MB block)
  benchmark_allocations(1024 * 1024, 100);

  // Test large allocations (Workspace)
  benchmark_allocations(32 * 1024 * 1024, 50);

  return 0;
}
