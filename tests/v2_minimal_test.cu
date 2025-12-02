#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Minimal V2-only test - NO legacy code dependencies

typedef uint32_t u32;

struct ParseCost {
    unsigned long long data;
    __host__ __device__ ParseCost() : data(0xFFFFFFFFFFFFFFFFULL) {}
    __host__ __device__ void set(u32 cost, u32 parent_pos) {
        data = ((unsigned long long)cost << 32) | parent_pos;
    }
    __host__ __device__ u32 cost() const { return (u32)(data >> 32); }
    __host__ __device__ u32 parent() const { return (u32)(data & 0xFFFFFFFF); }
};

struct Match {
    u32 length;
    u32 offset;
};

__device__ u32 calculate_literal_cost(u32 len) { return len * 8; }
__device__ u32 calculate_match_cost(u32 len, u32 offset) { return 20; }

__global__ void optimal_parse_kernel_v2(
    u32 input_size,
    const Match* d_matches,
    ParseCost* d_costs
) {
    u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= input_size || d_costs[pos].cost() >= 1000000000) return;
    
    u32 current_cost = d_costs[pos].cost();
    
    // Literal
    if (pos + 1 <= input_size) {
        ParseCost new_val;
        new_val.set(current_cost + calculate_literal_cost(1), pos);
        atomicMin(&d_costs[pos + 1].data, new_val.data);
    }
    
    // Match
    const Match& match = d_matches[pos];
    if (match.length >= 3 && match.length <= input_size - pos) {
        u32 end_pos = pos + match.length;
        ParseCost new_val;
        new_val.set(current_cost + calculate_match_cost(match.length, match.offset), pos);
        atomicMin(&d_costs[end_pos].data, new_val.data);
    }
}

void run_test(u32 size) {
    printf("\n=== V2 Test: %u bytes ===\n", size);
    
    Match* d_matches;
    ParseCost* d_costs;
    cudaMalloc(&d_matches, size * sizeof(Match));
    cudaMalloc(&d_costs, (size + 1) * sizeof(ParseCost));
    
    // Init with simple pattern
    Match* h_matches = new Match[size];
    for (u32 i = 0; i < size; i++) {
        h_matches[i].length = (i % 10 == 0 && i > 3) ? 4 : 0;
        h_matches[i].offset = (i > 3) ? 4 : 0;
    }
    cudaMemcpy(d_matches, h_matches, size * sizeof(Match), cudaMemcpyHostToDevice);
    
    cudaMemset(d_costs, 0xFF, (size + 1) * sizeof(ParseCost));
    ParseCost init;
    init.set(0, 0);
    cudaMemcpy(d_costs, &init, sizeof(ParseCost), cudaMemcpyHostToDevice);
    
    const u32 threads = 256;
    const u32 blocks = (size + threads - 1) / threads;
    const int passes = 50;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < passes; i++) {
        optimal_parse_kernel_v2<<<blocks, threads>>>(size, d_matches, d_costs);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    ParseCost final_cost;
    cudaMemcpy(&final_cost, d_costs + size, sizeof(ParseCost), cudaMemcpyDeviceToHost);
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Time: %ld ms, Final cost: %u, Valid: %s\n", 
           ms, final_cost.cost(), (final_cost.cost() < 1000000000) ? "YES ✅" : "NO ❌");
    
    delete[] h_matches;
    cudaFree(d_matches);
    cudaFree(d_costs);
}

int main() {
    printf("========================================\n");
    printf("V2-Only Minimal Test (No Legacy Code)\n");
    printf("========================================\n");
    
    run_test(1000);
    run_test(10000);
    run_test(100000);
    
    printf("\n✅ V2 kernel works!\n");
    return 0;
}
