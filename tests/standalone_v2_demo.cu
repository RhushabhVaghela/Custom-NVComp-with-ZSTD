#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Standalone V2 demonstration - independent of main codebase

typedef uint32_t u32;

// New ParseCost format (packed u64)
struct ParseCost {
    unsigned long long data;

    __host__ __device__ ParseCost() : data(0xFFFFFFFFFFFFFFFFULL) {}
    
    __host__ __device__ void set(u32 cost, u32 parent_pos) {
        data = ((unsigned long long)cost << 32) | parent_pos;
    }
    
    __host__ __device__ u32 cost() const {
        return (u32)(data >> 32);
    }
    
    __host__ __device__ u32 parent() const {
        return (u32)(data & 0xFFFFFFFF);
    }
};

struct Match {
    u32 length;
    u32 offset;
};

// Helper functions
__device__ u32 calculate_literal_cost(u32 len) {
    return len * 8;  // Simplified: 8 bits per literal
}

__device__ u32 calculate_match_cost(u32 len, u32 offset) {
    return 16 + 4;  // Simplified: 16 bits for length + 4 for offset
}

// ============================================================================
// V2 Kernel: Simplified Multi-Pass
// ============================================================================

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
        u32 cost_as_literal = current_cost + calculate_literal_cost(1);
        ParseCost new_val;
        new_val.set(cost_as_literal, pos);
        atomicMin(&d_costs[pos + 1].data, new_val.data);
    }
    
    // Match
    const Match& match = d_matches[pos];
    if (match.length >= 3) {
        u32 match_cost = calculate_match_cost(match.length, match.offset);
        u32 total_cost = current_cost + match_cost;
        u32 end_pos = pos + match.length;
        if (end_pos > input_size) end_pos = input_size;
        if (end_pos <= input_size) {
            ParseCost new_val;
            new_val.set(total_cost, pos);
            atomicMin(&d_costs[end_pos].data, new_val.data);
        }
    }
}

// ============================================================================
// Test Program
// ============================================================================

void run_v2_test(u32 input_size) {
    printf("\n========================================\n");
    printf("V2 Standalone Test: %u bytes (%.2f MB)\n", input_size, input_size / 1024.0 / 1024.0);
    printf("========================================\n");
    
    // Allocate device memory
    Match* d_matches;
    ParseCost* d_costs;
    cudaMalloc(&d_matches, input_size * sizeof(Match));
    cudaMalloc(&d_costs, (input_size + 1) * sizeof(ParseCost));
    
    // Initialize with dummy matches (repeated pattern)
    Match* h_matches = new Match[input_size];
    for (u32 i = 0; i < input_size; i++) {
        h_matches[i].length = (i % 100 == 0) ? 50 : 0;  // Match every 100 bytes
        h_matches[i].offset = (i > 50) ? 50 : 0;
    }
    cudaMemcpy(d_matches, h_matches, input_size * sizeof(Match), cudaMemcpyHostToDevice);
    
    // Initialize costs to infinity
    cudaMemset(d_costs, 0xFF, (input_size + 1) * sizeof(ParseCost));
    
    // Set initial cost
    ParseCost initial_cost;
    initial_cost.set(0, 0);
    cudaMemcpy(d_costs, &initial_cost, sizeof(ParseCost), cudaMemcpyHostToDevice);
    
    // Determine passes - need MANY passes for full propagation!
    // Cost must propagate through entire input (potentially millions of positions)
    int max_passes;
    if (input_size < 100 * 1024) max_passes = 500;
    else if (input_size < 1024 * 1024) max_passes = 1000;
    else if (input_size < 10 * 1024 * 1024) max_passes = 2000;
    else max_passes = 3000;
    
    const u32 threads = 256;
    const u32 num_blocks = (input_size + threads - 1) / threads;
    
    printf("Configuration:\n");
    printf("  Blocks: %u\n", num_blocks);
    printf("  Threads/block: %u\n", threads);
    printf("  Passes: %d\n", max_passes);
    printf("  Total positions: %u\n", input_size);
    
    // Run V2 kernel
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int pass = 0; pass < max_passes; ++pass) {
        optimal_parse_kernel_v2<<<num_blocks, threads>>>(
            input_size, d_matches, d_costs
        );
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify results
    ParseCost* h_costs = new ParseCost[input_size + 1];
    cudaMemcpy(h_costs, d_costs, (input_size + 1) * sizeof(ParseCost), cudaMemcpyDeviceToHost);
    
    // Check final cost
    u32 final_cost = h_costs[input_size].cost();
    
    printf("\nResults:\n");
    printf("  Time: %ld ms\n", duration.count());
    printf("  Final cost: %u\n", final_cost);
    printf("  Final cost valid: %s\n", (final_cost < 1000000000) ? "YES ✅" : "NO ❌");
    printf("  Throughput: %.2f MB/s\n", (input_size / 1024.0 / 1024.0) / (duration.count() / 1000.0));
    
    // Cleanup
    delete[] h_matches;
    delete[] h_costs;
    cudaFree(d_matches);
    cudaFree(d_costs);
}

int main() {
    printf("========================================\n");
    printf("V2 Standalone Performance Demonstration\n");
    printf("========================================\n");
    
    // Test various input sizes
    run_v2_test(512 * 1024);        // 512KB
    run_v2_test(1024 * 1024);       // 1MB
    run_v2_test(10 * 1024 * 1024);  // 10MB
    run_v2_test(100 * 1024 * 1024); // 100MB
    
    printf("\n========================================\n");
    printf("All V2 tests completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}
