#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

typedef uint32_t u32;
typedef uint8_t u8;

// ============================================================================
// ParseCost - New Format
// ============================================================================

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

// ============================================================================
// Simple LZ77 Match Finder (CPU)
// ============================================================================

void find_matches_simple(const u8* input, u32 input_size, Match* matches) {
    const u32 MIN_MATCH = 3;
    const u32 MAX_MATCH = 128;
    const u32 MAX_OFFSET = 32768;
    
    for (u32 i = 0; i < input_size; i++) {
        matches[i].length = 0;
        matches[i].offset = 0;
        
        u32 best_len = 0;
        u32 best_off = 0;
        
        u32 search_start = (i > MAX_OFFSET) ? (i - MAX_OFFSET) : 0;
        
        for (u32 j = search_start; j < i; j++) {
            u32 len = 0;
            while (len < MAX_MATCH && 
                   i + len < input_size && 
                   input[j + len] == input[i + len]) {
                len++;
            }
            
            if (len >= MIN_MATCH && len > best_len) {
                best_len = len;
                best_off = i - j;
            }
        }
        
        if (best_len >= MIN_MATCH) {
            matches[i].length = best_len;
            matches[i].offset = best_off;
        }
    }
}

// ============================================================================
// Cost Functions
// ============================================================================

__device__ u32 calculate_literal_cost(u32 len) {
    return len * 8;  // 8 bits per byte
}

__device__ u32 calculate_match_cost(u32 len, u32 offset) {
    return 20;  // Simplified: fixed overhead for match
}

// ============================================================================
// V2 Kernel
// ============================================================================

__global__ void optimal_parse_kernel_v2(
    u32 input_size,
    const Match* d_matches,
    ParseCost* d_costs
) {
    u32 pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= input_size) return;
    if (d_costs[pos].cost() >= 1000000000) return;
    
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
// Test Data Generation
// ============================================================================

void generate_test_data(u8* data, u32 size) {
    // Repeated pattern for good compression
    const char* pattern = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    u32 pattern_len = strlen(pattern);
    
    for (u32 i = 0; i < size; i++) {
        data[i] = pattern[i % pattern_len];
    }
}

// ============================================================================
// Main Test
// ============================================================================

void run_v2_test_real(u32 input_size) {
    printf("\n========================================\n");
    printf("V2 Test with Real Matches: %.2f MB\n", input_size / 1024.0 / 1024.0);
    printf("========================================\n");
    
    // Generate test data
    u8* h_input = new u8[input_size];
    generate_test_data(h_input, input_size);
    
    // Find matches (CPU)
    printf("Finding matches (CPU)...\n");
    Match* h_matches = new Match[input_size];
    auto match_start = std::chrono::high_resolution_clock::now();
    find_matches_simple(h_input, input_size, h_matches);
    auto match_end = std::chrono::high_resolution_clock::now();
    auto match_time = std::chrono::duration_cast<std::chrono::milliseconds>(match_end - match_start);
    printf("  Match finding: %ld ms\n", match_time.count());
    
    // Copy to device
    Match* d_matches;
    ParseCost* d_costs;
    cudaMalloc(&d_matches, input_size * sizeof(Match));
    cudaMalloc(&d_costs, (input_size + 1) * sizeof(ParseCost));
    cudaMemcpy(d_matches, h_matches, input_size * sizeof(Match), cudaMemcpyHostToDevice);
    
    // Initialize costs
    cudaMemset(d_costs, 0xFF, (input_size + 1) * sizeof(ParseCost));
    ParseCost initial_cost;
    initial_cost.set(0, 0);
    cudaMemcpy(d_costs, &initial_cost, sizeof(ParseCost), cudaMemcpyHostToDevice);
    
    // Configure kernel
    const u32 threads = 256;
    const u32 num_blocks = (input_size + threads - 1) / threads;
    int max_passes = 100;  // Start conservative
    
    printf("\nRunning V2 kernel...\n");
    printf("  Blocks: %u, Threads: %u, Passes: %d\n", num_blocks, threads, max_passes);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int pass = 0; pass < max_passes; ++pass) {
        optimal_parse_kernel_v2<<<num_blocks, threads>>>(input_size, d_matches, d_costs);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results
    ParseCost* h_costs = new ParseCost[input_size + 1];
    cudaMemcpy(h_costs, d_costs, (input_size + 1) * sizeof(ParseCost), cudaMemcpyDeviceToHost);
    
    u32 final_cost = h_costs[input_size].cost();
    
    printf("\nResults:\n");
    printf("  Time: %ld ms\n", duration.count());
    printf("  Final cost: %u\n", final_cost);
    printf("  Valid: %s\n", (final_cost < 1000000000) ? "YES ✅" : "NO ❌");
    if (final_cost < 1000000000) {
        printf("  Throughput: %.2f MB/s\n", (input_size / 1024.0 / 1024.0) / (duration.count() / 1000.0));
        
        // Reconstruct path
        std::vector<u32> path;
        u32 pos = input_size;
        while (pos > 0 && path.size() < 10) {
            path.push_back(pos);
            u32 parent = h_costs[pos].parent();
            if (parent >= pos) break;
            pos = parent;
        }
        printf("  Sample path: ");
        for (int i = path.size() - 1; i >= 0; i--) {
            printf("%u ", path[i]);
        }
        printf("\n");
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_matches;
    delete[] h_costs;
    cudaFree(d_matches);
    cudaFree(d_costs);
}

int main() {
    printf("========================================\n");
    printf("V2 Isolated Test with Real LZ77 Matches\n");
    printf("========================================\n");
    
    run_v2_test_real(100 * 1024);     // 100KB
    run_v2_test_real(1024 * 1024);    // 1MB
    run_v2_test_real(10 * 1024 * 1024); // 10MB
    
    printf("\n========================================\n");
    printf("V2 tests completed!\n");
    printf("========================================\n");
    
    return 0;
}
