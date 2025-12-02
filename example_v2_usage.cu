#include "lz77_v2_optimal_parse.h"
#include <stdio.h>
#include <chrono>

/**
 * @file example_usage.cu
 * @brief Example showing how to use the V2 optimal parse library
 */

using namespace LZ77_V2;

void example_usage() {
    const u32 input_size = 100000;
    
    printf("V2 Optimal Parse Example\n");
    printf("========================\n\n");
    
    // 1. Allocate device memory
    Match* d_matches;
    ParseCost* d_costs;
    cudaMalloc(&d_matches, input_size * sizeof(Match));
    cudaMalloc(&d_costs, (input_size + 1) * sizeof(ParseCost));
    
    // 2. Initialize matches (your match finder goes here)
    Match* h_matches = new Match[input_size];
    for (u32 i = 0; i < input_size; i++) {
        // Example: match every 10 bytes
        h_matches[i].length = (i % 10 == 0 && i > 10) ? 5 : 0;
        h_matches[i].offset = (i > 10) ? 10 : 0;
    }
    cudaMemcpy(d_matches, h_matches, input_size * sizeof(Match), 
               cudaMemcpyHostToDevice);
    
    // 3. Run V2 optimal parse
    printf("Running V2 optimal parse...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int result = run_optimal_parse(input_size, d_matches, d_costs);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (result == 0) {
        printf("✅ Success! Time: %ld ms\n", ms.count());
        
        // 4. Get final cost
        ParseCost final_cost;
        cudaMemcpy(&final_cost, d_costs + input_size, sizeof(ParseCost),
                   cudaMemcpyDeviceToHost);
        
        printf("Final cost: %u\n", final_cost.cost());
        printf("Valid: %s\n", (final_cost.cost() < 1000000000) ? "YES" : "NO");
    } else {
        printf("❌ Error running optimal parse\n");
    }
    
    // 5. Cleanup
    delete[] h_matches;
    cudaFree(d_matches);
    cudaFree(d_costs);
}

int main() {
    example_usage();
    return 0;
}
