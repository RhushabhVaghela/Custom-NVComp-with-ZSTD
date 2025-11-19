// workspace_usage.cu
#include "workspace_manager.h"
#include <iostream>

using namespace compression;

// Mock compress function
Status compress_with_workspace(
    const u8* d_input,
    size_t input_size,
    u8* d_output,
    size_t* output_size,
    CompressionWorkspace& workspace
) {
    std::cout << "[Compress] Using pre-allocated workspace" << std::endl;
    
    // ✅ NO ALLOCATIONS - just use pre-partitioned buffers
    u32* hash_table = workspace.d_hash_table;
    u32* chain_table = workspace.d_chain_table;
    lz77::Match* matches = workspace.d_matches;
    lz77::ParseCost* costs = workspace.d_costs;
    
    std::cout << "[Compress] Hash table at: " << hash_table << std::endl;
    std::cout << "[Compress] Matches at: " << matches << std::endl;
    std::cout << "[Compress] Costs at: " << costs << std::endl;
    
    // TODO: Actual compression logic here
    // - Initialize hash table
    // - Find matches
    // - Compute optimal parse
    // - Encode sequences
    
    *output_size = input_size / 2; // Mock compression
    return Status::SUCCESS;
}

int main() {
    std::cout << "=== Workspace Pre-Allocation Pattern Demo ===" << std::endl;
    
    // Configuration
    CompressionConfig config;
    config.window_log = 20;  // 1MB window
    config.hash_log = 17;    // 128K hash table
    config.chain_log = 17;
    
    size_t max_block_size = 128 * 1024;  // 128 KB blocks
    
    // ✅ STEP 1: Allocate workspace ONCE (initialization)
    CompressionWorkspace workspace;
    Status status = allocate_compression_workspace(workspace, max_block_size, config);
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to allocate workspace!" << std::endl;
        return 1;
    }
    
    // Allocate test data
    size_t input_size = 64 * 1024;
    u8* d_input = nullptr;
    u8* d_output = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size * 2);
    
    // ✅ STEP 2: Compress multiple times WITHOUT re-allocating
    std::cout << "\n=== Running 10 compression operations ===" << std::endl;
    for (int i = 0; i < 10; i++) {
        size_t output_size = 0;
        status = compress_with_workspace(d_input, input_size, d_output, &output_size, workspace);
        std::cout << "  Compression " << (i+1) << ": " << output_size << " bytes" << std::endl;
    }
    
    // ✅ STEP 3: Free workspace ONCE (cleanup)
    free_compression_workspace(workspace);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
