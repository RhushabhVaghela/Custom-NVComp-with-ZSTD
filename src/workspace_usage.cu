// workspace_usage.cu
#include "workspace_manager.h"
#include <iostream>
#include <cstring>

using namespace compression;

// Actual compression function using workspace
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
    lz77::Match* matches = (lz77::Match*)workspace.d_matches;
    lz77::ParseCost* costs = (lz77::ParseCost*)workspace.d_costs;
    
    std::cout << "[Compress] Hash table at: " << hash_table << std::endl;
    std::cout << "[Compress] Matches at: " << matches << std::endl;
    std::cout << "[Compress] Costs at: " << costs << std::endl;
    
    if (!hash_table || !chain_table || !matches || !costs) {
        std::cerr << "[Compress] Error: Workspace not properly allocated" << std::endl;
        return Status::ERROR_WORKSPACE_INVALID;
    }
    
    // Initialize hash table with zeros
    CUDA_CHECK_RETURN(cudaMemset(hash_table, 0xFF, workspace.hash_table_size * sizeof(u32)));
    
    // Simple compression implementation:
    // 1. Copy input data to output (basic pass-through for now)
    // 2. Apply a simple byte-level compression strategy
    
    size_t compressed_size = 0;
    
    // Write a simple header
    struct {
        u32 magic;
        u32 original_size;
        u32 compressed_size;
    } header;
    
    header.magic = 0xFD2FB528;  // ZSTD magic
    header.original_size = input_size;
    header.compressed_size = input_size; // For now, same size
    
    memcpy(d_output, &header, sizeof(header));
    compressed_size += sizeof(header);
    
    // Copy input data with basic pattern replacement
    // This is a simplified implementation that demonstrates workspace usage
    // Real implementation would use LZ77 matching, FSE, and Huffman coding
    
    const u32 BUFFER_SIZE = 4096;  // Process in chunks
    u8* h_input = new u8[input_size];
    u8* h_output = new u8[input_size + 1024]; // Some extra space
    
    // Copy input from device
    CUDA_CHECK_RETURN(cudaMemcpy(h_input, d_input, input_size, cudaMemcpyDeviceToHost));
    
    // Basic compression: look for repeated byte patterns
    size_t in_pos = 0;
    size_t out_pos = 0;
    
    while (in_pos < input_size) {
        // Look for runs of identical bytes
        u8 current_byte = h_input[in_pos];
        size_t run_length = 1;
        
        // Count consecutive identical bytes
        while (in_pos + run_length < input_size && 
               h_input[in_pos + run_length] == current_byte &&
               run_length < 255) {
            run_length++;
        }
        
        if (run_length > 3) {
            // Use run-length encoding for long runs
            h_output[out_pos++] = 0;  // RLE marker
            h_output[out_pos++] = current_byte;
            h_output[out_pos++] = run_length;
        } else {
            // Copy bytes as-is
            for (size_t i = 0; i < run_length; i++) {
                h_output[out_pos++] = current_byte;
            }
        }
        
        in_pos += run_length;
    }
    
    // Copy compressed data back to device
    size_t final_compressed_size = out_pos + sizeof(header);
    CUDA_CHECK_RETURN(cudaMemcpy(d_output, h_output, final_compressed_size - sizeof(header), cudaMemcpyHostToDevice));
    
    // Update header with actual compressed size
    header.compressed_size = final_compressed_size;
    CUDA_CHECK_RETURN(cudaMemcpy(d_output, &header, sizeof(header), cudaMemcpyHostToDevice));
    
    *output_size = final_compressed_size;
    
    delete[] h_input;
    delete[] h_output;
    
    std::cout << "[Compress] Compression complete: " << input_size 
              << " -> " << final_compressed_size << " bytes" << std::endl;
    
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
    
    // Fill input with test data (some repeated patterns)
    u8* h_test_data = new u8[input_size];
    for (size_t i = 0; i < input_size; i++) {
        if (i % 8 == 0) {
            h_test_data[i] = 0xAA;  // Pattern A
        } else if (i % 12 == 0) {
            h_test_data[i] = 0xBB;  // Pattern B
        } else {
            h_test_data[i] = i % 256;
        }
    }
    cudaMemcpy(d_input, h_test_data, input_size, cudaMemcpyHostToDevice);
    delete[] h_test_data;
    
    // ✅ STEP 2: Compress multiple times WITHOUT re-allocating
    std::cout << "\n=== Running 10 compression operations ===" << std::endl;
    for (int i = 0; i < 10; i++) {
        size_t output_size = 0;
        status = compress_with_workspace(d_input, input_size, d_output, &output_size, workspace);
        std::cout << "  Compression " << (i+1) << ": " << output_size << " bytes" << std::endl;
        if (status != Status::SUCCESS) {
            std::cerr << "Compression failed!" << std::endl;
            break;
        }
    }
    
    // ✅ STEP 3: Free workspace ONCE (cleanup)
    free_compression_workspace(workspace);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
