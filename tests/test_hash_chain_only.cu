#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_zstd_lz77.h"
#include "cuda_zstd_types.h"

using namespace cuda_zstd;

int main() {
    std::cout << "=== Hash Chain Building Unit Test ===" << std::endl;
    
    // Setup test input
    const size_t input_size = 4096;
    std::vector<uint8_t> h_input(input_size);
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = (uint8_t)(i % 256);
    }
    
    // Allocate device input
    uint8_t* d_input = nullptr;
    cudaError_t err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_input: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    
    // Setup LZ77 context
    lz77::LZ77Context lz77_ctx;
    memset(&lz77_ctx, 0, sizeof(lz77_ctx));
    
    lz77::LZ77Config config;
    config.hash_log = 14;  // 16K hash table
    config.chain_log = 14; // 16K chain table
    config.min_match = 3;
    config.search_depth = 8;
    
    std::cout << "Initializing LZ77 context..." << std::endl;
    auto status = lz77::init_lz77_context(lz77_ctx, config, input_size);
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to init LZ77 context" << std::endl;
        cudaFree(d_input);
        return 1;
    }
    
    // Allocate hash and chain tables
    size_t hash_size = (1 << config.hash_log);
    size_t chain_size = (1 << config.chain_log);
    
    uint32_t* d_hash_table = nullptr;
    uint32_t* d_chain_table = nullptr;
    
    err = cudaMalloc(&d_hash_table, hash_size * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate hash table: " << cudaGetErrorString(err) << std::endl;
        lz77::free_lz77_context(lz77_ctx);
        cudaFree(d_input);
        return 1;
    }
    
    err = cudaMalloc(&d_chain_table, chain_size * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate chain table: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_hash_table);
        lz77::free_lz77_context(lz77_ctx);
        cudaFree(d_input);
        return 1;
    }
    
    // Initialize tables to 0xFF
    std::cout << "Initializing hash/chain tables..." << std::endl;
    cudaMemset(d_hash_table, 0xFF, hash_size * sizeof(uint32_t));
    cudaMemset(d_chain_table, 0xFF, chain_size * sizeof(uint32_t));
    
    // Set up hash and chain tables in context
    lz77_ctx.hash_table.table = d_hash_table;
    lz77_ctx.hash_table.size = hash_size;
    lz77_ctx.chain_table.prev = d_chain_table;
    lz77_ctx.chain_table.size = chain_size;
    
    std::cout << "Calling lz77::find_matches..." << std::endl;
    
    // Create a dummy workspace for find_matches
    // It needs d_matches and d_costs if we were doing full compression, 
    // but find_matches mainly uses hash/chain tables.
    // Wait, find_matches writes to d_matches! We need to allocate that.
    
    size_t matches_size = input_size * sizeof(lz77::Match);
    void* d_matches = nullptr;
    cudaMalloc(&d_matches, matches_size);
    
    // We need a CompressionWorkspace
    CompressionWorkspace workspace;
    workspace.d_hash_table = d_hash_table;
    workspace.hash_table_size = hash_size;
    workspace.d_chain_table = d_chain_table;
    workspace.chain_table_size = chain_size;
    workspace.d_matches = d_matches;
    workspace.max_matches = input_size;
    
    // Call find_matches
    status = lz77::find_matches(
        lz77_ctx,
        d_input,
        input_size,
        nullptr, // no dict
        &workspace,
        nullptr, 0, // no window
        0 // default stream
    );
    
    if (status != Status::SUCCESS) {
        std::cerr << "find_matches failed: " << (int)status << std::endl;
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sync failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_chain_table);
        cudaFree(d_hash_table);
        lz77::free_lz77_context(lz77_ctx);
        cudaFree(d_input);
        return 1;
    }
    
    std::cout << "✅ Hash chain setup successful!" << std::endl;
    std::cout << "   Hash table size: " << hash_size << std::endl;
    std::cout << "   Chain table size: " << chain_size << std::endl;
    std::cout << "   Input size: " << input_size << std::endl;
    
    // Cleanup
    cudaFree(d_chain_table);
    cudaFree(d_hash_table);
    lz77::free_lz77_context(lz77_ctx);
    cudaFree(d_input);
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}
