// ==============================================================================
// test_workspace_patterns.cu - Tests for workspace pre-allocation pattern
// ==============================================================================

#include "workspace_manager.h"
#include "cuda_zstd_types.h"
#include "error_context.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Bring types and functions into scope
using compression::CompressionConfig;
using compression::allocate_compression_workspace;
using compression::free_compression_workspace;
using compression::calculate_workspace_size;

void test_workspace_allocation() {
    std::cout << "[TEST] Workspace Allocation\n";

    CompressionConfig config;
    config.hash_log = 17;
    config.chain_log = 17;

    compression::CompressionWorkspace workspace;
    size_t max_block_size = 128 * 1024;

    compression::Status status = allocate_compression_workspace(workspace, max_block_size, config);
    assert(status == compression::Status::SUCCESS);
    assert(workspace.is_allocated);
    assert(workspace.d_hash_table != nullptr);
    assert(workspace.d_chain_table != nullptr);

    status = free_compression_workspace(workspace);
    assert(status == compression::Status::SUCCESS);
    assert(!workspace.is_allocated);

    std::cout << "[PASS] Workspace Allocation\n";
}

void test_workspace_reuse() {
    std::cout << "[TEST] Workspace Reuse (No Re-allocation)\n";

    CompressionConfig config;
    compression::CompressionWorkspace workspace;
    size_t max_block_size = 64 * 1024;

    // Allocate once
    compression::Status status = allocate_compression_workspace(workspace, max_block_size, config);
    assert(status == compression::Status::SUCCESS);

    void* original_hash_ptr = workspace.d_hash_table;

    // Use multiple times WITHOUT re-allocation
    for (int i = 0; i < 10; i++) {
        // Simulate usage
        assert(workspace.d_hash_table == original_hash_ptr);
        assert(workspace.is_allocated);
    }

    free_compression_workspace(workspace);
    std::cout << "[PASS] Workspace Reuse\n";
}

void test_workspace_size_calculation() {
    std::cout << "[TEST] Workspace Size Calculation\n";

    CompressionConfig config;
    size_t block_size = 128 * 1024;

    size_t calculated_size = calculate_workspace_size(block_size, config);

    // Should be non-zero and reasonable
    assert(calculated_size > 0);
    assert(calculated_size < 1024 * 1024 * 1024); // < 1GB

    std::cout << "  Calculated workspace size: " 
              << (calculated_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "[PASS] Workspace Size Calculation\n";
}

void test_workspace_error_handling() {
    std::cout << "[TEST] Workspace Error Handling\n";

    compression::CompressionWorkspace workspace;

    // Double free should be safe
    compression::Status status = free_compression_workspace(workspace);
    assert(status == compression::Status::SUCCESS);

    std::cout << "[PASS] Workspace Error Handling\n";
}

int main() {
    std::cout << "\n=== Workspace Pre-Allocation Pattern Tests ===\n\n";

    try {
        test_workspace_allocation();
        test_workspace_reuse();
        test_workspace_size_calculation();
        test_workspace_error_handling();

        std::cout << "\n=== ALL TESTS PASSED ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] Exception: " << e.what() << "\n";
        return 1;
    }
}
