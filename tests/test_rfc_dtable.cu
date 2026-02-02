// Simple test to verify FSE_buildDTable_rfc debug output
#include <cstdio>
#include <vector>
#include "cuda_zstd_fse.h"
#include "cuda_zstd_fse_rfc.h"

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

int main() {
    printf("Testing FSE_buildDTable_rfc debug output\n");
    
    // Create a simple test table
    const u32 table_log = 6; // 64 entries
    const u32 table_size = 1u << table_log;
    
    // Simple normalized distribution for testing
    std::vector<u16> normalized(36, 1); // 36 symbols, 1 each = 36
    normalized[0] = table_size - 35;    // Make sum = 64
    
    // Build decode table using RFC version
    FSEDecodeTable h_table;
    h_table.newState = new u16[table_size];
    h_table.symbol = new u8[table_size];
    h_table.nbBits = new u8[table_size];
    h_table.table_log = table_log;
    h_table.table_size = table_size;
    
    printf("\n=== Calling FSE_buildDTable_rfc ===\n");
    Status status = FSE_buildDTable_rfc(normalized.data(), 35, table_size, h_table);
    
    if (status == Status::SUCCESS) {
        printf("\n=== Table built successfully ===\n");
        printf("newState[0] = %u\n", h_table.newState[0]);
        printf("newState[1] = %u\n", h_table.newState[1]);
    } else {
        printf("Table build failed with status %d\n", (int)status);
    }
    
    delete[] h_table.newState;
    delete[] h_table.symbol;
    delete[] h_table.nbBits;
    
    return (status == Status::SUCCESS) ? 0 : 1;
}
