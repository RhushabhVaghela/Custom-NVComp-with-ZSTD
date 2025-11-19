// ============================================================================
// test_metadata_roundtrip.cu - RFC 8878 Frame Header Parsing & Metadata
// ============================================================================

#include "cuda_zstd_manager.h"
#include "cuda_error_checking.h"
#include "cuda_zstd_types.h"
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>

using namespace cuda_zstd;

// ============================================================================
// Test: Metadata Roundtrip Compression/Decompression
// ============================================================================

Status test_metadata_roundtrip() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TEST: RFC 8878 Frame Header Metadata Roundtrip\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Initialize all pointers to nullptr
    byte_t *d_input = nullptr, *d_compressed = nullptr, *d_decompressed = nullptr;
    void *d_compress_workspace = nullptr, *d_decompress_workspace = nullptr;
    
    try {
        // === Step 1: Create test data ===
        const size_t DATA_SIZE = 8192;  // 8 KB of test data
        std::vector<byte_t> h_original_data(DATA_SIZE);
        
        // Fill with recognizable pattern
        for (size_t i = 0; i < DATA_SIZE; i++) {
            h_original_data[i] = static_cast<byte_t>(i % 256);
        }
        
        std::cout << " Created test data: " << DATA_SIZE << " bytes\n";
        std::cout << "    Pattern: 0x00, 0x01, 0x02, ..., 0xFF (repeating)\n\n";
        
        // === Step 2: Allocate GPU memory with safe error handling ===
        if (!safe_cuda_malloc((void**)&d_input, DATA_SIZE)) {
            std::cerr << " ✗ FAILED: Failed to allocate input buffer\n";
            return Status::ERROR_CUDA_ERROR;
        }

        if (!safe_cuda_malloc((void**)&d_compressed, DATA_SIZE * 2)) {
            std::cerr << " ✗ FAILED: Failed to allocate compressed buffer\n";
            safe_cuda_free(d_input);
            return Status::ERROR_CUDA_ERROR;
        }

        if (!safe_cuda_malloc((void**)&d_decompressed, DATA_SIZE)) {
            std::cerr << " ✗ FAILED: Failed to allocate decompressed buffer\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Allocated GPU memory:\n";
        std::cout << "    Input buffer: " << DATA_SIZE << " bytes\n";
        std::cout << "    Compressed buffer: " << (DATA_SIZE * 2) << " bytes\n";
        std::cout << "    Decompressed buffer: " << DATA_SIZE << " bytes\n\n";
        
        // === Step 3: Copy input to device ===
        if (!safe_cuda_memcpy(d_input, h_original_data.data(), DATA_SIZE, cudaMemcpyHostToDevice)) {
            std::cerr << " ✗ FAILED: Failed to copy input data to device\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Copied input data to GPU\n\n";
        
        // === Step 4: Create compression manager with safe error handling ===
        std::unique_ptr<ZstdManager> manager;
        try {
            manager = create_manager(5);  // Level 5
            if (!manager) {
                std::cerr << " ✗ FAILED: Failed to create compression manager\n";
                safe_cuda_free(d_input);
                safe_cuda_free(d_compressed);
                safe_cuda_free(d_decompressed);
                return Status::ERROR_CUDA_ERROR;
            }
        } catch (const std::exception& e) {
            std::cerr << " ✗ FAILED: Manager creation failed: " << e.what() << "\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Created compression manager (level 5)\n\n";
        
        // === Step 5: Calculate workspace size ===
        size_t compress_workspace_size = manager->get_compress_temp_size(DATA_SIZE);
        
        if (!safe_cuda_malloc(&d_compress_workspace, compress_workspace_size)) {
            std::cerr << " ✗ FAILED: Failed to allocate compression workspace\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Allocated compression workspace: " << compress_workspace_size 
                  << " bytes\n\n";
        
        // === Step 6: Compress ===
        std::cout << " COMPRESSION PHASE:\n";
        std::cout << "    Compressing " << DATA_SIZE << " bytes...\n";
        
        size_t compressed_size = DATA_SIZE * 2;
        
        Status status = manager->compress(
            d_input,
            DATA_SIZE,
            d_compressed,
            &compressed_size,
            d_compress_workspace,
            compress_workspace_size,
            nullptr,  // No dictionary
            0,        // Dictionary size
            0         // Default stream
        );
        
        if (status != Status::SUCCESS) {
            std::cerr << " ✗ FAILED: Compression failed with status "
                     << static_cast<int>(status) << "\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            return status;
        }
        
        std::cout << "    Compression successful!\n";
        std::cout << "    Original size: " << DATA_SIZE << " bytes\n";
        std::cout << "    Compressed size: " << compressed_size << " bytes\n";
        std::cout << "    Compression ratio: " << std::fixed << std::setprecision(2)
                  << (100.0 * compressed_size / DATA_SIZE) << "%\n\n";
        
        // === Step 7: Copy compressed data to host for inspection ===
        std::vector<byte_t> h_compressed_data(compressed_size);
        if (!safe_cuda_memcpy(h_compressed_data.data(), d_compressed, 
                     compressed_size, cudaMemcpyDeviceToHost)) {
            std::cerr << " ✗ FAILED: Failed to copy compressed data to host\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Copied compressed data to host for inspection\n\n";
        
        // === Step 8: Parse and Verify Frame Header ===
        std::cout << " FRAME HEADER VERIFICATION (RFC 8878):\n";
        
        // Check magic number
        u32 magic_number;
        memcpy(&magic_number, h_compressed_data.data(), 4);
        
        std::cout << "    Magic Number: 0x" << std::hex << magic_number << std::dec << "\n";
        
        if (magic_number != 0xFD2FB528) {
            std::cerr << " ✗ FAILED: Invalid Zstd magic number\n";
            std::cerr << "    Expected: 0xFD2FB528\n";
            std::cerr << "    Got: 0x" << std::hex << magic_number << std::dec << "\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            return Status::ERROR_CORRUPTED_DATA;
        }
        
        std::cout << "    ✓ Magic number is correct (0xFD2FB528)\n";
        
        // Parse Frame Header Descriptor
        byte_t fhd = h_compressed_data[4];
        std::cout << "    Frame Header Descriptor: 0x" << std::hex << (int)fhd << std::dec << "\n";
        
        bool single_segment = (fhd >> 5) & 0x01;
        bool has_checksum = (fhd >> 2) & 0x01;
        bool has_dict_id = (fhd & 0x03) != 0;
        
        std::cout << "    Flags:\n";
        std::cout << "      - Single Segment: " << (single_segment ? "Yes" : "No") << "\n";
        std::cout << "      - Has Checksum: " << (has_checksum ? "Yes" : "No") << "\n";
        std::cout << "      - Has Dictionary ID: " << (has_dict_id ? "Yes" : "No") << "\n";
        
        // Parse content size
        u32 fcs_field_size = (fhd >> 6) & 0x03;
        u32 parsed_content_size = 0;
        u32 header_offset = 5;  // After magic + FHD
        
        if (fcs_field_size == 0 && single_segment) {
            parsed_content_size = h_compressed_data[header_offset];
            std::cout << "    Content Size Field: " << parsed_content_size << " bytes (1 byte)\n";
        } else if (fcs_field_size == 1) {
            u16 size_val;
            memcpy(&size_val, h_compressed_data.data() + header_offset, 2);
            parsed_content_size = size_val + 256;
            std::cout << "    Content Size Field: " << parsed_content_size << " bytes (2 bytes)\n";
        } else if (fcs_field_size == 2) {
            memcpy(&parsed_content_size, h_compressed_data.data() + header_offset, 4);
            std::cout << "    Content Size Field: " << parsed_content_size << " bytes (4 bytes)\n";
        } else if (fcs_field_size == 3) {
            u64 size_val;
            memcpy(&size_val, h_compressed_data.data() + 5, 8);
            parsed_content_size = (u32)size_val;
            std::cout << "    Content Size Field: " << parsed_content_size << " bytes (8 bytes)\n";
        }
        
        if (parsed_content_size != DATA_SIZE) {
            std::cerr << " ✗ FAILED: Content size mismatch in frame header\n";
            std::cerr << "    Expected: " << DATA_SIZE << " bytes\n";
            std::cerr << "    Got: " << parsed_content_size << " bytes\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            return Status::ERROR_CORRUPTED_DATA;
        }
        
        std::cout << "    ✓ Content size is correct\n\n";
        
        // === Step 9: Calculate decompression workspace ===
        size_t decompress_workspace_size = manager->get_decompress_temp_size(compressed_size);
        
        if (!safe_cuda_malloc(&d_decompress_workspace, decompress_workspace_size)) {
            std::cerr << " ✗ FAILED: Failed to allocate decompression workspace\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            return Status::ERROR_CUDA_ERROR;
        }
        
        std::cout << " Allocated decompression workspace: " << decompress_workspace_size 
                  << " bytes\n\n";
        
        // === Step 10: Decompress ===
        std::cout << " DECOMPRESSION PHASE:\n";
        std::cout << "     Decompressing " << compressed_size << " bytes...\n";
        
        size_t decompressed_size = DATA_SIZE;
        
        status = manager->decompress(
            d_compressed,
            compressed_size,
            d_decompressed,
            &decompressed_size,
            d_decompress_workspace,
            decompress_workspace_size
        );
        
        if (status != Status::SUCCESS) {
            std::cerr << " ✗ FAILED: Decompression failed with status "
                     << static_cast<int>(status) << "\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            safe_cuda_free(d_decompress_workspace);
            return status;
        }
        
        std::cout << "     Decompression successful!\n";
        std::cout << "     Decompressed size: " << decompressed_size << " bytes\n\n";
        
        // === Step 11: Verify roundtrip ===
        std::cout << " ROUNDTRIP VERIFICATION:\n";
        
        std::vector<byte_t> h_decompressed_data(decompressed_size);
        if (!safe_cuda_memcpy(h_decompressed_data.data(), d_decompressed, 
                     decompressed_size, cudaMemcpyDeviceToHost)) {
            std::cerr << " ✗ FAILED: Failed to copy decompressed data to host\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            safe_cuda_free(d_decompress_workspace);
            return Status::ERROR_CUDA_ERROR;
        }
        
        // Check sizes match
        if (decompressed_size != DATA_SIZE) {
            std::cerr << " ✗ FAILED: Decompressed size mismatch\n";
            std::cerr << "    Expected: " << DATA_SIZE << " bytes\n";
            std::cerr << "    Got: " << decompressed_size << " bytes\n";
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            safe_cuda_free(d_decompress_workspace);
            return Status::ERROR_CORRUPTED_DATA;
        }
        
        std::cout << "    ✓ Size matches original (" << DATA_SIZE << " bytes)\n";
        
        // Check data matches byte-by-byte
        bool data_matches = true;
        for (size_t i = 0; i < DATA_SIZE; i++) {
            if (h_original_data[i] != h_decompressed_data[i]) {
                std::cerr << " ✗ FAILED: Data mismatch at byte " << i << "\n";
                std::cerr << "    Expected: 0x" << std::hex << (int)h_original_data[i] 
                         << ", Got: 0x" << (int)h_decompressed_data[i] << std::dec << "\n";
                data_matches = false;
                break;
            }
        }
        
        if (!data_matches) {
            safe_cuda_free(d_input);
            safe_cuda_free(d_compressed);
            safe_cuda_free(d_decompressed);
            safe_cuda_free(d_compress_workspace);
            safe_cuda_free(d_decompress_workspace);
            return Status::ERROR_CORRUPTED_DATA;
        }
        
        std::cout << "    ✓ Data matches exactly (byte-by-byte verification)\n";
        
        // === SUCCESS ===
        std::cout << " ✓ PASSED: RFC 8878 Frame Header Metadata Roundtrip Test\n";
        std::cout << std::string(70, '=') << "\n\n";
        
        // === Cleanup ===
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_compress_workspace);
        safe_cuda_free(d_decompress_workspace);
        
        return Status::SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << " ✗ EXCEPTION: " << e.what() << "\n";
        // Cleanup on exception
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        safe_cuda_free(d_decompressed);
        safe_cuda_free(d_compress_workspace);
        safe_cuda_free(d_decompress_workspace);
        return Status::ERROR_INTERNAL;
    }
}

// ============================================================================
// main() - Test Entry Point
// ============================================================================

int main() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CUDA ZSTD - Frame Header Metadata Test\n";
    std::cout << std::string(70, '=') << "\n";
    
    // Skip on CPU-only environments; otherwise print device info
    SKIP_IF_NO_CUDA_RET(0);
    check_cuda_device();
    
    // Run test
    test_metadata_roundtrip();
    
    std::cout << "\n✓ All tests completed\n\n";
    
    return 0;
}
