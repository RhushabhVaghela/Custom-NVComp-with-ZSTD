// test_fse_reference_comparison.cu
// Purpose: Debug 256KB content mismatch by comparing GPU FSE encoding with a scalar host reference.

#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"
#include "cuda_error_checking.h"
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// =================================================================================================
// SCALAR REFERENCE ENCODER (Host)
// =================================================================================================

struct FSEEncodeSymbolHost {
    u16 newState;
    u32 nbBits;
    u8 symbol;
};

// Helper to fill buffer with random data (Same seed as test_coverage_gaps)
void fill_random(std::vector<byte_t>& buffer) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = (byte_t)dist(rng);
    }
}

// Host-side bitstream writer
struct BitStreamHost {
    std::vector<byte_t> buffer;
    u32 bit_pos;
    u64 bit_buffer;
    u32 bits_in_buffer;

    BitStreamHost(size_t capacity) : buffer(capacity), bit_pos(0), bit_buffer(0), bits_in_buffer(0) {}

    void addBits(u32 value, u32 numBits) {
        u32 mask = (1u << numBits) - 1;
        bit_buffer |= (u64)(value & mask) << bits_in_buffer;
        bits_in_buffer += numBits;

        while (bits_in_buffer >= 8) {
            buffer[bit_pos++] = (byte_t)(bit_buffer & 0xFF);
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }

    void flush() {
        while (bits_in_buffer > 0) {
            buffer[bit_pos++] = (byte_t)(bit_buffer & 0xFF);
            bit_buffer >>= 8;
            if (bits_in_buffer >= 8) bits_in_buffer -= 8; else bits_in_buffer = 0;
        }
    }
};

// Scalar FSE Compressor with Chunking
// Replicates the GPU's parallel chunking logic
void FSE_compress_scalar_host_chunked(
    const std::vector<byte_t>& input,
    const FSEEncodeTable& h_table,
    u32 table_log,
    std::vector<byte_t>& output,
    u32& output_size_bits
) {
    u32 table_size = 1u << table_log;
    u32 num_chunks = 64;
    u32 input_size = (u32)input.size();
    u32 symbols_per_chunk = (input_size + num_chunks - 1) / num_chunks;
    
    // 1. Setup Pass: Find start states for each chunk
    // Runs in reverse over the whole file
    std::vector<u32> chunk_start_states(num_chunks);
    u32 state = table_size;
    
    // Iterate FORWARD to match GPU setup kernel
    for (u32 i = 0; i < input_size; ++i) {
        u32 chunk_id = i / symbols_per_chunk;
        
        // Capture state at the start of the chunk
        if (i == chunk_id * symbols_per_chunk) {
             chunk_start_states[chunk_id] = state;
             printf("Scalar Setup Chunk %u: start_state=%u\n", chunk_id, state);
        }
        
        u8 symbol = input[i];
        const FSEEncodeTable::FSEEncodeSymbol& symInfo = h_table.d_symbol_table[symbol];
        u32 old_state = state;
        u32 nbBitsOut = (state + symInfo.deltaNbBits) >> 16;
        u32 nextStateIndex = (state >> nbBitsOut) + symInfo.deltaFindState;
        state = h_table.d_next_state[nextStateIndex];
        
        if (i < 5) {
            printf("Scalar Setup i=%u: sym=%u, state_in=%u, deltaNbBits=%u, deltaFindState=%u, nbBitsOut=%u, nextStateIndex=%u, state_out=%u\n",
                   i, symbol, old_state, (u32)(symInfo.deltaNbBits), (u32)(symInfo.deltaFindState), nbBitsOut, nextStateIndex, state);
        }
    }
    
    // 2. Parallel Encode Pass (Simulated)
    std::vector<std::vector<byte_t>> chunk_bitstreams(num_chunks);
    std::vector<u32> chunk_bit_counts(num_chunks);
    
    for (u32 chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        u32 in_idx_start = chunk_id * symbols_per_chunk;
        u32 in_idx_end = std::min((chunk_id + 1) * symbols_per_chunk, input_size);
        
        if (in_idx_start >= in_idx_end) continue;
        
        BitStreamHost bs(symbols_per_chunk * 2 + 100);
        u32 chunk_state = chunk_start_states[chunk_id];
        // Encode chunk - Forward
        u32 chunk_total_bits = 0;
        for (u32 i = in_idx_start; i < in_idx_end; ++i) {
            u8 symbol = input[i];
            const FSEEncodeTable::FSEEncodeSymbol& symInfo = h_table.d_symbol_table[symbol];
            u32 nbBitsOut = (chunk_state + symInfo.deltaNbBits) >> 16;
            // Write bits
            bs.addBits(chunk_state, nbBitsOut);
            chunk_total_bits += nbBitsOut;

            // Update state
          u32 old_state = chunk_state;
            u32 val = chunk_state & ((1u << nbBitsOut) - 1);

            u32 idx = (chunk_state >> nbBitsOut) + symInfo.deltaFindState;
            chunk_state = h_table.d_next_state[idx];
            
            if (chunk_id == 0 && i < 5) {
                 printf("Scalar Chunk 0: i=%d, sym=%u, state_in=%u, nbBits=%u, bits=%u, state_out=%u\n",
                        i, symbol, old_state, nbBitsOut, val, chunk_state);
            }
            if (chunk_id == 63) {
                 printf("Scalar Chunk 63: i=%d, sym=%u, state_in=%u, nbBits=%u, bits=%u, state_out=%u\n",
                        i, symbol, old_state, nbBitsOut, val, chunk_state);
            }
        }
        
        // Final state (only for last chunk)
        if (chunk_id == num_chunks - 1) {
            bs.addBits(chunk_state, table_log);
            chunk_total_bits += table_log;
            bs.addBits(1, 1); // Stop bit
            chunk_total_bits += 1;
        }
        
        bs.flush();
        chunk_bitstreams[chunk_id] = bs.buffer;
        chunk_bit_counts[chunk_id] = chunk_total_bits; // Exact bits
        
        // Resize buffer to actual bytes written
        chunk_bitstreams[chunk_id].resize(bs.bit_pos);
        
        if (chunk_id < 5 || chunk_id > 60) printf("Scalar Encode Chunk %u: total_bits=%u, end_state=%u\n", chunk_id, chunk_bit_counts[chunk_id], chunk_state);
    }
    
    // 3. Concatenate Bitstreams
    // The GPU uses `fse_parallel_bitstream_copy_kernel` which handles bit-level concatenation.
    // We must replicate that.
    
    BitStreamHost final_bs(input_size * 2); // Large enough
    
    for (u32 chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        // We append the bits from the chunk bitstream.
        // Since we only have the BYTES, we need to be careful.
        // The chunk bitstream is flushed to bytes.
        // But `chunk_bit_counts[chunk_id]` tells us the valid bits.
        
        const std::vector<byte_t>& chunk_data = chunk_bitstreams[chunk_id];
        u32 bits_to_copy = chunk_bit_counts[chunk_id];
        
        u32 byte_idx = 0;
        u32 bits_read = 0;
        
        while (bits_read < bits_to_copy) {
            u32 bits_in_this_byte = std::min(8u, bits_to_copy - bits_read);
            u8 val = chunk_data[byte_idx++];
            
            // The bits are in the low part of the byte?
            // BitStreamHost fills from LSB.
            // So we just take the low `bits_in_this_byte` bits.
            final_bs.addBits(val, bits_in_this_byte);
            
            bits_read += bits_in_this_byte;
        }
    }
    
    final_bs.flush();
    output = final_bs.buffer;
    output.resize(final_bs.bit_pos);
    output_size_bits = final_bs.bit_pos * 8 - final_bs.bits_in_buffer;
}

// =================================================================================================
// TEST MAIN
// =================================================================================================

int main() {
    printf("=== FSE Reference Comparison Test ===\n");
    
    // 1. Generate Data
    const u32 data_size = 256 * 1024;
    std::vector<byte_t> h_input(data_size);
    fill_random(h_input);
    
    printf("Generated %u bytes of random data\n", data_size);
    
    // 2. Prepare GPU resources
    byte_t* d_input = nullptr;
    byte_t* d_output = nullptr;
    u32* d_output_sizes = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_output, data_size * 2));
    CUDA_CHECK(cudaMalloc(&d_output_sizes, sizeof(u32)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice));
    
    // 3. Run GPU Encoding
    printf("Running GPU Encoding...\n");
    const byte_t* d_inputs_arr[] = { d_input };
    u32 input_sizes_arr[] = { data_size };
    byte_t* d_outputs_arr[] = { d_output };
    
    Status status = encode_fse_batch(
        (const byte_t**)d_inputs_arr,
        input_sizes_arr,
        (byte_t**)d_outputs_arr,
        d_output_sizes,
        1,
        0
    );
    
    if (status != Status::SUCCESS) {
        printf("❌ GPU Encoding failed: %d\n", (int)status);
        return 1;
    }
    
    u32 gpu_output_size = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_output_size, d_output_sizes, sizeof(u32), cudaMemcpyDeviceToHost));
    
    std::vector<byte_t> gpu_output(gpu_output_size);
    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, gpu_output_size, cudaMemcpyDeviceToHost));
    
    printf("GPU Output Size: %u bytes\n", gpu_output_size);
    
    // 4. Run Scalar Reference Encoding
    printf("Running Scalar Reference Encoding...\n");
    
    // We need to build the SAME table as the GPU used.
    // We can reuse the logic from `encode_fse_batch` but we need to extract the table building part.
    // Or we can just call `FSE_buildCTable_Host` ourselves with the same stats.
    
    // Analyze stats
    FSEStats stats;
    stats.total_count = data_size;
    std::vector<u32> freqs(256, 0);
    for (byte_t b : h_input) freqs[b]++;
    
    stats.max_symbol = 0;
    stats.unique_symbols = 0;
    for (int i = 0; i < 256; i++) {
        stats.frequencies[i] = freqs[i];
        if (freqs[i] > 0) {
            stats.max_symbol = i;
            stats.unique_symbols++;
        }
    }
    
    u32 table_log = select_optimal_table_log(stats.frequencies, stats.total_count, stats.max_symbol, stats.unique_symbols);
    u32 table_size = 1u << table_log;
    
    std::vector<u16> h_normalized(stats.max_symbol + 1);
    normalize_frequencies_accurate(stats.frequencies, stats.total_count, table_size, h_normalized.data(), stats.max_symbol, nullptr);
    
    FSEEncodeTable h_table;
    FSE_buildCTable_Host(h_normalized.data(), stats.max_symbol, table_log, &h_table);
    
    // DEBUG: Search for state 590
    for (u32 i = 0; i < table_size; i++) {
        if (h_table.d_next_state[i] == 590) {
            printf("CTable: d_next_state[%u] = 590. Symbol=?, subState=?\n", i);
        }
    }
    
    std::vector<byte_t> scalar_output;
    u32 scalar_bits = 0;
    FSE_compress_scalar_host_chunked(h_input, h_table, table_log, scalar_output, scalar_bits);
    
    printf("Scalar Output Size: %zu bytes\n", scalar_output.size());
    
    // 5. Compare
    // Note: GPU output has a header. Scalar output (as implemented above) is just the bitstream.
    // We need to skip the header in GPU output.
    // Header size = 4 (Log) + 4 (Size) + 4 (MaxSym) + TableSize
    u32 header_size = 12 + (stats.max_symbol + 1) * 2;
    
    printf("Comparing bitstreams (skipping %u byte header)...\n", header_size);
    
    if (gpu_output_size - header_size != scalar_output.size()) {
        printf("❌ Size mismatch! GPU (data): %u, Scalar: %zu\n", gpu_output_size - header_size, scalar_output.size());
    }

    printf("Last 5 bytes Scalar: ");
    for (size_t i = scalar_output.size() - 5; i < scalar_output.size(); i++) printf("%02x ", scalar_output[i]);
    printf("\n");

    printf("Last 5 bytes GPU:    ");
    for (size_t i = gpu_output.size() - 5; i < gpu_output.size(); i++) printf("%02x ", gpu_output[i]);
    printf("\n");

    
    u32 mismatch_count = 0;
    for (size_t i = 0; i < scalar_output.size(); i++) {
        if (i + header_size >= gpu_output.size()) break;
        
        byte_t gpu_byte = gpu_output[i + header_size];
        byte_t scalar_byte = scalar_output[i];
        
        if (gpu_byte != scalar_byte) {
            printf("Mismatch at byte %zu: GPU=0x%02x, Scalar=0x%02x\n", i, gpu_byte, scalar_byte);
            mismatch_count++;
            if (mismatch_count > 10) break;
        }
    }
    
    if (mismatch_count == 0) {
        printf("✅ Bitstreams Match!\n");
    } else {
        printf("❌ Bitstreams Diverge!\n");
    }
    
    // Cleanup
    delete[] h_table.d_symbol_table;
    delete[] h_table.d_next_state;
    delete[] h_table.d_state_to_symbol;
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_sizes);
    
    return 0;
}
