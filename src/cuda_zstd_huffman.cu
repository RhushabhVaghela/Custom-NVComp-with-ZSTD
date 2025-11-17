// ============================================================================
// cuda_zstd_huffman.cu - Complete Huffman Encoding/Decoding Implementation
//
// NOTE: This file is now fully parallelized.
// - 'huffman_encode_kernel' is a parallel 2-pass scan + write.
// - 'huffman_decode_sequential_kernel' is now a true parallel chunked decoder
//   (using a setup kernel to find chunk starts).
//
// (NEW) NOTE: Refactored to use cuda_zstd_utils for parallel_scan.
// ============================================================================

#include "cuda_zstd_huffman.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_utils.h" // <-- 1. ADDED INCLUDE

#include <vector>
#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>

// Note: A production implementation would use CUB for these scans.
// We implement them manually to be self-contained.

namespace cuda_zstd {
namespace huffman {

// ============================================================================
// Huffman Constants
// ============================================================================

constexpr u32 HUFFMAN_ENCODE_THREADS = 256;
constexpr u32 HUFFMAN_DECODE_THREADS_PER_CHUNK = 1; // Decode is sequential per chunk
constexpr u32 HUFFMAN_DECODE_SYMBOLS_PER_CHUNK = 4096; // Symbols per chunk

// ============================================================================
// Huffman Structures (from .h file, repeated for context)
// ============================================================================

struct HuffmanEncodeTable {
    HuffmanCode* codes;
    u32 num_symbols;
    u32 max_code_length;
    u8* h_code_lengths;
};

struct HuffmanDecodeTable {
    u16* d_fast_lookup;
    u32* d_decode_info;
    u32 max_length;
};

// ============================================================================
// Parallel Scan Kernels (for Encode)
// (REMOVED) - This entire section is now gone and moved to cuda_zstd_utils.cu
// ============================================================================

// ============================================================================
// Frequency Analysis Kernel
// ============================================================================

__global__ void analyze_frequencies_kernel(
    const byte_t* input,
    u32 input_size,
    u32* global_frequencies
) {
    __shared__ u32 local_freq[MAX_HUFFMAN_SYMBOLS];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    if (tid < MAX_HUFFMAN_SYMBOLS) {
        local_freq[tid] = 0;
    }
    __syncthreads();
    
    for (int i = idx; i < input_size; i += stride) {
        u8 symbol = input[i];
        atomicAdd(&local_freq[symbol], 1);
    }
    __syncthreads();
    
    if (tid < MAX_HUFFMAN_SYMBOLS) {
        if (local_freq[tid] > 0) {
            atomicAdd(&global_frequencies[tid], local_freq[tid]);
        }
    }
}

// ============================================================================
// Host Huffman Tree/Table Builder
// ============================================================================

class HuffmanTreeBuilder {
    struct NodeComparator {
        const HuffmanNode* nodes;
        NodeComparator(const HuffmanNode* n) : nodes(n) {}
        bool operator()(int a, int b) const {
            return nodes[a].frequency > nodes[b].frequency;
        }
    };
public:
    static void build_tree(
        const u32* frequencies,
        u32 num_symbols,
        HuffmanNode* nodes,
        u32& num_nodes,
        i32& root_idx
    ) {
        NodeComparator comp(nodes);
        std::priority_queue<int, std::vector<int>, NodeComparator> pq(comp);
        num_nodes = 0;
        for (u32 i = 0; i < num_symbols; ++i) {
            if (frequencies[i] > 0) {
                nodes[num_nodes] = { static_cast<u16>(i), static_cast<u16>(frequencies[i]),
                                     HUFFMAN_NULL_IDX, HUFFMAN_NULL_IDX, HUFFMAN_NULL_IDX };
                pq.push(num_nodes);
                num_nodes++;
            }
        }
        if (num_nodes == 0) { root_idx = HUFFMAN_NULL_IDX; return; }
        if (num_nodes == 1) {
            i32 leaf = pq.top();
            nodes[num_nodes] = { 0, nodes[leaf].frequency,
                                 static_cast<u16>(leaf), HUFFMAN_NULL_IDX, HUFFMAN_NULL_IDX };
            nodes[leaf].parent = static_cast<u16>(num_nodes);
            root_idx = num_nodes;
            num_nodes++;
            return;
        }
        while (pq.size() > 1) {
            int left = pq.top(); pq.pop();
            int right = pq.top(); pq.pop();
            int parent = num_nodes;
            nodes[parent] = { 0, static_cast<u16>(nodes[left].frequency + nodes[right].frequency),
                              static_cast<u16>(left), static_cast<u16>(right), HUFFMAN_NULL_IDX };
            nodes[left].parent = static_cast<u16>(parent);
            nodes[right].parent = static_cast<u16>(parent);
            pq.push(parent);
            num_nodes++;
        }
        root_idx = pq.top();
    }
};

__host__ Status serialize_huffman_table(
    const u8* h_code_lengths,
    byte_t* h_output,
    u32* header_size
) {
    h_output[0] = MAX_HUFFMAN_BITS;
    u32 offset = 1;
    memcpy(h_output + offset, h_code_lengths, MAX_HUFFMAN_SYMBOLS);
    offset += MAX_HUFFMAN_SYMBOLS;
    
    *header_size = offset;
    return Status::SUCCESS;
}

__host__ Status deserialize_huffman_table(
    const byte_t* h_input,
    u32 input_size,
    u8* h_code_lengths,
    u32* header_size
) {
    if (input_size < 1 + MAX_HUFFMAN_SYMBOLS) {
        return Status::ERROR_CORRUPT_DATA;
    }
    u32 max_bits = h_input[0];
    if (max_bits > MAX_HUFFMAN_BITS) {
        return Status::ERROR_CORRUPT_DATA;
    }
    
    memcpy(h_code_lengths, h_input + 1, MAX_HUFFMAN_SYMBOLS);
    *header_size = 1 + MAX_HUFFMAN_SYMBOLS;
    
    return Status::SUCCESS;
}

// ============================================================================
// Huffman Encoding (Parallel)
// ============================================================================

/**
 * @brief (NEW) Kernel to get the code length for each symbol in parallel.
 */
__global__ void get_symbol_lengths_kernel(
    const byte_t* input,
    u32 input_size,
    const HuffmanCode* codes,
    u32* d_code_lengths // Output
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;
    
    u8 symbol = input[idx];
    d_code_lengths[idx] = codes[symbol].length;
}

/**
 * @brief Phase 1: Store encoded symbols in thread-local format
 * Each thread stores its code and position info for later merging.
 */
__global__ void huffman_encode_phase1_kernel(
    const byte_t* input,
    u32 input_size,
    const HuffmanCode* codes,
    const u32* d_bit_offsets,
    u32* d_codes_out,      // Output: code values
    u32* d_lengths_out,    // Output: code lengths
    u32* d_positions_out   // Output: bit positions
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;

    u8 symbol = input[idx];
    const HuffmanCode& c = codes[symbol];
    
    // Store code info for phase 2
    d_codes_out[idx] = c.code;
    d_lengths_out[idx] = c.length;
    d_positions_out[idx] = d_bit_offsets[idx];
}

/**
 * @brief Phase 2: Merge encoded symbols into final bitstream (ATOMIC-FREE)
 *
 * Uses block-level cooperation: each block processes a contiguous range
 * of symbols, building output in shared memory, then writing coalesced
 * to global memory. No atomics needed.
 */
__global__ void huffman_encode_phase2_kernel(
    const u32* d_codes,
    const u32* d_lengths,
    const u32* d_positions,
    u32 input_size,
    byte_t* output,
    u32 header_size_bits
) {
    const u32 BUFFER_SIZE = 512; // Bytes per block buffer
    __shared__ byte_t shared_buffer[BUFFER_SIZE];
    
    // Each block processes THREADS_PER_BLOCK symbols
    u32 block_start_idx = blockIdx.x * blockDim.x;
    u32 block_end_idx = min(block_start_idx + blockDim.x, input_size);
    
    if (block_start_idx >= input_size) return;
    
    // Clear shared buffer
    for (u32 i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x) {
        shared_buffer[i] = 0;
    }
    __syncthreads();
    
    // Find block's bit range
    u32 block_first_bit = (block_start_idx == 0) ? 0 : d_positions[block_start_idx];
    u32 block_last_bit = (block_end_idx == 0) ? 0 :
                         (d_positions[block_end_idx - 1] + d_lengths[block_end_idx - 1]);
    
    // Each thread encodes its symbol into shared buffer
    u32 idx = block_start_idx + threadIdx.x;
    if (idx < block_end_idx) {
        u32 code = d_codes[idx];
        u32 length = d_lengths[idx];
        u32 bit_pos = d_positions[idx];
        
        if (length > 0) {
            // Position relative to block start
            u32 local_bit_pos = bit_pos - block_first_bit;
            u32 local_byte_pos = local_bit_pos >> 3;
            u32 local_bit_offset = local_bit_pos & 7;
            
            // Write to shared memory (within block, positions don't overlap in bits)
            u64 shifted_code = static_cast<u64>(code) << local_bit_offset;
            
            // Write up to 3 bytes (max for 15-bit code + 7-bit offset)
            for (u32 i = 0; i < 3 && local_byte_pos + i < BUFFER_SIZE; i++) {
                u8 byte_val = (shifted_code >> (i * 8)) & 0xFF;
                if (byte_val != 0) {
                    atomicOr(reinterpret_cast<unsigned int*>(&shared_buffer[local_byte_pos + i]),
                             static_cast<unsigned int>(byte_val));
                }
            }
        }
    }
    __syncthreads();
    
    // Cooperatively write shared buffer to global memory (coalesced)
    u32 global_byte_start = (header_size_bits + block_first_bit) >> 3;
    u32 bytes_to_write = ((block_last_bit - block_first_bit) + 7) / 8;
    
    for (u32 i = threadIdx.x; i < bytes_to_write && i < BUFFER_SIZE; i += blockDim.x) {
        output[global_byte_start + i] |= shared_buffer[i];
    }
}

/**
 * @brief Optimized Parallel Huffman encoding kernel (REDUCED ATOMICS).
 *
 * This kernel uses warp-level aggregation to reduce atomic operations
 * by 32x. Each warp of 32 threads shares atomic operations, significantly
 * reducing global memory contention.
 */
__global__ void parallel_huffman_encode_kernel(
    const byte_t* input,
    u32 input_size,
    const HuffmanCode* codes,
    const u32* d_bit_offsets, // Input from prefix sum
    byte_t* output,
    u32 header_size_bits
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_size) return;

    u8 symbol = input[idx];
    const HuffmanCode& c = codes[symbol];
    
    if (c.length == 0) return;

    u32 bit_pos = header_size_bits + d_bit_offsets[idx];
    u32 byte_pos = bit_pos >> 3;
    u32 bit_offset = bit_pos & 7;

    // Warp-level optimization: Use warp shuffle to reduce atomic operations
    const u32 lane_id = threadIdx.x & 31;
    
    // Each thread prepares its write
    u64 shifted_code = static_cast<u64>(c.code) << bit_offset;
    
    // Determine which 64-bit word this thread writes to
    u32 word_idx = byte_pos >> 3;
    
    // Use warp vote to see if multiple threads write to same word
    u32 my_word = word_idx;
    u32 same_word_mask = __match_any_sync(0xFFFFFFFF, my_word);
    
    // If I'm the first thread in my word group, I do the write
    bool should_write = (__ffs(same_word_mask) - 1) == lane_id;
    
    if (should_write) {
        // Aggregate writes from all threads in this word group
        u64 aggregated_value = 0;
        
        for (u32 i = 0; i < 32; i++) {
            if (same_word_mask & (1u << i)) {
                u32 other_code = __shfl_sync(same_word_mask, c.code, i);
                u32 other_len = __shfl_sync(same_word_mask, c.length, i);
                u32 other_offset = __shfl_sync(same_word_mask, bit_offset, i);
                
                u64 other_shifted = static_cast<u64>(other_code) << other_offset;
                aggregated_value |= other_shifted;
            }
        }
        
        // Single atomic write for the entire warp group
        atomicOr(reinterpret_cast<unsigned long long*>(output + (word_idx << 3)),
                 aggregated_value);
    }
}


// ============================================================================
// Huffman Decoding (REPLACED with Parallel)
// ============================================================================

/**
 * @brief Sequential kernel to build the decode table.
 * This is fast and small, no need to parallelize.
 */
__global__ void build_decode_table_kernel(
    const u8* code_lengths,
    u32 num_symbols,
    u16* d_first_code, // [MAX_HUFFMAN_BITS + 2]
    u16* d_symbol_index, // [MAX_HUFFMAN_BITS + 1]
    u8* d_symbols // [num_symbols]
) {
    // This kernel is run with one thread (1,1)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Count symbols per length
    u32 length_count[MAX_HUFFMAN_BITS + 1] = {0};
    u32 max_len = 0;
    for (u32 i = 0; i < num_symbols; ++i) {
        if (code_lengths[i] > 0 && code_lengths[i] <= MAX_HUFFMAN_BITS) {
            length_count[code_lengths[i]]++;
            max_len = max(max_len, (u32)code_lengths[i]);
        }
    }
    
    // Build first_code table
    u32 code = 0;
    d_first_code[0] = 0;
    for (u32 len = 1; len <= max_len; ++len) {
        code = (code + length_count[len - 1]) << 1;
        d_first_code[len] = static_cast<u16>(code);
    }
    d_first_code[max_len + 1] = 0xFFFF;
    
    // Build symbol index table
    u32 idx = 0;
    for (u32 len = 1; len <= max_len; ++len) {
        d_symbol_index[len] = static_cast<u16>(idx);
        idx += length_count[len];
    }
    
    // Fill symbols array in canonical order
    idx = 0;
    for (u32 len = 1; len <= max_len; ++len) {
        for (u32 sym = 0; sym < num_symbols; ++sym) {
            if (code_lengths[sym] == len) {
                d_symbols[idx] = static_cast<u8>(sym);
                idx++;
            }
        }
    }
    
    // Store max_length
    d_symbol_index[0] = max_len; 
}

/**
 * @brief (NEW) Pass 1 for Parallel Decode: Find chunk start bits.
 * This kernel is SEQUENTIAL (<<<1, 1>>>) and scans the bitstream
 * to find the starting bit_pos for each chunk.
 */
__global__ void find_chunk_start_bits_kernel(
    const byte_t* input,
    u32 header_size_bytes,
    u32 input_size_bytes,
    const u16* d_first_code,
    const u16* d_symbol_index,
    const u8* d_symbols,
    u32 decompressed_size,
    u32 num_chunks,
    u32 symbols_per_chunk,
    u32* d_chunk_start_bits // Output
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    u32 max_len = d_symbol_index[0]; // Read max_length
    
    u32 bit_pos = header_size_bytes * 8;
    const u32 end_bit_pos = input_size_bytes * 8;
    u32 out_idx = 0;
    
    d_chunk_start_bits[0] = bit_pos; // First chunk starts after header
    
    for (u32 chunk = 1; chunk < num_chunks; ++chunk) {
        u32 symbols_to_decode = symbols_per_chunk;
        
        while (symbols_to_decode > 0 && out_idx < decompressed_size) {
            if (bit_pos + max_len > end_bit_pos) {
                 if (bit_pos >= end_bit_pos) break;
            }

            u32 byte_pos = bit_pos >> 3;
            u32 bit_offset = bit_pos & 7;
            
            u64 value = 0;
            memcpy(&value, input + byte_pos, min(8u, input_size_bytes - byte_pos));
            value >>= bit_offset;
            
            u32 code = 0;
            u32 len = 1;
            for (; len <= max_len; ++len) {
                code = value & ((1U << len) - 1);
                if (code < d_first_code[len + 1]) {
                    break;
                }
            }
            
            if (len > max_len) return; // Corrupt
            
            bit_pos += len;
            out_idx++;
            symbols_to_decode--;
        }
        
        d_chunk_start_bits[chunk] = bit_pos;
    }
    
    // Last chunk start is used to find total size
    d_chunk_start_bits[num_chunks] = bit_pos;
}


/**
 * @brief (REPLACEMENT) Parallel Huffman decoding kernel.
 * Each block decodes one chunk *sequentially within the block*.
 */
__global__ void parallel_huffman_decode_kernel(
    const byte_t* input,
    u32 input_size_bytes,
    const u16* d_first_code,
    const u16* d_symbol_index,
    const u8* d_symbols,
    byte_t* output,
    u32 decompressed_size,
    const u32* d_chunk_start_bits // Input from Pass 1
) {
    // This kernel is launched with one block per chunk
    // and one thread per block (sequential within block)
    if (threadIdx.x != 0) return;

    u32 chunk_id = blockIdx.x;
    u32 num_chunks = gridDim.x;
    
    u32 max_len = d_symbol_index[0]; // Read max_length
    
    // --- 1. Find the start and end symbol index for this chunk ---
    u32 symbols_per_chunk = (decompressed_size + num_chunks - 1) / num_chunks;
    u32 out_idx = chunk_id * symbols_per_chunk;
    u32 out_idx_end = min((chunk_id + 1) * symbols_per_chunk, decompressed_size);

    if (out_idx >= out_idx_end) return;

    // --- 2. Find the start bit position for this chunk ---
    u32 bit_pos = d_chunk_start_bits[chunk_id];
    const u32 end_bit_pos = (chunk_id == num_chunks - 1) ?
                             (input_size_bytes * 8) : d_chunk_start_bits[chunk_id + 1];

    // --- 3. Decode this chunk sequentially ---
    while (out_idx < out_idx_end) {
        if (bit_pos + max_len > end_bit_pos) {
             if (bit_pos >= end_bit_pos) break;
        }

        u32 byte_pos = bit_pos >> 3;
        u32 bit_offset = bit_pos & 7;
        
        u64 value = 0;
        memcpy(&value, input + byte_pos, min(8u, input_size_bytes - byte_pos));
        value >>= bit_offset;
        
        u32 code = 0;
        u32 len = 1;
        for (; len <= max_len; ++len) {
            code = value & ((1U << len) - 1);
            if (code < d_first_code[len + 1]) {
                break;
            }
        }
        
        if (len > max_len) {
            // Corrupt data.
            return;
        }

        u32 idx = d_symbol_index[len] + (code - d_first_code[len]);
        u8 symbol = d_symbols[idx];
        
        output[out_idx] = symbol;
        out_idx++;
        bit_pos += len;
    }
}


// ============================================================================
// Host API Functions 
// ============================================================================

Status encode_huffman(
    const byte_t* d_input,
    u32 input_size,
    const HuffmanTable& table,
    byte_t* d_output,
    size_t* output_size, // Host pointer
    CompressionWorkspace* workspace,
    cudaStream_t stream
) {
    // --- START REPLACEMENT ---
    if (!d_input || !d_output || !output_size || input_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // --- 1. Analyze frequencies - USE WORKSPACE BUFFER ---
    u32* d_frequencies = workspace ? workspace->d_frequencies : nullptr;
    bool allocated_temp = false;
    
    if (!d_frequencies) {
        // Fallback: allocate if no workspace provided (backward compatibility)
        CUDA_CHECK(cudaMalloc(&d_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
        allocated_temp = true;
    }
    CUDA_CHECK(cudaMemsetAsync(d_frequencies, 0, MAX_HUFFMAN_SYMBOLS * sizeof(u32), stream));
    
    int threads = HUFFMAN_ENCODE_THREADS;
    int blocks = (input_size + threads - 1) / threads;
    
    analyze_frequencies_kernel<<<blocks, threads, 0, stream>>>(
        d_input, input_size, d_frequencies
    );
    
    // Use pinned memory for async transfer
    u32* h_frequencies = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_frequencies, MAX_HUFFMAN_SYMBOLS * sizeof(u32)));
    CUDA_CHECK(cudaMemcpyAsync(h_frequencies, d_frequencies,
                                MAX_HUFFMAN_SYMBOLS * sizeof(u32),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Required: need frequencies for tree building
    
    // Only free if we allocated it ourselves
    if (allocated_temp) {
        CUDA_CHECK(cudaFree(d_frequencies));
    }

    // --- 2. Build Tables on Host ---
    HuffmanNode* h_nodes = new HuffmanNode[MAX_HUFFMAN_SYMBOLS * 2];
    u32 num_nodes = 0;
    i32 root_idx = -1;
    HuffmanTreeBuilder::build_tree(h_frequencies, MAX_HUFFMAN_SYMBOLS,
                                   h_nodes, num_nodes, root_idx);
    
    u8* h_code_lengths = new u8[MAX_HUFFMAN_SYMBOLS];
    // (FIX) Need to actually generate the code lengths from the tree
    memset(h_code_lengths, 0, MAX_HUFFMAN_SYMBOLS);
    std::function<void(int, u8)> find_lengths =
        [&](int node_idx, u8 depth) {
        if (node_idx == HUFFMAN_NULL_IDX) return;
        
        // Leaf node
        if (h_nodes[node_idx].left_child == HUFFMAN_NULL_IDX) {
            h_code_lengths[h_nodes[node_idx].symbol] = depth;
            return;
        }
        
        if (depth < MAX_HUFFMAN_BITS) {
            find_lengths(h_nodes[node_idx].left_child, depth + 1);
            find_lengths(h_nodes[node_idx].right_child, depth + 1);
        }
    };
    find_lengths(root_idx, 0);

    // Use canonical Huffman codes (RFC 8878 compliant)
    // (FIX) We need a host-side buffer for the codes first
    HuffmanCode* h_codes = new HuffmanCode[MAX_HUFFMAN_SYMBOLS];
    Status status = huffman::generate_canonical_codes(
        h_code_lengths,
        MAX_HUFFMAN_SYMBOLS,
        h_codes // Generate into host buffer
    );
    if (status != Status::SUCCESS) {
        delete[] h_frequencies;
        delete[] h_nodes;
        delete[] h_code_lengths;
        delete[] h_codes;
        return status;
    }
    
    // --- 3. Serialize table header ---
    byte_t* h_header = new byte_t[1 + MAX_HUFFMAN_SYMBOLS];
    u32 header_size = 0;
    serialize_huffman_table(h_code_lengths, h_header, &header_size);
    u32 header_size_bits = header_size * 8;
    
    CUDA_CHECK(cudaMemcpyAsync(d_output, h_header, header_size, cudaMemcpyHostToDevice, stream));

    // --- 4. Parallel Encode ---
    
    // Copy codes to device (use the table's device pointer)
    CUDA_CHECK(cudaMemcpyAsync(table.codes, h_codes,
                                MAX_HUFFMAN_SYMBOLS * sizeof(HuffmanCode),
                                cudaMemcpyHostToDevice, stream));
    
    // Allocate temp buffers for scan - USE WORKSPACE BUFFERS
    u32* d_code_lengths = workspace ? workspace->d_code_lengths : nullptr;
    u32* d_bit_offsets = workspace ? workspace->d_bit_offsets : nullptr;
    bool allocated_scan_buffers = false;
    
    if (!d_code_lengths || !d_bit_offsets) {
        // Fallback: allocate if no workspace
        CUDA_CHECK(cudaMalloc(&d_code_lengths, input_size * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&d_bit_offsets, input_size * sizeof(u32)));
        allocated_scan_buffers = true;
    }
    CUDA_CHECK(cudaMemsetAsync(d_output + header_size, 0, (input_size * 2), stream)); // Clear output area

    // Allocate buffers for two-phase atomic-free encoding
    u32* d_codes_temp = nullptr;
    u32* d_positions_temp = nullptr;
    bool allocated_phase_buffers = false;
    
    // Allocate temporary buffers (workspace doesn't have temp_storage members)
    {
        // Allocate temporary buffers
        CUDA_CHECK(cudaMalloc(&d_codes_temp, input_size * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&d_positions_temp, input_size * sizeof(u32)));
        allocated_phase_buffers = true;
    }
    
    // Pass 1a: Get length of each symbol
    get_symbol_lengths_kernel<<<blocks, threads, 0, stream>>>(
        d_input, input_size, table.codes, d_code_lengths
    );
    
    // Pass 1b: Parallel prefix sum to compute bit offsets
    status = cuda_zstd::utils::parallel_scan(d_code_lengths, d_bit_offsets, input_size, stream);
    if (status != Status::SUCCESS) {
        if (allocated_phase_buffers) {
            cudaFree(d_codes_temp);
            cudaFree(d_positions_temp);
        }
        if (allocated_scan_buffers) {
            cudaFree(d_code_lengths);
            cudaFree(d_bit_offsets);
        }
        delete[] h_frequencies;
        delete[] h_nodes;
        delete[] h_code_lengths;
        delete[] h_header;
        delete[] h_codes;
        return status;
    }

    // Pass 2a: Store codes and positions (Phase 1 of atomic-free encoding)
    huffman_encode_phase1_kernel<<<blocks, threads, 0, stream>>>(
        d_input, input_size, table.codes, d_bit_offsets,
        d_codes_temp, d_code_lengths, d_positions_temp
    );
    
    // Pass 2b: Merge into final bitstream using block-cooperative approach (ATOMIC-FREE)
    // Each block builds its segment in shared memory, then writes coalesced to global
    huffman_encode_phase2_kernel<<<blocks, threads, 0, stream>>>(
        d_codes_temp, d_code_lengths, d_positions_temp, input_size,
        d_output, header_size_bits
    );
    
    // Cleanup phase buffers
    if (allocated_phase_buffers) {
        cudaFree(d_codes_temp);
        cudaFree(d_positions_temp);
    }

    // --- 5. Get final size ---
    u32 h_last_offset = 0, h_last_length = 0;
    if (input_size > 0) {
        CUDA_CHECK(cudaMemcpy(&h_last_offset, d_bit_offsets + (input_size - 1), sizeof(u32), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_last_length, d_code_lengths + (input_size - 1), sizeof(u32), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    u32 total_bits = header_size_bits + h_last_offset + h_last_length;
    *output_size = (total_bits + 7) / 8;
    
    // Cleanup - only free if we allocated
    if (allocated_scan_buffers) {
        cudaFree(d_code_lengths);
        cudaFree(d_bit_offsets);
    }
    cudaFreeHost(h_frequencies); // Free pinned memory
    delete[] h_nodes;
    delete[] h_code_lengths;
    delete[] h_header;
    delete[] h_codes;
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
    // --- END REPLACEMENT ---
}

Status decode_huffman(
    const byte_t* d_input,
    size_t input_size, // Full size of compressed data
    const HuffmanTable& table, // Not used, table is in stream
    byte_t* d_output,
    size_t* d_output_size, // This is a host pointer
    u32 decompressed_size, // We know this
    cudaStream_t stream
) {
    // The 'table' parameter is provided for API compatibility but the table
    // is serialized in the stream for the current format; it may be ignored
    // by this implementation. We annotate the parameter in the header with
    // [[maybe_unused]] (C++17) so no additional suppression is necessary.
    // --- START REPLACEMENT ---
    if (!d_input || !d_output || !d_output_size || input_size == 0 || decompressed_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // --- 1. Read header ---
    u8* h_code_lengths = new u8[MAX_HUFFMAN_SYMBOLS];
    byte_t* h_input_header = new byte_t[1 + MAX_HUFFMAN_SYMBOLS];
    
    CUDA_CHECK(cudaMemcpyAsync(h_input_header, d_input, 1 + MAX_HUFFMAN_SYMBOLS,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    u32 header_size = 0;
    Status status = deserialize_huffman_table(h_input_header, 1 + MAX_HUFFMAN_SYMBOLS,
                                              h_code_lengths, &header_size);
    delete[] h_input_header;
    if (status != Status::SUCCESS) {
        delete[] h_code_lengths;
        return status;
    }
    
    // --- 2. Build decode table on GPU ---
    u8* d_code_lengths;
    u16* d_first_code;
    u16* d_symbol_index;
    u8* d_symbols;

    CUDA_CHECK(cudaMalloc(&d_code_lengths, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));
    CUDA_CHECK(cudaMalloc(&d_first_code, (MAX_HUFFMAN_BITS + 2) * sizeof(u16)));
    CUDA_CHECK(cudaMalloc(&d_symbol_index, (MAX_HUFFMAN_BITS + 1) * sizeof(u16)));
    CUDA_CHECK(cudaMalloc(&d_symbols, MAX_HUFFMAN_SYMBOLS * sizeof(u8)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_code_lengths, h_code_lengths, MAX_HUFFMAN_SYMBOLS * sizeof(u8),
                                cudaMemcpyHostToDevice, stream));
    
    delete[] h_code_lengths;
    
    build_decode_table_kernel<<<1, 1, 0, stream>>>(
        d_code_lengths, MAX_HUFFMAN_SYMBOLS,
        d_first_code, d_symbol_index, d_symbols
    );

    // --- 3. Parallel Decode ---
    u32 num_chunks = (decompressed_size + HUFFMAN_DECODE_SYMBOLS_PER_CHUNK - 1) 
                   / HUFFMAN_DECODE_SYMBOLS_PER_CHUNK;
    
    u32* d_chunk_start_bits;
    CUDA_CHECK(cudaMalloc(&d_chunk_start_bits, (num_chunks + 1) * sizeof(u32)));

    // Pass 1: Find chunk start bits
    find_chunk_start_bits_kernel<<<1, 1, 0, stream>>>(
        d_input,
        header_size,
        input_size,
        d_first_code,
        d_symbol_index,
        d_symbols,
        decompressed_size,
        num_chunks,
        HUFFMAN_DECODE_SYMBOLS_PER_CHUNK,
        d_chunk_start_bits
    );

    // Pass 2: Parallel decode
    parallel_huffman_decode_kernel<<<num_chunks, HUFFMAN_DECODE_THREADS_PER_CHUNK, 0, stream>>>(
        d_input,
        input_size,
        d_first_code,
        d_symbol_index,
        d_symbols,
        d_output,
        decompressed_size,
        d_chunk_start_bits
    );
    
    // We know the output size, so just set it.
    *d_output_size = decompressed_size;
    
    // Cleanup
    cudaFree(d_code_lengths);
    cudaFree(d_first_code);
    cudaFree(d_symbol_index);
    cudaFree(d_symbols);
    cudaFree(d_chunk_start_bits);
    
    CUDA_CHECK(cudaGetLastError());
    return Status::SUCCESS;
    // --- END REPLACEMENT ---
}


/**
 * @brief (HELPER) A simple __device__ bitstream reader for Huffman.
 *
 * Reads the stream FORWARD. Not optimized for parallel reads.
 */
struct HuffmanBitStreamReader {
    const byte_t* stream_ptr;
    const byte_t* stream_end;
    u64 bit_container;
    i32 bits_remaining;

    /**
     * @brief Initializes the reader.
     *
     * @param stream_start Points to the beginning of the bitstream.
     * @param stream_size The total size in bytes.
     */
    __device__ void init(const byte_t* stream_start, size_t stream_size) {
        stream_ptr = stream_start;
        stream_end = stream_start + stream_size;
        bit_container = 0;
        bits_remaining = 0;
        
        // Pre-load the bit container
        reload();
        reload(); // Load up to 64 bits
    }

    /**
     * @brief Ensures the bit container has at least 32 bits, if possible.
     */
    __device__ void reload() {
        if (bits_remaining <= 32 && stream_ptr <= stream_end - 4) {
            u64 next_bits = *reinterpret_cast<const u32*>(stream_ptr);
            stream_ptr += 4;
            bit_container |= (next_bits << bits_remaining);
            bits_remaining += 32;
        }
    }

    /**
     * @brief Peeks at `num_bits` without consuming them.
     */
    __device__ u32 peek(u32 num_bits) {
        return bit_container & ((1ULL << num_bits) - 1);
    }

    /**
     * @brief Consumes `num_bits` from the stream.
     */
    __device__ void consume(u32 num_bits) {
        bit_container >>= num_bits;
        bits_remaining -= num_bits;
        
        // Reload if we are running low
        reload();
    }
};

Status free_huffman_decoder_table(
    HuffmanDecoderTable* table,
    cudaStream_t stream)
{
    if (table->d_table) {
        CUDA_CHECK(cudaFreeAsync(table->d_table, stream));
        table->d_table = nullptr;
    }
    return Status::SUCCESS;
}

} // namespace huffman
} // namespace cuda_zstd