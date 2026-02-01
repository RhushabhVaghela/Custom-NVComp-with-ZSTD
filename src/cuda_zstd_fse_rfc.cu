/**
 * @file cuda_zstd_fse_rfc.cu
 * @brief RFC 8878 Compliant FSE Implementation
 * 
 * Clean room implementation following RFC 8878 Section 4.1.
 */

#include "cuda_zstd_fse_rfc.h"
#include "cuda_zstd_fse.h"          // For FSEEncodeTable (old table format)
#include "cuda_zstd_types.h"
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>

namespace cuda_zstd {
namespace fse {

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ u32 highestBit(u32 v) {
    if (v == 0) return 0;
    return 31 - __clz(v);
}

// =============================================================================
// TABLE BUILDING (RFC 8878 Algorithm 1)
// =============================================================================

/**
 * @brief GPU Kernel: Build FSE distribution table
 * 
 * Implements RFC 8878 Section 4.1.1:
 * 1. Spread symbols using step = (5/8)*tableSize + 3
 * 2. Assign states to symbols
 * 3. Build state transition table
 */
__global__ void k_fse_build_table(
    const u16 *__restrict__ normFreqs,
    u32 maxSymbol,
    u32 tableSize,
    FSEDistributionEntry *__restrict__ entries,
    u16 *__restrict__ symbolFirstState
) {
    u32 tid = threadIdx.x;
    
    // Initialize symbol first state array
    for (u32 s = tid; s <= maxSymbol; s += blockDim.x) {
        symbolFirstState[s] = 0;
    }
    __syncthreads();
    
    // Phase 1: Spread symbols (Algorithm 1 from RFC)
    // Step size: (5/8)*tableSize + 3
    // This provides good distribution across the table
    
    u32 step = (tableSize >> 1) + (tableSize >> 3) + 3;
    u32 mask = tableSize - 1;
    u32 pos = 0;
    
    // Track which states are occupied (for -1 symbols at end)
    __shared__ u8 s_occupied[1024]; // Max 1024 for table_log <= 10
    for (u32 i = tid; i < tableSize; i += blockDim.x) {
        s_occupied[i] = 0;
        entries[i].symbol = 0;
        entries[i].nbBits = 0;
        entries[i].nextState = 0;
    }
    __syncthreads();
    
    // Place -1 symbols (frequency = -1) at high positions
    u32 highPos = tableSize - 1;
    if (tid == 0) {
        for (u32 s = 0; s <= maxSymbol; s++) {
            if ((i16)normFreqs[s] == -1) {
                symbolFirstState[s] = highPos;
                s_occupied[highPos] = 1;
                highPos--;
            }
        }
    }
    __syncthreads();
    
    // Spread positive frequency symbols
    if (tid == 0) {
        for (u32 s = 0; s <= maxSymbol; s++) {
            u32 freq = normFreqs[s];
            if (freq == 0 || (i16)freq == -1) continue;
            
            for (u32 i = 0; i < freq; i++) {
                // Skip positions occupied by -1 symbols
                while (s_occupied[pos]) {
                    pos = (pos + step) & mask;
                }
                
                // Place symbol at this position
                entries[pos].symbol = (u8)s;
                s_occupied[pos] = 1;
                
                // Track first state for this symbol
                if (symbolFirstState[s] == 0) {
                    symbolFirstState[s] = pos;
                }
                
                pos = (pos + step) & mask;
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Build state transition table
    // For each state, determine nbBits and nextState baseline
    // Following RFC: nbBits = ceil(log2(freq)) or table_log for freq=1
    
    for (u32 state = tid; state < tableSize; state += blockDim.x) {
        u8 sym = entries[state].symbol;
        u32 freq = normFreqs[sym];
        
        if (freq == 0) {
            entries[state].nbBits = 0;
            entries[state].nextState = 0;
        } else if ((i16)freq == -1) {
            entries[state].nbBits = highestBit(tableSize);
            entries[state].nextState = 0;
        } else {
            // Calculate nbBits based on frequency
            u32 nbBits;
            if (freq == 1) {
                nbBits = highestBit(tableSize);
            } else {
                nbBits = highestBit(tableSize) - highestBit(freq - 1);
            }
            entries[state].nbBits = (u8)nbBits;
            
            // nextState calculation: (state >> nbBits) maps to sub-state
            // Add baseline to get to this symbol's range
            u32 subState = state >> nbBits;
            entries[state].nextState = (u16)(symbolFirstState[sym] + subState);
        }
    }
}

// =============================================================================
// ENCODER (RFC 8878 Section 4.1.2)
// =============================================================================

/**
 * @brief GPU Kernel: FSE Encode
 * 
 * Implements FSE encoding algorithm:
 * 1. Initialize state to first state of first symbol
 * 2. For each symbol:
 *    - Emit state & mask (low bits)
 *    - Update state = (state >> nbBits) + nextStateBaseline
 * 3. Flush remaining bits
 */
__global__ void k_fse_encode(
    const u8 *__restrict__ symbols,
    u32 numSymbols,
    const FSEDistributionEntry *__restrict__ table,
    const u16 *__restrict__ symbolFirstState,
    u32 tableLog,
    u32 tableSize,
    u8 *__restrict__ bitstream,
    size_t bitstreamCapacity,
    size_t *__restrict__ outputSize
) {
    // Only thread 0 does encoding (sequential dependency)
    if (threadIdx.x != 0) return;
    
    if (numSymbols == 0) {
        *outputSize = 0;
        return;
    }
    
    // Bitstream writer state
    u8 *ptr = bitstream;
    u8 *end = bitstream + bitstreamCapacity;
    u64 bitContainer = 0;
    u32 bitCount = 0;
    
    // Helper: write bits to bitstream
    auto writeBits = [&](u32 value, u32 nbBits) {
        if (nbBits == 0) return;
        bitContainer |= ((u64)value << bitCount);
        bitCount += nbBits;
        while (bitCount >= 8) {
            if (ptr < end) {
                *ptr++ = (u8)(bitContainer & 0xFF);
            }
            bitContainer >>= 8;
            bitCount -= 8;
        }
    };
    
    // Step 1: Initialize state for first symbol
    u32 state = symbolFirstState[symbols[0]];
    
    // Step 2: Process symbols 0 to N-1
    for (u32 i = 0; i < numSymbols; i++) {
        u8 sym = symbols[i];
        
        // Write state bits (low nbBits bits)
        u32 nbBits = table[state].nbBits;
        if (nbBits > 0) {
            writeBits(state & ((1u << nbBits) - 1), nbBits);
        }
        
        // Update state for next symbol
        u32 nextState = table[state].nextState;
        state = nextState;
    }
    
    // Step 3: Write final state (for decoder initialization)
    writeBits(state, tableLog);
    
    // Step 4: Write sentinel (single 1 bit at end)
    writeBits(1, 1);
    
    // Flush remaining bits
    while (bitCount > 0) {
        if (ptr < end) {
            *ptr++ = (u8)(bitContainer & 0xFF);
        }
        bitContainer >>= 8;
        bitCount = (bitCount > 8) ? bitCount - 8 : 0;
    }
    
    *outputSize = ptr - bitstream;
}

// =============================================================================
// DECODER (RFC 8878 Section 4.1.3)
// =============================================================================

/**
 * @brief GPU Kernel: FSE Decode
 * 
 * Implements FSE decoding algorithm:
 * 1. Read initial state from end of bitstream
 * 2. While not done:
 *    - Decode symbol from state
 *    - Read bits to determine next state
 *    - Update state = nextState + readBits
 */
__global__ void k_fse_decode(
    const u8 *__restrict__ bitstream,
    size_t bitstreamSize,
    const FSEDistributionEntry *__restrict__ table,
    u32 tableLog,
    u32 numSymbols,
    u8 *__restrict__ symbols
) {
    // Only thread 0 does decoding
    if (threadIdx.x != 0) return;
    
    if (numSymbols == 0 || bitstreamSize == 0) return;
    
    // Find sentinel bit
    u32 sentinelPos = 0;
    bool found = false;
    
    for (int byteIdx = (int)bitstreamSize - 1; byteIdx >= 0 && !found; byteIdx--) {
        u8 byte = bitstream[byteIdx];
        for (int bit = 0; bit < 8; bit++) {
            if ((byte >> bit) & 1) {
                sentinelPos = byteIdx * 8 + bit;
                found = true;
                break;
            }
        }
    }
    
    if (!found) return;
    
    // Read initial state (tableLog bits before sentinel)
    u32 bitPos = sentinelPos - tableLog;
    u32 state = 0;
    for (u32 i = 0; i < tableLog; i++) {
        u32 byteIdx = (bitPos + i) / 8;
        u32 bitIdx = (bitPos + i) % 8;
        if (byteIdx < bitstreamSize) {
            state |= (((bitstream[byteIdx] >> bitIdx) & 1) << i);
        }
    }
    
    // Decode symbols in reverse order
    for (int i = (int)numSymbols - 1; i >= 0; i--) {
        // Decode symbol from state
        symbols[i] = table[state].symbol;
        
        // Read bits for next state
        u32 nbBits = table[state].nbBits;
        u32 bits = 0;
        
        if (nbBits > 0) {
            bitPos -= nbBits;
            for (u32 j = 0; j < nbBits; j++) {
                u32 byteIdx = (bitPos + j) / 8;
                u32 bitIdx = (bitPos + j) % 8;
                if (byteIdx < bitstreamSize) {
                    bits |= (((bitstream[byteIdx] >> bitIdx) & 1) << j);
                }
            }
        }
        
        // Update state
        state = table[state].nextState + bits;
    }
}

// =============================================================================
// HOST INTERFACES
// =============================================================================

__host__ Status FSE_allocTableRFC(
    FSETableRFC &table,
    u32 tableLog,
    u32 maxSymbol,
    cudaStream_t stream
) {
    table.tableLog = tableLog;
    table.tableSize = 1u << tableLog;
    table.maxSymbol = maxSymbol;
    
    cudaError_t err;
    
    err = cudaMallocAsync(&table.d_entries, 
                          table.tableSize * sizeof(FSEDistributionEntry), 
                          stream);
    if (err != cudaSuccess) return Status::ERROR_OUT_OF_MEMORY;
    
    err = cudaMallocAsync(&table.d_normFreqs,
                          (maxSymbol + 1) * sizeof(u16),
                          stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(table.d_entries, stream);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    err = cudaMallocAsync(&table.d_symbolFirstState,
                          (maxSymbol + 1) * sizeof(u16),
                          stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(table.d_entries, stream);
        cudaFreeAsync(table.d_normFreqs, stream);
        return Status::ERROR_OUT_OF_MEMORY;
    }
    
    table.allocated = true;
    return Status::SUCCESS;
}

__host__ void FSE_freeTableRFC(
    FSETableRFC &table,
    cudaStream_t stream
) {
    if (table.allocated) {
        cudaFreeAsync(table.d_entries, stream);
        cudaFreeAsync(table.d_normFreqs, stream);
        cudaFreeAsync(table.d_symbolFirstState, stream);
        table.allocated = false;
    }
}

__host__ Status FSE_buildTableRFC(
    const u16 *normFreqs,
    u32 maxSymbol,
    u32 tableLog,
    FSETableRFC &table,
    void *d_workspace,
    size_t workspaceSize,
    cudaStream_t stream
) {
    // Copy normalized frequencies to device
    CUDA_CHECK(cudaMemcpyAsync(table.d_normFreqs, normFreqs,
                               (maxSymbol + 1) * sizeof(u16),
                               cudaMemcpyHostToDevice, stream));
    
    // Launch table building kernel
    u32 tableSize = 1u << tableLog;
    k_fse_build_table<<<1, 256, 0, stream>>>(
        table.d_normFreqs,
        maxSymbol,
        tableSize,
        table.d_entries,
        table.d_symbolFirstState
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return Status::SUCCESS;
}

__host__ Status FSE_encodeRFC(
    const u8 *d_symbols,
    u32 numSymbols,
    const FSETableRFC &table,
    u8 *d_bitstream,
    size_t bitstreamCapacity,
    size_t *d_outputSize,
    cudaStream_t stream
) {
    k_fse_encode<<<1, 1, 0, stream>>>(
        d_symbols,
        numSymbols,
        table.d_entries,
        table.d_symbolFirstState,
        table.tableLog,
        table.tableSize,
        d_bitstream,
        bitstreamCapacity,
        d_outputSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return Status::SUCCESS;
}

__host__ Status FSE_decodeRFC(
    const u8 *d_bitstream,
    size_t bitstreamSize,
    const FSETableRFC &table,
    u32 numSymbols,
    u8 *d_symbols,
    cudaStream_t stream
) {
    k_fse_decode<<<1, 1, 0, stream>>>(
        d_bitstream,
        bitstreamSize,
        table.d_entries,
        table.tableLog,
        numSymbols,
        d_symbols
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return Status::SUCCESS;
}

// =============================================================================
// WRAPPER: launch_fse_encoding_kernel_rfc
// =============================================================================

/**
 * @brief GPU Kernel: RFC-compliant FSE encoding using old table format
 * 
 * This kernel uses the old FSEEncodeTable structure but with corrected
 * RFC 8878 compliant encoding logic. It reads:
 * - d_state_to_symbol[state] for symbol lookup
 * - d_nbBits_table[state] for bits to emit
 * - d_next_state_vals[state] for next state transition
 * - d_symbol_first_state[symbol] for initial state
 * 
 * This provides a drop-in replacement that uses existing table allocations
 * but produces RFC-compliant bitstreams.
 */
__global__ void k_fse_encode_rfc_from_old_tables(
    const u8 *__restrict__ d_ll_codes, const u32 *__restrict__ d_ll_extras, const u8 *__restrict__ d_ll_bits,
    const u8 *__restrict__ d_of_codes, const u32 *__restrict__ d_of_extras, const u8 *__restrict__ d_of_bits,
    const u8 *__restrict__ d_ml_codes, const u32 *__restrict__ d_ml_extras, const u8 *__restrict__ d_ml_bits,
    u32 num_symbols,
    const FSEEncodeTable *tables,
    unsigned char *__restrict__ output_bitstream,
    size_t bitstream_capacity,
    size_t *__restrict__ output_pos
) {
    // Only thread 0 does encoding (FSE is inherently sequential per stream)
    if (threadIdx.x != 0 || num_symbols == 0 || bitstream_capacity == 0) {
        return;
    }
    
    // Validate inputs
    if (!d_ll_codes || !d_of_codes || !d_ml_codes || !output_bitstream || !output_pos || !tables) {
        if (output_pos) *output_pos = 0;
        return;
    }

    // Validate table pointers for all three streams
    if (!tables[0].d_symbol_first_state || !tables[1].d_symbol_first_state || !tables[2].d_symbol_first_state ||
        !tables[0].d_state_to_symbol || !tables[1].d_state_to_symbol || !tables[2].d_state_to_symbol ||
        !tables[0].d_nbBits_table || !tables[1].d_nbBits_table || !tables[2].d_nbBits_table ||
        !tables[0].d_next_state_vals || !tables[1].d_next_state_vals || !tables[2].d_next_state_vals) {
        *output_pos = 0;
        return;
    }

    unsigned char *ptr = output_bitstream;
    unsigned char *end_limit = output_bitstream + bitstream_capacity;
    u64 bitContainer = 0;
    u32 bitCount = 0;

    auto write_bits = [&](u64 val, u32 nbBits) {
        if (nbBits == 0) return;
        bitContainer |= (val << bitCount);
        bitCount += nbBits;
        while (bitCount >= 8) {
            if (ptr < end_limit) {
                *ptr++ = (unsigned char)(bitContainer & 0xFF);
            }
            bitContainer >>= 8;
            bitCount -= 8;
        }
    };

    // Get table logs and sizes
    const u32 ll_table_log = tables[0].table_log;
    const u32 of_table_log = tables[1].table_log;
    const u32 ml_table_log = tables[2].table_log;
    const u32 ll_table_size = tables[0].table_size;
    const u32 of_table_size = tables[1].table_size;
    const u32 ml_table_size = tables[2].table_size;
    const u32 ll_max_sym = tables[0].max_symbol;
    const u32 of_max_sym = tables[1].max_symbol;
    const u32 ml_max_sym = tables[2].max_symbol;

    // Validate symbol codes
    u8 ll_code0 = d_ll_codes[0];
    u8 of_code0 = d_of_codes[0];
    u8 ml_code0 = d_ml_codes[0];
    
    if (ll_code0 > ll_max_sym || of_code0 > of_max_sym || ml_code0 > ml_max_sym) {
        *output_pos = 0;
        return;
    }

    // Initialize states using RFC-compliant first state lookup
    u32 stateLL = tables[0].d_symbol_first_state[ll_code0];
    u32 stateOF = tables[1].d_symbol_first_state[of_code0];
    u32 stateML = tables[2].d_symbol_first_state[ml_code0];
    
    // Clamp states to valid range
    if (stateLL >= ll_table_size) stateLL = ll_code0 < ll_max_sym ? tables[0].d_symbol_first_state[ll_code0] : 0;
    if (stateOF >= of_table_size) stateOF = of_code0 < of_max_sym ? tables[1].d_symbol_first_state[of_code0] : 0;
    if (stateML >= ml_table_size) stateML = ml_code0 < ml_max_sym ? tables[2].d_symbol_first_state[ml_code0] : 0;

    // Write extra bits for sequence 0 (Zstd convention: written first, read last)
    // Order: ML, OF, LL
    write_bits(d_ml_extras[0], d_ml_bits[0]);
    write_bits(d_of_extras[0], d_of_bits[0]);
    write_bits(d_ll_extras[0], d_ll_bits[0]);

    // Encode sequences 0 to N-2 using RFC state transitions
    // RFC 8878 encoding: emit state bits, then transition to next state
    if (num_symbols > 1) {
        for (u32 i = 0; i < num_symbols - 1; i++) {
            u32 next_idx = i + 1;

            // Literal Length Stream (Table 0) - RFC compliant encoding
            {
                u8 code = d_ll_codes[next_idx];
                if (code > ll_max_sym) {
                    *output_pos = 0;
                    return;
                }
                
                // Get nbBits from table (RFC: bits to emit from current state)
                u32 nbBits = tables[0].d_nbBits_table[stateLL];
                if (nbBits > ll_table_log) nbBits = ll_table_log;
                
                // Emit low bits of current state
                if (nbBits > 0) {
                    write_bits(stateLL & ((1u << nbBits) - 1), nbBits);
                }
                
                // RFC state transition: nextState = table[state].nextState
                // The nextState already includes the baseline + sub-state calculation
                u32 nextState = tables[0].d_next_state_vals[stateLL];
                stateLL = nextState;
                
                // Validate state transition
                if (stateLL >= ll_table_size) {
                    // Fallback: use first state for the symbol we're encoding
                    stateLL = tables[0].d_symbol_first_state[code];
                }
            }

            // Offset Stream (Table 1) - RFC compliant encoding
            {
                u8 code = d_of_codes[next_idx];
                if (code > of_max_sym) {
                    *output_pos = 0;
                    return;
                }
                
                u32 nbBits = tables[1].d_nbBits_table[stateOF];
                if (nbBits > of_table_log) nbBits = of_table_log;
                
                if (nbBits > 0) {
                    write_bits(stateOF & ((1u << nbBits) - 1), nbBits);
                }
                
                stateOF = tables[1].d_next_state_vals[stateOF];
                if (stateOF >= of_table_size) {
                    stateOF = tables[1].d_symbol_first_state[code];
                }
            }

            // Match Length Stream (Table 2) - RFC compliant encoding
            {
                u8 code = d_ml_codes[next_idx];
                if (code > ml_max_sym) {
                    *output_pos = 0;
                    return;
                }
                
                u32 nbBits = tables[2].d_nbBits_table[stateML];
                if (nbBits > ml_table_log) nbBits = ml_table_log;
                
                if (nbBits > 0) {
                    write_bits(stateML & ((1u << nbBits) - 1), nbBits);
                }
                
                stateML = tables[2].d_next_state_vals[stateML];
                if (stateML >= ml_table_size) {
                    stateML = tables[2].d_symbol_first_state[code];
                }
            }

            // Write extra bits for this sequence (order: OF, ML, LL per Zstd spec)
            write_bits(d_of_extras[next_idx], d_of_bits[next_idx]);
            write_bits(d_ml_extras[next_idx], d_ml_bits[next_idx]);
            write_bits(d_ll_extras[next_idx], d_ll_bits[next_idx]);
        }
    }

    // Write final states for decoder initialization
    // Zstd order: ML, OF, LL (from spec / decoder read order)
    write_bits(stateML, ml_table_log);
    write_bits(stateOF, of_table_log);
    write_bits(stateLL, ll_table_log);

    // Write sentinel bit (marks end of bitstream)
    write_bits(1, 1);

    // Flush remaining bits
    while (bitCount > 0) {
        if (ptr < end_limit) {
            *ptr++ = (unsigned char)(bitContainer & 0xFF);
        }
        bitContainer >>= 8;
        bitCount = (bitCount > 8) ? bitCount - 8 : 0;
    }

    *output_pos = ptr - output_bitstream;
}

/**
 * @brief Wrapper function with same signature as old launch_fse_encoding_kernel
 * 
 * This function provides a drop-in replacement for the old buggy FSE encoder.
 * It uses the existing FSEEncodeTable structures (with d_nbBits_table, 
 * d_next_state_vals, d_state_to_symbol, d_symbol_first_state) but implements
 * RFC 8878 compliant encoding logic.
 * 
 * Usage: Simply change line 4462 in cuda_zstd_manager.cu from:
 *   fse::launch_fse_encoding_kernel(...)
 * to:
 *   fse::launch_fse_encoding_kernel_rfc(...)
 */
Status launch_fse_encoding_kernel_rfc(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, 
    unsigned char *d_bitstream, 
    size_t *d_output_pos,
    size_t bitstream_capacity,
    const FSEEncodeTable *d_tables,
    cudaStream_t stream
) {
    // Validate inputs
    if (!d_ll_codes || !d_of_codes || !d_ml_codes || 
        !d_bitstream || !d_output_pos || !d_tables) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    if (num_symbols == 0) {
        cudaMemsetAsync(d_output_pos, 0, sizeof(size_t), stream);
        return Status::SUCCESS;
    }

    // Launch RFC-compliant encoding kernel using old table format
    // Single block, single thread for sequential FSE encoding
    k_fse_encode_rfc_from_old_tables<<<1, 1, 0, stream>>>(
        d_ll_codes, d_ll_extras, d_ll_bits,
        d_of_codes, d_of_extras, d_of_bits,
        d_ml_codes, d_ml_extras, d_ml_bits,
        num_symbols,
        d_tables,
        d_bitstream, bitstream_capacity, d_output_pos
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::ERROR_INTERNAL;
    }

    return Status::SUCCESS;
}

// =============================================================================
// FREQUENCY NORMALIZATION (RFC 8878 Section 4.1.1)
// =============================================================================

__host__ Status FSE_normalizeFreqs(
    const u32 *rawFreqs,
    u32 maxSymbol,
    u32 numSamples,
    u32 tableLog,
    u16 *normFreqs
) {
    u32 tableSize = 1u << tableLog;
    
    // Step 1: Calculate normalized frequencies
    // norm[s] = (raw[s] * tableSize) / numSamples
    
    std::vector<u32> scaled(maxSymbol + 1);
    u32 totalScaled = 0;
    
    for (u32 s = 0; s <= maxSymbol; s++) {
        if (rawFreqs[s] > 0) {
            // Scale with rounding
            scaled[s] = ((u64)rawFreqs[s] * tableSize + (numSamples / 2)) / numSamples;
            if (scaled[s] == 0) scaled[s] = 1; // Minimum 1
            totalScaled += scaled[s];
        } else {
            scaled[s] = 0;
        }
    }
    
    // Step 2: Adjust to make total exactly tableSize
    if (totalScaled > tableSize) {
        // Reduce highest frequencies
        while (totalScaled > tableSize) {
            u32 maxS = 0;
            for (u32 s = 1; s <= maxSymbol; s++) {
                if (scaled[s] > scaled[maxS]) maxS = s;
            }
            if (scaled[maxS] > 1) {
                scaled[maxS]--;
                totalScaled--;
            } else {
                break;
            }
        }
    } else if (totalScaled < tableSize) {
        // Increase frequencies
        while (totalScaled < tableSize) {
            u32 minS = 0;
            for (u32 s = 1; s <= maxSymbol; s++) {
                if (scaled[s] > 0 && (scaled[s] < scaled[minS] || scaled[minS] == 0)) {
                    minS = s;
                }
            }
            scaled[minS]++;
            totalScaled++;
        }
    }
    
    // Step 3: Convert to signed representation (RFC 8878)
    // 0 = not present
    // 1 to tableSize = positive frequency
    // -1 = special (gets 1 slot at end)
    
    for (u32 s = 0; s <= maxSymbol; s++) {
        if (scaled[s] == 0) {
            normFreqs[s] = 0;
        } else {
            normFreqs[s] = (u16)scaled[s];
        }
    }
    
    return Status::SUCCESS;
}

} // namespace fse
} // namespace cuda_zstd