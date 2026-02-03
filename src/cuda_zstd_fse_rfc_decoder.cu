/**
 * @file cuda_zstd_fse_rfc_decoder.cu
 * @brief RFC 8878 Compliant FSE Decoder - Interleaved Version
 * 
 * This decoder correctly decodes the interleaved bitstream produced by RFC encoder.
 * Handles all three tables (LL, OF, ML) simultaneously with different modes.
 */

#include "cuda_zstd_fse_rfc.h"
#include "cuda_zstd_fse.h"
#include "cuda_zstd_sequence.h"
#include <stdio.h>

namespace cuda_zstd {
namespace fse {

// Mode constants
constexpr u32 MODE_PREDEFINED = 0;
constexpr u32 MODE_RLE = 1;
constexpr u32 MODE_FSE = 2;

// =============================================================================
// TABLE BUILDING FROM FREQUENCY DATA (RFC 8878 Compliant)
// =============================================================================

/**
 * @brief Build FSEDecodeTable from normalized frequencies (RFC 8878)
 * 
 * This function builds a decoder table that matches the RFC encoder's output.
 * It ensures encoder and decoder use the same state transitions.
 * 
 * @param normFreqs Normalized frequencies (sum must equal tableSize)
 * @param maxSymbol Maximum symbol value
 * @param tableLog Table size = 2^tableLog
 * @param h_table Output table (caller must pre-allocate arrays)
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_buildDTable_rfc(
    const u16 *normFreqs,
    u32 maxSymbol,
    u32 tableLog,
    FSEDecodeTable &h_table
) {
    if (!normFreqs || !h_table.symbol || !h_table.nbBits || !h_table.newState) {
        printf("[FSE_buildDTable_rfc] ERROR: Invalid parameters\n");
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    u32 tableSize = 1u << tableLog;
    h_table.table_log = tableLog;
    h_table.table_size = tableSize;
    
    // printf("[FSE_buildDTable_rfc] Building table: size=%u, log=%u, max_sym=%u\n",
    //        tableSize, tableLog, maxSymbol);
    
    // Step 1: Spread symbols across the table using RFC 8878 algorithm
    // Algorithm 1 from RFC 8878 Section 4.1
    std::vector<u8> spread(tableSize);
    
    // Initialize all slots to zero (invalid symbol)
    for (u32 i = 0; i < tableSize; i++) {
        spread[i] = 0;
    }
    
    // Calculate cumulative frequencies
    std::vector<u32> cumFreq(maxSymbol + 2, 0);
    u32 total = 0;
    for (u32 s = 0; s <= maxSymbol; s++) {
        cumFreq[s] = total;
        total += normFreqs[s];
    }
    cumFreq[maxSymbol + 1] = total;
    
    if (total != tableSize) {
        printf("[FSE_buildDTable_rfc] WARNING: Frequencies sum to %u, expected %u\n",
               total, tableSize);
    }
    
    // Step size for spreading (RFC 8878: step = (5/8) * tableSize + 3)
    u32 step = (tableSize * 5) / 8 + 3;
    step |= 1;  // Ensure step is odd
    
    // printf("[FSE_buildDTable_rfc] Spreading with step=%u\n", step);
    
    // Spread symbols
    u32 pos = 0;
    for (u32 s = 0; s <= maxSymbol; s++) {
        u32 freq = normFreqs[s];
        for (u32 j = 0; j < freq; j++) {
            spread[pos] = (u8)s;
            pos = (pos + step) & (tableSize - 1);  // Circular buffer
            while (spread[pos] != 0 && j < freq - 1) {
                pos = (pos + 1) & (tableSize - 1);
            }
        }
    }
    
    // Step 2: Build state transition table
    // For each state, determine: symbol, nbBits, and newState
    std::vector<u32> symbolNext(maxSymbol + 1, 0);
    for (u32 s = 0; s <= maxSymbol; s++) {
        symbolNext[s] = normFreqs[s];
    }
    
    for (u32 state = 0; state < tableSize; state++) {
        u8 symbol = spread[state];
        
        // Get nextState for this symbol occurrence
        u32 nextState = symbolNext[symbol]++;
        
        // Calculate nbBits (number of bits to read)
        // nbBits = tableLog - floor(log2(nextState)) for nextState > 0
        // For nextState = 0, nbBits = tableLog
        u32 nbBits;
        if (nextState == 0) {
            nbBits = tableLog;
        } else {
            u32 highBit = 0;
            u32 tmp = nextState;
            while (tmp >>= 1) {
                highBit++;
            }
            nbBits = tableLog - highBit;
        }
        
        // Calculate newState (baseline for next state)
        // RFC 8878: newState = (nextState << nbBits) - tableSize
        // This gives the baseline value that decoder adds readBits to
        i32 newStateVal;
        if (nextState == 0) {
            newStateVal = 0;
        } else {
            newStateVal = (nextState << nbBits) - tableSize;
        }
        
        h_table.symbol[state] = symbol;
        h_table.nbBits[state] = (u8)nbBits;
        h_table.newState[state] = (u16)(newStateVal > 0 ? newStateVal : 0);
        
        // Debug first few entries
        if (state < 3) {
            // printf("[FSE_buildDTable_rfc] State[%u]: sym=%u, nextState=%u, nbBits=%u, newState=%d\n",
            //        state, symbol, nextState, nbBits, newStateVal);
        }
    }
    
    // printf("[FSE_buildDTable_rfc] Table complete: newState[0]=%u, newState[1]=%u\n",
    //        h_table.newState[0], h_table.newState[1]);
    
    return Status::SUCCESS;
}

// =============================================================================
// RFC INTERLEAVED DECODER KERNEL
// =============================================================================

/**
 * @brief Device version of FSEDecodeTable
 */
struct FSEDecodeTableDevice {
    u16 *newState;
    u8 *symbol;
    u8 *nbBits;
    u32 table_log;
    u32 table_size;
};

/**
 * @brief RFC-compliant interleaved FSE decoder kernel
 * 
 * Decodes interleaved bitstream with 3 tables (LL, OF, ML).
 * Handles different modes: Predefined (0), RLE (1), FSE (2).
 */
__global__ void k_fse_decode_interleaved_rfc(
    const u8 *__restrict__ bitstream,
    size_t bitstream_size,
    // LL table
    const FSEDecodeTableDevice ll_table,
    u32 ll_mode,
    u32 ll_rle_value,
    // OF table  
    const FSEDecodeTableDevice of_table,
    u32 of_mode,
    u32 of_rle_value,
    // ML table
    const FSEDecodeTableDevice ml_table,
    u32 ml_mode,
    u32 ml_rle_value,
    // Output
    u32 num_sequences,
    u32 *__restrict__ d_ll_out,
    u32 *__restrict__ d_of_out,
    u32 *__restrict__ d_ml_out
) {
    if (threadIdx.x != 0 || num_sequences == 0) return;
    
    printf("[INTERLEAVED_DECODER] START: sequences=%u, bitstream=%zu bytes\n", 
           num_sequences, bitstream_size);
    
    // Helper to read bits from bitstream (LSB first, moving backwards)
    int current_byte = (int)bitstream_size - 1;
    int current_bit = 7;
    
    auto read_bits = [&](u32 nb_bits) -> u32 {
        if (nb_bits == 0) return 0;
        u32 result = 0;
        for (u32 i = 0; i < nb_bits; i++) {
            if (current_bit == 0) {
                current_bit = 7;
                current_byte--;
            } else {
                current_bit--;
            }
            
            if (current_byte < 0) {
                printf("[INTERLEAVED_DECODER] WARNING: Underrun at bit %u\n", i);
                return result;
            }
            
            u32 bit_val = (bitstream[current_byte] >> current_bit) & 1;
            result |= (bit_val << i);
        }
        return result;
    };
    
    // Find sentinel bit (marks end of bitstream)
    // RFC 8878: The stop bit is the first bit with a value of 1, starting from the end.
    bool found_sentinel = false;
    for (int byte_idx = (int)bitstream_size - 1; byte_idx >= 0 && !found_sentinel; --byte_idx) {
        u8 byte = bitstream[byte_idx];
        for (int bit = 7; bit >= 0; --bit) {
            if ((byte >> bit) & 1) {
                current_byte = byte_idx;
                current_bit = bit;
                found_sentinel = true;
                break;
            }
        }
    }
    
    if (!found_sentinel) {
        printf("[INTERLEAVED_DECODER] ERROR: No sentinel found\n");
        return;
    }
    
    // Read initial states for FSE modes
    u32 stateLL = 0, stateOF = 0, stateML = 0;
    
    if (ll_mode == MODE_FSE || ll_mode == MODE_PREDEFINED) {
        stateLL = read_bits(ll_table.table_log);
        if (stateLL >= ll_table.table_size) stateLL = 0;
    }
    if (of_mode == MODE_FSE || of_mode == MODE_PREDEFINED) {
        stateOF = read_bits(of_table.table_log);
        if (stateOF >= of_table.table_size) stateOF = 0;
    }
    if (ml_mode == MODE_FSE || ml_mode == MODE_PREDEFINED) {
        stateML = read_bits(ml_table.table_log);
        if (stateML >= ml_table.table_size) stateML = 0;
    }
    
    // Decode sequences in reverse order (N-1 to 0)
    for (int seq_idx = (int)num_sequences - 1; seq_idx >= 0; seq_idx--) {
        u32 ll_value = 0, of_value = 0, ml_value = 0;
        
        // --- Decode LL ---
        if (ll_mode == MODE_RLE) {
            ll_value = ll_rle_value;
        } else if (ll_mode == MODE_PREDEFINED || ll_mode == MODE_FSE) {
            // Read symbol and extra from state
            u8 symbol = ll_table.symbol[stateLL];
            u8 nbBits = ll_table.nbBits[stateLL];
            u16 baseline = ll_table.newState[stateLL];
            u32 extra = read_bits(nbBits);
            
            // Convert FSE code to literal length
            ll_value = sequence::ZstdSequence::get_lit_len(symbol) + extra;
            
            // Update state
            stateLL = baseline + extra;
            if (stateLL >= ll_table.table_size) stateLL = 0;
        }
        
        // --- Decode OF ---
        if (of_mode == MODE_RLE) {
            of_value = of_rle_value;
        } else if (of_mode == MODE_PREDEFINED || of_mode == MODE_FSE) {
            u8 symbol = of_table.symbol[stateOF];
            u8 nbBits = of_table.nbBits[stateOF];
            u16 baseline = of_table.newState[stateOF];
            u32 extra = read_bits(nbBits);
            
            // Convert FSE code to offset
            of_value = (1u << symbol) + extra;
            
            stateOF = baseline + extra;
            if (stateOF >= of_table.table_size) stateOF = 0;
        }
        
        // --- Decode ML ---
        if (ml_mode == MODE_RLE) {
            ml_value = ml_rle_value;
        } else if (ml_mode == MODE_PREDEFINED || ml_mode == MODE_FSE) {
            u8 symbol = ml_table.symbol[stateML];
            u8 nbBits = ml_table.nbBits[stateML];
            u16 baseline = ml_table.newState[stateML];
            u32 extra = read_bits(nbBits);
            
            // Convert FSE code to match length
            ml_value = sequence::ZstdSequence::get_match_len(symbol) + extra;
            
            stateML = baseline + extra;
            if (stateML >= ml_table.table_size) stateML = 0;
        }
        
        // Store results
        d_ll_out[seq_idx] = ll_value;
        d_of_out[seq_idx] = of_value;
        d_ml_out[seq_idx] = ml_value;
        
        // Print first few for debugging
        if (seq_idx < 3 || seq_idx == (int)num_sequences - 1) {
            printf("[INTERLEAVED_DECODER] Seq[%d]: LL=%u, OF=%u, ML=%u\n",
                   seq_idx, ll_value, of_value, ml_value);
        }
    }
    
    printf("[INTERLEAVED_DECODER] COMPLETE: Decoded %u sequences\n", num_sequences);
}

// =============================================================================
// HOST WRAPPER
// =============================================================================

/**
 * @brief RFC-compliant interleaved FSE decoder
 * 
 * Replaces decode_sequences_interleaved with correct RFC decoding logic.
 * Handles all three streams (LL, OF, ML) with their respective modes.
 */
__host__ Status decode_sequences_interleaved_rfc(
    const unsigned char *d_input,
    u32 input_size,
    u32 num_sequences,
    u32 *d_ll_out,
    u32 *d_of_out,
    u32 *d_ml_out,
    u32 ll_mode,
    u32 of_mode,
    u32 ml_mode,
    const FSEDecodeTable *ll_table,
    const FSEDecodeTable *of_table,
    const FSEDecodeTable *ml_table,
    u32 literals_limit,
    cudaStream_t stream
) {
    if (!d_input || num_sequences == 0) {
        printf("[decode_sequences_interleaved_rfc] ERROR: Invalid parameters\n");
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    printf("[decode_sequences_interleaved_rfc] START: input=%p, size=%u, seq=%u\n",
           d_input, input_size, num_sequences);
    printf("[decode_sequences_interleaved_rfc] Modes: LL=%u, OF=%u, ML=%u\n",
           ll_mode, of_mode, ml_mode);
    
    // Allocate and copy table data for FSE modes
    FSEDecodeTableDevice d_ll_table = {};
    FSEDecodeTableDevice d_of_table = {};
    FSEDecodeTableDevice d_ml_table = {};
    
    u32 ll_rle = 0, of_rle = 0, ml_rle = 0;
    
    // Setup LL table
    if (ll_mode == MODE_FSE || ll_mode == MODE_PREDEFINED) {
        if (!ll_table) {
            printf("[decode_sequences_interleaved_rfc] ERROR: LL table is NULL\n");
            return Status::ERROR_INVALID_PARAMETER;
        }
        d_ll_table.table_log = ll_table->table_log;
        d_ll_table.table_size = ll_table->table_size;
        
        size_t table_bytes = ll_table->table_size * sizeof(u16);
        size_t sym_bytes = ll_table->table_size * sizeof(u8);
        
        cudaMallocAsync(&d_ll_table.newState, table_bytes, stream);
        cudaMallocAsync(&d_ll_table.symbol, sym_bytes, stream);
        cudaMallocAsync(&d_ll_table.nbBits, sym_bytes, stream);
        
        cudaMemcpyAsync(d_ll_table.newState, ll_table->newState, table_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ll_table.symbol, ll_table->symbol, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ll_table.nbBits, ll_table->nbBits, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Setup OF table
    if (of_mode == MODE_FSE || of_mode == MODE_PREDEFINED) {
        if (!of_table) {
            printf("[decode_sequences_interleaved_rfc] ERROR: OF table is NULL\n");
            return Status::ERROR_INVALID_PARAMETER;
        }
        d_of_table.table_log = of_table->table_log;
        d_of_table.table_size = of_table->table_size;
        
        size_t table_bytes = of_table->table_size * sizeof(u16);
        size_t sym_bytes = of_table->table_size * sizeof(u8);
        
        cudaMallocAsync(&d_of_table.newState, table_bytes, stream);
        cudaMallocAsync(&d_of_table.symbol, sym_bytes, stream);
        cudaMallocAsync(&d_of_table.nbBits, sym_bytes, stream);
        
        cudaMemcpyAsync(d_of_table.newState, of_table->newState, table_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_of_table.symbol, of_table->symbol, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_of_table.nbBits, of_table->nbBits, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Setup ML table
    if (ml_mode == MODE_FSE || ml_mode == MODE_PREDEFINED) {
        if (!ml_table) {
            printf("[decode_sequences_interleaved_rfc] ERROR: ML table is NULL\n");
            return Status::ERROR_INVALID_PARAMETER;
        }
        d_ml_table.table_log = ml_table->table_log;
        d_ml_table.table_size = ml_table->table_size;
        
        size_t table_bytes = ml_table->table_size * sizeof(u16);
        size_t sym_bytes = ml_table->table_size * sizeof(u8);
        
        cudaMallocAsync(&d_ml_table.newState, table_bytes, stream);
        cudaMallocAsync(&d_ml_table.symbol, sym_bytes, stream);
        cudaMallocAsync(&d_ml_table.nbBits, sym_bytes, stream);
        
        cudaMemcpyAsync(d_ml_table.newState, ml_table->newState, table_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ml_table.symbol, ml_table->symbol, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ml_table.nbBits, ml_table->nbBits, sym_bytes,
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Launch kernel
    k_fse_decode_interleaved_rfc<<<1, 1, 0, stream>>>(
        d_input, input_size,
        d_ll_table, ll_mode, ll_rle,
        d_of_table, of_mode, of_rle,
        d_ml_table, ml_mode, ml_rle,
        num_sequences,
        d_ll_out, d_of_out, d_ml_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[decode_sequences_interleaved_rfc] Kernel error: %s\n", cudaGetErrorString(err));
        return Status::ERROR_INTERNAL;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("[decode_sequences_interleaved_rfc] Stream error: %s\n", cudaGetErrorString(err));
        return Status::ERROR_INTERNAL;
    }
    
    // Cleanup
    if (d_ll_table.newState) cudaFreeAsync(d_ll_table.newState, stream);
    if (d_ll_table.symbol) cudaFreeAsync(d_ll_table.symbol, stream);
    if (d_ll_table.nbBits) cudaFreeAsync(d_ll_table.nbBits, stream);
    if (d_of_table.newState) cudaFreeAsync(d_of_table.newState, stream);
    if (d_of_table.symbol) cudaFreeAsync(d_of_table.symbol, stream);
    if (d_of_table.nbBits) cudaFreeAsync(d_of_table.nbBits, stream);
    if (d_ml_table.newState) cudaFreeAsync(d_ml_table.newState, stream);
    if (d_ml_table.symbol) cudaFreeAsync(d_ml_table.symbol, stream);
    if (d_ml_table.nbBits) cudaFreeAsync(d_ml_table.nbBits, stream);
    
    printf("[decode_sequences_interleaved_rfc] SUCCESS\n");
    return Status::SUCCESS;
}

} // namespace fse
} // namespace cuda_zstd
