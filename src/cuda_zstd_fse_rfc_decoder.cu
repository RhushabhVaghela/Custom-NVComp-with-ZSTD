#include "cuda_zstd_fse_rfc.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_safe_alloc.h"
#include <cstdio>

namespace cuda_zstd {
namespace fse {

// Zstandard sequence compression modes
enum SequenceMode {
    MODE_PREDEFINED = 0,
    MODE_RLE = 1,
    MODE_FSE = 2,
    MODE_REPEAT = 3
};

// Device-side table representation
struct FSEDecodeTableDevice {
    u16 *newState;
    u8 *symbol;
    u8 *nbBits;
    u8 *nbAdditionalBits;
    u32 *baseValue;
    u32 table_log;
    u32 table_size;
};

// Error codes for device-side errors
enum DeviceError {
    ERROR_NONE = 0,
    ERROR_EMPTY_BITSTREAM = 1,
    ERROR_SENTINEL_MISSING = 2,
    ERROR_INVALID_TABLE_STATE = 3,
    ERROR_LITERAL_OVERFLOW = 4,
    ERROR_BITSTREAM_UNDERFLOW = 5
};

// =============================================================================
// DEVICE KERNEL
// =============================================================================

/**
 * @brief Interleaved FSE decoder kernel (RFC 8878 compliant)
 */
__global__ void k_fse_decode_interleaved_rfc(
    const unsigned char *bitstream,
    u32 bitstream_size,
    FSEDecodeTableDevice ll_table, u32 ll_mode, u32 ll_rle_value,
    FSEDecodeTableDevice of_table, u32 of_mode, u32 of_rle_value,
    FSEDecodeTableDevice ml_table, u32 ml_mode, u32 ml_rle_value,
    u32 num_sequences,
    u32 *d_ll_out, u32 *d_of_out, u32 *d_ml_out,
    u32 literals_limit,
    u32 *d_error_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bool decode_ll = (ll_mode == MODE_FSE || ll_mode == MODE_PREDEFINED || ll_mode == MODE_REPEAT);
    bool decode_of = (of_mode == MODE_FSE || of_mode == MODE_PREDEFINED || of_mode == MODE_REPEAT);
    bool decode_ml = (ml_mode == MODE_FSE || ml_mode == MODE_PREDEFINED || ml_mode == MODE_REPEAT);
    
    bool rle_ll = (ll_mode == MODE_RLE);
    bool rle_of = (of_mode == MODE_RLE);
    bool rle_ml = (ml_mode == MODE_RLE);

    bool needs_bitstream = decode_ll || decode_of || decode_ml;
    sequence::FSEBitStreamReader reader;
    if (needs_bitstream) {
        if (bitstream_size == 0) {
            if (d_error_flag) *d_error_flag = ERROR_EMPTY_BITSTREAM;
            return;
        }

        u8 last_byte = bitstream[bitstream_size - 1];
        if (last_byte == 0) {
            if (d_error_flag) *d_error_flag = ERROR_SENTINEL_MISSING;
            return;
        }
        u32 sentinel_bit = 0;
        for (u32 bit = 7; bit > 0; --bit) {
            if (last_byte & (1u << bit)) {
                sentinel_bit = bit;
                break;
            }
        }
        u32 sentinel_pos = (bitstream_size - 1) * 8u + sentinel_bit;

        reader = sequence::FSEBitStreamReader(bitstream, sentinel_pos,
                                              (u32)bitstream_size,
                                              (u8)sentinel_bit);
    }

    u32 stateLL = 0;
    u32 stateOF = 0;
    u32 stateML = 0;

    // Initial states were pushed ML -> OF -> LL, so we pop LL -> OF -> ML
    if (decode_ll) {
        stateLL = reader.read(ll_table.table_log);
        stateLL %= ll_table.table_size;
    }
    if (decode_of) {
        stateOF = reader.read(of_table.table_log);
        stateOF %= of_table.table_size;
    }
    if (decode_ml) {
        stateML = reader.read(ml_table.table_log);
        stateML %= ml_table.table_size;
    }

    u64 total_lit_len = 0;

    for (u32 seq = 0; seq < num_sequences; seq++) {
        if (decode_ll && stateLL >= ll_table.table_size) {
            if (d_error_flag) *d_error_flag = ERROR_INVALID_TABLE_STATE;
            return;
        }
        if (decode_of && stateOF >= of_table.table_size) {
            if (d_error_flag) *d_error_flag = ERROR_INVALID_TABLE_STATE;
            return;
        }
        if (decode_ml && stateML >= ml_table.table_size) {
            if (d_error_flag) *d_error_flag = ERROR_INVALID_TABLE_STATE;
            return;
        }

        const bool emit_ll = decode_ll || rle_ll;
        const bool emit_of = decode_of || rle_of;
        const bool emit_ml = decode_ml || rle_ml;

        u32 ll_sym = emit_ll ? (decode_ll ? ll_table.symbol[stateLL] : ll_rle_value) : 0;
        u32 of_sym = emit_of ? (decode_of ? of_table.symbol[stateOF] : of_rle_value) : 0;
        u32 ml_sym = emit_ml ? (decode_ml ? ml_table.symbol[stateML] : ml_rle_value) : 0;

        u32 of_extra = 0;
        u32 ml_extra = 0;
        u32 ll_extra = 0;

        // --- ORDER OF READS (matching libzstd) ---
        // 1. OF Extra bits
        if (decode_of) {
            u32 nb = of_table.nbAdditionalBits[stateOF];
            of_extra = (nb > 0) ? reader.read(nb) : 0;
        }
        // 2. ML Extra bits
        if (decode_ml) {
            u32 nb = ml_table.nbAdditionalBits[stateML];
            ml_extra = (nb > 0) ? reader.read(nb) : 0;
        }
        // 3. LL Extra bits
        if (decode_ll) {
            u32 nb = ll_table.nbAdditionalBits[stateLL];
            ll_extra = (nb > 0) ? reader.read(nb) : 0;
        }

        if (reader.underflow) {
            if (d_error_flag) *d_error_flag = ERROR_BITSTREAM_UNDERFLOW;
            return;
        }

        // --- Value Calculation & Output (BEFORE state transitions) ---
        // Values must use pre-transition states for baseValue[] lookups
        if (emit_of) {
            if (decode_of) {
                u32 base = of_table.baseValue[stateOF];
                d_of_out[seq] = base + of_extra;
            } else if (of_sym <= 2) {
                d_of_out[seq] = of_sym + 1;
            } else {
                u32 base = sequence::ZstdSequence::get_offset(of_sym);
                d_of_out[seq] = base + of_extra + 3;
            }
        }

        if (emit_ml) {
            u32 base = (ml_mode == MODE_PREDEFINED && decode_ml)
                           ? ml_table.baseValue[stateML]
                           : sequence::ZstdSequence::get_match_len(ml_sym);
            d_ml_out[seq] = base + ml_extra;
        }

        if (emit_ll) {
            u32 base = (ll_mode == MODE_PREDEFINED && decode_ll)
                               ? ll_table.baseValue[stateLL]
                               : sequence::ZstdSequence::get_lit_len(ll_sym);
            u32 val_ll = base + ll_extra;
            
            // Validation: Allow slight overflow for debug, but enforce limit if requested
            if (literals_limit > 0 && (total_lit_len + val_ll > literals_limit)) {
                if (d_error_flag) *d_error_flag = ERROR_LITERAL_OVERFLOW;
                return;
            }
            
            d_ll_out[seq] = val_ll;
            total_lit_len += val_ll;
        }

        // 4. Update States (Order: LL, then ML, then OF) — standard ZSTD decode order
        if (seq < num_sequences - 1) {
            if (decode_ll) {
                u8 nb = ll_table.nbBits[stateLL];
                u32 bits = nb ? reader.read(nb) : 0u;
                stateLL = ll_table.newState[stateLL] + bits;
            }
            if (decode_ml) {
                u8 nb = ml_table.nbBits[stateML];
                u32 bits = nb ? reader.read(nb) : 0u;
                stateML = ml_table.newState[stateML] + bits;
            }
            if (decode_of) {
                u8 nb = of_table.nbBits[stateOF];
                u32 bits = nb ? reader.read(nb) : 0u;
                stateOF = of_table.newState[stateOF] + bits;
            }
        }

        if (reader.underflow) {
            if (d_error_flag) *d_error_flag = ERROR_BITSTREAM_UNDERFLOW;
            return;
        }
    }
}

// =============================================================================
// HOST WRAPPER
// =============================================================================

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
    u32 ll_rle_value,
    u32 of_rle_value,
    u32 ml_rle_value,
    u32 literals_limit,
    cudaStream_t stream
) {
    if (!d_input || num_sequences == 0) return Status::ERROR_INVALID_PARAMETER;
    
    FSEDecodeTableDevice d_ll_table = {};
    FSEDecodeTableDevice d_of_table = {};
    FSEDecodeTableDevice d_ml_table = {};
    
    auto copy_table = [&](const FSEDecodeTable *src, FSEDecodeTableDevice &dst) {
        if (!src) return;
        dst.table_log = src->table_log;
        dst.table_size = src->table_size;
        if (cuda_zstd::safe_cuda_malloc_async(&dst.newState, src->table_size * sizeof(u16), stream) != cudaSuccess) return;
        if (cuda_zstd::safe_cuda_malloc_async(&dst.symbol, src->table_size * sizeof(u8), stream) != cudaSuccess) return;
        if (cuda_zstd::safe_cuda_malloc_async(&dst.nbBits, src->table_size * sizeof(u8), stream) != cudaSuccess) return;
        if (cuda_zstd::safe_cuda_malloc_async(&dst.nbAdditionalBits, src->table_size * sizeof(u8), stream) != cudaSuccess) return;
        if (cuda_zstd::safe_cuda_malloc_async(&dst.baseValue, src->table_size * sizeof(u32), stream) != cudaSuccess) return;
        cudaMemcpyAsync(dst.newState, src->newState, src->table_size * sizeof(u16), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst.symbol, src->symbol, src->table_size * sizeof(u8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst.nbBits, src->nbBits, src->table_size * sizeof(u8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst.nbAdditionalBits, src->nbAdditionalBits, src->table_size * sizeof(u8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dst.baseValue, src->baseValue, src->table_size * sizeof(u32), cudaMemcpyHostToDevice, stream);
    };

    if (ll_mode == MODE_FSE || ll_mode == MODE_PREDEFINED || ll_mode == MODE_REPEAT) copy_table(ll_table, d_ll_table);
    if (of_mode == MODE_FSE || of_mode == MODE_PREDEFINED || of_mode == MODE_REPEAT) copy_table(of_table, d_of_table);
    if (ml_mode == MODE_FSE || ml_mode == MODE_PREDEFINED || ml_mode == MODE_REPEAT) copy_table(ml_table, d_ml_table);
    
    u32 *d_error_flag = nullptr;
    if (cuda_zstd::safe_cuda_malloc_async(&d_error_flag, sizeof(u32), stream) != cudaSuccess) return Status::ERROR_OUT_OF_MEMORY;
    cudaMemsetAsync(d_error_flag, 0, sizeof(u32), stream);

    k_fse_decode_interleaved_rfc<<<1, 1, 0, stream>>>(
        d_input, input_size,
        d_ll_table, ll_mode, ll_rle_value,
        d_of_table, of_mode, of_rle_value,
        d_ml_table, ml_mode, ml_rle_value,
        num_sequences,
        d_ll_out, d_of_out, d_ml_out,
        literals_limit, d_error_flag
    );
    
    u32 h_error = 0;
    cudaMemcpyAsync(&h_error, d_error_flag, sizeof(u32), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFreeAsync(d_error_flag, stream);
    auto free_table = [&](FSEDecodeTableDevice &t) {
        if (t.newState) cudaFreeAsync(t.newState, stream);
        if (t.symbol) cudaFreeAsync(t.symbol, stream);
        if (t.nbBits) cudaFreeAsync(t.nbBits, stream);
        if (t.nbAdditionalBits) cudaFreeAsync(t.nbAdditionalBits, stream);
        if (t.baseValue) cudaFreeAsync(t.baseValue, stream);
    };
    free_table(d_ll_table);
    free_table(d_of_table);
    free_table(d_ml_table);

    if (h_error != 0) return Status::ERROR_CORRUPT_DATA;
    return Status::SUCCESS;
}

__host__ Status FSE_buildDTable_rfc(
    const u16 *normFreqs,
    u32 maxSymbol,
    u32 tableLog,
    FSEDecodeTable &h_table
) {
    return FSE_buildDTable_Host(normFreqs, maxSymbol, 1u << tableLog, h_table);
}

} // namespace fse
} // namespace cuda_zstd
