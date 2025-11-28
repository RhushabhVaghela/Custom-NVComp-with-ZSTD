// ==========================================================================

// cuda_zstd_fse.h - Production FSE (Finite State Entropy) Implementation
// ==============================================================================
// RFC 8878 compliant with GPU optimizations
// Features: Adaptive tables, accurate normalization, symbol reordering, multi-table
// ==============================================================================

#ifndef CUDA_ZSTD_FSE_H_
#define CUDA_ZSTD_FSE_H_

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>
#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#ifdef __cplusplus
namespace cuda_zstd {
namespace fse {

// ==============================================================================
// CONSTANTS
// ==============================================================================

constexpr u32 FSE_MIN_TABLELOG = 5;
constexpr u32 FSE_MAX_TABLELOG = 12;    // ✅ NOW USED!
constexpr u32 FSE_DEFAULT_TABLELOG = 6;
constexpr u32 FSE_MAX_SYMBOL_VALUE = 255;
constexpr u32 FSE_ACCURACY_LOG = 6;     // ✅ NOW USED!

// Multi-table identifiers
enum class TableType : u8 {
    LITERALS = 0,
    MATCH_LENGTHS = 1,
    OFFSETS = 2,
    CUSTOM = 3
};

// ==============================================================================
// STRUCTURES
// ==============================================================================

/**
 * @brief (REPLACEMENT) This struct now holds a true Zstd-style CTable.
 * The CTable contains (nbBits, newStateBase) for each *symbol*.
 */
struct alignas(16) FSEEncodeTable {
    // (NEW) CTable format: { newStateBase, nbBits }
    struct FSEEncodeSymbol {
        u16 newStateBase;
        u8  nbBits;
    };

    FSEEncodeSymbol* d_symbol_table; // [max_symbol + 1]
    u32  table_log;
    u32  table_size;
    u32  max_symbol; // (NEW) Need to store this
    
    // RFC 8878 State Table: For correct initial state calculation
    u8*  d_state_to_symbol;     // [table_size] state→symbol mapping
    u16* d_symbol_first_state;  // [max_symbol+1] first state for each symbol
};

struct alignas(16) FSEDecodeTable {
    u16* newState;
    u8*  symbol;
    u8*  nbBits;
    u32  table_log;
    u32  table_size;
};

struct FSEStats {
    u32 frequencies[FSE_MAX_SYMBOL_VALUE + 1];
    u32 total_count;
    u32 max_symbol;
    u32 unique_symbols;
    u32 recommended_log;
    f32 entropy;
};

struct MultiTableFSE {
    FSEEncodeTable tables[4] = {};
    u32 active_tables = 0;
    MultiTableFSE() : tables{}, active_tables(0) {}
    // Defensive clear: does not free device memory, only clears logical state.
    inline void clear() {
        for (int i = 0; i < 4; ++i) {
            tables[i].d_symbol_table = nullptr;
            tables[i].table_log = 0;
            tables[i].table_size = 0;
            tables[i].max_symbol = 0;
        }
        active_tables = 0;
    }
};

// ============================================================================
// FSE Decompression Structures (NEW)
// ============================================================================

/**
 * @brief Represents a single entry in the FSE decoding table.
 *
 * It stores the symbol to emit and the next state transition information.
 */
struct FSEDecoderEntry {
    u16 next_state_base; // Base value for the next state
    u8  num_bits;        // Number of bits to read from the stream
    u8  symbol;          // The decoded symbol
};

/**
 * @brief GPU-resident FSE decoding table.
 *
 * This table is built from the normalized counts read from the
 * Zstd block header.
 */
struct FSEDecoderTable {
    // The table_log determines the size of this table (1 << table_log)
    FSEDecoderEntry* d_table; 
    u32 table_log;
    u32 max_symbol_value;
};


// ============================================================================
// FSE Decompression Host Functions (NEW)
// ============================================================================

/**
 * @brief Allocates and builds the FSE decoder table on the GPU.
 *
 * (NEW) This function now builds the table on the HOST (CPU)
 * and asynchronously copies it to the device.
 * 
 * @param h_normalized_counts Host array of normalized counts
 * @param num_counts Number of entries in the counts array
 * @param max_symbol_value The highest symbol value present
 * @param table_log The log2 size of the table to create
 * @param d_table_out Pointer to the FSEDecoderTable struct (which will be populated)
 * @param stream CUDA stream
 * @return Status
 */
Status build_fse_decoder_table(
    const i16* h_normalized_counts,
    u32 num_counts,
    u32 max_symbol_value,
    u32 table_log,
    FSEDecoderTable* d_table_out,
    cudaStream_t stream
);

/**
 * @brief Builds the FSE Encoding Table (CTable) on the host.
 */
Status FSE_buildCTable_Host(
    const u16* h_normalized,
    u32 max_symbol,
    u32 table_log,
    FSEEncodeTable* h_table
);

/**
 * @brief Builds the FSE Decoding Table (DTable) on the host.
 */
Status FSE_buildDTable_Host(
    const u16* h_normalized,
    u32 max_symbol,
    u32 table_size,
    FSEDecodeTable& h_table
);

/**
 * @brief Frees the memory associated with an FSEDecoderTable.
 */
Status free_fse_decoder_table(
    FSEDecoderTable* table,
    cudaStream_t stream
);

// ==============================================================================
// FEATURE 1: Adaptive Table Log Selection
// ==============================================================================

__host__ u32 select_optimal_table_log(
    const u32* frequencies,
    u32 total_count,
    u32 max_symbol,
    u32 unique_symbols
);

__host__ f32 calculate_entropy(
    const u32* frequencies,
    u32 total_count,
    u32 max_symbol
);

// ==============================================================================
// FEATURE 2: Accurate Normalization
// ==============================================================================

__host__ Status normalize_frequencies_accurate(
    const u32* frequencies,
    u32 total_count,
    u32 max_symbol,
    u16* normalized,
    u32 table_log,
    u32* actual_table_size
);

__host__ void apply_probability_correction(
    u16* normalized,
    const u32* frequencies,
    u32 max_symbol,
    u32 table_size
);

// ==============================================================================
// FEATURE 3: GPU-Optimized Symbol Reordering
// ==============================================================================

__host__ Status reorder_symbols_for_gpu(
    FSEEncodeTable& table,
    const u16* normalized,
    u32 max_symbol
);

__device__ __forceinline__ u32 get_coalesced_state(
    const FSEEncodeTable& table,
    u8 symbol,
    u32 thread_id
);

// ==============================================================================
// FEATURE 4: Multi-Table FSE
// ==============================================================================

__host__ Status create_multi_table_fse(
    MultiTableFSE& multi_table,
    const byte_t* input,
    u32 input_size,
    cudaStream_t stream = 0
);

__host__ Status encode_with_table_type(
    const byte_t* d_input,
    u32 input_size,
    byte_t* d_output,
    u32* d_output_size,
    TableType type,
    const MultiTableFSE& multi_table,
    cudaStream_t stream = 0
);

// ==============================================================================
// HELPER FUNCTIONS
// ==============================================================================

__host__ const u16* get_predefined_norm(
    TableType table_type,
    u32* max_symbol,
    u32* table_log
);

// ==============================================================================
// ENHANCED CORE API
// ==============================================================================
__host__ Status FSE_buildCTable_Host(
    const u16* h_norm,
    u32 max_symbol,
    u32 table_log,
    FSEEncodeTable* table
);

__host__ Status FSE_buildDTable_Simple_Host(
    const u16* h_normalized,
    u32 max_symbol,
    u32 table_log,
    u8** h_state_to_symbol_out
);


__host__ Status encode_fse_advanced(
    const byte_t* d_input,
    u32 input_size,
    byte_t* d_output,
    u32* d_output_size,
    TableType table_type = TableType::LITERALS,
    bool auto_table_log = true,
    bool accurate_norm = true,
    bool gpu_optimize = true,
    cudaStream_t stream = 0
);

__host__ Status encode_fse_batch(
    const byte_t** d_inputs,
    const u32* input_sizes,
    byte_t** d_outputs,
    u32* d_output_sizes,
    u32 num_blocks,
    cudaStream_t stream = 0
);

__host__ Status decode_fse(
    const byte_t* d_input,
    u32 input_size,
    byte_t* d_output,
    u32* d_output_size, // Host pointer
    cudaStream_t stream = 0
);

/**
 * @brief Decodes a stream using a predefined Zstd table.
 * THIS IS THE CORRECTED DECLARATION.
 */
__host__ Status decode_fse_predefined(
    const byte_t* d_input,    // <-- ADDED: The bitstream to read from
    u32 input_size,         // <-- ADDED: The size of the bitstream
    byte_t* d_output,       // <-- The destination buffer (e.g., d_literal_lengths)
    u32 num_sequences,      // <-- The number of symbols to decode
    u32* h_decoded_count,   // <-- Host pointer for the count of decoded symbols
    TableType table_type,   // <-- Which predefined table to use
    cudaStream_t stream = 0
);

/**
 * @brief (NEW) Encodes sequences using FSE tables (Tier 1).
 * Encodes in reverse order and interleaves LL, ML, OF streams.
 */
/**
 * @brief (NEW) Parallel Preparation Kernel: Pre-calculate codes and extra bits
 */
__global__ void fse_prepare_sequences_kernel(
    const u32* d_literal_lengths,
    const u32* d_match_lengths,
    const u32* d_offsets,
    u32 num_sequences,
    u8* d_ll_codes,
    u8* d_ml_codes,
    u8* d_of_codes,
    u32* d_ll_extras,
    u32* d_ml_extras,
    u32* d_of_extras,
    u8* d_ll_num_bits,
    u8* d_ml_num_bits,
    u8* d_of_num_bits
);

/**
 * @brief (NEW) Serial Encoding Kernel: Consumes pre-calculated codes
 */
__global__ void fse_encode_sequences_serial_kernel(
    const u8* d_ll_codes,
    const u8* d_ml_codes,
    const u8* d_of_codes,
    const u32* d_ll_extras,
    const u32* d_ml_extras,
    const u32* d_of_extras,
    const u8* d_ll_num_bits,
    const u8* d_ml_num_bits,
    const u8* d_of_num_bits,
    u32 num_sequences,
    const FSEEncodeTable* d_ll_table,
    const FSEEncodeTable* d_ml_table,
    const FSEEncodeTable* d_of_table,
    byte_t* d_output,
    u32* d_output_size_bytes,
    u32 max_output_size,
    u16* d_state_table
);

/**
 * @brief (NEW) Decodes sequences using FSE tables (Tier 1).
 * Reverse of fse_encode_sequences_kernel.
 */
/**
 * @brief (NEW) Serial State Update Kernel: Decodes codes and extra bits
 */
__global__ void fse_decode_states_serial_kernel(
    const byte_t* d_input,
    u32 input_size,
    u32 num_sequences,
    const FSEEncodeTable* d_ll_table,
    const FSEEncodeTable* d_ml_table,
    const FSEEncodeTable* d_of_table,
    const u8* d_ll_decode,
    const u8* d_ml_decode,
    const u8* d_of_decode,
    u8* d_ll_codes,
    u8* d_ml_codes,
    u8* d_of_codes,
    u32* d_ll_extras,
    u32* d_ml_extras,
    u32* d_of_extras,
    u32* d_bytes_read
);

/**
 * @brief (NEW) Parallel Reconstruction Kernel: Reconstructs values from codes
 */
__global__ void fse_reconstruct_sequences_kernel(
    const u8* d_ll_codes,
    const u8* d_ml_codes,
    const u8* d_of_codes,
    const u32* d_ll_extras,
    const u32* d_ml_extras,
    const u32* d_of_extras,
    u32 num_sequences,
    u32* d_literal_lengths,
    u32* d_match_lengths,
    u32* d_offsets
);

// ==============================================================================
// STATISTICS & ANALYSIS
// ==============================================================================

__host__ Status analyze_block_statistics(
    const byte_t* d_input,
    u32 input_size,
    FSEStats* stats,
    cudaStream_t stream = 0
);

__host__ void print_fse_stats(const FSEStats& stats);

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

__host__ Status validate_fse_table(const FSEEncodeTable& table);
__host__ void free_fse_table(FSEEncodeTable& table);
__host__ void free_multi_table(MultiTableFSE& multi_table);

// ==============================================================================
// PREDEFINED TABLES
// ==============================================================================

namespace predefined {
    extern const u16 default_ll_norm[36];
    extern const u16 default_of_norm[29];
    extern const u16 default_ml_norm[53];
}

} // namespace fse
} // namespace cuda_zstd

#endif // __cplusplus

#endif // CUDA_ZSTD_FSE_H
