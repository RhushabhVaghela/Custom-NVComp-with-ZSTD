/**
 * @file cuda_zstd_fse_rfc.h
 * @brief RFC 8878 Compliant FSE Implementation
 * 
 * Clean implementation following RFC 8878 Section 4.1 exactly.
 * No legacy code, no workarounds - pure spec compliance.
 */

#ifndef CUDA_ZSTD_FSE_RFC_H_
#define CUDA_ZSTD_FSE_RFC_H_

#include "cuda_zstd_types.h"
#include "cuda_zstd_internal.h"

// Forward declaration for wrapper compatibility
struct FSEEncodeTable;

namespace cuda_zstd {
namespace fse {

// RFC 8878 Table Parameters
static constexpr u32 FSE_RFC_TABLELOG_MIN = 5;   // 32 states minimum
static constexpr u32 FSE_RFC_TABLELOG_MAX = 20;  // 1M states maximum  
static constexpr u32 FSE_RFC_MAX_SYMBOL = 255;   // Max symbol value

/**
 * @brief FSE Distribution Table Entry (RFC 8878 compliant)
 * 
 * Each table entry maps a state to:
 * - The symbol emitted when in that state
 * - Number of bits to read for next state
 * - Baseline next state value
 */
struct FSEDistributionEntry {
    u8 symbol;      // Symbol emitted from this state
    u8 nbBits;      // Bits to read: 0 to table_log
    u16 nextState;  // Baseline next state (add read bits)
};

/**
 * @brief Complete FSE Table (RFC 8878 compliant)
 */
struct FSETableRFC {
    u32 tableLog;           // log2(tableSize)
    u32 tableSize;          // 1 << tableLog
    u32 maxSymbol;          // Maximum symbol value
    
    // Distribution table: indexed by state (0 to tableSize-1)
    FSEDistributionEntry *d_entries;
    
    // Symbol frequencies (normalized, sum = tableSize)
    u16 *d_normFreqs;
    
    // For encoding: first state for each symbol
    u16 *d_symbolFirstState;
    
    // Device memory allocated?
    bool allocated = false;
};

/**
 * @brief Build FSE table from normalized frequencies (RFC 8878)
 * 
 * Implements Algorithm 1 from RFC 8878 Section 4.1:
 * 1. Verify normalized frequencies sum to tableSize
 * 2. Spread symbols using step = (5/8)*tableSize + 3
 * 3. Build state transition table
 * 
 * @param normFreqs Normalized frequencies (sum must equal tableSize)
 * @param maxSymbol Maximum symbol value
 * @param tableLog Table size = 2^tableLog
 * @param table Output table structure
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_buildTableRFC(
    const u16 *normFreqs,
    u32 maxSymbol,
    u32 tableLog,
    FSETableRFC &table,
    void *d_workspace,
    size_t workspaceSize,
    cudaStream_t stream
);

// FSE_encodeRFC removed — used broken k_fse_encode kernel.
// Production encoder: k_encode_fse_interleaved in cuda_zstd_fse_encoding_kernel.cu.

/**
 * @brief Decode FSE bitstream (RFC 8878 compliant)
 * 
 * Decodes bitstream back to symbols.
 * 
 * @param bitstream Input bitstream
 * @param bitstreamSize Size in bytes
 * @param table FSE table
 * @param numSymbols Expected number of symbols
 * @param symbols Output symbol array
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_decodeRFC(
    const u8 *d_bitstream,
    size_t bitstreamSize,
    const FSETableRFC &table,
    u32 numSymbols,
    u8 *d_symbols,
    cudaStream_t stream
);

/**
 * @brief Normalize frequencies for FSE table (RFC 8878)
 * 
 * Converts raw frequencies to normalized frequencies that sum
 * exactly to tableSize.
 * 
 * @param rawFreqs Raw symbol frequencies
 * @param maxSymbol Maximum symbol value
 * @param numSamples Total count (sum of rawFreqs)
 * @param tableLog Target table size = 2^tableLog
 * @param normFreqs Output normalized frequencies
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_normalizeFreqs(
    const u32 *rawFreqs,
    u32 maxSymbol,
    u32 numSamples,
    u32 tableLog,
    u16 *normFreqs
);

/**
 * @brief Allocate device memory for FSE table
 */
__host__ Status FSE_allocTableRFC(
    FSETableRFC &table,
    u32 tableLog,
    u32 maxSymbol,
    cudaStream_t stream
);

/**
 * @brief Free device memory for FSE table
 */
__host__ void FSE_freeTableRFC(
    FSETableRFC &table,
    cudaStream_t stream
);

// =============================================================================
// WRAPPER: Backward-compatible API
// =============================================================================

// launch_fse_encoding_kernel_rfc removed — dead code (never called).
// Production encoder: k_encode_fse_interleaved in cuda_zstd_fse_encoding_kernel.cu.

/**
 * @brief RFC-compliant FSE decoding wrapper
 *
 * Decodes bitstream produced by the production FSE encoder.
 * 
 * @param d_bitstream Device pointer to bitstream data
 * @param bitstream_size Size of bitstream in bytes
 * @param h_tables Host pointer to FSEDecodeTable (contains table data in host memory)
 * @param num_symbols Number of sequences to decode
 * @param d_ll_codes Output: LL symbol codes (device memory)
 * @param d_of_codes Output: OF symbol codes (device memory)
 * @param d_ml_codes Output: ML symbol codes (device memory)
 * @param d_ll_extras Output: LL extra bits (device memory)
 * @param d_of_extras Output: OF extra bits (device memory)
 * @param d_ml_extras Output: ML extra bits (device memory)
 * @param d_ll_bits Output: LL nbBits (device memory)
 * @param d_of_bits Output: OF nbBits (device memory)
 * @param d_ml_bits Output: ML nbBits (device memory)
 * @param stream CUDA stream
 * @return Status SUCCESS or error code
 */
__host__ Status launch_fse_decoding_kernel_rfc(
    const u8 *d_bitstream,
    size_t bitstream_size,
    const FSEDecodeTable *h_tables,
    u32 num_symbols,
    u8 *d_ll_codes,
    u8 *d_of_codes,
    u8 *d_ml_codes,
    u32 *d_ll_extras,
    u32 *d_of_extras,
    u32 *d_ml_extras,
    u8 *d_ll_bits,
    u8 *d_of_bits,
    u8 *d_ml_bits,
    cudaStream_t stream
);

/**
 * @brief RFC-compliant interleaved FSE decoder
 * 
 * Replaces decode_sequences_interleaved with correct RFC decoding.
 * Handles all three streams (LL, OF, ML) simultaneously with different modes.
 * 
 * @param d_input Device pointer to compressed bitstream
 * @param input_size Size of input in bytes
 * @param num_sequences Number of sequences to decode
 * @param d_ll_out Output: literal lengths (device memory)
 * @param d_of_out Output: offsets (device memory)
 * @param d_ml_out Output: match lengths (device memory)
 * @param ll_mode Literal length mode (0=Predefined, 1=RLE, 2=FSE)
 * @param of_mode Offset mode (0=Predefined, 1=RLE, 2=FSE)
 * @param ml_mode Match length mode (0=Predefined, 1=RLE, 2=FSE)
 * @param ll_table Literal length decode table (for mode 0 or 2)
 * @param of_table Offset decode table (for mode 0 or 2)
 * @param ml_table Match length decode table (for mode 0 or 2)
 * @param literals_limit Maximum literal length limit
 * @param stream CUDA stream
 * @return Status SUCCESS or error code
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
    u32 ll_rle_value,
    u32 of_rle_value,
    u32 ml_rle_value,
    u32 literals_limit,
    cudaStream_t stream
);

/**
 * @brief Build RFC-compliant FSE decoder table from normalized frequencies
 * 
 * This function ensures encoder and decoder use consistent state transitions
 * by building the decoder table from the same frequency data used by the encoder.
 * 
 * @param normFreqs Normalized frequencies (array of size maxSymbol+1)
 * @param maxSymbol Maximum symbol value
 * @param tableLog Table logarithm (table size = 2^tableLog)
 * @param h_table Output table structure (arrays must be pre-allocated by caller)
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_buildDTable_rfc(
    const u16 *normFreqs,
    u32 maxSymbol,
    u32 tableLog,
    FSEDecodeTable &h_table
);

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_RFC_H_
