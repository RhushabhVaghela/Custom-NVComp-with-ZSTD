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

/**
 * @brief Encode symbols using FSE (RFC 8878 compliant)
 * 
 * Encodes symbol sequence into bitstream.
 * 
 * @param symbols Input symbol array
 * @param numSymbols Number of symbols
 * @param table FSE table (built with FSE_buildTableRFC)
 * @param bitstream Output bitstream buffer
 * @param bitstreamCapacity Max output size
 * @param outputSize Actual output size (written by kernel)
 * @return Status SUCCESS or error code
 */
__host__ Status FSE_encodeRFC(
    const u8 *d_symbols,
    u32 numSymbols,
    const FSETableRFC &table,
    u8 *d_bitstream,
    size_t bitstreamCapacity,
    size_t *d_outputSize,
    cudaStream_t stream
);

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

/**
 * @brief RFC-compliant FSE encoding wrapper (drop-in replacement for old API)
 * 
 * This function has the same signature as the old launch_fse_encoding_kernel
 * but implements RFC 8878 compliant encoding logic using existing table data.
 * 
 * To migrate: Change line 4462 in cuda_zstd_manager.cu from:
 *   fse::launch_fse_encoding_kernel(...)
 * to:
 *   fse::launch_fse_encoding_kernel_rfc(...)
 * 
 * The function reads from existing FSEEncodeTable structures but uses
 * corrected encoding logic that follows RFC 8878 specification.
 */
__host__ Status launch_fse_encoding_kernel_rfc(
    const u8 *d_ll_codes, const u32 *d_ll_extras, const u8 *d_ll_bits,
    const u8 *d_of_codes, const u32 *d_of_extras, const u8 *d_of_bits,
    const u8 *d_ml_codes, const u32 *d_ml_extras, const u8 *d_ml_bits,
    u32 num_symbols, 
    unsigned char *d_bitstream, 
    size_t *d_output_pos,
    size_t bitstream_capacity,
    const FSEEncodeTable *d_tables,
    cudaStream_t stream
);

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_RFC_H_