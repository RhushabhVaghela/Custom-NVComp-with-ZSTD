// cuda_zstd_fse_encoding_kernel.h - GPU FSE Encoding Kernel Wrapper
#ifndef CUDA_ZSTD_FSE_ENCODING_KERNEL_H_
#define CUDA_ZSTD_FSE_ENCODING_KERNEL_H_

#include "cuda_zstd_fse.h"
#include "cuda_zstd_types.h"

namespace cuda_zstd {
namespace fse {

/**
 * @brief Launches the GPU FSE Encoding Kernel.
 *
 * Supports two modes:
 * 1. Single-Block: input_size < 128KB (Legacy/Small).
 * 2. Multi-Block: input_size > 128KB (Streaming/Large).
 *
 * @param d_sequences Device pointer to Sequence/Raw input.
 * @param num_sequences Number of sequences (or elements).
 * @param d_bitstream Output bitstream buffer.
 * @param bitstream_capacity Max size of output.
 * @param tables FSE Encoding Tables (CTables on Device).
 * @param stream CUDA stream.
 */
Status
launch_fse_encoding_kernel(const u32 *d_ll, const u32 *d_of, const u32 *d_ml,
                           u32 num_sequences, byte_t *d_bitstream,
                           size_t *d_output_pos, size_t bitstream_capacity,
                           const FSEEncodeTable *d_tables, // Expects array of 3
                           cudaStream_t stream);

/**
 * @brief Builds FSE CTable on the Device directly (Phase 2a).
 */
Status FSE_buildCTable_Device(const u32 *d_normalized_counters, u32 max_symbol,
                              u32 table_log, FSEEncodeTable *d_table,
                              void *d_workspace, size_t workspace_size,
                              cudaStream_t stream);

} // namespace fse
} // namespace cuda_zstd

#endif
