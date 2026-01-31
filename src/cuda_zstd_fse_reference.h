/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 */

#ifndef CUDA_ZSTD_FSE_REFERENCE_H
#define CUDA_ZSTD_FSE_REFERENCE_H

#include "cuda_zstd_types.h"
#include <vector>

namespace cuda_zstd {
namespace fse {

struct FSE_CTable_Entry {
  u32 deltaNbBits;
  u32 deltaFindState;
  u16 nextState; // Base state for the next symbol
};

void build_fse_ctable_reference(std::vector<FSE_CTable_Entry> &table,
                                const std::vector<short> &normalized_counts,
                                unsigned table_log);

void fse_encode_step(u32 &state, u32 symbol, u32 next_symbol,
                     const std::vector<FSE_CTable_Entry> &ctable,
                     std::vector<u8> &bitstream, u64 &bitContainer,
                     u32 &bitCount);

} // namespace fse
} // namespace cuda_zstd

#endif // CUDA_ZSTD_FSE_REFERENCE_H
