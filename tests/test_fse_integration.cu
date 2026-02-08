#include "cuda_zstd_fse.h"
#include "cuda_zstd_fse_encoding_kernel.h"
#include "cuda_zstd_types.h"
#include "cuda_zstd_safe_alloc.h"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace cuda_zstd;

void check(Status s, const char *msg) {
  if (s != Status::SUCCESS) {
    std::cerr << "FAILED: " << msg << " Status=" << (int)s << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

int main() {
  std::cout << "Testing FSE Integation (Builder + Kernel)..." << std::endl;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // 1. Build Tables on GPU
  fse::FSEEncodeTable *d_tables;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_tables, 3 * sizeof(fse::FSEEncodeTable)));

  u32 valid_max_ll = 0, valid_max_of = 0, valid_max_ml = 0;
  std::vector<u32> ll_norm_vec, of_norm_vec, ml_norm_vec;

  auto build_table_and_get_norm = [&](fse::TableType type, int idx,
                                      std::vector<u32> &norm_vec,
                                      u32 &max_s_out) {
    u32 max_s, t_log;
    const u16 *h_norm = fse::get_predefined_norm(type, &max_s, &t_log);
    max_s_out = max_s;

    norm_vec.resize(max_s + 1);
    std::vector<u32> h_norm_u32(max_s + 1);
    for (u32 i = 0; i <= max_s; ++i) {
      h_norm_u32[i] = h_norm[i];
      norm_vec[i] = h_norm[i];
    }

    u32 *d_norm;
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_norm, (max_s + 1) * sizeof(u32)));
    CHECK_CUDA(cudaMemcpy(d_norm, h_norm_u32.data(), (max_s + 1) * sizeof(u32),
                          cudaMemcpyHostToDevice));

    fse::FSEEncodeTable h_desc;
    h_desc.max_symbol = max_s;
    h_desc.table_log = t_log;
    u32 table_size = 1 << t_log;
    h_desc.table_size = table_size;

    CHECK_CUDA(
        cuda_zstd::safe_cuda_malloc(&h_desc.d_symbol_table,
                   (max_s + 1) * sizeof(fse::FSEEncodeTable::FSEEncodeSymbol)));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&h_desc.d_next_state, table_size * sizeof(u16)));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&h_desc.d_nbBits_table, table_size * sizeof(u8)));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&h_desc.d_state_to_symbol, table_size * sizeof(u8)));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&h_desc.d_symbol_first_state, (max_s + 1) * sizeof(u16)));
    CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&h_desc.d_next_state_vals, table_size * sizeof(u16)));

    // Copy descriptor to array
    CHECK_CUDA(cudaMemcpy(&d_tables[idx], &h_desc, sizeof(fse::FSEEncodeTable),
                          cudaMemcpyHostToDevice));

    // Run Builder
    fse::FSE_buildCTable_Device(d_norm, max_s, t_log, &d_tables[idx], nullptr,
                                0, stream);
  };

  build_table_and_get_norm(fse::TableType::LITERALS, 0, ll_norm_vec,
                           valid_max_ll);
  build_table_and_get_norm(fse::TableType::OFFSETS, 1, of_norm_vec,
                           valid_max_of);
  build_table_and_get_norm(fse::TableType::MATCH_LENGTHS, 2, ml_norm_vec,
                           valid_max_ml);

  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 2. Setup Input Data (Sequences)
  int num_refs = 127;
  std::vector<u32> h_ll(num_refs);
  std::vector<u32> h_of(num_refs);
  std::vector<u32> h_ml(num_refs);

  auto get_valid_sym = [](const std::vector<u32> &norms, int seed) {
    int range = norms.size();
    for (int k = 0; k < range; ++k) {
      u32 sym = (seed + k) % range;
      if (norms[sym] > 0)
        return sym;
    }
    return (u32)0;
  };

  for (int i = 0; i < num_refs; ++i) {
    h_ll[i] = get_valid_sym(ll_norm_vec, i);
    h_of[i] = get_valid_sym(of_norm_vec, i);
    h_ml[i] = get_valid_sym(ml_norm_vec, i);
  }

  u8 *d_ll, *d_of, *d_ml;
  u32 *d_ll_extras, *d_of_extras, *d_ml_extras;
  u8 *d_ll_bits, *d_of_bits, *d_ml_bits;

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ll, num_refs));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_of, num_refs));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ml, num_refs));

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ll_extras, num_refs * 4));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_of_extras, num_refs * 4));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ml_extras, num_refs * 4));
  CHECK_CUDA(cudaMemset(d_ll_extras, 0, num_refs * 4));
  CHECK_CUDA(cudaMemset(d_of_extras, 0, num_refs * 4));
  CHECK_CUDA(cudaMemset(d_ml_extras, 0, num_refs * 4));

  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ll_bits, num_refs));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_of_bits, num_refs));
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_ml_bits, num_refs));
  CHECK_CUDA(cudaMemset(d_ll_bits, 0, num_refs));
  CHECK_CUDA(cudaMemset(d_of_bits, 0, num_refs));
  CHECK_CUDA(cudaMemset(d_ml_bits, 0, num_refs));

  // Update inputs on device
  std::vector<u8> h_ll_u8(num_refs), h_of_u8(num_refs), h_ml_u8(num_refs);
  for (int i = 0; i < num_refs; ++i) {
    h_ll_u8[i] = (u8)h_ll[i];
    h_of_u8[i] = (u8)h_of[i];
    h_ml_u8[i] = (u8)h_ml[i];
  }

  CHECK_CUDA(
      cudaMemcpy(d_ll, h_ll_u8.data(), num_refs, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_of, h_of_u8.data(), num_refs, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_ml, h_ml_u8.data(), num_refs, cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 3. Launch Encoding
  size_t *d_pos;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_pos, sizeof(size_t)));
  CHECK_CUDA(cudaMemset(d_pos, 0, sizeof(size_t)));

  size_t capacity = num_refs * 8 + 512;
  byte_t *d_bitstream;
  CHECK_CUDA(cuda_zstd::safe_cuda_malloc(&d_bitstream, capacity));

  Status launchStatus = fse::launch_fse_encoding_kernel(
      d_ll, d_ll_extras, d_ll_bits, d_of, d_of_extras, d_of_bits, d_ml,
      d_ml_extras, d_ml_bits, num_refs, d_bitstream, d_pos, capacity, d_tables,
      stream);

  check(launchStatus, "Kernel Launch");

  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 4. Verify Output
  size_t h_pos_val;
  CHECK_CUDA(
      cudaMemcpy(&h_pos_val, d_pos, sizeof(size_t), cudaMemcpyDeviceToHost));

  std::cout << "Output Size: " << h_pos_val << " bytes" << std::endl;

  if (h_pos_val == 0 && num_refs > 0) {
    std::cerr << "FAILED: Output size is 0" << std::endl;
    return 1;
  }

  if (h_pos_val > 0) {
    std::cout << "SUCCESS: Integration Verified." << std::endl;
    // Verify bitstream content?
    // For now, just non-zero output ensures kernel ran and produced data.
  }

  // Cleanup (Simplified for test)
  cudaFree(d_tables);
  cudaFree(d_ll);
  cudaFree(d_of);
  cudaFree(d_ml);
  cudaFree(d_ll_extras);
  cudaFree(d_of_extras);
  cudaFree(d_ml_extras);
  cudaFree(d_ll_bits);
  cudaFree(d_of_bits);
  cudaFree(d_ml_bits);
  cudaFree(d_pos);
  cudaFree(d_bitstream);
  cudaStreamDestroy(stream);

  return 0;
}
