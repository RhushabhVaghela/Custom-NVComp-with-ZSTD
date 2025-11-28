#ifndef CUDA_ZSTD_DEBUG_H
#define CUDA_ZSTD_DEBUG_H

#include <stdint.h>

// Uncomment the following line to enable debug output
// #define CUDA_ZSTD_DEBUG

#ifdef __CUDACC__
#ifndef CUDA_ZSTD_DEBUG_CU
extern __device__ uint32_t g_debug_print_counter;
extern __device__ uint32_t g_debug_print_limit;
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void set_device_debug_print_limit(uint32_t limit);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ZSTD_DEBUG_H
