#define CUDA_ZSTD_DEBUG_CU
#include "cuda_zstd_debug.h"

__device__ uint32_t g_debug_print_counter = 0;
__device__ uint32_t g_debug_print_limit = 0;

// Kernel to atomically reset and set the debug print limit. Using a kernel
// avoids name-mangling issues with cudaMemcpyToSymbol and works across
// separate translation units.
extern "C" __global__ void set_debug_print_limit_kernel(uint32_t limit) {
	g_debug_print_counter = 0u;
	g_debug_print_limit = limit;
}

// Host wrapper to call kernel; this keeps the kernel launch in a .cu
// translation unit and avoids putting <<<>>> calls in headers.
void set_device_debug_print_limit(uint32_t limit) {
	set_debug_print_limit_kernel<<<1, 1>>>(limit);
	cudaDeviceSynchronize();
}

