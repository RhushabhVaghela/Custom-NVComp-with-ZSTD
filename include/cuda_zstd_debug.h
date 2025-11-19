#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

using u32 = uint32_t;

// Device-side counters for throttling kernel-level debug prints
extern "C" {
    extern __device__ u32 g_debug_print_counter;
    extern __device__ u32 g_debug_print_limit;
}

// Host helper: reset (zero) the debug counter and set limit
// Host wrapper declaration. This is implemented in src/cuda_zstd_debug.cu
extern void set_device_debug_print_limit(u32 limit);

static inline void reset_debug_print_counter(u32 limit) {
    set_device_debug_print_limit(limit);
}
