/*
jit_kernel.cu (FIXED - nvCOMP 5.0 ZSTD with proper function closing)
Implements full GPU-based Zstandard compression and decompression
using nvCOMP 5.0 actual C++ Manager API
*/

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept> 
#include <exception>
#include <cmath>
#include <iomanip>

// --- CORRECTED: nvCOMP 5.0 C++ Manager API Includes ---
#include <nvcomp.hpp>

// --- CORRECTED: nvCOMP 5.0 C++ Manager API Includes ---
#include <nvcomp.hpp>
#include <nvcomp/zstd.hpp>              // ✅ Zstd specific header
#include <nvcomp/nvcompManagerFactory.hpp>  // ✅ Factory for decompression

#define NV_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
              << cudaGetErrorString(err) << std::endl; \
    throw std::runtime_error("CUDA error in kernel"); \
  } \
} while(0)

// === CPU Utility Function ===
// This is a "host" (CPU) function. It cannot be called from Python.
std::string format_bytes_cpp_host(long long size_in_bytes) {
    if (size_in_bytes <= 0) {
        return "0 B";
    }
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
    
    int i = static_cast<int>(floor(log(size_in_bytes) / log(1024)));
    double converted_size = size_in_bytes / pow(1024, i);
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << converted_size << " " << units[i];
    return ss.str();
}

// ====================================================================
// 🔥 GPU DENSE ZSTD COMPRESSION (Corrected ZstdManager API)
// ====================================================================

torch::Tensor jit_compress_zstd_v1_launcher(torch::Tensor input_tensor, int zstd_level) {
  std::cout << "[JIT Kernel .cu] 🔥 Launching GPU Zstd COMPRESSION..." << std::endl;

  c10::cuda::CUDAStream stream = c10::cuda::getDefaultCUDAStream();
  cudaStream_t raw_stream = stream.stream();

  const uint8_t* input_data = static_cast<uint8_t*>(input_tensor.data_ptr());
  const size_t input_size = input_tensor.nbytes();

  std::cout << "[JIT Kernel .cu] Input size: " << format_bytes_cpp_host(input_size) << std::endl;

  try {
    constexpr size_t chunk_size = 64 * 1024;

    // --- START: KERNEL FIX ---
    // NOTE: The C++ Manager API in this nvcomp_12 version does not support
    // setting 'level' on the nvcompBatchedZstdCompressOpts_t struct,
    // and has no constructor overload for an integer level.
    // The passed 'zstd_level' will be ignored, and nvCOMP's default level will be used.
    std::cout << "[JIT Kernel .cu] Using Zstd compression (default level). Passed level " 
              << zstd_level << " is unused by this C++ Manager API." << std::endl;

    // Create the compression options struct and zero-initialize it 
    nvcompBatchedZstdCompressOpts_t compress_opts = {}; 

    // Create decompression options struct (needed for constructor)
    nvcompBatchedZstdDecompressOpts_t decompress_opts;

    // Create the manager using the options struct 
    nvcomp::ZstdManager manager(
        chunk_size,
        compress_opts,      // Pass the default struct
        decompress_opts,                // Default decompress options (default-initialized)
        raw_stream
    );
    // --- END: KERNEL FIX ---

    std::cout << "[JIT Kernel .cu] Manager created successfully" << std::endl;
    
    nvcomp::CompressionConfig comp_config = manager.configure_compression(input_size);
    const size_t max_compressed_size = comp_config.max_compressed_buffer_size;
    
    std::cout << "[JIT Kernel .cu] Max compressed size: " << max_compressed_size << " bytes" << "-->" << format_bytes_cpp_host(max_compressed_size) << std::endl;
    
    // ... (rest of the function is correct) ...
    
    torch::Tensor output_buffer = torch::empty(
      max_compressed_size,
      input_tensor.options().dtype(torch::kUInt8)
    );
    
    uint8_t* output_data = static_cast<uint8_t*>(output_buffer.data_ptr());

    torch::Tensor d_output_size_tensor = torch::empty(
        {1}, 
        torch::TensorOptions().dtype(torch::kInt64).device(input_tensor.device())
    );
    size_t* d_output_size_ptr = static_cast<size_t*>(d_output_size_tensor.data_ptr());

    manager.compress(input_data, output_data, comp_config, d_output_size_ptr);
    NV_CHECK(cudaStreamSynchronize(raw_stream));
    size_t h_output_size = static_cast<size_t>(d_output_size_tensor.item<int64_t>());
    
    std::cout << "[JIT Kernel .cu] ✅ GPU Compression complete. Compressed size: "
              << h_output_size << " bytes" << "-->" << format_bytes_cpp_host(h_output_size) << std::endl;
    
    return output_buffer.slice(0, 0, h_output_size).clone();
    
  } catch (const std::exception& e) {
    std::cerr << "[JIT Kernel .cu] ❌ Compression error: " << e.what() << std::endl;
    throw;
  }
}

// ====================================================================
// 🔥 GPU DENSE ZSTD DECOMPRESSION (Corrected Manager Factory API)
// ====================================================================

torch::Tensor jit_decompress_zstd_v1_launcher(
  torch::Tensor compressed_tensor,
  int64_t uncompressed_size_bytes
) {
  std::cout << "[JIT Kernel .cu] 🔥 Launching GPU Zstd DECOMPRESSION..." << std::endl;
  
  // ✅ FIXED: Use c10::cuda namespace with proper header
  c10::cuda::CUDAStream stream = c10::cuda::getDefaultCUDAStream();
  cudaStream_t raw_stream = stream.stream();
  
  const uint8_t* compressed_data = static_cast<uint8_t*>(compressed_tensor.data_ptr());
  const size_t compressed_size = compressed_tensor.nbytes();
  
  std::cout << "[JIT Kernel .cu] Compressed size: " << format_bytes_cpp_host(compressed_size) << std::endl;
  std::cout << "[JIT Kernel .cu] Uncompressed size: " << format_bytes_cpp_host(uncompressed_size_bytes) << std::endl;
  
  try {
    // ✅ CORRECTED: Use create_manager factory for decompression
    // This factory automatically detects compression format from header
    // ✅ ADDED: Full template type specification
    std::shared_ptr<nvcomp::nvcompManagerBase> manager = 
      nvcomp::create_manager(compressed_data, raw_stream);
    
    std::cout << "[JIT Kernel .cu] Manager created successfully for decompression" << std::endl;
    
    // Allocate output buffer on GPU
    torch::Tensor output_buffer = torch::empty(
      {uncompressed_size_bytes},
      compressed_tensor.options().dtype(torch::kUInt8)
    );
    
    uint8_t* output_data = static_cast<uint8_t*>(output_buffer.data_ptr());
    
    // ✅ CORRECTED: Use configure_decompression() to get config from compressed buffer
    nvcomp::DecompressionConfig decomp_config = 
      manager->configure_decompression(compressed_data);
    
    // ✅ CORRECTED: Perform decompression using decompress()
    manager->decompress(output_data, compressed_data, decomp_config);
    
    // Synchronize the stream
    NV_CHECK(cudaStreamSynchronize(raw_stream));
    
    std::cout << "[JIT Kernel .cu] ✅ GPU Decompression complete. Output size: "
              << uncompressed_size_bytes << " bytes" << "-->" << format_bytes_cpp_host(uncompressed_size_bytes) << std::endl;
    
    return output_buffer;
    
  } catch (const std::exception& e) {
    std::cerr << "[JIT Kernel .cu] ❌ Decompression error: " << e.what() << std::endl;
    throw;
  }
}

// ====================================================================
// 🔥 GPU SCATTER-ADD KERNEL (for sparse operations)
// ====================================================================

template<typename T>
__global__ void JIT_Scatter_Add_Kernel_V1(
  T* __restrict__ W_base_flat,
  const int64_t* __restrict__ indices_decomp,
  const T* __restrict__ values_decomp,
  const int64_t* __restrict__ chunk_map,
  int N_chunks
) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N_chunks; i += gridDim.x * blockDim.x) {
    int64_t offset = chunk_map[i * 2 + 0];
    int64_t num_in_chunk = chunk_map[i * 2 + 1];
    for (int64_t j = 0; j < num_in_chunk; ++j) {
      int64_t global_idx = indices_decomp[offset + j];
      T value = values_decomp[offset + j];
      atomicAdd(&W_base_flat[global_idx], value);
    }
  }
}

template<typename T>
void decompress_and_scatter_add_launcher(
  torch::Tensor W_base,
  torch::Tensor comp_indices,
  torch::Tensor comp_values,
  torch::Tensor chunk_map,
  torch::Tensor decomp_metadata
) {
  std::cout << "[JIT Kernel .cu] 🔥 Launching scatter-add (SPARSE)..." << std::endl;
  std::cout << "[JIT Kernel .cu] ⚠️ GPU SPARSE decompression is not implemented." << std::endl;
  std::cout << "[JIT Kernel .cu] Only DENSE ZSTD compression/decompression is available." << std::endl;
  throw std::runtime_error(
    "GPU SPARSE decompression kernel is not implemented. "
    "Only GPU DENSE ZSTD compression/decompression is available."
  );
}

// Explicit template instantiations for float and double
template void decompress_and_scatter_add_launcher<float>(
  torch::Tensor W_base,
  torch::Tensor comp_indices,
  torch::Tensor comp_values,
  torch::Tensor chunk_map,
  torch::Tensor decomp_metadata
);

template void decompress_and_scatter_add_launcher<double>(
  torch::Tensor W_base,
  torch::Tensor comp_indices,
  torch::Tensor comp_values,
  torch::Tensor chunk_map,
  torch::Tensor decomp_metadata
);

/*
====================================================================
🔥 GPU EALE LOSSLESS RECONSTRUCTION KERNEL
====================================================================
*/

__global__ void eale_lossless_reconstruct_kernel(
    const int8_t* W_Base_Packed,   // The 8-bit Base Model data (VRAM)
    const float* W_Residual,      // The 32-bit decompressed Residual (JIT)
    void* W_Recon_Output,         // Output: The final reconstructed weight
    const torch::ScalarType output_dtype, // Dtype of original tensor (fp32, fp16, bf16)
    const float scale_factor,     // The scale factor from metadata
    const size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // 1. Dequantize the 8-bit Base to float32
        // (This reverses the (round(W_Orig * scale) / scale) logic)
        float W_Base_HP = ((float)W_Base_Packed[idx]) / scale_factor;
        
        // 2. Perform 100% Lossless Addition in float32
        float recon_val_fp32 = W_Base_HP + W_Residual[idx];

        // 3. Cast to the *original* target dtype
        if (output_dtype == torch::kFloat16) {
            ((__half*)W_Recon_Output)[idx] = __float2half(recon_val_fp32);
        } else if (output_dtype == torch::kBFloat16) {
            ((__nv_bfloat16*)W_Recon_Output)[idx] = __float2bfloat16(recon_val_fp32);
        } else { // Default to float32
            ((float*)W_Recon_Output)[idx] = recon_val_fp32;
        }
    }
}

// Launcher function
void eale_lossless_reconstruct_launcher(
    torch::Tensor W_Base_Packed,   // int8
    torch::Tensor W_Residual,      // float32
    torch::Tensor W_Recon_Output,  // Target Dtype (fp32, fp16, bf16)
    float scale_factor
) {
    // Ensure tensors are on CUDA
    if (!W_Base_Packed.is_cuda() || !W_Residual.is_cuda() || !W_Recon_Output.is_cuda()) {
        throw std::runtime_error("EALE Kernel: All tensors must be on CUDA.");
    }
    
    // Validate dtypes
    if (W_Base_Packed.scalar_type() != torch::kInt8) {
        throw std::runtime_error("EALE Kernel: W_Base_Packed must be int8.");
    }
    if (W_Residual.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("EALE Kernel: W_Residual must be float32.");
    }

    const size_t num_elements = W_Recon_Output.numel();
    if (num_elements == 0) return;

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // Kernel launch configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Get the output data type
    const torch::ScalarType output_dtype = W_Recon_Output.scalar_type();

    // Launch the kernel
    eale_lossless_reconstruct_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        W_Base_Packed.data_ptr<int8_t>(),
        W_Residual.data_ptr<float>(),
        W_Recon_Output.data_ptr(), // Use void*
        output_dtype,
        scale_factor,
        num_elements
    );
    
    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}