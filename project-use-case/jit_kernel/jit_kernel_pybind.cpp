/*
jit_decompress_pybind.cpp (FIXED FOR nvCOMP 5.0 C++ Manager API)

PyBind11 wrapper for the nvCOMP 5.0 kernel with enhanced error handling.

** UPDATED **
- Added C++ helper functions to get Zstandard compression level bounds.
- Bound these functions to Python for runtime querying.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cmath>      // <-- ADD THIS
#include <iomanip>    // <-- ADD THIS

// === Forward declaration for DENSE ZSTD ===

// --- FIX: Added int zstd_level ---
torch::Tensor jit_compress_zstd_v1_launcher(torch::Tensor input_tensor, int zstd_level);

torch::Tensor jit_decompress_zstd_v1_launcher(
    torch::Tensor compressed_tensor,
    int64_t uncompressed_size_bytes
);

// === Forward declaration for SPARSE ===

template<typename T>
void decompress_and_scatter_add_launcher(
    torch::Tensor W_base,
    torch::Tensor comp_indices,
    torch::Tensor comp_values,
    torch::Tensor chunk_map,
    torch::Tensor decomp_metadata
);

// --- FIX: Added EALE launcher ---
void eale_lossless_reconstruct_launcher(
    torch::Tensor W_Base_Packed,
    torch::Tensor W_Residual,
    torch::Tensor W_Recon_Output,
    float scale_factor
);

// Helper function to format tensor info
std::string format_tensor_info(const torch::Tensor& t, const std::string& name) {
    std::stringstream ss;
    ss << "\n  " << name << ":";
    ss << "\n  Shape: [";
    for (int i = 0; i < t.dim(); ++i) {
        ss << t.size(i);
        if (i < t.dim() - 1) ss << ", ";
    }
    ss << "]";
    ss << "\n  Device: " << t.device();
    ss << "\n  DType: " << t.dtype();
    ss << "\n  Numel: " << t.numel();
    return ss.str();
}

// === EXISTING SPARSE Function ===

torch::Tensor jit_apply_v1_full_gpu(
    torch::Tensor W_base,
    torch::Tensor comp_indices,
    torch::Tensor comp_values,
    torch::Tensor chunk_map,
    torch::Tensor decomp_metadata
) {
    std::cout << "[JIT Kernel Pybind] C++ bridge 'jit_apply_v1_full_gpu' called." << std::endl;
    
    try {
        std::cout << "[JIT Kernel Pybind] Validating inputs..." << std::endl;
        
        if (!W_base.is_cuda()) {
            throw std::runtime_error(format_tensor_info(W_base, "W_base") + " is not on CUDA device");
        }
        if (!comp_indices.is_cuda()) {
            throw std::runtime_error(format_tensor_info(comp_indices, "comp_indices") + " is not on CUDA device");
        }
        if (!comp_values.is_cuda()) {
            throw std::runtime_error(format_tensor_info(comp_values, "comp_values") + " is not on CUDA device");
        }
        if (!chunk_map.is_cuda()) {
            throw std::runtime_error(format_tensor_info(chunk_map, "chunk_map") + " is not on CUDA device");
        }
        // Note: The PDF contains a bug here, checking if decomp_metadata.is_cuda() is true.
        // Your original logic is correct: metadata should likely be on CPU.
        // We will keep your original (correct) logic.
        if (decomp_metadata.is_cuda()) {
            throw std::runtime_error(format_tensor_info(decomp_metadata, "decomp_metadata") + " should be on CPU");
        }
        
        std::cout << "[JIT Kernel Pybind] Input validation passed." << std::endl;
        
        auto W_reconstructed = W_base.clone();
        
        std::cout << "[JIT Kernel Pybind] Launching kernel with W_base shape: ";
        for (int i = 0; i < W_base.dim(); ++i) {
            std::cout << W_base.size(i);
            if (i < W_base.dim() - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        
        decompress_and_scatter_add_launcher<float>(
            W_reconstructed,
            comp_indices,
            comp_values,
            chunk_map,
            decomp_metadata
        );
        
        std::cout << "[JIT Kernel Pybind] Kernel execution completed successfully." << std::endl;
        return W_reconstructed;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[JIT Kernel Pybind] ❌ EXCEPTION CAUGHT:\n";
        std::cerr << "Error message: " << e.what() << "\n";
        throw std::runtime_error(std::string("nvCOMP kernel failed: ") + e.what());
    }
}

// === GPU DENSE ZSTD COMPRESSION ===

// --- FIX: Added int zstd_level ---
torch::Tensor jit_compress_zstd_v1(torch::Tensor delta_gpu, int zstd_level) {
    if (!delta_gpu.is_cuda()) {
        throw std::runtime_error("Input tensor for compression must be on CUDA device.");
    }
    
    if (!delta_gpu.is_contiguous()) {
        delta_gpu = delta_gpu.contiguous();
    }
    
    // --- FIX: Pass zstd_level ---
    return jit_compress_zstd_v1_launcher(delta_gpu, zstd_level);
}

// === GPU DENSE ZSTD DECOMPRESSION ===

torch::Tensor jit_decompress_zstd_v1(
    torch::Tensor compressed_gpu,
    int64_t uncompressed_size_bytes
) {
    if (!compressed_gpu.is_cuda()) {
        throw std::runtime_error("Compressed tensor for decompression must be on CUDA device.");
    }
    
    if (compressed_gpu.dtype() != torch::kUInt8) {
        throw std::runtime_error("Compressed tensor must be of dtype torch.uint8.");
    }
    
    if (!compressed_gpu.is_contiguous()) {
        compressed_gpu = compressed_gpu.contiguous();
    }
    
    return jit_decompress_zstd_v1_launcher(compressed_gpu, uncompressed_size_bytes);
}

// === UTILITIES ===

// --- ADDED THIS FUNCTION ---
std::string format_bytes_cpp(long long size_in_bytes) {
    if (size_in_bytes <= 0) {
        return "0 B";
    }
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
    
    // Calculate the index of the appropriate unit
    int i = static_cast<int>(floor(log(size_in_bytes) / log(1024)));
    
    // Calculate the converted size
    double converted_size = size_in_bytes / pow(1024, i);
    
    // Format the output string
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << converted_size << " " << units[i];
    return ss.str();
}

bool validate_nvcomp_format(torch::Tensor comp_data) {
    std::cout << "[JIT Kernel Pybind] Format validation: ";
    
    if (!comp_data.is_cuda()) {
        std::cout << "❌ Data not on CUDA device" << std::endl;
        return false;
    }
    
    if (comp_data.numel() == 0) {
        std::cout << "❌ Empty compressed data" << std::endl;
        return false;
    }
    
    if (comp_data.numel() < 16) {
        std::cout << "⚠️  Suspiciously small data (" << comp_data.numel() << " bytes)" << std::endl;
    }
    
    std::cout << "✅ Basic format validation passed (" << comp_data.numel() << " bytes)" << std::endl;
    return true;
}

std::string get_backend_version() {
    return "JIT Kernel v2.3.0 (Manager API + Zstd)";
}

std::string get_system_diagnostics() {
    std::stringstream ss;
    ss << "JIT Kernel Backend Diagnostics:\n";
    ss << "  PyBind Version: v14 + Manager API\n";
    ss << "  CUDA Support: " << (torch::cuda::is_available() ? "YES" : "NO") << "\n";
    ss << "  CUDA Device Count: " << torch::cuda::device_count() << "\n";
    
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    ss << "  Current Device: " << device << "\n";
    
    if (err == cudaSuccess) {
        cudaDeviceProp prop;
        // --- FIX: PDF had a typo 'a' here, corrected to 'cudaSuccess' ---
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            ss << "  Device Name: " << prop.name << "\n";
        } else {
            ss << "  Device Name: [cudaGetDeviceProperties error]\n";
        }
    } else {
        ss << "  Device Name: [cudaGetDevice error]\n";
    }
    
    return ss.str();
}


// ===================================================================
// 🔥 START: OPTIONAL HELPER FUNCTIONS (from PDF)
// ===================================================================

// === COMPRESSION LEVEL INFO ===

int get_zstd_min_level() {
    // Zstandard minimum compression level (negative levels are for speed)
    return -5; 
}

int get_zstd_max_level() {
    // Zstandard maximum "ultra" compression level
    return 22; 
}

int get_zstd_default_level() {
    // A balanced default (as suggested in PDF)
    return 6; 
}

std::string get_compression_level_info(int current_level) {
    std::stringstream ss;
    ss << "nvCOMP_12 Zstandard Compression Configuration:\n";
    ss << "  Current Level:   " << current_level << "\n";
    ss << "  Minimum Level:   " << get_zstd_min_level() << " (fastest)\n";
    ss << "  Maximum Level:   " << get_zstd_max_level() << " (best compression)\n";
    ss << "  Default Level:   " << get_zstd_default_level() << " (balanced)\n\n";

    if (current_level < 0) {
        ss << "  Mode: fast (prioritizes speed over compression ratio)\n";
    } else if (current_level <= 9) {
        ss << "  Mode: balanced (standard levels)\n";
    } else {
        ss << "  Mode: maximum (high compression, slower, high memory use)\n";
    }
    return ss.str();
}

// ===================================================================
// 🔥 END: OPTIONAL HELPER FUNCTIONS
// ===================================================================


// === PYBIND MODULE ===

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "JIT GPU Kernel with nvCOMP 5.0 Manager API (Zstd)";
    
    // === SPARSE ===
    m.def("jit_apply_v1_full_gpu", &jit_apply_v1_full_gpu,
          "JIT Full-GPU Kernel (SPARSE scatter-add)",
          pybind11::arg("W_base"),
          pybind11::arg("comp_indices"),
          pybind11::arg("comp_values"),
          pybind11::arg("chunk_map"),
          pybind11::arg("decomp_metadata"));
    
    m.def("apply_full_gpu", &jit_apply_v1_full_gpu,
          "Alias for jit_apply_v1_full_gpu (compatibility)",
          pybind11::arg("W_base"),
          pybind11::arg("comp_indices"),
          pybind11::arg("comp_values"),
          pybind11::arg("chunk_map"),
          pybind11::arg("decomp_metadata"));

    // --- FIX: Added EALE kernel binding ---
    m.def("eale_reconstruct_launcher", &eale_lossless_reconstruct_launcher,
          "EALE 100% Lossless Reconstruction Kernel (W_Base + W_Residual)",
          pybind11::arg("W_Base_Packed"),
          pybind11::arg("W_Residual"),
          pybind11::arg("W_Recon_Output"),
          pybind11::arg("scale_factor"));
    
    // === DENSE ZSTD ===
    // --- FIX: Added zstd_level argument ---
    m.def("jit_compress_zstd_v1", &jit_compress_zstd_v1,
          "JIT Full-GPU Zstd Compressor (DENSE)",
          pybind11::arg("delta_gpu"),
          pybind11::arg("zstd_level"));
    
    m.def("jit_decompress_zstd_v1", &jit_decompress_zstd_v1,
          "JIT Full-GPU Zstd Decompressor (DENSE)",
          pybind11::arg("compressed_gpu"),
          pybind11::arg("uncompressed_size_bytes"));
    
    // === UTILITIES ===
    m.def("validate_format", &validate_nvcomp_format,
          "Basic format validation for compressed data",
          pybind11::arg("comp_data"));
    
    m.def("get_version", &get_backend_version,
          "Get backend version string");
    
    m.def("get_diagnostics", &get_system_diagnostics,
          "Get system diagnostics information");
          
    // --- ADDED THIS ---
    m.def("format_bytes", &format_bytes_cpp,
          "Format bytes to a human-readable string (KB, MB, GB)",
          pybind11::arg("size_in_bytes"));

    // ===================================================================
    // 🔥 START: BINDINGS FOR OPTIONAL HELPER FUNCTIONS
    // ===================================================================
    m.def("get_zstd_min_level", &get_zstd_min_level,
          "Get minimum Zstandard compression level");

    m.def("get_zstd_max_level", &get_zstd_max_level,
          "Get maximum Zstandard compression level");

    m.def("get_zstd_default_level", &get_zstd_default_level,
          "Get default Zstandard compression level");

    m.def("get_compression_level_info", &get_compression_level_info,
          "Get detailed compression level information string",
          pybind11::arg("current_level"));
    // ===================================================================
    // 🔥 END: BINDINGS
    // ===================================================================
}