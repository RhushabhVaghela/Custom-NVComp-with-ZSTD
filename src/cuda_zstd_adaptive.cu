// ============================================================================
// cuda_zstd_adaptive.cu - Adaptive Compression Level Implementation
// ============================================================================

#include "cuda_zstd_adaptive.h"
#include "cuda_zstd_utils.h"
#include <cmath>
#include <algorithm>

namespace cuda_zstd {
namespace adaptive {

// ============================================================================
// Analysis Kernels
// ============================================================================

__global__ void analyze_entropy_kernel(
    const byte_t* data,
    uint32_t size,
    uint32_t* d_frequencies
) {
    __shared__ uint32_t s_freq[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    // Initialize shared memory
    if (tid < 256) {
        s_freq[tid] = 0;
    }
    __syncthreads();
    
    // Count frequencies in shared memory
    for (uint32_t i = idx; i < size; i += stride) {
        atomicAdd(&s_freq[data[i]], 1);
    }
    __syncthreads();
    
    // Reduce to global memory
    if (tid < 256) {
        if (s_freq[tid] > 0) {
            atomicAdd(&d_frequencies[tid], s_freq[tid]);
        }
    }
}

__global__ void analyze_repetition_kernel(
    const byte_t* data,
    uint32_t size,
    uint32_t* d_repeated_count,
    uint32_t* d_max_run
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    for (uint32_t i = idx; i < size - 1; i += stride) {
        if (data[i] == data[i + 1]) {
            atomicAdd(d_repeated_count, 1);
            
            // Count run length
            uint32_t run = 1;
            uint32_t j = i;
            while (j + 1 < size && data[j] == data[j + 1] && run < 256) {
                run++;
                j++;
            }
            atomicMax(d_max_run, run);
        }
    }
}

__global__ void analyze_patterns_kernel(
    const byte_t* data,
    uint32_t size,
    uint32_t* d_pattern_score
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    // Look for simple patterns (2-byte, 4-byte repeats)
    for (uint32_t i = idx; i < size - 4; i += stride) {
        // Check 2-byte pattern
        if (data[i] == data[i + 2] && data[i + 1] == data[i + 3]) {
            atomicAdd(d_pattern_score, 1);
        }
        
        // Check 4-byte pattern
        if (i + 8 < size) {
            bool match = true;
            for (int j = 0; j < 4; j++) {
                if (data[i + j] != data[i + 4 + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                atomicAdd(d_pattern_score, 2);
            }
        }
    }
}

// ============================================================================
// AdaptiveLevelSelector Implementation
// ============================================================================

AdaptiveLevelSelector::AdaptiveLevelSelector()
    : preference_(AdaptivePreference::BALANCED) {
}

AdaptiveLevelSelector::AdaptiveLevelSelector(AdaptivePreference pref)
    : preference_(pref) {
}

AdaptiveLevelSelector::~AdaptiveLevelSelector() {
}

Status AdaptiveLevelSelector::analyze_entropy(
    const byte_t* d_data,
    size_t size,
    float& entropy,
    cudaStream_t stream
) {
    uint32_t* d_frequencies = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frequencies, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_frequencies, 0, 256 * sizeof(uint32_t), stream));
    
    uint32_t threads = 256;
    uint32_t blocks = std::min(static_cast<uint32_t>((size + threads - 1) / threads), 1024u);
    
    analyze_entropy_kernel<<<blocks, threads, 0, stream>>>(
        d_data, size, d_frequencies
    );
    
    // Copy frequencies to host
    uint32_t h_frequencies[256];
    // Copy frequencies to host - use synchronous copy to regular host memory.
    // cudaMemcpyAsync requires pinned host memory for device->host copies; the
    // tests and callers allocate a stack buffer (non-pinned). Using cudaMemcpy
    // avoids undefined behavior on some driver/host configurations.
    CUDA_CHECK(cudaMemcpy(h_frequencies, d_frequencies, 256 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_frequencies));
    
    // Calculate Shannon entropy
    entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (h_frequencies[i] > 0) {
            float prob = (float)h_frequencies[i] / size;
            entropy -= prob * log2f(prob);
        }
    }
    
    return Status::SUCCESS;
}

Status AdaptiveLevelSelector::analyze_repetition(
    const byte_t* d_data,
    size_t size,
    float& ratio,
    cudaStream_t stream
) {
    uint32_t* d_repeated_count = nullptr;
    uint32_t* d_max_run = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_repeated_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_max_run, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_repeated_count, 0, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_max_run, 0, sizeof(uint32_t), stream));
    
    uint32_t threads = 256;
    uint32_t blocks = std::min(static_cast<uint32_t>((size + threads - 1) / threads), 1024u);
    
    analyze_repetition_kernel<<<blocks, threads, 0, stream>>>(
        d_data, size, d_repeated_count, d_max_run
    );
    
    uint32_t h_repeated_count = 0;
    // Use synchronous copies for host-local variables; Async would require
    // pinned host memory to be safe.
    CUDA_CHECK(cudaMemcpy(&h_repeated_count, d_repeated_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_characteristics_.max_run_length, d_max_run, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_repeated_count));
    CUDA_CHECK(cudaFree(d_max_run));
    
    ratio = (float)h_repeated_count / size;
    return Status::SUCCESS;
}

Status AdaptiveLevelSelector::analyze_compressibility(
    const byte_t* d_data,
    size_t size,
    float& comp,
    cudaStream_t stream
) {
    // Estimate compressibility based on entropy and repetition
    // compressibility = 1.0 means highly compressible, 0.0 means incompressible
    
    float max_entropy = 8.0f;  // Maximum possible entropy for byte data
    float entropy_factor = 1.0f - (last_characteristics_.entropy / max_entropy);
    float repetition_factor = last_characteristics_.repetition_ratio;
    
    // Combine factors
    comp = (entropy_factor * 0.6f) + (repetition_factor * 0.4f);
    comp = std::max(0.0f, std::min(1.0f, comp));
    
    return Status::SUCCESS;
}

Status AdaptiveLevelSelector::analyze_patterns(
    const byte_t* d_data,
    size_t size,
    DataCharacteristics& chars,
    cudaStream_t stream
) {
    uint32_t* d_pattern_score = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pattern_score, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_pattern_score, 0, sizeof(uint32_t), stream));
    
    uint32_t threads = 256;
    uint32_t blocks = std::min(static_cast<uint32_t>((size + threads - 1) / threads), 1024u);
    
    analyze_patterns_kernel<<<blocks, threads, 0, stream>>>(
        d_data, size, d_pattern_score
    );
    
    uint32_t h_pattern_score = 0;
    CUDA_CHECK(cudaMemcpy(&h_pattern_score, d_pattern_score, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_pattern_score));
    
    chars.has_patterns = (h_pattern_score > size / 100);  // More than 1% patterns
    chars.is_random = (chars.entropy > 7.5f && !chars.has_patterns);
    
    return Status::SUCCESS;
}

int AdaptiveLevelSelector::select_level_from_characteristics(
    const DataCharacteristics& chars,
    AdaptivePreference pref
) {
    // Decision tree based on data characteristics and user preference
    
    // Random data: use low level (fast)
    if (chars.is_random) {
        return (pref == AdaptivePreference::SPEED) ? 1 : 3;
    }
    
    // Highly repetitive: can use higher levels efficiently
    if (chars.repetition_ratio > 0.5f || chars.max_run_length > 100) {
        if (pref == AdaptivePreference::SPEED) return 6;
        if (pref == AdaptivePreference::BALANCED) return 12;
        return 19;  // RATIO preference
    }
    
    // High compressibility: worth using good compression
    if (chars.compressibility > 0.7f) {
        if (pref == AdaptivePreference::SPEED) return 5;
        if (pref == AdaptivePreference::BALANCED) return 9;
        return 16;
    }
    
    // Medium compressibility
    if (chars.compressibility > 0.4f) {
        if (pref == AdaptivePreference::SPEED) return 3;
        if (pref == AdaptivePreference::BALANCED) return 6;
        return 12;
    }
    
    // Low compressibility: use fast compression
    if (pref == AdaptivePreference::SPEED) return 1;
    if (pref == AdaptivePreference::BALANCED) return 3;
    return 6;  // Even for RATIO, don't waste time
}

Strategy AdaptiveLevelSelector::select_strategy_from_level(int level) {
    return CompressionConfig::level_to_strategy(level);
}

const char* AdaptiveLevelSelector::generate_reasoning(
    const DataCharacteristics& chars,
    int level
) {
    if (chars.is_random) {
        return "Data appears random (high entropy), using fast compression";
    }
    if (chars.repetition_ratio > 0.5f) {
        return "Highly repetitive data detected, using high compression level";
    }
    if (chars.compressibility > 0.7f) {
        return "Highly compressible data, good compression ratio expected";
    }
    if (chars.compressibility > 0.4f) {
        return "Moderately compressible data, balanced level selected";
    }
    return "Low compressibility, prioritizing speed";
}

Status AdaptiveLevelSelector::select_level(
    const void* d_data,
    size_t data_size,
    AdaptiveResult& result,
    cudaStream_t stream
) {
    if (!d_data || data_size == 0) {
        return Status::ERROR_INVALID_PARAMETER;
    }
    
    // Determine sample size
    size_t actual_sample = std::min(sample_size_, data_size);
    const byte_t* d_sample = static_cast<const byte_t*>(d_data);
    
    // Reset characteristics
    last_characteristics_ = DataCharacteristics();
    
    // Analyze entropy
    Status status = analyze_entropy(d_sample, actual_sample, 
                                    last_characteristics_.entropy, stream);
    if (status != Status::SUCCESS) return status;
    
    // Analyze repetition
    status = analyze_repetition(d_sample, actual_sample,
                               last_characteristics_.repetition_ratio, stream);
    if (status != Status::SUCCESS) return status;
    
    // Analyze compressibility
    status = analyze_compressibility(d_sample, actual_sample,
                                     last_characteristics_.compressibility, stream);
    if (status != Status::SUCCESS) return status;
    
    // Analyze patterns
    status = analyze_patterns(d_sample, actual_sample,
                             last_characteristics_, stream);
    if (status != Status::SUCCESS) return status;
    
    // Select level based on characteristics
    int level = select_level_from_characteristics(last_characteristics_, preference_);
    
    // Fill result
    result.recommended_level = level;
    result.recommended_strategy = select_strategy_from_level(level);
    result.characteristics = last_characteristics_;
    result.confidence = 0.85f;  // High confidence with full analysis
    result.reasoning = generate_reasoning(last_characteristics_, level);
    
    return Status::SUCCESS;
}

void AdaptiveLevelSelector::set_preference(AdaptivePreference pref) {
    preference_ = pref;
}

void AdaptiveLevelSelector::set_sample_size(size_t size) {
    sample_size_ = size;
}

void AdaptiveLevelSelector::enable_quick_analysis(bool enable) {
    quick_mode_ = enable;
}

const DataCharacteristics& AdaptiveLevelSelector::get_last_characteristics() const {
    return last_characteristics_;
}

void AdaptiveLevelSelector::reset_statistics() {
    last_characteristics_ = DataCharacteristics();
}

// ============================================================================
// Standalone Functions
// ============================================================================

Status select_adaptive_level(
    const void* d_data,
    size_t data_size,
    int& recommended_level,
    AdaptivePreference preference,
    cudaStream_t stream
) {
    AdaptiveLevelSelector selector(preference);
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result, stream);
    if (status == Status::SUCCESS) {
        recommended_level = result.recommended_level;
    }
    
    return status;
}

Status analyze_and_select(
    const void* d_data,
    size_t data_size,
    AdaptiveResult& result,
    AdaptivePreference preference,
    cudaStream_t stream
) {
    AdaptiveLevelSelector selector(preference);
    return selector.select_level(d_data, data_size, result, stream);
}

Status is_compressible(
    const void* d_data,
    size_t data_size,
    bool& compressible,
    float& estimated_ratio,
    cudaStream_t stream
) {
    AdaptiveLevelSelector selector;
    AdaptiveResult result;
    
    Status status = selector.select_level(d_data, data_size, result, stream);
    if (status != Status::SUCCESS) {
        return status;
    }
    
    compressible = (result.characteristics.compressibility > 0.3f);
    estimated_ratio = 1.0f + (result.characteristics.compressibility * 3.0f);
    
    return Status::SUCCESS;
}

} // namespace adaptive
} // namespace cuda_zstd