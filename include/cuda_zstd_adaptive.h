// ============================================================================
// cuda_zstd_adaptive.h - Adaptive Compression Level Selection
// ============================================================================

#ifndef CUDA_ZSTD_ADAPTIVE_H_
#define CUDA_ZSTD_ADAPTIVE_H_

#include "cuda_zstd_types.h"
#include <cuda_runtime.h>

namespace cuda_zstd {
namespace adaptive {

// ============================================================================
// Adaptive Level Selection Configuration
// ============================================================================

enum class AdaptivePreference {
    SPEED = 0,      // Prioritize compression speed
    BALANCED = 1,   // Balance speed and ratio
    RATIO = 2       // Prioritize compression ratio
};

struct DataCharacteristics {
    float entropy = 0.0f;              // Shannon entropy (0-8 bits)
    float repetition_ratio = 0.0f;     // Ratio of repeated bytes (0-1)
    float compressibility = 0.0f;      // Estimated compressibility (0-1)
    uint32_t unique_bytes = 0;         // Number of unique byte values
    uint32_t max_run_length = 0;       // Longest run of repeated bytes
    uint32_t avg_match_length = 0;     // Average LZ77 match length
    bool has_patterns = false;         // Contains repetitive patterns
    bool is_random = false;            // Appears random (high entropy)
};

struct AdaptiveResult {
    int recommended_level = 3;                    // Recommended compression level (1-22)
    Strategy recommended_strategy = Strategy::GREEDY;
    DataCharacteristics characteristics;
    float confidence = 0.0f;                      // Confidence in recommendation (0-1)
    const char* reasoning = nullptr;              // Human-readable reasoning
};

// ============================================================================
// Adaptive Level Selector
// ============================================================================

class AdaptiveLevelSelector {
public:
    AdaptiveLevelSelector();
    explicit AdaptiveLevelSelector(AdaptivePreference pref);
    ~AdaptiveLevelSelector();
    
    // Main selection interface
    Status select_level(
        const void* d_data,
        size_t data_size,
        AdaptiveResult& result,
        cudaStream_t stream = 0
    );
    
    // Configuration
    void set_preference(AdaptivePreference pref);
    void set_sample_size(size_t size);  // How much data to sample
    void enable_quick_analysis(bool enable);  // Fast mode vs thorough
    
    // Statistics
    const DataCharacteristics& get_last_characteristics() const;
    void reset_statistics();
    
private:
    AdaptivePreference preference_ = AdaptivePreference::BALANCED;
    size_t sample_size_ = 64 * 1024;  // Sample first 64KB by default
    bool quick_mode_ = false;
    DataCharacteristics last_characteristics_;
    
    // Analysis kernels
    Status analyze_entropy(const byte_t* d_data, size_t size, float& entropy, cudaStream_t stream);
    Status analyze_repetition(const byte_t* d_data, size_t size, float& ratio, cudaStream_t stream);
    Status analyze_compressibility(const byte_t* d_data, size_t size, float& comp, cudaStream_t stream);
    Status analyze_patterns(const byte_t* d_data, size_t size, DataCharacteristics& chars, cudaStream_t stream);
    
    // Decision logic
    int select_level_from_characteristics(const DataCharacteristics& chars, AdaptivePreference pref);
    Strategy select_strategy_from_level(int level);
    const char* generate_reasoning(const DataCharacteristics& chars, int level);
};

// ============================================================================
// Standalone Functions
// ============================================================================

// Quick analysis and level selection (single function)
Status select_adaptive_level(
    const void* d_data,
    size_t data_size,
    int& recommended_level,
    AdaptivePreference preference = AdaptivePreference::BALANCED,
    cudaStream_t stream = 0
);

// Detailed analysis with full results
Status analyze_and_select(
    const void* d_data,
    size_t data_size,
    AdaptiveResult& result,
    AdaptivePreference preference = AdaptivePreference::BALANCED,
    cudaStream_t stream = 0
);

// Check if data is compressible (quick test)
Status is_compressible(
    const void* d_data,
    size_t data_size,
    bool& compressible,
    float& estimated_ratio,
    cudaStream_t stream = 0
);

} // namespace adaptive
} // namespace cuda_zstd

#endif // CUDA_ZSTD_ADAPTIVE_H