// ==============================================================================
// common_types.h - Common types and definitions
// ==============================================================================

#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <cstdint>
#include <cstddef>

namespace compression {

// Type aliases
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;
using byte_t = unsigned char;

// Status codes
enum class Status : u32 {
    SUCCESS = 0,
    ERROR_GENERIC = 1,
    ERROR_INVALID_PARAMETER = 2,
    ERROR_OUT_OF_MEMORY = 3,
    ERROR_CUDA_ERROR = 4,
    ERROR_ALLOCATION_FAILED = 5,
    ERROR_TIMEOUT = 6,
    ERROR_NOT_INITIALIZED = 7,
    ERROR_INVALID_STATE = 8,
    ERROR_BUFFER_TOO_SMALL = 9,
    ERROR_CORRUPT_DATA = 10
};

// Convert status to string
inline const char* status_to_string(Status status) {
    switch (status) {
        case Status::SUCCESS: return "SUCCESS";
        case Status::ERROR_GENERIC: return "ERROR_GENERIC";
        case Status::ERROR_INVALID_PARAMETER: return "ERROR_INVALID_PARAMETER";
        case Status::ERROR_OUT_OF_MEMORY: return "ERROR_OUT_OF_MEMORY";
        case Status::ERROR_CUDA_ERROR: return "ERROR_CUDA_ERROR";
        case Status::ERROR_ALLOCATION_FAILED: return "ERROR_ALLOCATION_FAILED";
        case Status::ERROR_TIMEOUT: return "ERROR_TIMEOUT";
        case Status::ERROR_NOT_INITIALIZED: return "ERROR_NOT_INITIALIZED";
        case Status::ERROR_INVALID_STATE: return "ERROR_INVALID_STATE";
        case Status::ERROR_BUFFER_TOO_SMALL: return "ERROR_BUFFER_TOO_SMALL";
        case Status::ERROR_CORRUPT_DATA: return "ERROR_CORRUPT_DATA";
        default: return "UNKNOWN_ERROR";
    }
}

} // namespace compression

#endif // COMMON_TYPES_H
