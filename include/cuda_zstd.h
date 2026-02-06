/**
 * @file cuda_zstd.h
 * @brief Umbrella header for the CUDA-ZSTD GPU compression library.
 *
 * Including this single header provides access to the complete public API:
 *   - Types, status codes, and error handling (cuda_zstd_types.h)
 *   - Manager interface for compress/decompress (cuda_zstd_manager.h)
 *   - FSE (Finite State Entropy) table operations (cuda_zstd_fse.h)
 *   - Huffman coding support (cuda_zstd_huffman.h)
 *   - Dictionary compression (cuda_zstd_dictionary.h)
 *   - CUDA stream pool for async operations (cuda_zstd_stream_pool.h)
 *   - RAII device memory wrapper (cuda_zstd_cuda_ptr.h)
 *
 * For advanced/internal usage (sequence coding, LZ77, hashing, etc.),
 * include the specific headers directly.
 */
#ifndef CUDA_ZSTD_H_
#define CUDA_ZSTD_H_

// Core types, status codes, and error utilities
#include "cuda_zstd_types.h"

// Manager: the primary compress/decompress API
#include "cuda_zstd_manager.h"

// FSE table construction and decoding
#include "cuda_zstd_fse.h"

// Huffman coding
#include "cuda_zstd_huffman.h"

// Dictionary support
#include "cuda_zstd_dictionary.h"

// CUDA stream pool for multi-stream async I/O
#include "cuda_zstd_stream_pool.h"

// RAII wrapper for device memory
#include "cuda_zstd_cuda_ptr.h"

#endif // CUDA_ZSTD_H_
