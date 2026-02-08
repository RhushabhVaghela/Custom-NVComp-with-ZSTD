// ============================================================================
// binding.cpp — pybind11 bindings for the cuda_zstd GPU compression library
//
// This file exposes the C++ API to Python as the `cuda_zstd._core` extension
// module.  The high-level Python API in `cuda_zstd/__init__.py` wraps these
// raw bindings with ergonomic helpers (bytes in → bytes out, NumPy/CuPy, etc).
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <zstd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Parent library headers
#include "cuda_zstd.h"
#include "cuda_zstd_hybrid.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_safe_alloc.h"
#include "cuda_zstd_types.h"

namespace py = pybind11;

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Throw a Python RuntimeError from a cuda_zstd::Status code.
[[noreturn]] void throw_status(cuda_zstd::Status st, const char *context = "") {
    const char *msg = cuda_zstd::status_to_string(st);
    throw std::runtime_error(
        std::string(context) + (context[0] ? ": " : "") + (msg ? msg : "unknown error"));
}

/// Check a Status and throw on failure.
inline void check_status(cuda_zstd::Status st, const char *context = "") {
    if (st != cuda_zstd::Status::SUCCESS) {
        throw_status(st, context);
    }
}

/// RAII wrapper for cudaMalloc / cudaFree.
struct DeviceBuffer {
    void *ptr = nullptr;
    size_t size = 0;

    explicit DeviceBuffer(size_t n) : size(n) {
        if (n == 0) return;
        cudaError_t err = cuda_zstd::safe_cuda_malloc(&ptr, n);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("safe_cuda_malloc failed: ") + cudaGetErrorString(err));
        }
    }
    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr;
        o.size = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr;
            size = o.size;
            o.ptr = nullptr;
            o.size = 0;
        }
        return *this;
    }
};

/// Copy host → device.
inline void h2d(void *dst, const void *src, size_t n) {
    cudaError_t err = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") +
                                 cudaGetErrorString(err));
}

/// Copy device → host.
inline void d2h(void *dst, const void *src, size_t n) {
    cudaError_t err = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") +
                                 cudaGetErrorString(err));
}

/// Extract a contiguous buffer view from any Python buffer-protocol object.
struct BufferInfo {
    const uint8_t *data;
    size_t size;
    py::buffer_info info;

    explicit BufferInfo(py::buffer buf) : info(buf.request()) {
        if (info.ndim != 1 && info.ndim != 0) {
            // Allow multi-dim arrays — just treat as flat bytes.
        }
        data = static_cast<const uint8_t *>(info.ptr);
        size = static_cast<size_t>(info.size * info.itemsize);
    }
};

/// Try to get a CuPy device pointer (returns nullptr if not CuPy).
/// This avoids an unnecessary H2D copy when the data is already on the GPU.
struct CuPyInfo {
    void *device_ptr = nullptr;
    size_t nbytes = 0;
    bool is_cupy = false;
};

CuPyInfo try_get_cupy_ptr(py::object obj) {
    CuPyInfo info;
    try {
        // Check for CuPy ndarray by looking for __cuda_array_interface__
        if (py::hasattr(obj, "__cuda_array_interface__")) {
            py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
            py::tuple data_tuple = iface["data"].cast<py::tuple>();
            info.device_ptr = reinterpret_cast<void *>(data_tuple[0].cast<uintptr_t>());
            info.nbytes = obj.attr("nbytes").cast<size_t>();
            info.is_cupy = true;
        }
    } catch (...) {
        info.is_cupy = false;
    }
    return info;
}

} // anonymous namespace

// ============================================================================
// Python wrapper for ZstdBatchManager
// ============================================================================

/// Thin wrapper that owns a ZstdBatchManager and handles GPU memory
/// transparently so the Python user never needs to touch device pointers.
class PyManager {
public:
    /// Construct with compression level and optional config overrides.
    explicit PyManager(int level = 3) {
        cuda_zstd::CompressionConfig cfg =
            cuda_zstd::CompressionConfig::from_level(level);
        mgr_ = cuda_zstd::create_batch_manager(level);
        if (!mgr_) throw std::runtime_error("Failed to create ZstdBatchManager");
        check_status(mgr_->configure(cfg), "configure");
    }

    /// Construct from a full CompressionConfig.
    explicit PyManager(const cuda_zstd::CompressionConfig &cfg) {
        mgr_ = cuda_zstd::create_batch_manager(cfg.level);
        if (!mgr_) throw std::runtime_error("Failed to create ZstdBatchManager");
        check_status(mgr_->configure(cfg), "configure");
    }

    // ------------------------------------------------------------------
    // Single-buffer compress
    // ------------------------------------------------------------------

    /// Compress host data (bytes / bytearray / numpy array) → bytes.
    py::bytes compress(py::object input) {
        // Check for CuPy (already on GPU)
        CuPyInfo cupy = try_get_cupy_ptr(input);

        const uint8_t *h_src = nullptr;
        size_t src_size = 0;
        DeviceBuffer d_src_buf(0);

        if (cupy.is_cupy) {
            // Data is already on GPU — use device pointer directly.
            src_size = cupy.nbytes;
        } else {
            // Extract host buffer via Python buffer protocol.
            BufferInfo buf(input.cast<py::buffer>());
            h_src = buf.data;
            src_size = buf.size;
            d_src_buf = DeviceBuffer(src_size);
            h2d(d_src_buf.ptr, h_src, src_size);
        }

        void *d_src = cupy.is_cupy ? cupy.device_ptr : d_src_buf.ptr;

        // Allocate output on GPU (worst-case bound).
        size_t max_dst = mgr_->get_max_compressed_size(src_size);
        DeviceBuffer d_dst(max_dst);

        // Workspace.
        size_t ws_size = mgr_->get_compress_temp_size(src_size);
        DeviceBuffer d_ws(ws_size);

        // Compress.
        size_t dst_size = max_dst;
        check_status(
            mgr_->compress(d_src, src_size, d_dst.ptr, &dst_size,
                           d_ws.ptr, ws_size, nullptr, 0, /*stream=*/0),
            "compress");

        // Copy result back to host.
        std::string result(dst_size, '\0');
        d2h(result.data(), d_dst.ptr, dst_size);
        return py::bytes(result);
    }

    // ------------------------------------------------------------------
    // Single-buffer decompress
    // ------------------------------------------------------------------

    /// Decompress host data → bytes.
    py::bytes decompress(py::object input) {
        CuPyInfo cupy = try_get_cupy_ptr(input);

        const uint8_t *h_src = nullptr;
        size_t src_size = 0;
        DeviceBuffer d_src_buf(0);

        if (cupy.is_cupy) {
            src_size = cupy.nbytes;
        } else {
            BufferInfo buf(input.cast<py::buffer>());
            h_src = buf.data;
            src_size = buf.size;
            d_src_buf = DeviceBuffer(src_size);
            h2d(d_src_buf.ptr, h_src, src_size);
        }

        void *d_src = cupy.is_cupy ? cupy.device_ptr : d_src_buf.ptr;

        // Query decompressed size from the compressed header.
        size_t decompressed_size = 0;
        check_status(
            cuda_zstd::get_decompressed_size(d_src, src_size, &decompressed_size),
            "get_decompressed_size");

        if (decompressed_size == 0) {
            // Fallback: use a generous estimate.
            decompressed_size = src_size * 16;
        }

        DeviceBuffer d_dst(decompressed_size);
        size_t ws_size = mgr_->get_decompress_temp_size(src_size);
        DeviceBuffer d_ws(ws_size);

        size_t out_size = decompressed_size;
        check_status(
            mgr_->decompress(d_src, src_size, d_dst.ptr, &out_size,
                             d_ws.ptr, ws_size, /*stream=*/0),
            "decompress");

        std::string result(out_size, '\0');
        d2h(result.data(), d_dst.ptr, out_size);
        return py::bytes(result);
    }

    // ------------------------------------------------------------------
    // Batch compress
    // ------------------------------------------------------------------

    /// Compress a list of buffers in one GPU launch → list[bytes].
    std::vector<py::bytes> compress_batch(std::vector<py::buffer> inputs) {
        const size_t n = inputs.size();
        if (n == 0) return {};

        // Parse inputs on host.
        std::vector<BufferInfo> bufs;
        bufs.reserve(n);
        for (auto &b : inputs) bufs.emplace_back(b);

        // Allocate device memory for all inputs.
        std::vector<DeviceBuffer> d_srcs;
        d_srcs.reserve(n);
        for (auto &b : bufs) {
            d_srcs.emplace_back(b.size);
            h2d(d_srcs.back().ptr, b.data, b.size);
        }

        // Allocate device memory for all outputs.
        std::vector<size_t> max_sizes(n);
        for (size_t i = 0; i < n; ++i)
            max_sizes[i] = mgr_->get_max_compressed_size(bufs[i].size);

        std::vector<DeviceBuffer> d_dsts;
        d_dsts.reserve(n);
        for (size_t i = 0; i < n; ++i)
            d_dsts.emplace_back(max_sizes[i]);

        // Build BatchItem array.
        std::vector<cuda_zstd::BatchItem> items(n);
        for (size_t i = 0; i < n; ++i) {
            items[i].input_ptr = d_srcs[i].ptr;
            items[i].output_ptr = d_dsts[i].ptr;
            items[i].input_size = bufs[i].size;
            items[i].output_size = max_sizes[i];
        }

        // Workspace.
        std::vector<size_t> uncompressed_sizes(n);
        for (size_t i = 0; i < n; ++i)
            uncompressed_sizes[i] = bufs[i].size;
        size_t ws_size = mgr_->get_batch_compress_temp_size(uncompressed_sizes);
        DeviceBuffer d_ws(ws_size);

        check_status(
            mgr_->compress_batch(items, d_ws.ptr, ws_size, /*stream=*/0),
            "compress_batch");

        // Copy results back.
        std::vector<py::bytes> results;
        results.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            check_status(items[i].status, "compress_batch item");
            std::string buf(items[i].output_size, '\0');
            d2h(buf.data(), d_dsts[i].ptr, items[i].output_size);
            results.push_back(py::bytes(buf));
        }
        return results;
    }

    // ------------------------------------------------------------------
    // Batch decompress
    // ------------------------------------------------------------------

    std::vector<py::bytes> decompress_batch(std::vector<py::buffer> inputs) {
        const size_t n = inputs.size();
        if (n == 0) return {};

        std::vector<BufferInfo> bufs;
        bufs.reserve(n);
        for (auto &b : inputs) bufs.emplace_back(b);

        std::vector<DeviceBuffer> d_srcs;
        d_srcs.reserve(n);
        for (auto &b : bufs) {
            d_srcs.emplace_back(b.size);
            h2d(d_srcs.back().ptr, b.data, b.size);
        }

        // Query decompressed sizes.
        std::vector<size_t> dec_sizes(n);
        for (size_t i = 0; i < n; ++i) {
            cuda_zstd::Status st = cuda_zstd::get_decompressed_size(
                d_srcs[i].ptr, bufs[i].size, &dec_sizes[i]);
            if (st != cuda_zstd::Status::SUCCESS || dec_sizes[i] == 0)
                dec_sizes[i] = bufs[i].size * 16; // fallback
        }

        std::vector<DeviceBuffer> d_dsts;
        d_dsts.reserve(n);
        for (size_t i = 0; i < n; ++i)
            d_dsts.emplace_back(dec_sizes[i]);

        std::vector<cuda_zstd::BatchItem> items(n);
        for (size_t i = 0; i < n; ++i) {
            items[i].input_ptr = d_srcs[i].ptr;
            items[i].output_ptr = d_dsts[i].ptr;
            items[i].input_size = bufs[i].size;
            items[i].output_size = dec_sizes[i];
        }

        std::vector<size_t> compressed_sizes(n);
        for (size_t i = 0; i < n; ++i)
            compressed_sizes[i] = bufs[i].size;
        size_t ws_size = mgr_->get_batch_decompress_temp_size(compressed_sizes);
        DeviceBuffer d_ws(ws_size);

        check_status(
            mgr_->decompress_batch(items, d_ws.ptr, ws_size, /*stream=*/0),
            "decompress_batch");

        std::vector<py::bytes> results;
        results.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            check_status(items[i].status, "decompress_batch item");
            std::string buf(items[i].output_size, '\0');
            d2h(buf.data(), d_dsts[i].ptr, items[i].output_size);
            results.push_back(py::bytes(buf));
        }
        return results;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    int get_level() const { return mgr_->get_compression_level(); }

    void set_level(int level) {
        check_status(mgr_->set_compression_level(level), "set_compression_level");
    }

    cuda_zstd::CompressionStats get_stats() const { return mgr_->get_stats(); }

    void reset_stats() { mgr_->reset_stats(); }

    cuda_zstd::CompressionConfig get_config() const { return mgr_->get_config(); }

private:
    std::unique_ptr<cuda_zstd::ZstdBatchManager> mgr_;
};

// ============================================================================
// Python wrapper for HybridEngine
// ============================================================================

/// Thin wrapper that owns a HybridEngine and handles HOST→HOST compression
/// transparently so the Python user just passes bytes in and gets bytes out.
class PyHybridEngine {
public:
    /// Construct with compression level (default 3).
    explicit PyHybridEngine(int level = 3) {
        cuda_zstd::HybridConfig cfg;
        cfg.compression_level = level;
        engine_ = std::make_unique<cuda_zstd::HybridEngine>(cfg);
    }

    /// Construct from a HybridConfig.
    explicit PyHybridEngine(const cuda_zstd::HybridConfig &cfg) {
        engine_ = std::make_unique<cuda_zstd::HybridEngine>(cfg);
    }

    // ------------------------------------------------------------------
    // Compress: bytes in → bytes out  (HOST→HOST, auto-routed)
    // ------------------------------------------------------------------

    py::bytes compress(py::buffer input) {
        BufferInfo buf(input);
        if (buf.size == 0) return py::bytes("", 0);

        size_t max_out = engine_->get_max_compressed_size(buf.size);
        std::string output(max_out, '\0');
        size_t out_size = max_out;

        cuda_zstd::Status st = engine_->compress(
            buf.data, buf.size,
            output.data(), &out_size,
            cuda_zstd::DataLocation::HOST,
            cuda_zstd::DataLocation::HOST);
        check_status(st, "hybrid compress");

        output.resize(out_size);
        return py::bytes(output);
    }

    // ------------------------------------------------------------------
    // Decompress: bytes in → bytes out  (HOST→HOST, auto-routed)
    // ------------------------------------------------------------------

    py::bytes decompress(py::buffer input) {
        BufferInfo buf(input);
        if (buf.size == 0) return py::bytes("", 0);

        // Use ZSTD_getFrameContentSize for exact decompressed size when
        // available.  This avoids allocation guessing and fixes small-data
        // decompression where the hybrid engine's cpu_decompress path
        // receives a non-zero output_size and skips its own frame-header
        // check.
        size_t out_size;
        unsigned long long frame_size =
            ZSTD_getFrameContentSize(buf.data, buf.size);
        if (frame_size != ZSTD_CONTENTSIZE_UNKNOWN &&
            frame_size != ZSTD_CONTENTSIZE_ERROR) {
            out_size = static_cast<size_t>(frame_size);
        } else {
            out_size = buf.size * 16;
            if (out_size < 1024) out_size = 1024;
        }

        // Try up to 3 times with increasing buffer sizes
        for (int attempt = 0; attempt < 3; ++attempt) {
            std::string output(out_size, '\0');
            size_t actual_size = out_size;

            cuda_zstd::Status st = engine_->decompress(
                buf.data, buf.size,
                output.data(), &actual_size,
                cuda_zstd::DataLocation::HOST,
                cuda_zstd::DataLocation::HOST);

            if (st == cuda_zstd::Status::SUCCESS) {
                output.resize(actual_size);
                return py::bytes(output);
            }

            if (st == cuda_zstd::Status::ERROR_BUFFER_TOO_SMALL) {
                out_size *= 4;
                continue;
            }

            // Other error — throw immediately
            check_status(st, "hybrid decompress");
        }

        throw std::runtime_error("hybrid decompress: buffer too small after 3 attempts");
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    int get_level() const { return engine_->get_config().compression_level; }

    void set_level(int level) {
        check_status(engine_->set_compression_level(level),
                     "set_compression_level");
    }

    cuda_zstd::CompressionStats get_stats() const {
        return engine_->get_stats();
    }

    void reset_stats() { engine_->reset_stats(); }

    cuda_zstd::HybridConfig get_config() const {
        return engine_->get_config();
    }

    /// Query which backend would be chosen for a given data size.
    cuda_zstd::ExecutionBackend query_routing(size_t data_size) const {
        return engine_->query_routing(
            data_size,
            cuda_zstd::DataLocation::HOST,
            cuda_zstd::DataLocation::HOST,
            /*is_compression=*/true);
    }

private:
    std::unique_ptr<cuda_zstd::HybridEngine> engine_;
};

// ============================================================================
// Free functions (convenience — create a temporary manager per call)
// ============================================================================

namespace {

py::bytes compress_simple_py(py::object input, int level) {
    PyManager mgr(level);
    return mgr.compress(input);
}

py::bytes decompress_simple_py(py::object input) {
    PyManager mgr(3);
    return mgr.decompress(input);
}

std::vector<py::bytes> compress_batch_py(std::vector<py::buffer> inputs, int level) {
    PyManager mgr(level);
    return mgr.compress_batch(inputs);
}

std::vector<py::bytes> decompress_batch_py(std::vector<py::buffer> inputs) {
    PyManager mgr(3);
    return mgr.decompress_batch(inputs);
}

/// Hybrid convenience: compress bytes using HybridEngine.
py::bytes hybrid_compress_py(py::buffer input, int level) {
    PyHybridEngine engine(level);
    return engine.compress(input);
}

/// Hybrid convenience: decompress bytes using HybridEngine.
py::bytes hybrid_decompress_py(py::buffer input) {
    PyHybridEngine engine(3);
    return engine.decompress(input);
}

} // anonymous namespace

// ============================================================================
// Validation and estimation utilities
// ============================================================================

namespace {

/// Validate compressed data on the GPU.
bool validate_compressed_data_py(py::buffer data, bool check_checksum) {
    BufferInfo buf(data);
    DeviceBuffer d_buf(buf.size);
    h2d(d_buf.ptr, buf.data, buf.size);

    cuda_zstd::Status st =
        cuda_zstd::validate_compressed_data(d_buf.ptr, buf.size, check_checksum);
    return (st == cuda_zstd::Status::SUCCESS);
}

/// Estimate the compressed output size.
size_t estimate_compressed_size_py(size_t uncompressed_size, int level) {
    return cuda_zstd::estimate_compressed_size(uncompressed_size, level);
}

} // anonymous namespace

// ============================================================================
// CUDA device queries
// ============================================================================

namespace {

py::dict get_cuda_device_info() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaGetDeviceProperties failed: ") +
            cudaGetErrorString(err));
    }

    py::dict info;
    info["name"] = std::string(prop.name);
    info["compute_capability"] =
        std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info["total_memory_mb"] =
        static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);
    info["multiprocessor_count"] = prop.multiProcessorCount;
    info["device_index"] = device;
    return info;
}

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

} // anonymous namespace

// ============================================================================
// pybind11 module definition
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "cuda_zstd._core — low-level pybind11 bindings for GPU ZSTD";

    // ------------------------------------------------------------------
    // Enums
    // ------------------------------------------------------------------

    py::enum_<cuda_zstd::Status>(m, "Status")
        .value("SUCCESS", cuda_zstd::Status::SUCCESS)
        .value("ERROR_GENERIC", cuda_zstd::Status::ERROR_GENERIC)
        .value("ERROR_INVALID_PARAMETER", cuda_zstd::Status::ERROR_INVALID_PARAMETER)
        .value("ERROR_OUT_OF_MEMORY", cuda_zstd::Status::ERROR_OUT_OF_MEMORY)
        .value("ERROR_CUDA_ERROR", cuda_zstd::Status::ERROR_CUDA_ERROR)
        .value("ERROR_INVALID_MAGIC", cuda_zstd::Status::ERROR_INVALID_MAGIC)
        .value("ERROR_CORRUPT_DATA", cuda_zstd::Status::ERROR_CORRUPT_DATA)
        .value("ERROR_BUFFER_TOO_SMALL", cuda_zstd::Status::ERROR_BUFFER_TOO_SMALL)
        .value("ERROR_UNSUPPORTED_VERSION", cuda_zstd::Status::ERROR_UNSUPPORTED_VERSION)
        .value("ERROR_COMPRESSION", cuda_zstd::Status::ERROR_COMPRESSION)
        .value("ERROR_DECOMPRESSION", cuda_zstd::Status::ERROR_DECOMPRESSION)
        .export_values();

    py::enum_<cuda_zstd::Strategy>(m, "Strategy")
        .value("FAST", cuda_zstd::Strategy::FAST)
        .value("DFAST", cuda_zstd::Strategy::DFAST)
        .value("GREEDY", cuda_zstd::Strategy::GREEDY)
        .value("LAZY", cuda_zstd::Strategy::LAZY)
        .value("LAZY2", cuda_zstd::Strategy::LAZY2)
        .value("BTLAZY2", cuda_zstd::Strategy::BTLAZY2)
        .value("BTOPT", cuda_zstd::Strategy::BTOPT)
        .value("BTULTRA", cuda_zstd::Strategy::BTULTRA)
        .export_values();

    py::enum_<cuda_zstd::ChecksumPolicy>(m, "ChecksumPolicy")
        .value("NONE", cuda_zstd::ChecksumPolicy::NO_COMPUTE_NO_VERIFY)
        .value("COMPUTE", cuda_zstd::ChecksumPolicy::COMPUTE_NO_VERIFY)
        .value("COMPUTE_AND_VERIFY", cuda_zstd::ChecksumPolicy::COMPUTE_AND_VERIFY)
        .export_values();

    py::enum_<cuda_zstd::HybridMode>(m, "HybridMode")
        .value("AUTO", cuda_zstd::HybridMode::AUTO)
        .value("PREFER_CPU", cuda_zstd::HybridMode::PREFER_CPU)
        .value("PREFER_GPU", cuda_zstd::HybridMode::PREFER_GPU)
        .value("FORCE_CPU", cuda_zstd::HybridMode::FORCE_CPU)
        .value("FORCE_GPU", cuda_zstd::HybridMode::FORCE_GPU)
        .value("ADAPTIVE", cuda_zstd::HybridMode::ADAPTIVE)
        .export_values();

    py::enum_<cuda_zstd::DataLocation>(m, "DataLocation")
        .value("HOST", cuda_zstd::DataLocation::HOST)
        .value("DEVICE", cuda_zstd::DataLocation::DEVICE)
        .value("MANAGED", cuda_zstd::DataLocation::MANAGED)
        .value("UNKNOWN", cuda_zstd::DataLocation::UNKNOWN)
        .export_values();

    py::enum_<cuda_zstd::ExecutionBackend>(m, "ExecutionBackend")
        .value("CPU_LIBZSTD", cuda_zstd::ExecutionBackend::CPU_LIBZSTD)
        .value("GPU_KERNELS", cuda_zstd::ExecutionBackend::GPU_KERNELS)
        .value("CPU_PARALLEL", cuda_zstd::ExecutionBackend::CPU_PARALLEL)
        .value("GPU_BATCH", cuda_zstd::ExecutionBackend::GPU_BATCH)
        .export_values();

    // ------------------------------------------------------------------
    // CompressionConfig
    // ------------------------------------------------------------------

    py::class_<cuda_zstd::CompressionConfig>(m, "CompressionConfig")
        .def(py::init<>())
        .def_readwrite("level", &cuda_zstd::CompressionConfig::level)
        .def_readwrite("strategy", &cuda_zstd::CompressionConfig::strategy)
        .def_readwrite("window_log", &cuda_zstd::CompressionConfig::window_log)
        .def_readwrite("hash_log", &cuda_zstd::CompressionConfig::hash_log)
        .def_readwrite("chain_log", &cuda_zstd::CompressionConfig::chain_log)
        .def_readwrite("search_log", &cuda_zstd::CompressionConfig::search_log)
        .def_readwrite("min_match", &cuda_zstd::CompressionConfig::min_match)
        .def_readwrite("block_size", &cuda_zstd::CompressionConfig::block_size)
        .def_readwrite("enable_ldm", &cuda_zstd::CompressionConfig::enable_ldm)
        .def_readwrite("checksum", &cuda_zstd::CompressionConfig::checksum)
        .def_static("from_level", &cuda_zstd::CompressionConfig::from_level,
                     py::arg("level"))
        .def_static("optimal", &cuda_zstd::CompressionConfig::optimal,
                     py::arg("input_size"))
        .def_static("get_default", &cuda_zstd::CompressionConfig::get_default)
        .def("__repr__", [](const cuda_zstd::CompressionConfig &c) {
            return "<CompressionConfig level=" + std::to_string(c.level) +
                   " block_size=" + std::to_string(c.block_size) + ">";
        });

    // ------------------------------------------------------------------
    // HybridConfig
    // ------------------------------------------------------------------

    py::class_<cuda_zstd::HybridConfig>(m, "HybridConfig")
        .def(py::init<>())
        .def_readwrite("mode", &cuda_zstd::HybridConfig::mode)
        .def_readwrite("cpu_size_threshold", &cuda_zstd::HybridConfig::cpu_size_threshold)
        .def_readwrite("gpu_device_threshold", &cuda_zstd::HybridConfig::gpu_device_threshold)
        .def_readwrite("enable_profiling", &cuda_zstd::HybridConfig::enable_profiling)
        .def_readwrite("compression_level", &cuda_zstd::HybridConfig::compression_level)
        .def_readwrite("cpu_thread_count", &cuda_zstd::HybridConfig::cpu_thread_count)
        .def_readwrite("use_pinned_memory", &cuda_zstd::HybridConfig::use_pinned_memory)
        .def_readwrite("overlap_transfers", &cuda_zstd::HybridConfig::overlap_transfers)
        .def("__repr__", [](const cuda_zstd::HybridConfig &c) {
            return "<HybridConfig mode=" + std::to_string(static_cast<int>(c.mode)) +
                   " level=" + std::to_string(c.compression_level) + ">";
        });

    // ------------------------------------------------------------------
    // CompressionStats
    // ------------------------------------------------------------------

    py::class_<cuda_zstd::CompressionStats>(m, "CompressionStats")
        .def(py::init<>())
        .def_readonly("input_bytes", &cuda_zstd::CompressionStats::input_bytes)
        .def_readonly("output_bytes", &cuda_zstd::CompressionStats::output_bytes)
        .def_readonly("num_blocks", &cuda_zstd::CompressionStats::num_blocks)
        .def_readonly("compression_time_ms",
                      &cuda_zstd::CompressionStats::compression_time_ms)
        .def_readonly("decompression_time_ms",
                      &cuda_zstd::CompressionStats::decompression_time_ms)
        .def("get_ratio", &cuda_zstd::CompressionStats::get_ratio)
        .def("get_compression_throughput_gbps",
             &cuda_zstd::CompressionStats::get_compression_throughput_gbps)
        .def("__repr__", [](const cuda_zstd::CompressionStats &s) {
            return "<CompressionStats ratio=" +
                   std::to_string(s.get_ratio()) +
                   " in=" + std::to_string(s.input_bytes) +
                   " out=" + std::to_string(s.output_bytes) + ">";
        });

    // ------------------------------------------------------------------
    // PyManager (the main class users interact with)
    // ------------------------------------------------------------------

    py::class_<PyManager>(m, "Manager")
        .def(py::init<int>(), py::arg("level") = 3,
             "Create a GPU compression manager.\n\n"
             "Args:\n"
             "    level: Compression level 1-22 (default 3).\n"
             "           Higher = better ratio but slower.")
        .def(py::init<const cuda_zstd::CompressionConfig &>(),
             py::arg("config"),
             "Create a manager from a CompressionConfig.")
        .def("compress", &PyManager::compress, py::arg("data"),
             "Compress data on GPU.\n\n"
             "Args:\n"
             "    data: bytes, bytearray, numpy array, or CuPy array.\n\n"
             "Returns:\n"
             "    Compressed data as bytes.")
        .def("decompress", &PyManager::decompress, py::arg("data"),
             "Decompress GPU-ZSTD compressed data.\n\n"
             "Args:\n"
             "    data: Compressed bytes/buffer.\n\n"
             "Returns:\n"
             "    Decompressed data as bytes.")
        .def("compress_batch", &PyManager::compress_batch, py::arg("inputs"),
             "Compress a batch of buffers in a single GPU launch.\n\n"
             "Args:\n"
             "    inputs: list of bytes/bytearray/numpy arrays.\n\n"
             "Returns:\n"
             "    list[bytes] of compressed data.")
        .def("decompress_batch", &PyManager::decompress_batch, py::arg("inputs"),
             "Decompress a batch of compressed buffers.\n\n"
             "Args:\n"
             "    inputs: list of compressed bytes/buffers.\n\n"
             "Returns:\n"
             "    list[bytes] of decompressed data.")
        .def_property("level", &PyManager::get_level, &PyManager::set_level,
                      "Compression level (1-22).")
        .def("get_stats", &PyManager::get_stats,
             "Return compression statistics since last reset.")
        .def("reset_stats", &PyManager::reset_stats,
             "Reset compression/decompression statistics.")
        .def("get_config", &PyManager::get_config,
             "Return the current CompressionConfig.")
        .def("__enter__", [](PyManager &self) -> PyManager & { return self; })
        .def("__exit__",
             [](PyManager & /*self*/, py::object /*exc_type*/,
                py::object /*exc_val*/, py::object /*exc_tb*/) {
                 // Resources freed by destructor.
             })
        .def("__repr__", [](const PyManager &m) {
            return "<cuda_zstd.Manager level=" + std::to_string(m.get_level()) + ">";
        });

    // ------------------------------------------------------------------
    // PyHybridEngine (CPU/GPU auto-routing compression)
    // ------------------------------------------------------------------

    py::class_<PyHybridEngine>(m, "HybridEngine")
        .def(py::init<int>(), py::arg("level") = 3,
             "Create a hybrid CPU/GPU compression engine.\n\n"
             "The hybrid engine automatically routes work to CPU (libzstd)\n"
             "or GPU kernels based on data size and location.\n\n"
             "Args:\n"
             "    level: Compression level 1-22 (default 3).")
        .def(py::init<const cuda_zstd::HybridConfig &>(),
             py::arg("config"),
             "Create a hybrid engine from a HybridConfig.")
        .def("compress", &PyHybridEngine::compress, py::arg("data"),
             "Compress data using automatic CPU/GPU routing.\n\n"
             "Args:\n"
             "    data: bytes, bytearray, or numpy array.\n\n"
             "Returns:\n"
             "    Compressed data as bytes.")
        .def("decompress", &PyHybridEngine::decompress, py::arg("data"),
             "Decompress data using automatic CPU/GPU routing.\n\n"
             "Args:\n"
             "    data: Compressed bytes/buffer.\n\n"
             "Returns:\n"
             "    Decompressed data as bytes.")
        .def_property("level", &PyHybridEngine::get_level, &PyHybridEngine::set_level,
                      "Compression level (1-22).")
        .def("get_stats", &PyHybridEngine::get_stats,
             "Return compression statistics since last reset.")
        .def("reset_stats", &PyHybridEngine::reset_stats,
             "Reset compression/decompression statistics.")
        .def("get_config", &PyHybridEngine::get_config,
             "Return the current HybridConfig.")
        .def("query_routing", &PyHybridEngine::query_routing,
             py::arg("data_size"),
             "Query which backend would handle data of the given size.\n\n"
             "Args:\n"
             "    data_size: Size in bytes.\n\n"
             "Returns:\n"
             "    ExecutionBackend enum value.")
        .def("__enter__", [](PyHybridEngine &self) -> PyHybridEngine & { return self; })
        .def("__exit__",
             [](PyHybridEngine & /*self*/, py::object /*exc_type*/,
                py::object /*exc_val*/, py::object /*exc_tb*/) {
                 // Resources freed by destructor.
             })
        .def("__repr__", [](const PyHybridEngine &e) {
            return "<cuda_zstd.HybridEngine level=" + std::to_string(e.get_level()) + ">";
        });

    // ------------------------------------------------------------------
    // Free functions
    // ------------------------------------------------------------------

    m.def("compress", &compress_simple_py,
          py::arg("data"), py::arg("level") = 3,
          "Compress data on GPU (convenience function).\n\n"
          "Creates a temporary Manager for each call. For repeated\n"
          "compression, create a Manager and reuse it.\n\n"
          "Args:\n"
          "    data: bytes, bytearray, numpy array, or CuPy array.\n"
          "    level: Compression level 1-22 (default 3).\n\n"
          "Returns:\n"
          "    Compressed bytes.");

    m.def("decompress", &decompress_simple_py,
          py::arg("data"),
          "Decompress GPU-ZSTD compressed data (convenience function).\n\n"
          "Args:\n"
          "    data: Compressed bytes/buffer.\n\n"
          "Returns:\n"
          "    Decompressed bytes.");

    m.def("compress_batch", &compress_batch_py,
          py::arg("inputs"), py::arg("level") = 3,
          "Compress a list of buffers in a single GPU batch.\n\n"
          "Args:\n"
          "    inputs: list of bytes/bytearray/numpy arrays.\n"
          "    level: Compression level 1-22 (default 3).\n\n"
          "Returns:\n"
          "    list[bytes] of compressed data.");

    m.def("decompress_batch", &decompress_batch_py,
          py::arg("inputs"),
          "Decompress a list of compressed buffers in a single GPU batch.\n\n"
          "Args:\n"
          "    inputs: list of compressed bytes/buffers.\n\n"
          "Returns:\n"
          "    list[bytes] of decompressed data.");

    m.def("hybrid_compress", &hybrid_compress_py,
          py::arg("data"), py::arg("level") = 3,
          "Compress data using the hybrid CPU/GPU engine (convenience).\n\n"
          "Creates a temporary HybridEngine. For repeated calls,\n"
          "create a HybridEngine and reuse it.\n\n"
          "Args:\n"
          "    data: bytes, bytearray, or numpy array.\n"
          "    level: Compression level 1-22 (default 3).\n\n"
          "Returns:\n"
          "    Compressed bytes.");

    m.def("hybrid_decompress", &hybrid_decompress_py,
          py::arg("data"),
          "Decompress data using the hybrid CPU/GPU engine (convenience).\n\n"
          "Args:\n"
          "    data: Compressed bytes/buffer.\n\n"
          "Returns:\n"
          "    Decompressed bytes.");

    // ------------------------------------------------------------------
    // Validation & estimation utilities
    // ------------------------------------------------------------------

    m.def("validate_compressed_data", &validate_compressed_data_py,
          py::arg("data"), py::arg("check_checksum") = true,
          "Validate compressed data integrity.\n\n"
          "Uploads the data to the GPU and checks the compressed format.\n\n"
          "Args:\n"
          "    data: Compressed bytes/buffer to validate.\n"
          "    check_checksum: Whether to verify the checksum (default True).\n\n"
          "Returns:\n"
          "    True if the data is valid, False otherwise.");

    m.def("estimate_compressed_size", &estimate_compressed_size_py,
          py::arg("uncompressed_size"), py::arg("level") = 3,
          "Estimate the maximum compressed output size.\n\n"
          "This is a conservative upper bound useful for pre-allocating buffers.\n\n"
          "Args:\n"
          "    uncompressed_size: Size of the uncompressed data in bytes.\n"
          "    level: Compression level 1-22 (default 3).\n\n"
          "Returns:\n"
          "    Estimated maximum compressed size in bytes.");

    // ------------------------------------------------------------------
    // CUDA utilities
    // ------------------------------------------------------------------

    m.def("is_cuda_available", &is_cuda_available,
          "Check if a CUDA-capable GPU is available.");

    m.def("get_cuda_device_info", &get_cuda_device_info,
          "Return a dict with CUDA device properties (name, compute cap, memory, etc).");

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------

    m.attr("MIN_LEVEL") = py::int_(cuda_zstd::MIN_COMPRESSION_LEVEL);
    m.attr("MAX_LEVEL") = py::int_(cuda_zstd::MAX_COMPRESSION_LEVEL);
    m.attr("DEFAULT_LEVEL") = py::int_(cuda_zstd::DEFAULT_COMPRESSION_LEVEL);
    m.attr("__version__") = "1.0.0";  // keep in sync with root pyproject.toml
}
