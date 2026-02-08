// ============================================================================
// benchmark_hybrid.cu
//
// Benchmarks the Hybrid CPU/GPU engine across all routing paths, data sizes,
// and data patterns. Measures throughput (MB/s), compression ratio, and
// transfer vs compute time breakdown.
//
// Paths tested:
//   HOST->HOST     (CPU-only, no transfers)
//   HOST->DEVICE   (compress on CPU, upload result)
//   DEVICE->HOST   (download, compress on CPU)
//   DEVICE->DEVICE (GPU kernels, no transfers)
//
// Data patterns:
//   Random         (incompressible)
//   Repetitive     (highly compressible, repeating 64-byte block)
//   Sequential     (moderate compressibility, ascending bytes)
//   Text-like      (printable ASCII with structure)
//
// Data sizes:
//   1 KB, 64 KB, 256 KB, 1 MB, 10 MB, 100 MB
//
// Usage:
//   ./benchmark_hybrid
//   ./benchmark_hybrid --quick          (fewer sizes/patterns)
//   ./benchmark_hybrid --csv results.csv
//
// ============================================================================

#include "cuda_zstd_hybrid.h"
#include "cuda_zstd_safe_alloc.h"
#include <cuda_runtime.h>
#include <zstd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace cuda_zstd;

// ============================================================================
// Helpers
// ============================================================================

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return;                                                                  \
    }                                                                          \
  } while (0)

struct BenchResult {
  std::string pattern;
  size_t data_size;
  std::string path;         // "HOST->HOST", "HOST->DEV", etc.
  std::string backend;      // "CPU_LIBZSTD" or "GPU_KERNELS"
  double compress_mbps;
  double decompress_mbps;
  double compress_ratio;
  double compress_transfer_ms;
  double compress_compute_ms;
  double decompress_transfer_ms;
  double decompress_compute_ms;
  bool   compress_ok;
  bool   decompress_ok;
};

static std::string backend_to_str(ExecutionBackend b) {
  switch (b) {
    case ExecutionBackend::CPU_LIBZSTD:  return "CPU_LIBZSTD";
    case ExecutionBackend::GPU_KERNELS:  return "GPU_KERNELS";
    case ExecutionBackend::CPU_PARALLEL: return "CPU_PARALLEL";
    case ExecutionBackend::GPU_BATCH:    return "GPU_BATCH";
    default:                             return "UNKNOWN";
  }
}

static std::string size_to_str(size_t bytes) {
  if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + " KB";
  }
  return std::to_string(bytes) + " B";
}

// ============================================================================
// Data generators
// ============================================================================

static std::vector<unsigned char> gen_random(size_t n) {
  std::vector<unsigned char> v(n);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < n; i++) v[i] = static_cast<unsigned char>(dist(rng));
  return v;
}

static std::vector<unsigned char> gen_repetitive(size_t n) {
  std::vector<unsigned char> v(n);
  const char block[] = "ABCDEFGHIJKLMNOP0123456789abcdef"
                       "ABCDEFGHIJKLMNOP0123456789abcdef";
  for (size_t i = 0; i < n; i++) v[i] = static_cast<unsigned char>(block[i % 64]);
  return v;
}

static std::vector<unsigned char> gen_sequential(size_t n) {
  std::vector<unsigned char> v(n);
  for (size_t i = 0; i < n; i++) v[i] = static_cast<unsigned char>(i & 0xFF);
  return v;
}

static std::vector<unsigned char> gen_textlike(size_t n) {
  std::vector<unsigned char> v(n);
  std::mt19937 rng(123);
  // Printable ASCII with word/line structure
  const char charset[] = "abcdefghijklmnopqrstuvwxyz "
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
                         "0123456789.,;:!? \n";
  std::uniform_int_distribution<int> dist(0, (int)sizeof(charset) - 2);
  for (size_t i = 0; i < n; i++) v[i] = static_cast<unsigned char>(charset[dist(rng)]);
  return v;
}

// ============================================================================
// Benchmark a single (pattern, size, path) combination
// ============================================================================

static BenchResult bench_one(
    HybridEngine &engine,
    const std::vector<unsigned char> &data,
    const std::string &pattern_name,
    DataLocation src_loc,
    DataLocation dst_loc,
    const std::string &path_label,
    int iterations)
{
  BenchResult r{};
  r.pattern   = pattern_name;
  r.data_size = data.size();
  r.path      = path_label;

  size_t comp_cap = ZSTD_compressBound(data.size());

  // --- Prepare source buffer ---
  const void *src_ptr = data.data();
  void *d_src = nullptr;
  if (src_loc == DataLocation::DEVICE) {
    auto err = cuda_zstd::safe_cuda_malloc(&d_src, data.size());
    if (err != cudaSuccess) { r.compress_ok = false; r.decompress_ok = false; return r; }
    cudaMemcpy(d_src, data.data(), data.size(), cudaMemcpyHostToDevice);
    src_ptr = d_src;
  }

  // --- Prepare dest buffer ---
  void *comp_buf = nullptr;
  if (dst_loc == DataLocation::DEVICE) {
    auto err = cuda_zstd::safe_cuda_malloc(&comp_buf, comp_cap);
    if (err != cudaSuccess) {
      if (d_src) cudaFree(d_src);
      r.compress_ok = false; r.decompress_ok = false;
      return r;
    }
  } else {
    comp_buf = malloc(comp_cap);
    if (!comp_buf) {
      if (d_src) cudaFree(d_src);
      r.compress_ok = false; r.decompress_ok = false;
      return r;
    }
  }

  // =========== COMPRESS ===========
  double total_comp_ms = 0;
  double total_comp_transfer = 0;
  double total_comp_compute = 0;
  size_t compressed_size = 0;
  r.compress_ok = true;

  for (int it = 0; it < iterations; it++) {
    size_t out_size = comp_cap;
    HybridResult hr{};
    auto t0 = std::chrono::high_resolution_clock::now();
    Status st = engine.compress(
        src_ptr, data.size(),
        comp_buf, &out_size,
        src_loc, dst_loc, &hr);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (st != Status::SUCCESS) {
      r.compress_ok = false;
      break;
    }
    compressed_size = out_size;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_comp_ms += ms;
    total_comp_transfer += hr.transfer_time_ms;
    total_comp_compute  += hr.compute_time_ms;
    r.backend = backend_to_str(hr.backend_used);
  }

  if (r.compress_ok && iterations > 0) {
    double avg_ms = total_comp_ms / iterations;
    r.compress_mbps = (avg_ms > 0) ? (data.size() / (1024.0 * 1024.0)) / (avg_ms / 1000.0) : 0;
    r.compress_ratio = (compressed_size > 0) ? (double)data.size() / compressed_size : 0;
    r.compress_transfer_ms = total_comp_transfer / iterations;
    r.compress_compute_ms  = total_comp_compute / iterations;
  }

  // =========== DECOMPRESS ===========
  r.decompress_ok = r.compress_ok;
  if (!r.compress_ok) {
    // Cleanup
    if (dst_loc == DataLocation::DEVICE) cudaFree(comp_buf);
    else free(comp_buf);
    if (d_src) cudaFree(d_src);
    return r;
  }

  // Prepare decompression output
  void *decomp_buf = nullptr;
  DataLocation decomp_loc = src_loc; // decompress back to original location
  if (decomp_loc == DataLocation::DEVICE) {
    auto err = cuda_zstd::safe_cuda_malloc(&decomp_buf, data.size());
    if (err != cudaSuccess) {
      r.decompress_ok = false;
      if (dst_loc == DataLocation::DEVICE) cudaFree(comp_buf);
      else free(comp_buf);
      if (d_src) cudaFree(d_src);
      return r;
    }
  } else {
    decomp_buf = malloc(data.size());
    if (!decomp_buf) {
      r.decompress_ok = false;
      if (dst_loc == DataLocation::DEVICE) cudaFree(comp_buf);
      else free(comp_buf);
      if (d_src) cudaFree(d_src);
      return r;
    }
  }

  double total_decomp_ms = 0;
  double total_decomp_transfer = 0;
  double total_decomp_compute = 0;

  for (int it = 0; it < iterations; it++) {
    size_t out_size = data.size();
    HybridResult hr{};
    auto t0 = std::chrono::high_resolution_clock::now();
    Status st = engine.decompress(
        comp_buf, compressed_size,
        decomp_buf, &out_size,
        dst_loc, decomp_loc, &hr);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (st != Status::SUCCESS) {
      r.decompress_ok = false;
      break;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_decomp_ms += ms;
    total_decomp_transfer += hr.transfer_time_ms;
    total_decomp_compute  += hr.compute_time_ms;
  }

  if (r.decompress_ok && iterations > 0) {
    double avg_ms = total_decomp_ms / iterations;
    r.decompress_mbps = (avg_ms > 0) ? (data.size() / (1024.0 * 1024.0)) / (avg_ms / 1000.0) : 0;
    r.decompress_transfer_ms = total_decomp_transfer / iterations;
    r.decompress_compute_ms  = total_decomp_compute / iterations;

    // Verify correctness on last iteration
    if (decomp_loc == DataLocation::HOST) {
      if (memcmp(decomp_buf, data.data(), data.size()) != 0) {
        r.decompress_ok = false;
        std::cerr << "  [MISMATCH] " << pattern_name << " " << size_to_str(data.size())
                  << " " << path_label << std::endl;
      }
    } else {
      // Download and verify
      std::vector<unsigned char> verify(data.size());
      cudaMemcpy(verify.data(), decomp_buf, data.size(), cudaMemcpyDeviceToHost);
      if (memcmp(verify.data(), data.data(), data.size()) != 0) {
        r.decompress_ok = false;
        std::cerr << "  [MISMATCH] " << pattern_name << " " << size_to_str(data.size())
                  << " " << path_label << std::endl;
      }
    }
  }

  // Cleanup
  if (decomp_loc == DataLocation::DEVICE) cudaFree(decomp_buf);
  else free(decomp_buf);
  if (dst_loc == DataLocation::DEVICE) cudaFree(comp_buf);
  else free(comp_buf);
  if (d_src) cudaFree(d_src);

  return r;
}

// ============================================================================
// Print results table
// ============================================================================

static void print_header() {
  std::cout << std::left
            << std::setw(12) << "Pattern"
            << std::setw(10) << "Size"
            << std::setw(14) << "Path"
            << std::setw(14) << "Backend"
            << std::setw(12) << "Comp MB/s"
            << std::setw(12) << "Dec MB/s"
            << std::setw(8)  << "Ratio"
            << std::setw(10) << "Xfer(ms)"
            << std::setw(10) << "Comp(ms)"
            << std::setw(6)  << "OK"
            << std::endl;
  std::cout << std::string(108, '-') << std::endl;
}

static void print_row(const BenchResult &r) {
  std::cout << std::left
            << std::setw(12) << r.pattern
            << std::setw(10) << size_to_str(r.data_size)
            << std::setw(14) << r.path
            << std::setw(14) << r.backend
            << std::setw(12) << std::fixed << std::setprecision(1) << r.compress_mbps
            << std::setw(12) << std::fixed << std::setprecision(1) << r.decompress_mbps
            << std::setw(8)  << std::fixed << std::setprecision(2) << r.compress_ratio
            << std::setw(10) << std::fixed << std::setprecision(2) << r.compress_transfer_ms
            << std::setw(10) << std::fixed << std::setprecision(2) << r.compress_compute_ms
            << std::setw(6)  << (r.compress_ok && r.decompress_ok ? "PASS" : "FAIL")
            << std::endl;
}

static void write_csv(const std::vector<BenchResult> &results, const std::string &path) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "Cannot open " << path << " for writing" << std::endl;
    return;
  }
  f << "pattern,data_size_bytes,path,backend,compress_mbps,decompress_mbps,"
       "compress_ratio,compress_transfer_ms,compress_compute_ms,"
       "decompress_transfer_ms,decompress_compute_ms,compress_ok,decompress_ok\n";
  for (auto &r : results) {
    f << r.pattern << "," << r.data_size << "," << r.path << "," << r.backend << ","
      << std::fixed << std::setprecision(2)
      << r.compress_mbps << "," << r.decompress_mbps << ","
      << r.compress_ratio << ","
      << r.compress_transfer_ms << "," << r.compress_compute_ms << ","
      << r.decompress_transfer_ms << "," << r.decompress_compute_ms << ","
      << (r.compress_ok ? 1 : 0) << "," << (r.decompress_ok ? 1 : 0) << "\n";
  }
  f.close();
  std::cout << "\nCSV written to: " << path << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  bool quick_mode = false;
  std::string csv_path;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--quick") {
      quick_mode = true;
    } else if (arg == "--csv" && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: benchmark_hybrid [--quick] [--csv output.csv]\n";
      return 0;
    }
  }

  // Check VRAM availability
  size_t free_vram = 0, total_vram = 0;
  cudaMemGetInfo(&free_vram, &total_vram);
  std::cout << "============================================================\n";
  std::cout << "   Hybrid CPU/GPU Compression Benchmark\n";
  std::cout << "============================================================\n";
  std::cout << "GPU VRAM: " << free_vram / (1024*1024) << " MB free / "
            << total_vram / (1024*1024) << " MB total\n";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU: " << prop.name << "\n";
  std::cout << "Mode: " << (quick_mode ? "QUICK" : "FULL") << "\n";
  std::cout << "============================================================\n\n";

  // Data sizes to test
  std::vector<size_t> sizes;
  if (quick_mode) {
    sizes = {1024, 64*1024, 1024*1024};
  } else {
    sizes = {1024, 64*1024, 256*1024, 1024*1024, 10*1024*1024, 100*1024*1024};
  }

  // Check that largest size fits in VRAM (with safety margin)
  size_t max_size = sizes.back();
  // Need: input + compressed + decompressed = ~3x max_size on device
  size_t vram_needed = max_size * 4;
  if (vram_needed + VRAM_SAFETY_BUFFER_BYTES > free_vram) {
    // Trim sizes that won't fit
    while (!sizes.empty()) {
      size_t needed = sizes.back() * 4 + VRAM_SAFETY_BUFFER_BYTES;
      if (needed <= free_vram) break;
      std::cout << "Skipping " << size_to_str(sizes.back())
                << " (would exceed VRAM safety margin)\n";
      sizes.pop_back();
    }
    if (sizes.empty()) {
      std::cerr << "Not enough VRAM for any test size\n";
      return 1;
    }
  }

  // Pattern generators
  struct PatternDef {
    std::string name;
    std::vector<unsigned char> (*gen)(size_t);
  };
  std::vector<PatternDef> patterns;
  if (quick_mode) {
    patterns = {{"random", gen_random}, {"repetitive", gen_repetitive}};
  } else {
    patterns = {
      {"random",     gen_random},
      {"repetitive", gen_repetitive},
      {"sequential", gen_sequential},
      {"textlike",   gen_textlike},
    };
  }

  // Path combinations
  struct PathDef {
    std::string label;
    DataLocation src;
    DataLocation dst;
  };
  std::vector<PathDef> paths = {
    {"HOST->HOST", DataLocation::HOST, DataLocation::HOST},
    {"HOST->DEV",  DataLocation::HOST, DataLocation::DEVICE},
    {"DEV->HOST",  DataLocation::DEVICE, DataLocation::HOST},
    {"DEV->DEV",   DataLocation::DEVICE, DataLocation::DEVICE},
  };

  int iterations = quick_mode ? 1 : 3;

  // Create engine in AUTO mode
  HybridConfig cfg{};
  cfg.mode = HybridMode::AUTO;
  cfg.compression_level = 3;
  cfg.enable_profiling = true;
  auto engine_ptr = create_hybrid_engine(cfg);
  HybridEngine &engine = *engine_ptr;

  std::vector<BenchResult> all_results;

  for (auto &pat : patterns) {
    std::cout << "\n--- Pattern: " << pat.name << " ---\n\n";
    print_header();

    for (auto sz : sizes) {
      auto data = pat.gen(sz);

      for (auto &p : paths) {
        BenchResult r = bench_one(engine, data, pat.name, p.src, p.dst, p.label, iterations);
        print_row(r);
        all_results.push_back(r);
      }
    }
  }

  // =========== Summary ===========
  std::cout << "\n============================================================\n";
  std::cout << "   Summary\n";
  std::cout << "============================================================\n";

  int pass = 0, fail = 0;
  for (auto &r : all_results) {
    if (r.compress_ok && r.decompress_ok) pass++;
    else fail++;
  }
  std::cout << "Total benchmarks: " << all_results.size()
            << "  PASS: " << pass << "  FAIL: " << fail << "\n";

  // Best throughput per path
  std::cout << "\nBest compress throughput per path:\n";
  for (auto &p : paths) {
    double best = 0;
    std::string best_pat;
    size_t best_sz = 0;
    for (auto &r : all_results) {
      if (r.path == p.label && r.compress_ok && r.compress_mbps > best) {
        best = r.compress_mbps;
        best_pat = r.pattern;
        best_sz = r.data_size;
      }
    }
    if (best > 0) {
      std::cout << "  " << std::setw(14) << std::left << p.label
                << std::fixed << std::setprecision(1) << best << " MB/s"
                << " (" << best_pat << " " << size_to_str(best_sz) << ")\n";
    }
  }

  std::cout << "\nBest decompress throughput per path:\n";
  for (auto &p : paths) {
    double best = 0;
    std::string best_pat;
    size_t best_sz = 0;
    for (auto &r : all_results) {
      if (r.path == p.label && r.decompress_ok && r.decompress_mbps > best) {
        best = r.decompress_mbps;
        best_pat = r.pattern;
        best_sz = r.data_size;
      }
    }
    if (best > 0) {
      std::cout << "  " << std::setw(14) << std::left << p.label
                << std::fixed << std::setprecision(1) << best << " MB/s"
                << " (" << best_pat << " " << size_to_str(best_sz) << ")\n";
    }
  }

  // Routing decisions summary
  std::cout << "\nRouting decisions (AUTO mode):\n";
  int cpu_count = 0, gpu_count = 0;
  for (auto &r : all_results) {
    if (r.backend == "CPU_LIBZSTD") cpu_count++;
    else gpu_count++;
  }
  std::cout << "  CPU routed: " << cpu_count << "  GPU routed: " << gpu_count << "\n";

  if (!csv_path.empty()) {
    write_csv(all_results, csv_path);
  }

  std::cout << "\nDone.\n";
  return (fail > 0) ? 1 : 0;
}
