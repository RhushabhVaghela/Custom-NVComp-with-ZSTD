// ============================================================================
// test_hybrid.cu - Comprehensive Tests for Hybrid CPU/GPU Engine
// ============================================================================
//
// Tests routing correctness, all HybridMode values, data location combinations,
// cross-path round-trips, batch operations, edge cases, HybridResult metadata,
// query_routing, and the C API.
// ============================================================================

#include "cuda_zstd_hybrid.h"
#include "cuda_zstd_safe_alloc.h"
#include <cuda_runtime.h>
#include <zstd.h>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace cuda_zstd;

static int g_pass = 0;
static int g_fail = 0;

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "  FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ \
                << ")\n";                                                      \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define RUN_TEST(func)                                                         \
  do {                                                                         \
    std::cout << "  " << #func << "... ";                                      \
    if (func()) {                                                              \
      std::cout << "PASSED\n";                                                 \
      g_pass++;                                                                \
    } else {                                                                   \
      std::cout << "FAILED\n";                                                 \
      g_fail++;                                                                \
    }                                                                          \
  } while (0)

// ============================================================================
// Test data generators
// ============================================================================

static std::vector<uint8_t> make_repetitive(size_t size) {
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; i++)
    data[i] = static_cast<uint8_t>(i % 7);
  return data;
}

static std::vector<uint8_t> make_sequential(size_t size) {
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; i++)
    data[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
  return data;
}

static std::vector<uint8_t> make_zeros(size_t size) {
  return std::vector<uint8_t>(size, 0);
}

// ============================================================================
// Test 1: HOST -> HOST roundtrip (should route to CPU)
// ============================================================================

static bool test_host_to_host_roundtrip() {
  const size_t data_size = 64 * 1024; // 64 KB
  auto original = make_repetitive(data_size);

  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Compress
  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult comp_result;

  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             &comp_result);
  TEST_ASSERT(s == Status::SUCCESS, "compress failed");
  TEST_ASSERT(comp_size > 0 && comp_size < data_size, "unexpected compressed size");
  TEST_ASSERT(comp_result.backend_used == ExecutionBackend::CPU_LIBZSTD,
              "HOST->HOST should route to CPU");
  TEST_ASSERT(comp_result.input_bytes == data_size, "input_bytes wrong");
  TEST_ASSERT(comp_result.output_bytes == comp_size, "output_bytes wrong");
  TEST_ASSERT(comp_result.total_time_ms > 0.0, "total_time_ms should be > 0");
  TEST_ASSERT(comp_result.routing_reason != nullptr, "routing_reason null");

  // Decompress
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  HybridResult decomp_result;

  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &decomp_result);
  TEST_ASSERT(s == Status::SUCCESS, "decompress failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(decomp_result.backend_used == ExecutionBackend::CPU_LIBZSTD,
              "HOST->HOST decompress should route to CPU");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch after roundtrip");
  return true;
}

// ============================================================================
// Test 2: DEVICE -> DEVICE roundtrip (should route to GPU)
// ============================================================================

static bool test_device_to_device_roundtrip() {
  const size_t data_size = 32 * 1024; // 32 KB (< gpu_device_threshold=64KB)
  auto original = make_repetitive(data_size);

  void *d_input = nullptr, *d_compressed = nullptr, *d_decompressed = nullptr;
  if (safe_cuda_malloc(&d_input, data_size) != cudaSuccess)
    return false;
  size_t comp_cap = ZSTD_compressBound(data_size);
  if (safe_cuda_malloc(&d_compressed, comp_cap) != cudaSuccess) {
    cudaFree(d_input);
    return false;
  }
  if (safe_cuda_malloc(&d_decompressed, data_size) != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    return false;
  }

  cudaMemcpy(d_input, original.data(), data_size, cudaMemcpyHostToDevice);

  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Compress
  size_t comp_size = comp_cap;
  HybridResult comp_result;
  Status s = engine.compress(d_input, data_size, d_compressed, &comp_size,
                             DataLocation::DEVICE, DataLocation::DEVICE,
                             &comp_result);
  TEST_ASSERT(s == Status::SUCCESS, "GPU compress failed");
  TEST_ASSERT(comp_size > 0, "compressed size is 0");
  // Small device-to-device data should use GPU
  TEST_ASSERT(comp_result.backend_used == ExecutionBackend::GPU_KERNELS,
              "DEVICE->DEVICE small data should route to GPU");

  // Decompress
  size_t decomp_size = data_size;
  HybridResult decomp_result;
  s = engine.decompress(d_compressed, comp_size, d_decompressed, &decomp_size,
                        DataLocation::DEVICE, DataLocation::DEVICE,
                        &decomp_result);
  TEST_ASSERT(s == Status::SUCCESS, "GPU decompress failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");

  // Verify data
  std::vector<uint8_t> h_result(data_size);
  cudaMemcpy(h_result.data(), d_decompressed, data_size,
             cudaMemcpyDeviceToHost);
  TEST_ASSERT(memcmp(original.data(), h_result.data(), data_size) == 0,
              "data mismatch after GPU roundtrip");

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  return true;
}

// ============================================================================
// Test 3: HOST -> DEVICE roundtrip
// ============================================================================

static bool test_host_to_device_roundtrip() {
  const size_t data_size = 16 * 1024; // 16 KB
  auto original = make_sequential(data_size);

  void *d_compressed = nullptr;
  size_t comp_cap = ZSTD_compressBound(data_size);
  if (safe_cuda_malloc(&d_compressed, comp_cap) != cudaSuccess)
    return false;

  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Compress from host to device
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(original.data(), data_size, d_compressed,
                             &comp_size, DataLocation::HOST,
                             DataLocation::DEVICE, &result);
  TEST_ASSERT(s == Status::SUCCESS, "compress HOST->DEVICE failed");
  TEST_ASSERT(comp_size > 0, "compressed size is 0");

  // Decompress from device to host
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(d_compressed, comp_size, decompressed.data(),
                        &decomp_size, DataLocation::DEVICE,
                        DataLocation::HOST, &result);
  TEST_ASSERT(s == Status::SUCCESS, "decompress DEVICE->HOST failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");

  cudaFree(d_compressed);
  return true;
}

// ============================================================================
// Test 4: DEVICE -> HOST roundtrip
// ============================================================================

static bool test_device_to_host_roundtrip() {
  const size_t data_size = 16 * 1024;
  auto original = make_sequential(data_size);

  void *d_input = nullptr;
  if (safe_cuda_malloc(&d_input, data_size) != cudaSuccess)
    return false;
  cudaMemcpy(d_input, original.data(), data_size, cudaMemcpyHostToDevice);

  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Compress from device to host
  size_t comp_cap = ZSTD_compressBound(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(d_input, data_size, compressed.data(), &comp_size,
                             DataLocation::DEVICE, DataLocation::HOST, &result);
  TEST_ASSERT(s == Status::SUCCESS, "compress DEVICE->HOST failed");
  TEST_ASSERT(comp_size > 0, "compressed size is 0");

  // Decompress from host to host
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &result);
  TEST_ASSERT(s == Status::SUCCESS, "decompress failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");

  cudaFree(d_input);
  return true;
}

// ============================================================================
// Test 5: FORCE_CPU mode always uses CPU
// ============================================================================

static bool test_force_cpu_mode() {
  const size_t data_size = 8 * 1024;
  auto original = make_repetitive(data_size);

  void *d_input = nullptr, *d_output = nullptr;
  if (safe_cuda_malloc(&d_input, data_size) != cudaSuccess)
    return false;
  size_t comp_cap = ZSTD_compressBound(data_size);
  if (safe_cuda_malloc(&d_output, comp_cap) != cudaSuccess) {
    cudaFree(d_input);
    return false;
  }
  cudaMemcpy(d_input, original.data(), data_size, cudaMemcpyHostToDevice);

  HybridConfig config;
  config.mode = HybridMode::FORCE_CPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Even with DEVICE data, FORCE_CPU should use CPU
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(d_input, data_size, d_output, &comp_size,
                             DataLocation::DEVICE, DataLocation::DEVICE,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "FORCE_CPU compress failed");
  TEST_ASSERT(result.backend_used == ExecutionBackend::CPU_LIBZSTD,
              "FORCE_CPU should always use CPU");

  cudaFree(d_input);
  cudaFree(d_output);
  return true;
}

// ============================================================================
// Test 6: FORCE_GPU mode always uses GPU
// ============================================================================

static bool test_force_gpu_mode() {
  const size_t data_size = 8 * 1024;
  auto original = make_repetitive(data_size);

  HybridConfig config;
  config.mode = HybridMode::FORCE_GPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  // HOST data with FORCE_GPU should still use GPU (engine handles transfers)
  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "FORCE_GPU compress failed");
  TEST_ASSERT(result.backend_used == ExecutionBackend::GPU_KERNELS,
              "FORCE_GPU should always use GPU");

  // Decompress with FORCE_GPU
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &result);
  // Note: GPU decompress may fail for CPU-compressed Huffman data, in which
  // case the hybrid engine falls back to CPU. For FORCE_GPU this fallback
  // should NOT happen. Either way, the roundtrip should succeed or we get
  // a status error. Since the data was GPU-compressed (no Huffman), it should
  // decompress fine.
  TEST_ASSERT(s == Status::SUCCESS, "FORCE_GPU decompress failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");
  return true;
}

// ============================================================================
// Test 7: PREFER_CPU mode
// ============================================================================

static bool test_prefer_cpu_mode() {
  const size_t data_size = 16 * 1024;
  auto original = make_repetitive(data_size);

  HybridConfig config;
  config.mode = HybridMode::PREFER_CPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Host data should definitely use CPU
  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "PREFER_CPU compress failed");
  TEST_ASSERT(result.backend_used == ExecutionBackend::CPU_LIBZSTD,
              "PREFER_CPU HOST->HOST should use CPU");

  // Roundtrip verify
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &result);
  TEST_ASSERT(s == Status::SUCCESS, "decompress failed");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");
  return true;
}

// ============================================================================
// Test 8: PREFER_GPU mode with device data
// ============================================================================

static bool test_prefer_gpu_mode() {
  const size_t data_size = 16 * 1024;
  auto original = make_repetitive(data_size);

  void *d_input = nullptr, *d_compressed = nullptr, *d_decompressed = nullptr;
  if (safe_cuda_malloc(&d_input, data_size) != cudaSuccess)
    return false;
  size_t comp_cap = ZSTD_compressBound(data_size);
  if (safe_cuda_malloc(&d_compressed, comp_cap) != cudaSuccess) {
    cudaFree(d_input);
    return false;
  }
  if (safe_cuda_malloc(&d_decompressed, data_size) != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_compressed);
    return false;
  }
  cudaMemcpy(d_input, original.data(), data_size, cudaMemcpyHostToDevice);

  HybridConfig config;
  config.mode = HybridMode::PREFER_GPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  // Device data with PREFER_GPU should use GPU
  size_t comp_size = comp_cap;
  HybridResult result;
  Status s = engine.compress(d_input, data_size, d_compressed, &comp_size,
                             DataLocation::DEVICE, DataLocation::DEVICE,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "PREFER_GPU compress failed");
  TEST_ASSERT(result.backend_used == ExecutionBackend::GPU_KERNELS,
              "PREFER_GPU DEVICE->DEVICE should use GPU");

  // Decompress
  size_t decomp_size = data_size;
  s = engine.decompress(d_compressed, comp_size, d_decompressed, &decomp_size,
                        DataLocation::DEVICE, DataLocation::DEVICE, &result);
  TEST_ASSERT(s == Status::SUCCESS, "PREFER_GPU decompress failed");

  // Verify
  std::vector<uint8_t> h_result(data_size);
  cudaMemcpy(h_result.data(), d_decompressed, data_size,
             cudaMemcpyDeviceToHost);
  TEST_ASSERT(memcmp(original.data(), h_result.data(), data_size) == 0,
              "data mismatch");

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_decompressed);
  return true;
}

// ============================================================================
// Test 9: ADAPTIVE mode
// ============================================================================

static bool test_adaptive_mode() {
  const size_t data_size = 32 * 1024;
  auto original = make_repetitive(data_size);

  HybridConfig config;
  config.mode = HybridMode::ADAPTIVE;
  config.enable_profiling = true;
  config.compression_level = 3;
  HybridEngine engine(config);

  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;

  // Run a few iterations to let adaptive mode learn
  for (int i = 0; i < 5; i++) {
    comp_size = comp_cap;
    Status s = engine.compress(original.data(), data_size, compressed.data(),
                               &comp_size, DataLocation::HOST,
                               DataLocation::HOST, &result);
    TEST_ASSERT(s == Status::SUCCESS, "ADAPTIVE compress iteration failed");
  }

  // Check profiling data exists
  double cpu_throughput =
      engine.get_observed_throughput(ExecutionBackend::CPU_LIBZSTD, true);
  // Should have some profiling data after 5 iterations
  TEST_ASSERT(cpu_throughput > 0.0,
              "CPU throughput should be > 0 after profiling");

  // Verify roundtrip still works
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  Status s = engine.decompress(compressed.data(), comp_size,
                               decompressed.data(), &decomp_size,
                               DataLocation::HOST, DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "ADAPTIVE decompress failed");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");

  // Reset profiling
  engine.reset_profiling();
  cpu_throughput =
      engine.get_observed_throughput(ExecutionBackend::CPU_LIBZSTD, true);
  TEST_ASSERT(cpu_throughput == 0.0,
              "throughput should be 0 after reset_profiling");

  return true;
}

// ============================================================================
// Test 10: detect_location
// ============================================================================

static bool test_detect_location() {
  // Host pointer
  std::vector<uint8_t> h_data(64);
  DataLocation loc = HybridEngine::detect_location(h_data.data());
  TEST_ASSERT(loc == DataLocation::HOST, "host pointer should be HOST");

  // Device pointer
  void *d_data = nullptr;
  if (safe_cuda_malloc(&d_data, 64) != cudaSuccess)
    return false;
  loc = HybridEngine::detect_location(d_data);
  TEST_ASSERT(loc == DataLocation::DEVICE, "device pointer should be DEVICE");
  cudaFree(d_data);

  // Null pointer should be UNKNOWN or HOST
  loc = HybridEngine::detect_location(nullptr);
  // nullptr detection varies — just verify no crash
  TEST_ASSERT(loc == DataLocation::HOST || loc == DataLocation::UNKNOWN,
              "null ptr should be HOST or UNKNOWN");

  return true;
}

// ============================================================================
// Test 11: query_routing (advisory, no execution)
// ============================================================================

static bool test_query_routing() {
  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.cpu_size_threshold = 1024 * 1024;
  config.gpu_device_threshold = 64 * 1024;
  HybridEngine engine(config);

  // Host -> Host should always be CPU
  ExecutionBackend backend = engine.query_routing(
      1024, DataLocation::HOST, DataLocation::HOST, true);
  TEST_ASSERT(backend == ExecutionBackend::CPU_LIBZSTD,
              "HOST->HOST should query as CPU");

  // Device -> Device, small data should be GPU
  backend = engine.query_routing(
      32 * 1024, DataLocation::DEVICE, DataLocation::DEVICE, true);
  TEST_ASSERT(backend == ExecutionBackend::GPU_KERNELS,
              "DEVICE->DEVICE small should query as GPU");

  // Force CPU should always return CPU
  config.mode = HybridMode::FORCE_CPU;
  engine.configure(config);
  backend = engine.query_routing(
      32 * 1024, DataLocation::DEVICE, DataLocation::DEVICE, true);
  TEST_ASSERT(backend == ExecutionBackend::CPU_LIBZSTD,
              "FORCE_CPU should always query as CPU");

  return true;
}

// ============================================================================
// Test 12: get_max_compressed_size
// ============================================================================

static bool test_max_compressed_size() {
  HybridEngine engine;

  size_t max_size = engine.get_max_compressed_size(0);
  TEST_ASSERT(max_size > 0, "max compressed size for 0 should be > 0");

  size_t max_1mb = engine.get_max_compressed_size(1024 * 1024);
  TEST_ASSERT(max_1mb > 1024 * 1024,
              "max compressed size should exceed input for 1MB");

  return true;
}

// ============================================================================
// Test 13: Cross-path: GPU compress -> CPU decompress
// ============================================================================

static bool test_cross_path_gpu_compress_cpu_decompress() {
  const size_t data_size = 32 * 1024;
  auto original = make_repetitive(data_size);

  // Compress with FORCE_GPU
  HybridConfig config;
  config.mode = HybridMode::FORCE_GPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "GPU compress failed");

  // Decompress with FORCE_CPU
  config.mode = HybridMode::FORCE_CPU;
  engine.configure(config);

  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "CPU decompress of GPU-compressed data failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");
  return true;
}

// ============================================================================
// Test 14: Cross-path: CPU compress -> GPU decompress (with fallback)
// ============================================================================

static bool test_cross_path_cpu_compress_gpu_decompress() {
  const size_t data_size = 32 * 1024;
  auto original = make_repetitive(data_size);

  // Compress with FORCE_CPU
  HybridConfig config;
  config.mode = HybridMode::FORCE_CPU;
  config.compression_level = 3;
  HybridEngine engine(config);

  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "CPU compress failed");

  // Decompress with AUTO (GPU may fail on Huffman-encoded data and fall back)
  config.mode = HybridMode::AUTO;
  engine.configure(config);

  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  HybridResult result;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &result);
  TEST_ASSERT(s == Status::SUCCESS,
              "Decompress of CPU-compressed data failed (should fallback)");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");
  return true;
}

// ============================================================================
// Test 15: configure() and set_compression_level()
// ============================================================================

static bool test_configuration() {
  HybridEngine engine;

  // Default config should be AUTO
  HybridConfig cfg = engine.get_config();
  TEST_ASSERT(cfg.mode == HybridMode::AUTO, "default mode should be AUTO");
  TEST_ASSERT(cfg.compression_level == 3, "default level should be 3");

  // Set level
  Status s = engine.set_compression_level(9);
  TEST_ASSERT(s == Status::SUCCESS, "set_compression_level(9) should succeed");
  cfg = engine.get_config();
  TEST_ASSERT(cfg.compression_level == 9, "level should be 9 after set");

  // Invalid level
  s = engine.set_compression_level(0);
  TEST_ASSERT(s != Status::SUCCESS, "set_compression_level(0) should fail");

  s = engine.set_compression_level(23);
  TEST_ASSERT(s != Status::SUCCESS, "set_compression_level(23) should fail");

  // Reconfigure
  HybridConfig new_cfg;
  new_cfg.mode = HybridMode::FORCE_CPU;
  new_cfg.compression_level = 5;
  s = engine.configure(new_cfg);
  TEST_ASSERT(s == Status::SUCCESS, "configure() should succeed");
  cfg = engine.get_config();
  TEST_ASSERT(cfg.mode == HybridMode::FORCE_CPU, "mode should be FORCE_CPU");
  TEST_ASSERT(cfg.compression_level == 5, "level should be 5");

  return true;
}

// ============================================================================
// Test 16: Statistics tracking
// ============================================================================

static bool test_statistics() {
  const size_t data_size = 8 * 1024;
  auto original = make_repetitive(data_size);

  HybridEngine engine;

  // Reset and check zero
  engine.reset_stats();
  CompressionStats stats = engine.get_stats();
  // Stats may or may not have a frames field, so just ensure no crash

  // Run compress + decompress
  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  engine.compress(original.data(), data_size, compressed.data(), &comp_size,
                  DataLocation::HOST, DataLocation::HOST);

  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  engine.decompress(compressed.data(), comp_size, decompressed.data(),
                    &decomp_size, DataLocation::HOST, DataLocation::HOST);

  stats = engine.get_stats();
  // Verify some stats were recorded (implementation-dependent)
  // At minimum, no crash
  return true;
}

// ============================================================================
// Test 17: Factory functions
// ============================================================================

static bool test_factory_functions() {
  // Default config factory
  auto engine1 = create_hybrid_engine();
  TEST_ASSERT(engine1 != nullptr, "create_hybrid_engine() returned null");
  HybridConfig cfg = engine1->get_config();
  TEST_ASSERT(cfg.mode == HybridMode::AUTO, "factory default should be AUTO");

  // Level factory
  auto engine2 = create_hybrid_engine(7);
  TEST_ASSERT(engine2 != nullptr,
              "create_hybrid_engine(level) returned null");
  cfg = engine2->get_config();
  TEST_ASSERT(cfg.compression_level == 7, "factory level should be 7");

  // Config factory
  HybridConfig custom;
  custom.mode = HybridMode::FORCE_CPU;
  custom.compression_level = 12;
  auto engine3 = create_hybrid_engine(custom);
  TEST_ASSERT(engine3 != nullptr, "config factory returned null");
  cfg = engine3->get_config();
  TEST_ASSERT(cfg.mode == HybridMode::FORCE_CPU, "config mode mismatch");
  TEST_ASSERT(cfg.compression_level == 12, "config level mismatch");

  return true;
}

// ============================================================================
// Test 18: Convenience functions
// ============================================================================

static bool test_convenience_functions() {
  const size_t data_size = 16 * 1024;
  auto original = make_repetitive(data_size);

  size_t comp_cap = ZSTD_compressBound(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;

  Status s = hybrid_compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             3, &result);
  TEST_ASSERT(s == Status::SUCCESS, "hybrid_compress failed");
  TEST_ASSERT(comp_size > 0, "compressed size is 0");

  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = hybrid_decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST,
                        &result);
  TEST_ASSERT(s == Status::SUCCESS, "hybrid_decompress failed");
  TEST_ASSERT(decomp_size == data_size, "decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "data mismatch");
  return true;
}

// ============================================================================
// Test 19: Batch compress/decompress
// ============================================================================

static bool test_batch_operations() {
  const size_t count = 4;
  const size_t sizes[] = {1024, 4096, 8192, 16384};
  std::vector<std::vector<uint8_t>> originals(count);
  std::vector<std::vector<uint8_t>> compressed_bufs(count);
  std::vector<std::vector<uint8_t>> decompressed_bufs(count);

  std::vector<const void *> inputs(count);
  std::vector<size_t> input_sizes(count);
  std::vector<void *> outputs(count);
  std::vector<size_t> output_sizes(count);

  for (size_t i = 0; i < count; i++) {
    originals[i] = make_repetitive(sizes[i]);
    compressed_bufs[i].resize(ZSTD_compressBound(sizes[i]));
    inputs[i] = originals[i].data();
    input_sizes[i] = sizes[i];
    outputs[i] = compressed_bufs[i].data();
    output_sizes[i] = compressed_bufs[i].size();
  }

  HybridEngine engine;
  std::vector<BatchRoutingResult> results(count);

  Status s = engine.compress_batch(inputs.data(), input_sizes.data(),
                                   outputs.data(), output_sizes.data(), count,
                                   DataLocation::HOST, DataLocation::HOST,
                                   results.data());
  TEST_ASSERT(s == Status::SUCCESS, "batch compress failed");

  // Verify each item was routed
  for (size_t i = 0; i < count; i++) {
    TEST_ASSERT(results[i].status == Status::SUCCESS,
                "batch item compress failed");
    TEST_ASSERT(results[i].output_bytes > 0, "batch item output is 0");
  }

  // Batch decompress
  std::vector<const void *> decomp_inputs(count);
  std::vector<size_t> decomp_input_sizes(count);
  std::vector<void *> decomp_outputs(count);
  std::vector<size_t> decomp_output_sizes(count);

  for (size_t i = 0; i < count; i++) {
    decomp_inputs[i] = compressed_bufs[i].data();
    decomp_input_sizes[i] = output_sizes[i];
    decompressed_bufs[i].resize(sizes[i]);
    decomp_outputs[i] = decompressed_bufs[i].data();
    decomp_output_sizes[i] = sizes[i];
  }

  s = engine.decompress_batch(decomp_inputs.data(), decomp_input_sizes.data(),
                              decomp_outputs.data(), decomp_output_sizes.data(),
                              count, DataLocation::HOST, DataLocation::HOST,
                              results.data());
  TEST_ASSERT(s == Status::SUCCESS, "batch decompress failed");

  // Verify all items
  for (size_t i = 0; i < count; i++) {
    TEST_ASSERT(results[i].status == Status::SUCCESS,
                "batch item decompress failed");
    TEST_ASSERT(memcmp(originals[i].data(), decompressed_bufs[i].data(),
                       sizes[i]) == 0,
                "batch item data mismatch");
  }

  return true;
}

// ============================================================================
// Test 20: Edge case - zero-size input
// ============================================================================

static bool test_zero_size_input() {
  HybridEngine engine;
  std::vector<uint8_t> output(128);
  size_t out_size = output.size();

  // Zero-size compress — should either succeed with minimal output or return error
  Status s = engine.compress(nullptr, 0, output.data(), &out_size,
                             DataLocation::HOST, DataLocation::HOST);
  // Either success (ZSTD can compress empty frame) or an error — no crash
  (void)s;

  return true;
}

// ============================================================================
// Test 21: Move semantics
// ============================================================================

static bool test_move_semantics() {
  HybridConfig config;
  config.mode = HybridMode::FORCE_CPU;
  config.compression_level = 5;

  HybridEngine engine1(config);

  // Move construct
  HybridEngine engine2(std::move(engine1));

  HybridConfig cfg = engine2.get_config();
  TEST_ASSERT(cfg.mode == HybridMode::FORCE_CPU, "move ctor: mode wrong");
  TEST_ASSERT(cfg.compression_level == 5, "move ctor: level wrong");

  // Move assign
  HybridEngine engine3;
  engine3 = std::move(engine2);

  cfg = engine3.get_config();
  TEST_ASSERT(cfg.mode == HybridMode::FORCE_CPU, "move assign: mode wrong");
  TEST_ASSERT(cfg.compression_level == 5, "move assign: level wrong");

  // Verify engine3 still works
  const size_t data_size = 4096;
  auto original = make_repetitive(data_size);
  size_t comp_cap = ZSTD_compressBound(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  Status s = engine3.compress(original.data(), data_size, compressed.data(),
                              &comp_size, DataLocation::HOST,
                              DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "compress after move should work");

  return true;
}

// ============================================================================
// Test 22: HybridResult timing breakdown
// ============================================================================

static bool test_result_timing() {
  const size_t data_size = 64 * 1024;
  auto original = make_repetitive(data_size);

  HybridConfig config;
  config.mode = HybridMode::AUTO;
  config.compression_level = 3;
  HybridEngine engine(config);

  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;

  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "compress failed");

  // HOST->HOST: no transfers, so transfer_time should be ~0
  TEST_ASSERT(result.total_time_ms >= 0.0, "total_time_ms negative");
  TEST_ASSERT(result.compute_time_ms >= 0.0, "compute_time_ms negative");
  TEST_ASSERT(result.throughput_mbps > 0.0, "throughput should be > 0");
  TEST_ASSERT(result.compression_ratio > 1.0f,
              "repetitive data should compress (ratio > 1)");
  TEST_ASSERT(result.input_bytes == data_size, "input_bytes wrong");
  TEST_ASSERT(result.output_bytes == comp_size, "output_bytes wrong");

  return true;
}

// ============================================================================
// Test 23: Multiple data patterns
// ============================================================================

static bool test_multiple_data_patterns() {
  HybridEngine engine;

  struct TestCase {
    const char *name;
    std::vector<uint8_t> data;
  };

  TestCase cases[] = {
      {"zeros", make_zeros(4096)},
      {"repetitive", make_repetitive(4096)},
      {"sequential", make_sequential(4096)},
  };

  for (auto &tc : cases) {
    size_t comp_cap = engine.get_max_compressed_size(tc.data.size());
    std::vector<uint8_t> compressed(comp_cap);
    size_t comp_size = comp_cap;

    Status s =
        engine.compress(tc.data.data(), tc.data.size(), compressed.data(),
                        &comp_size, DataLocation::HOST, DataLocation::HOST);
    TEST_ASSERT(s == Status::SUCCESS, "compress failed for pattern");

    std::vector<uint8_t> decompressed(tc.data.size());
    size_t decomp_size = tc.data.size();
    s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                          &decomp_size, DataLocation::HOST, DataLocation::HOST);
    TEST_ASSERT(s == Status::SUCCESS, "decompress failed for pattern");
    TEST_ASSERT(decomp_size == tc.data.size(), "size mismatch for pattern");
    TEST_ASSERT(
        memcmp(tc.data.data(), decompressed.data(), tc.data.size()) == 0,
        "data mismatch for pattern");
  }
  return true;
}

// ============================================================================
// Test 24: Larger data (1MB) through hybrid engine
// ============================================================================

static bool test_larger_data() {
  const size_t data_size = 1024 * 1024; // 1MB
  auto original = make_sequential(data_size);

  HybridEngine engine;

  size_t comp_cap = engine.get_max_compressed_size(data_size);
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  HybridResult result;

  Status s = engine.compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST, DataLocation::HOST,
                             &result);
  TEST_ASSERT(s == Status::SUCCESS, "1MB compress failed");
  TEST_ASSERT(result.backend_used == ExecutionBackend::CPU_LIBZSTD,
              "1MB HOST data should route to CPU");

  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  s = engine.decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST);
  TEST_ASSERT(s == Status::SUCCESS, "1MB decompress failed");
  TEST_ASSERT(decomp_size == data_size, "1MB size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "1MB data mismatch");
  return true;
}

// ============================================================================
// Test 25: C API
// ============================================================================

static bool test_c_api() {
  const size_t data_size = 8 * 1024;
  auto original = make_repetitive(data_size);

  // Create engine
  cuda_zstd_hybrid_engine_t *engine = cuda_zstd_hybrid_create_default();
  TEST_ASSERT(engine != nullptr, "C API create_default returned null");

  // Max compressed size
  size_t comp_cap = cuda_zstd_hybrid_max_compressed_size(engine, data_size);
  TEST_ASSERT(comp_cap > 0, "C API max_compressed_size returned 0");

  // Query routing
  unsigned int routing = cuda_zstd_hybrid_query_routing(
      engine, data_size, 0 /*HOST*/, 0 /*HOST*/, 1 /*compression*/);
  TEST_ASSERT(routing == 0 /*CPU_LIBZSTD*/,
              "C API query_routing: HOST->HOST should be CPU");

  // Compress
  std::vector<uint8_t> compressed(comp_cap);
  size_t comp_size = comp_cap;
  cuda_zstd_hybrid_result_t result = {};

  int ret = cuda_zstd_hybrid_compress(engine, original.data(), data_size,
                                      compressed.data(), &comp_size,
                                      0 /*HOST*/, 0 /*HOST*/, &result, 0);
  TEST_ASSERT(ret == 0, "C API compress failed");
  TEST_ASSERT(comp_size > 0, "C API compressed size is 0");
  TEST_ASSERT(result.backend_used == 0 /*CPU_LIBZSTD*/,
              "C API should route HOST->HOST to CPU");
  TEST_ASSERT(result.total_time_ms > 0.0, "C API total_time_ms should be > 0");

  // Decompress
  std::vector<uint8_t> decompressed(data_size);
  size_t decomp_size = data_size;
  cuda_zstd_hybrid_result_t decomp_result = {};

  ret = cuda_zstd_hybrid_decompress(engine, compressed.data(), comp_size,
                                    decompressed.data(), &decomp_size,
                                    0 /*HOST*/, 0 /*HOST*/, &decomp_result, 0);
  TEST_ASSERT(ret == 0, "C API decompress failed");
  TEST_ASSERT(decomp_size == data_size, "C API decompressed size mismatch");
  TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
              "C API data mismatch");

  // Create with config
  cuda_zstd_hybrid_config_t cfg = {};
  cfg.mode = 3; // FORCE_CPU
  cfg.compression_level = 5;
  cuda_zstd_hybrid_engine_t *engine2 = cuda_zstd_hybrid_create(&cfg);
  TEST_ASSERT(engine2 != nullptr, "C API create with config returned null");

  // Destroy
  cuda_zstd_hybrid_destroy(engine2);
  cuda_zstd_hybrid_destroy(engine);

  return true;
}

// ============================================================================
// Test 26: Multiple compression levels
// ============================================================================

static bool test_compression_levels() {
  const size_t data_size = 8 * 1024;
  auto original = make_repetitive(data_size);

  int levels[] = {1, 3, 9, 15};
  for (int level : levels) {
    auto eng = create_hybrid_engine(level);
    size_t comp_cap = eng->get_max_compressed_size(data_size);
    std::vector<uint8_t> compressed(comp_cap);
    size_t comp_size = comp_cap;

    Status s = eng->compress(original.data(), data_size, compressed.data(),
                             &comp_size, DataLocation::HOST,
                             DataLocation::HOST);
    TEST_ASSERT(s == Status::SUCCESS, "compress at level failed");

    std::vector<uint8_t> decompressed(data_size);
    size_t decomp_size = data_size;
    s = eng->decompress(compressed.data(), comp_size, decompressed.data(),
                        &decomp_size, DataLocation::HOST, DataLocation::HOST);
    TEST_ASSERT(s == Status::SUCCESS, "decompress at level failed");
    TEST_ASSERT(memcmp(original.data(), decompressed.data(), data_size) == 0,
                "data mismatch at level");
  }
  return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "\n========================================\n";
  std::cout << "  Test: Hybrid CPU/GPU Engine\n";
  std::cout << "========================================\n\n";

  std::cout << "--- Data Location Roundtrips ---\n";
  RUN_TEST(test_host_to_host_roundtrip);
  RUN_TEST(test_device_to_device_roundtrip);
  RUN_TEST(test_host_to_device_roundtrip);
  RUN_TEST(test_device_to_host_roundtrip);

  std::cout << "\n--- Hybrid Modes ---\n";
  RUN_TEST(test_force_cpu_mode);
  RUN_TEST(test_force_gpu_mode);
  RUN_TEST(test_prefer_cpu_mode);
  RUN_TEST(test_prefer_gpu_mode);
  RUN_TEST(test_adaptive_mode);

  std::cout << "\n--- Detection & Queries ---\n";
  RUN_TEST(test_detect_location);
  RUN_TEST(test_query_routing);
  RUN_TEST(test_max_compressed_size);

  std::cout << "\n--- Cross-Path Roundtrips ---\n";
  RUN_TEST(test_cross_path_gpu_compress_cpu_decompress);
  RUN_TEST(test_cross_path_cpu_compress_gpu_decompress);

  std::cout << "\n--- Configuration ---\n";
  RUN_TEST(test_configuration);
  RUN_TEST(test_statistics);
  RUN_TEST(test_factory_functions);
  RUN_TEST(test_convenience_functions);

  std::cout << "\n--- Batch & Patterns ---\n";
  RUN_TEST(test_batch_operations);
  RUN_TEST(test_zero_size_input);
  RUN_TEST(test_move_semantics);
  RUN_TEST(test_result_timing);
  RUN_TEST(test_multiple_data_patterns);
  RUN_TEST(test_larger_data);

  std::cout << "\n--- API ---\n";
  RUN_TEST(test_c_api);
  RUN_TEST(test_compression_levels);

  std::cout << "\n========================================\n";
  std::cout << "  Results: " << g_pass << " passed, " << g_fail << " failed\n";
  std::cout << "========================================\n\n";

  return (g_fail > 0) ? 1 : 0;
}
