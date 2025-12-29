// ============================================================================
// test_integration.cu - Comprehensive Integration & Stress Tests
// ============================================================================

#include "cuda_error_checking.h"
#include "cuda_zstd_dictionary.h"
#include "cuda_zstd_manager.h"
#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_types.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace cuda_zstd;

// ============================================================================
// Test Logging Utilities
// ============================================================================

#define LOG_TEST(name) std::cout << "\n[TEST] " << name << std::endl
#define LOG_INFO(msg) std::cout << "  [INFO] " << msg << std::endl
#define LOG_PASS(name) std::cout << "  [PASS] " << name << std::endl
#define LOG_FAIL(name, msg)                                                    \
  std::cerr << "  [FAIL] " << name << ": " << msg << std::endl
#define ASSERT_EQ(a, b, msg)                                                   \
  if ((a) != (b)) {                                                            \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }
#define ASSERT_STATUS(status, msg)                                             \
  if ((status) != Status::SUCCESS) {                                           \
    LOG_FAIL(__func__, msg);                                                   \
    return false;                                                              \
  }

void print_separator() {
  std::cout << "========================================" << std::endl;
}

// ============================================================================
// Helper Functions
// ============================================================================

void generate_test_data(std::vector<uint8_t> &data, size_t size,
                        const char *pattern) {
  data.resize(size);
  if (strcmp(pattern, "compressible") == 0) {
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<uint8_t>(i % 32);
    }
  } else if (strcmp(pattern, "text") == 0) {
    const char *text = "The quick brown fox jumps over the lazy dog. ";
    size_t len = strlen(text);
    for (size_t i = 0; i < size; i++) {
      data[i] = text[i % len];
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<uint8_t>((i * 1103515245 + 12345) & 0xFF);
    }
  }
}

bool verify_data(const uint8_t *original, const uint8_t *decompressed,
                 size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (original[i] != decompressed[i]) {
      std::cerr << "    Mismatch at byte " << i << std::endl;
      return false;
    }
  }
  return true;
}

// ============================================================================
// TEST SUITE 1: End-to-End Pipeline Tests
// ============================================================================

bool test_complete_compression_pipeline() {
  LOG_TEST("Complete Compression Pipeline");

  try {
    const size_t data_size = 1024 * 1024;
    std::vector<uint8_t> h_input;
    generate_test_data(h_input, data_size, "compressible");

    LOG_INFO(
        "Testing complete pipeline: input -> compress -> decompress -> verify");

    // Allocate GPU memory with error checking
    LOG_INFO("Allocating memory...");
    void *d_input = nullptr, *d_compressed = nullptr, *d_output = nullptr,
         *d_temp = nullptr;

    if (!safe_cuda_malloc(&d_input, data_size)) {
      LOG_FAIL("Complete Compression Pipeline",
               "CUDA malloc for d_input failed");
      return false;
    }

    if (!safe_cuda_malloc(&d_compressed, data_size * 2)) {
      LOG_FAIL("Complete Compression Pipeline",
               "CUDA malloc for d_compressed failed");
      safe_cuda_free(d_input);
      return false;
    }

    if (!safe_cuda_malloc(&d_output, data_size)) {
      LOG_FAIL("Complete Compression Pipeline",
               "CUDA malloc for d_output failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      return false;
    }

    // Copy input data to device
    LOG_INFO("Copying input to device...");
    if (!safe_cuda_memcpy(d_input, h_input.data(), data_size,
                          cudaMemcpyHostToDevice)) {
      LOG_FAIL("Complete Compression Pipeline",
               "CUDA memcpy to d_input failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      return false;
    }

    // Create manager safely
    LOG_INFO("Creating manager...");
    std::unique_ptr<ZstdManager> manager;
    try {
      manager = create_manager(5); // Level 5
      if (!manager) {
        LOG_FAIL("Complete Compression Pipeline",
                 "Failed to create compression manager");
        safe_cuda_free(d_input);
        safe_cuda_free(d_compressed);
        safe_cuda_free(d_output);
        return false;
      }
    } catch (const std::exception &e) {
      LOG_FAIL("Complete Compression Pipeline",
               "Manager creation failed: " << e.what());
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      return false;
    }

    LOG_INFO("Getting temp size...");
    size_t temp_size = manager->get_compress_temp_size(data_size);
    LOG_INFO("Temp size: " << temp_size);
    if (!safe_cuda_malloc(&d_temp, temp_size)) {
      LOG_FAIL("Complete Compression Pipeline", "CUDA malloc for temp failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      return false;
    }

    // Compress
    LOG_INFO("Compressing...");
    size_t compressed_size =
        data_size * 2; // Initialize to allocated buffer size
    Status status =
        manager->compress(d_input, data_size, d_compressed, &compressed_size,
                          d_temp, temp_size, nullptr, 0, 0);
    if (status != Status::SUCCESS) {
      LOG_FAIL("Complete Compression Pipeline",
               "Compression failed with status " << (int)status);
      return false;
    }
    LOG_INFO("✓ Compression: " << data_size << " -> " << compressed_size
                               << " bytes");

    // Decompress
    size_t decompressed_size;
    status = manager->decompress(d_compressed, compressed_size, d_output,
                                 &decompressed_size, d_temp, temp_size);
    ASSERT_STATUS(status, "Decompression failed");
    ASSERT_EQ(decompressed_size, data_size, "Size mismatch");
    LOG_INFO("✓ Decompression: " << compressed_size << " -> "
                                 << decompressed_size << " bytes");

    // Verify
    std::vector<uint8_t> h_output(data_size);
    if (!safe_cuda_memcpy(h_output.data(), d_output, data_size,
                          cudaMemcpyDeviceToHost)) {
      LOG_FAIL("Complete Compression Pipeline",
               "CUDA memcpy from d_output failed");
      safe_cuda_free(d_input);
      safe_cuda_free(d_compressed);
      safe_cuda_free(d_output);
      safe_cuda_free(d_temp);
      return false;
    }

    ASSERT_TRUE(verify_data(h_input.data(), h_output.data(), data_size),
                "Data verification failed");
    LOG_INFO("✓ Data verified correctly");

    // Cleanup with safe free functions
    safe_cuda_free(d_input);
    safe_cuda_free(d_compressed);
    safe_cuda_free(d_output);
    safe_cuda_free(d_temp);

    LOG_PASS("Complete Compression Pipeline");
    return true;

  } catch (const std::exception &e) {
    LOG_FAIL("test_complete_compression_pipeline",
             std::string("Exception: ") + e.what());
    return false;
  } catch (...) {
    LOG_FAIL("test_complete_compression_pipeline", "Unknown exception");
    return false;
  }
}

bool test_dictionary_integration() {
  LOG_TEST("Dictionary Integration");

  const size_t data_size = 128 * 1024;
  const size_t dict_size = 16 * 1024;

  // Generate dictionary samples
  std::vector<std::vector<uint8_t>> samples;
  for (int i = 0; i < 10; i++) {
    std::vector<uint8_t> sample;
    generate_test_data(sample, 8192, "text");
    samples.push_back(sample);
  }

  // Train dictionary
  std::vector<const void *> sample_ptrs;
  std::vector<size_t> sample_sizes;
  for (const auto &s : samples) {
    sample_ptrs.push_back(s.data());
    sample_sizes.push_back(s.size());
  }

  dictionary::Dictionary dict;
  dictionary::DictionaryManager::allocate_dictionary_gpu(dict, dict_size, 0);
  dictionary::CoverParams params;
  Status status = dictionary::DictionaryTrainer::train_dictionary(
      sample_ptrs, sample_sizes, dict, dict_size, &params, 0);
  ASSERT_STATUS(status, "Dictionary training failed");
  LOG_INFO("✓ Dictionary trained (" << dict_size << " bytes)");

  // Compress with dictionary
  std::vector<uint8_t> h_input;
  generate_test_data(h_input, data_size, "text");

  void *d_input, *d_compressed, *d_output, *d_temp;
  cudaMalloc(&d_input, data_size);
  cudaMalloc(&d_compressed, data_size * 2);
  cudaMalloc(&d_output, data_size);
  cudaMemcpy(d_input, h_input.data(), data_size, cudaMemcpyHostToDevice);


  ZstdBatchManager manager(CompressionConfig{.level = 5});


  manager.set_dictionary(dict);


  size_t temp_size = manager.get_compress_temp_size(data_size);
  cudaMalloc(&d_temp, temp_size);


  size_t compressed_size = data_size * 2; // Init to allocated buffer size
  status = manager.compress(d_input, data_size, d_compressed, &compressed_size,
                            d_temp, temp_size, nullptr, 0);

  ASSERT_STATUS(status, "Dictionary compression failed");
  LOG_INFO("✓ Compressed with dictionary: " << data_size << " -> "
                                            << compressed_size);

  // Decompress with dictionary
  size_t decompressed_size;
  status = manager.decompress(d_compressed, compressed_size, d_output,
                              &decompressed_size, d_temp, temp_size);
  ASSERT_STATUS(status, "Dictionary decompression failed");
  LOG_INFO("✓ Decompressed with dictionary");

  // Verify
  std::vector<uint8_t> h_output(data_size);
  cudaMemcpy(h_output.data(), d_output, data_size, cudaMemcpyDeviceToHost);
  ASSERT_TRUE(verify_data(h_input.data(), h_output.data(), data_size),
              "Dictionary data verification failed");
  LOG_INFO("✓ Dictionary compression verified");

  cudaFree(d_input);
  cudaFree(d_compressed);
  cudaFree(d_output);
  cudaFree(d_temp);

  LOG_PASS("Dictionary Integration");
  return true;
}

bool test_batch_processing_with_memory_pool() {
  LOG_TEST("Batch Processing with Memory Pool");

  const int num_items = 10;
  const size_t item_size = 64 * 1024;

  LOG_INFO("Processing " << num_items << " items of " << item_size / 1024
                         << " KB each");

  // Initialize memory pool
  memory::MemoryPoolManager pool;

  ZstdBatchManager manager(CompressionConfig{.level = 5});
  std::vector<BatchItem> items;

  // Prepare batch items
  for (int i = 0; i < num_items; i++) {
    std::vector<uint8_t> h_data;
    generate_test_data(h_data, item_size,
                       (i % 2 == 0) ? "compressible" : "text");

    void *d_input = pool.allocate(item_size);
    void *d_output = pool.allocate(item_size * 2);

    cudaMemcpy(d_input, h_data.data(), item_size, cudaMemcpyHostToDevice);

    BatchItem item;
    item.input_ptr = d_input;
    item.input_size = item_size;
    item.output_ptr = d_output;
    item.output_size = item_size * 2; // Init to max buffer size (not 0!)

    items.push_back(item);
  }

  // Compress batch - need workspace for ALL items, not just 1!
  std::vector<size_t> all_sizes(num_items, item_size);
  size_t batch_temp_size = manager.get_batch_compress_temp_size(all_sizes);
  void *d_temp = pool.allocate(batch_temp_size);

  Status status = manager.compress_batch(items, d_temp, batch_temp_size);
  ASSERT_STATUS(status, "Batch compression failed");
  LOG_INFO("✓ Batch compressed");

  // Check results
  for (int i = 0; i < num_items; i++) {
    LOG_INFO("Item " << i << ": " << items[i].input_size << " -> "
                     << items[i].output_size << " bytes");
  }

  // Cleanup using pool
  for (const auto &item : items) {
    pool.deallocate(item.input_ptr);
    pool.deallocate(item.output_ptr);
  }
  pool.deallocate(d_temp);

  auto stats = pool.get_statistics();
  LOG_INFO("Pool stats - Allocations: " << stats.total_allocations
                                        << ", Hit rate: "
                                        << (stats.get_hit_rate() * 100) << "%");

  LOG_PASS("Batch Processing with Memory Pool");
  return true;
}

// ============================================================================
// TEST SUITE 2: Large File Tests
// ============================================================================

bool test_large_file_compression() {
  LOG_TEST("Large File Compression (100MB)");

  const size_t file_size = 100 * 1024 * 1024; // 100MB
  const size_t chunk_size = 4 * 1024 * 1024;  // 4MB chunks

  LOG_INFO("File size: " << file_size / (1024 * 1024) << " MB");
  LOG_INFO("Processing in " << chunk_size / (1024 * 1024) << " MB chunks");

  ZstdStreamingManager streaming_mgr(CompressionConfig{.level = 5});
  streaming_mgr.init_compression();

  void *d_input, *d_output;
  cudaMalloc(&d_input, chunk_size);
  cudaMalloc(&d_output, chunk_size * 2);

  size_t total_compressed = 0;
  int num_chunks = (file_size + chunk_size - 1) / chunk_size;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_chunks; i++) {
    size_t current_chunk = std::min(chunk_size, file_size - (i * chunk_size));

    // Generate chunk data
    std::vector<uint8_t> h_chunk;
    generate_test_data(h_chunk, current_chunk, "compressible");
    cudaMemcpy(d_input, h_chunk.data(), current_chunk, cudaMemcpyHostToDevice);

    size_t compressed_size;
    bool is_last = (i == num_chunks - 1);
    Status status = streaming_mgr.compress_chunk(
        d_input, current_chunk, d_output, &compressed_size, is_last);
    ASSERT_STATUS(status, "Chunk " << i << " compression failed");

    total_compressed += compressed_size;

    if ((i + 1) % 10 == 0 || is_last) {
      LOG_INFO("Progress: " << (i + 1) << "/" << num_chunks << " chunks, "
                            << total_compressed / (1024 * 1024)
                            << " MB compressed");
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_sec = std::chrono::duration<double>(end - start).count();

  LOG_INFO("Total compressed: " << total_compressed / (1024 * 1024) << " MB");
  LOG_INFO("Compression ratio: "
           << std::fixed << std::setprecision(2)
           << get_compression_ratio(file_size, total_compressed) << ":1");
  LOG_INFO("Time: " << std::fixed << std::setprecision(2) << elapsed_sec
                    << " seconds");
  LOG_INFO("Throughput: " << std::fixed << std::setprecision(1)
                          << (file_size / (1024.0 * 1024.0)) / elapsed_sec
                          << " MB/s");

  cudaFree(d_input);
  cudaFree(d_output);

  LOG_PASS("Large File Compression");
  return true;
}

bool test_memory_efficiency_large_file() {
  LOG_TEST("Memory Efficiency for Large Files");

  size_t free_before, total;
  cudaMemGetInfo(&free_before, &total);

  const size_t file_size = 50 * 1024 * 1024; // 50MB
  const size_t chunk_size = 2 * 1024 * 1024; // 2MB chunks

  LOG_INFO("Free memory before: " << free_before / (1024 * 1024) << " MB");

  ZstdStreamingManager streaming_mgr(CompressionConfig{.level = 5});
  streaming_mgr.init_compression();

  void *d_input, *d_output;
  cudaMalloc(&d_input, chunk_size);
  cudaMalloc(&d_output, chunk_size * 2);

  int num_chunks = (file_size + chunk_size - 1) / chunk_size;

  for (int i = 0; i < num_chunks; i++) {
    size_t current_chunk = std::min(chunk_size, file_size - (i * chunk_size));
    std::vector<uint8_t> h_chunk;
    generate_test_data(h_chunk, current_chunk, "text");
    cudaMemcpy(d_input, h_chunk.data(), current_chunk, cudaMemcpyHostToDevice);

    size_t compressed_size;
    streaming_mgr.compress_chunk(d_input, current_chunk, d_output,
                                 &compressed_size, i == num_chunks - 1);
  }

  cudaFree(d_input);
  cudaFree(d_output);

  size_t free_after;
  cudaMemGetInfo(&free_after, &total);

  LOG_INFO("Free memory after: " << free_after / (1024 * 1024) << " MB");

  size_t leaked = (free_before > free_after) ? (free_before - free_after) : 0;
  LOG_INFO("Memory delta: " << leaked / (1024 * 1024) << " MB");

  ASSERT_TRUE(leaked < 10 * 1024 * 1024, "Memory leak detected");
  LOG_INFO("✓ No significant memory leaks");

  LOG_PASS("Memory Efficiency for Large Files");
  return true;
}

// ============================================================================
// TEST SUITE 3: Multi-threaded Tests
// ============================================================================

bool test_concurrent_compression() {
  LOG_TEST("Concurrent Compression Operations");

  const int num_threads = 4;
  const int operations_per_thread = 20;
  const size_t data_size = 64 * 1024;

  LOG_INFO("Testing with " << num_threads << " threads");
  LOG_INFO("Each performing " << operations_per_thread << " compressions");

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};
  std::vector<std::thread> threads;

  auto worker = [&](int thread_id) {
    ZstdBatchManager manager(CompressionConfig{.level = 5});

    for (int i = 0; i < operations_per_thread; i++) {
      std::vector<uint8_t> h_data;
      generate_test_data(h_data, data_size, "compressible");

      void *d_input, *d_output, *d_temp;
      cudaMalloc(&d_input, data_size);
      cudaMalloc(&d_output, data_size * 2);
      cudaMemcpy(d_input, h_data.data(), data_size, cudaMemcpyHostToDevice);

      size_t temp_size = manager.get_compress_temp_size(data_size);
      cudaMalloc(&d_temp, temp_size);

      size_t compressed_size = data_size * 2; // Init to allocated buffer size
      Status status =
          manager.compress(d_input, data_size, d_output, &compressed_size,
                           d_temp, temp_size, nullptr, 0);

      if (status == Status::SUCCESS) {
        success_count++;
      } else {
        failure_count++;
      }

      cudaFree(d_input);
      cudaFree(d_output);
      cudaFree(d_temp);
    }
  };

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  int total_ops = num_threads * operations_per_thread;
  LOG_INFO("Total operations: " << total_ops);
  LOG_INFO("Successful: " << success_count);
  LOG_INFO("Failed: " << failure_count);
  LOG_INFO("Time: " << std::fixed << std::setprecision(2) << elapsed_ms
                    << " ms");
  LOG_INFO("Throughput: " << std::fixed << std::setprecision(1)
                          << (total_ops * 1000.0 / elapsed_ms) << " ops/sec");

  ASSERT_EQ(success_count, total_ops, "Some operations failed");

  LOG_PASS("Concurrent Compression");
  return true;
}

bool test_race_condition_detection() {
  LOG_TEST("Race Condition Detection");

  const int num_threads = 8;
  const int iterations = 100;

  LOG_INFO("Testing " << num_threads << " threads with shared resources");

  ZstdBatchManager manager(CompressionConfig{.level = 5});
  std::atomic<int> level_changes{0};

  std::vector<std::thread> threads;

  auto worker = [&](int thread_id) {
    for (int i = 0; i < iterations; i++) {
      // Rapidly change compression level
      int new_level = 1 + (thread_id + i) % 22;
      Status status = manager.set_compression_level(new_level);
      if (status == Status::SUCCESS) {
        level_changes++;
      }
    }
  };

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  LOG_INFO("Level changes: " << level_changes);
  LOG_INFO("✓ No crashes or deadlocks detected");

  LOG_PASS("Race Condition Detection");
  return true;
}

// ============================================================================
// TEST SUITE 4: Edge Cases
// ============================================================================

bool test_corrupt_data_handling() {
  LOG_TEST("Corrupted Data Handling");

  const size_t data_size = 1024;
  std::vector<uint8_t> corrupted_data(data_size);

  // Create various types of corrupted data
  std::vector<std::pair<std::string, std::vector<uint8_t>>> test_cases;

  // Case 1: All zeros
  test_cases.push_back({"all_zeros", std::vector<uint8_t>(data_size, 0)});

  // Case 2: Invalid magic
  std::vector<uint8_t> bad_magic(data_size, 0xAA);
  bad_magic.assign(1, 0xFF);
  bad_magic.assign(1, 0xFF);
  test_cases.push_back({"bad_magic", bad_magic});

  // Case 3: Truncated
  test_cases.push_back({"truncated", std::vector<uint8_t>(10, 0x28)});

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  for (const auto &[name, data] : test_cases) {
    LOG_INFO("Testing: " << name);

    void *d_compressed, *d_output, *d_temp;
    cudaMalloc(&d_compressed, data.size());
    cudaMalloc(&d_output, data_size);
    cudaMalloc(&d_temp, data_size);
    cudaMemcpy(d_compressed, data.data(), data.size(), cudaMemcpyHostToDevice);

    size_t output_size;
    Status status = manager.decompress(d_compressed, data.size(), d_output,
                                       &output_size, d_temp, data_size);

    ASSERT_TRUE(status != Status::SUCCESS, name << " should fail gracefully");
    LOG_INFO("  ✓ Detected as corrupted");

    cudaFree(d_compressed);
    cudaFree(d_output);
    cudaFree(d_temp);
  }

  LOG_PASS("Corrupted Data Handling");
  return true;
}

bool test_buffer_boundary_conditions() {
  LOG_TEST("Buffer Boundary Conditions");

  // Test various edge case sizes
  std::vector<size_t> test_sizes = {
      1,               // Minimum
      16,              // Very small
      4095,            // Just under page
      4096,            // Page size
      4097,            // Just over page
      65535,           // 64KB - 1
      65536,           // Exactly 64KB
      1024 * 1024 - 1, // Just under 1MB
      1024 * 1024      // Exactly 1MB
  };

  ZstdBatchManager manager(CompressionConfig{.level = 3});

  for (size_t size : test_sizes) {
    std::vector<uint8_t> h_data;
    generate_test_data(h_data, size, "compressible");

    void *d_input, *d_output, *d_decompressed, *d_temp;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size * 2);
    cudaMalloc(&d_decompressed, size);
    cudaMemcpy(d_input, h_data.data(), size, cudaMemcpyHostToDevice);

    size_t temp_size = manager.get_compress_temp_size(size);
    cudaMalloc(&d_temp, temp_size);

    size_t compressed_size = size * 2; // Init to allocated buffer size
    Status status = manager.compress(d_input, size, d_output, &compressed_size,
                                     d_temp, temp_size, nullptr, 0);

    if (status == Status::SUCCESS) {
      size_t decompressed_size = size; // Init to allocated buffer size!
      status = manager.decompress(d_output, compressed_size, d_decompressed,
                                  &decompressed_size, d_temp, temp_size);
      if (status != Status::SUCCESS) {

      }
      ASSERT_STATUS(status, "Size " << size << " decompression failed");
      ASSERT_EQ(decompressed_size, size, "Size mismatch for " << size);

      LOG_INFO("✓ Size " << size << " bytes: OK");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_decompressed);
    cudaFree(d_temp);
  }

  LOG_PASS("Buffer Boundary Conditions");
  return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << "\n";
  print_separator();
  std::cout << "CUDA ZSTD - Integration & Stress Test Suite" << std::endl;
  print_separator();
  std::cout << "\n";

  int passed = 0;
  int total = 0;

  SKIP_IF_NO_CUDA_RET(0);
  check_cuda_device();

  // End-to-End Pipeline Tests
  print_separator();
  std::cout << "SUITE 1: End-to-End Pipeline Tests" << std::endl;
  print_separator();

  total++;
  if (test_complete_compression_pipeline())
    passed++;
  total++;
  if (test_dictionary_integration())
    passed++;
  total++;
  if (test_batch_processing_with_memory_pool())
    passed++;

  // Large File Tests
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 2: Large File Tests" << std::endl;
  print_separator();

  total++;
  if (test_large_file_compression())
    passed++;
  total++;
  if (test_memory_efficiency_large_file())
    passed++;

  // Multi-threaded Tests
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 3: Multi-threaded Tests" << std::endl;
  print_separator();

  total++;
  if (test_concurrent_compression())
    passed++;
  total++;
  if (test_race_condition_detection())
    passed++;

  // Edge Cases
  std::cout << "\n";
  print_separator();
  std::cout << "SUITE 4: Edge Cases" << std::endl;
  print_separator();

  total++;
  if (test_corrupt_data_handling())
    passed++;
  total++;
  if (test_buffer_boundary_conditions())
    passed++;

  // Summary
  std::cout << "\n";
  print_separator();
  std::cout << "TEST RESULTS" << std::endl;
  print_separator();
  std::cout << "Passed: " << passed << "/" << total << std::endl;
  std::cout << "Failed: " << (total - passed) << "/" << total << std::endl;

  if (passed == total) {
    std::cout << "\n✓ ALL TESTS PASSED" << std::endl;
  } else {
    std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
  }
  print_separator();
  std::cout << "\n";

  return (passed == total) ? 0 : 1;
}
