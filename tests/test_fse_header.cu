/**
 * @file test_fse_header.cu
 * @brief Unit tests for FSE Header Parsing (Normalized Counts)
 *
 * Tests the real read_fse_header() from cuda_zstd_internal.h which reads
 * normalized frequency distributions from RFC 8878 compliant FSE header
 * bitstreams.
 */

#include "cuda_zstd_fse.h"
#include "cuda_zstd_internal.h"
#include "cuda_zstd_types.h"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cuda_zstd;
using namespace cuda_zstd::fse;

// Simple helper to print errors
#define CHECK(cond)                                                            \
  if (!(cond)) {                                                               \
    std::cerr << "Test failed at line " << __LINE__ << ": " << #cond           \
              << std::endl;                                                    \
    exit(1);                                                                   \
  }

/**
 * @brief Test basic FSE header structure validation
 *
 * This test validates that:
 * 1. Invalid inputs are rejected gracefully
 * 2. A known valid FSE header is correctly parsed
 */
void test_header_format_validation() {
  std::cout << "Testing FSE Header Format Validation..." << std::endl;

  // Test 1: Empty input should return ERROR_CORRUPT_DATA
  {
    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    // Test with zero size (non-null pointer to avoid crash)
    unsigned char dummy = 0;
    Status status =
        read_fse_header(&dummy, 0, normalized, &max_symbol, &table_log,
                        &bytes_read);
    CHECK(status != Status::SUCCESS);

    std::cout << "  - Empty input handling: PASS" << std::endl;
  }

  // Test 2: Valid FSE header — two symbols each with probability 16
  //
  // RFC 8878 FSE header format (bitstream, LSB-first):
  //   Bits 0-3:  Accuracy_Log = 0  =>  table_log = 0 + 5 = 5
  //   Bits 4-8:  val = 17  (nCount = 16, prob = 16)  =>  sym0 = 16
  //   Bits 9-12: raw = 15, bit 13: extra = 1  =>  val = 17, sym1 = 16
  //
  // Encoded as bytes: {0x10, 0x3F}
  {
    std::vector<unsigned char> header = {0x10, 0x3F};

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(header.data(), (u32)header.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    CHECK(status == Status::SUCCESS);
    CHECK(table_log == 5);
    CHECK(!normalized.empty());
    CHECK(normalized.size() == 2);
    CHECK(normalized[0] == 16);
    CHECK(normalized[1] == 16);
    CHECK(max_symbol == 1);
    CHECK(bytes_read == 2);

    // Verify normalized counts sum to 2^table_log
    u32 sum = 0;
    for (auto count : normalized) {
      if (count == (u16)-1)
        sum += 1; // Low probability symbol uses 1 slot
      else
        sum += count;
    }
    CHECK(sum == (1u << table_log));

    std::cout << "  - Valid header parsing (2-sym uniform): PASS" << std::endl;
  }
}

/**
 * @brief Test FSE header with a low-probability symbol (nCount = -1)
 *
 * RFC 8878: A normalized count of -1 (stored as (u16)-1 = 65535)
 * means the symbol has probability less than 1 and occupies exactly
 * 1 slot in the decode table.
 *
 * Bitstream: {0x00, 0x7E}
 *   Bits 0-3:  Accuracy_Log = 0  =>  table_log = 5
 *   Bits 4-8:  val = 0  =>  nCount = -1  =>  sym0 prob = 1 (low-prob marker)
 *   Bits 9-13: raw = 31, bit 14: extra = 1  =>  val = 32, nCount = 31  =>  sym1 = 31
 */
void test_low_probability_symbol() {
  std::cout << "Testing Low Probability Symbol (nCount=-1)..." << std::endl;

  std::vector<unsigned char> header = {0x00, 0x7E};

  std::vector<u16> normalized;
  u32 max_symbol = 0;
  u32 table_log = 0;
  u32 bytes_read = 0;

  Status status =
      read_fse_header(header.data(), (u32)header.size(), normalized,
                      &max_symbol, &table_log, &bytes_read);

  CHECK(status == Status::SUCCESS);
  CHECK(table_log == 5);
  CHECK(normalized.size() == 2);
  CHECK(normalized[0] == (u16)-1); // Low probability marker
  CHECK(normalized[1] == 31);
  CHECK(max_symbol == 1);
  CHECK(bytes_read == 2);

  // Verify sum: -1 prob uses 1 slot, sym1 uses 31 slots = 32 total
  u32 sum = 0;
  for (auto count : normalized) {
    if (count == (u16)-1)
      sum += 1;
    else
      sum += count;
  }
  CHECK(sum == (1u << table_log));

  std::cout << "  - Low probability symbol: PASS" << std::endl;
}

/**
 * @brief Test error handling for corrupted headers
 */
void test_corrupted_header_handling() {
  std::cout << "Testing Corrupted Header Handling..." << std::endl;

  // Test 1: Header with table_log > 15 (accuracy_log > 10)
  //   Accuracy_log of 11 => table_log = 16 => should be rejected
  //   Bits 0-3 = 11 = 0xB. Byte = 0x0B.
  {
    std::vector<unsigned char> header = {0x0B, 0x00};

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(header.data(), (u32)header.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    CHECK(status != Status::SUCCESS);
    std::cout << "  - Table log overflow detection: PASS" << std::endl;
  }

  // Test 2: Pseudo-random corrupted data
  {
    std::vector<unsigned char> corrupted(100);
    for (size_t i = 0; i < corrupted.size(); ++i) {
      corrupted[i] = static_cast<unsigned char>(i * 7 + 13);
    }

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(corrupted.data(), (u32)corrupted.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    // Should fail gracefully or at least not crash.
    // If it happens to succeed, verify output is sane.
    if (status != Status::SUCCESS) {
      std::cout << "  - Corrupted data detection: PASS" << std::endl;
    } else {
      CHECK(table_log >= 5 && table_log <= 15);
      std::cout << "  - Corrupted data handling (unexpected success but valid): "
                   "PASS"
                << std::endl;
    }
  }
}

/**
 * @brief Test normalized counts accuracy with known distributions
 */
void test_normalized_counts_accuracy() {
  std::cout << "Testing Normalized Counts Accuracy..." << std::endl;

  // Use the known-good test vector: {0x10, 0x3F}
  // Expected: table_log=5, counts=[16, 16], sum=32
  {
    std::vector<unsigned char> header = {0x10, 0x3F};

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(header.data(), (u32)header.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    CHECK(status == Status::SUCCESS);
    CHECK(table_log == 5);

    // Verify exactly 2 symbols with count 16 each
    CHECK(normalized.size() == 2);
    CHECK(normalized[0] == 16);
    CHECK(normalized[1] == 16);

    // Verify sum = 2^5 = 32
    u32 sum = 0;
    for (auto c : normalized) {
      sum += (c == (u16)-1) ? 1 : c;
    }
    CHECK(sum == 32);

    std::cout << "  - Uniform 2-symbol distribution: PASS" << std::endl;
  }

  // Test the low-prob + large-prob vector: {0x00, 0x7E}
  // Expected: table_log=5, counts=[(u16)-1, 31], effective sum=1+31=32
  {
    std::vector<unsigned char> header = {0x00, 0x7E};

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(header.data(), (u32)header.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    CHECK(status == Status::SUCCESS);
    CHECK(table_log == 5);
    CHECK(normalized.size() == 2);

    // sym0 is low-prob (-1 marker), sym1 = 31
    CHECK(normalized[0] == (u16)-1);
    CHECK(normalized[1] == 31);

    u32 sum = 0;
    for (auto c : normalized) {
      sum += (c == (u16)-1) ? 1 : c;
    }
    CHECK(sum == 32);

    std::cout << "  - Skewed distribution with low-prob symbol: PASS"
              << std::endl;
  }
}

/**
 * @brief Test that bytes_read output is correct
 */
void test_bytes_read_tracking() {
  std::cout << "Testing Bytes Read Tracking..." << std::endl;

  // Both test vectors are 2 bytes and should report bytes_read == 2
  {
    std::vector<unsigned char> header = {0x10, 0x3F};

    std::vector<u16> normalized;
    u32 max_symbol = 0;
    u32 table_log = 0;
    u32 bytes_read = 0;

    Status status =
        read_fse_header(header.data(), (u32)header.size(), normalized,
                        &max_symbol, &table_log, &bytes_read);

    CHECK(status == Status::SUCCESS);
    CHECK(bytes_read == 2);

    std::cout << "  - Bytes read tracking: PASS" << std::endl;
  }
}

/**
 * @brief Main test entry point
 */
int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "FSE Header Parsing Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    test_header_format_validation();
    test_low_probability_symbol();
    test_corrupted_header_handling();
    test_normalized_counts_accuracy();
    test_bytes_read_tracking();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All FSE Header tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
