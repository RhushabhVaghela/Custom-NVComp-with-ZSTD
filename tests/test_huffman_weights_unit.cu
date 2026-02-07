#include "cuda_zstd_huffman.h"
#include "cuda_zstd_internal.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <zstd.h>

// Helper to check CUDA errors
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err),  \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }
#endif

using namespace cuda_zstd;

void test_simple_weights() {
  printf("Running test_simple_weights...\n");

  // 1. Create compressible data (biased distribution to trigger FSE weights)
  std::vector<uint8_t> input_data;
  // Use a skewed distribution but large size
  for (int i = 0; i < 10000; i++) {
    int r = rand() % 100;
    if (r < 50)
      input_data.push_back('A');
    else if (r < 80)
      input_data.push_back('B');
    else if (r < 95)
      input_data.push_back('C');
    else
      input_data.push_back(rand() % 255);
  }

  // 2. Compress with libzstd
  std::vector<uint8_t> compressed(input_data.size() * 2);
  size_t c_size = ZSTD_compress(compressed.data(), compressed.size(),
                                input_data.data(), input_data.size(), 1);
  if (ZSTD_isError(c_size)) {
    fprintf(stderr, "ZSTD compression failed: %s\n", ZSTD_getErrorName(c_size));
    exit(1);
  }
  compressed.resize(c_size);

  printf("Compressed size: %zu\n", c_size);

  // 3. Locate the first block and Huffman header
  // Frame Header:
  // Magic (4)
  // Frame_Header_Descriptor (1)
  // Window_Descriptor (0 or 1 depending on FHD)
  // ...
  // This is brittle unless we parse properly.
  // Hack: Scan for the first byte that looks like a Huffman FSE header
  // (Accuracy Log <= 6) OR: Use libzstd to get frame header size? No public API
  // for that easily. Better: Magic Number (4) + 2 to 14 bytes. For small data,
  // it's usually Magic(4) + FHD(1) + WD(1) = 6 bytes. The Block Header is 3
  // bytes. Block Payload starts at offset ~9. Huffman Header is first byte of
  // payload.

  // Let's dump first 20 bytes to verify manually in logs if it fails
  printf("Header hex: ");
  for (int i = 0; i < 30; i++)
    printf("%02X ", compressed[i]);
  printf("\n");

  // Skip Frame Header (Assume standard simple frame: Magic+2 bytes)
  // ZSTD Magic: 28 B5 2F FD (Little Endian -> FD 2F B5 28)
  size_t offset = 4;
  uint8_t fhd = compressed[offset++];
  // If Single_Segment_Flag (bit 5) is set, Window_Descriptor is present?
  // RFC 8878:
  // FHD:
  // Dictionary_ID_Flag (2 bits)
  // Content_Checksum_Flag (1 bit)
  // Reserved_Bit (1 bit)
  // Single_Segment_Flag (1 bit)
  // Frame_Content_Size_Flag (2 bits)

  bool single_segment = (fhd >> 5) & 1;
  if (!single_segment) {
    offset++; // Window Descriptor
  }

  // Dictionary ID?
  int dict_id_flag = (fhd & 3);
  if (dict_id_flag == 1)
    offset += 1;
  else if (dict_id_flag == 2)
    offset += 2;
  else if (dict_id_flag == 3)
    offset += 4;

  // Frame Content Size?
  int fcs_flag = (fhd >> 6);
  if (fcs_flag == 0) {
    if (single_segment)
      offset += 1; // RFC 8878: FCS_Flag 0 + SS 1 means 1 byte
  } else if (fcs_flag == 1) {
    if (single_segment)
      offset += 2; // RFC 8878: Table 3 - FCS_Flag 1 + SS 1 means 2 bytes
    else
      offset += 2; // RFC 8878: Table 2 - FCS_Flag 1 + SS 0 means 2 bytes
  } else if (fcs_flag == 2)
    offset += 4; // 4 bytes
  else if (fcs_flag == 3)
    offset += 8; // 8 bytes

  printf("Estimated Frame Header End: %zu\n", offset);

  // Block Header (3 bytes)
  // Last_Block (1 bit)
  // Block_Type (2 bits)
  // Block_Size (21 bits)
  uint8_t bh0 = compressed[offset];
  uint8_t bh1 = compressed[offset + 1];
  uint8_t bh2 = compressed[offset + 2];
  offset += 3;

  int block_type = (bh0 >> 1) & 3;
  int last_block = bh0 & 1;
  printf("Block Type: %d (2=Compressed), Last: %d\n", block_type, last_block);

  if (block_type != 2) {
    fprintf(stderr, "Error: Not a compressed block (Type %d)\n", block_type);
    // It might be RLE or Raw if data is too simple, but we added noise.
    exit(1);
  }

  // Huffman Header
  // Literals_Section_Header
  uint8_t lit_header = compressed[offset];
  printf("Literals Header: %02X\n", lit_header);

  // Parse Literals Header to find Weight start
  // Huffman_Tree_Description (1 bit)? No.
  // Literals_Block_Type (2 bits)
  // Size_Format (2 bits)
  // Reg: (lit_header >> 2) & 3
  // Compressed Literals Block (2) or Treeless (1)?
  int lit_block_type = (lit_header & 3);
  printf("Lit Block Type: %d (0=Raw, 1=RLE, 2=Compressed, 3=Treeless)\n",
         lit_block_type);

  if (lit_block_type != 2) {
    fprintf(stderr, "Error: Not a Huffman compressed literals block\n");
    exit(1);
  }

  int size_format = (lit_header >> 2) & 3;
  int header_bytes = 0;

  // RFC 8878 Section 3.1.1.3.2: Compressed_Literals_Block header sizes
  // size_format 0 (single stream): 3 bytes (10-bit regen, 10-bit compressed)
  // size_format 1 (4 streams):     3 bytes (10-bit regen, 10-bit compressed)
  // size_format 2 (4 streams):     4 bytes (14-bit regen, 14-bit compressed)
  // size_format 3 (4 streams):     5 bytes (18-bit regen, 18-bit compressed)
  if (size_format == 0 || size_format == 1) {
    header_bytes = 3;
  } else if (size_format == 2) {
    header_bytes = 4;
  } else { // size_format == 3
    header_bytes = 5;
  }
  printf("Size Format: %d, Header Bytes: %d\n", size_format, header_bytes);

  offset += header_bytes; // Points to Huffman Tree header?

  // Only if using FSE table!
  // If the first bit of the tree description is 0, it's FSE.
  // If 1, it's direct weights? No.
  // Section 4.2.1:
  // <HeaderByte>
  // If HeaderByte < 128: FSE compression.
  // If HeaderByte >= 128: Direct weights.

  uint8_t huff_header_byte = compressed[offset];
  printf("Huffman Header Byte: %02X\n", huff_header_byte);

  if (huff_header_byte >= 128) {
    fprintf(stderr, "Direct weights used (uncompressed weights). Not useful "
                    "for FSE test.\n");
    // Try compressing with more entropy to force FSE?
    exit(0);
  }

  size_t fse_stream_size = huff_header_byte; // The byte IS the size
  printf("FSE Stream Size: %zu bytes\n", fse_stream_size);

  offset++; // Point to start of FSE NCount + bitstream data

  // Copy the FSE compressed data (same as how the real decoder calls it:
  //   decode_huffman_weights_fse(h_input + 1, header_byte, ...)
  // h_input+1 skips the header byte, header_byte = compressed size)
  size_t total_fse_buffer_size = fse_stream_size;
  std::vector<uint8_t> test_buffer(total_fse_buffer_size);
  memcpy(test_buffer.data(), &compressed[offset], total_fse_buffer_size);

  printf("Copied %zu bytes to test buffer.\n", total_fse_buffer_size);
  printf("FSE Data: ");
  for (size_t i = 0; i < total_fse_buffer_size; i++)
    printf("%02X ", test_buffer[i]);
  printf("\n");

  // 4. Run Decoder
  uint8_t weights[256] = {0};
  uint32_t num_symbols = 0;

  Status ret = huffman::decode_huffman_weights_fse(
      test_buffer.data(), (u32)total_fse_buffer_size, weights, &num_symbols);

  if (ret != Status::SUCCESS) {
    fprintf(stderr, "Decoding Failed with Status %d\n", (int)ret);
    exit(1);
  }

  printf("Decoded %u symbols.\n", num_symbols);
  printf("Weights: ");
  uint32_t weight_sum = 0;
  for (uint32_t i = 0; i < num_symbols; i++) {
    printf("%d ", weights[i]);
    if (weights[i] > 0) {
      weight_sum += (1u << (weights[i] - 1));
    }
  }
  printf("\n");

  printf("Weight Sum: %u (Expected power of 2, or gap should be power of 2)\n",
         weight_sum);

  // Check next power of 2
  uint32_t next_pow = 1;
  while (next_pow < weight_sum)
    next_pow <<= 1;

  printf("Next Power of 2: %u\n", next_pow);

  if (weight_sum != next_pow) {
    uint32_t gap = next_pow - weight_sum;
    printf("Gap detected: %u (Gap: %u)\n", gap, gap);

    // Check if gap is power of 2 (indicating valid Last Weight)
    if ((gap & (gap - 1)) == 0) {
      printf("SUCCESS: Gap is a power of 2. Valid Last Weight implied.\n");
    } else {
      printf("FAIL: Gap is NOT a power of 2.\n");
      exit(1);
    }
  } else {
    printf("SUCCESS: Perfect power of 2 sum.\n");
  }
}

int main() {
  test_simple_weights();
  return 0;
}
