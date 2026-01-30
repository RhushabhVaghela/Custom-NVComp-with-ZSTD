#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <zstd.h>

// Use internal zstd APIs for sequence access
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

int main() {
  // Create 1024-byte test data
  std::vector<uint8_t> input(1024);
  for (size_t i = 0; i < 1024; i++) {
    input[i] = static_cast<uint8_t>(i % 256);
  }

  // CPU compress
  std::vector<uint8_t> compressed(ZSTD_compressBound(1024));
  size_t comp_size = ZSTD_compress(compressed.data(), compressed.size(),
                                   input.data(), 1024, 1);
  if (ZSTD_isError(comp_size)) {
    printf("Compression error: %s\n", ZSTD_getErrorName(comp_size));
    return 1;
  }
  printf("Compressed 1024 bytes -> %zu bytes\n\n", comp_size);

  // Parse the sequence section manually
  // Frame header = 7 bytes (magic 4 + FHD 1 + FCS 2)
  // Block header = 3 bytes
  // Literal header + content starts at offset 10

  size_t pos = 10; // After frame + block headers

  // Literal header
  uint8_t lit_hdr = compressed[pos];
  uint8_t lit_type = lit_hdr & 3;
  uint8_t lit_fmt = (lit_hdr >> 2) & 3;

  uint32_t lit_regen = 0;
  uint32_t lit_hdr_size = 0;

  if (lit_fmt == 0) {
    lit_hdr_size = 1;
    lit_regen = (lit_hdr >> 3) & 0x1F;
  } else if (lit_fmt == 1) {
    lit_hdr_size = 2;
    lit_regen = (lit_hdr >> 4) | (compressed[pos + 1] << 4);
  }

  uint32_t lit_comp_size = (lit_type == 1) ? 1 : lit_regen;

  printf("=== LITERALS ===\n");
  printf("Type: %u, Format: %u\n", lit_type, lit_fmt);
  printf("Header Size: %u, Regen Size: %u, Comp Size: %u\n", lit_hdr_size,
         lit_regen, lit_comp_size);

  pos += lit_hdr_size + lit_comp_size;

  // Sequence header
  printf("\n=== SEQUENCES @ offset %zu ===\n", pos);

  uint8_t seq_hdr = compressed[pos];
  uint32_t num_seq = 0;
  uint32_t seq_hdr_size = 0;

  if (seq_hdr < 128) {
    num_seq = seq_hdr;
    seq_hdr_size = 1;
  } else if (seq_hdr < 255) {
    num_seq = ((seq_hdr - 128) << 8) | compressed[pos + 1];
    seq_hdr_size = 2;
  } else {
    num_seq = compressed[pos + 1] | (compressed[pos + 2] << 8);
    num_seq += 0x7F00;
    seq_hdr_size = 3;
  }
  printf("Num Sequences: %u\n", num_seq);
  pos += seq_hdr_size;

  if (num_seq > 0) {
    uint8_t mode = compressed[pos];
    uint8_t ll_mode = (mode >> 6) & 3;
    uint8_t of_mode = (mode >> 4) & 3;
    uint8_t ml_mode = (mode >> 2) & 3;
    printf("Mode Byte: 0x%02X\n", mode);
    printf("  LL_Mode=%u (0=Predefined)\n", ll_mode);
    printf("  OF_Mode=%u (0=Predefined)\n", of_mode);
    printf("  ML_Mode=%u (0=Predefined)\n", ml_mode);
    pos++;
  }

  // Dump remaining bitstream bytes
  size_t block_end = 7 + 3 + 266; // Frame hdr + block hdr + block content
  size_t bitstream_size = block_end - pos;

  printf("\n=== FSE BITSTREAM (%zu bytes) ===\n", bitstream_size);
  printf("Start offset: %zu, End offset: %zu\n", pos, block_end);

  for (size_t i = 0; i < bitstream_size && i < 32; i++) {
    printf("%02X ", compressed[pos + i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  printf("\n");

  // Last byte has sentinel
  uint8_t last_byte = compressed[block_end - 1];
  printf("Last byte: 0x%02X\n", last_byte);

  // Find sentinel bit
  int sentinel_bit = 7;
  while (sentinel_bit >= 0 && !((last_byte >> sentinel_bit) & 1)) {
    sentinel_bit--;
  }
  printf("Sentinel bit position in last byte: %d\n", sentinel_bit);

  // Calculate actual bitstream length
  size_t bitstream_bits = (bitstream_size - 1) * 8 + sentinel_bit;
  printf("Total bits (excluding sentinel): %zu\n", bitstream_bits);

  // For predefined tables, initial states are 6 bits each (LL, OF, ML)
  // Read backwards from before sentinel
  printf("\n=== EXPECTED SEQUENCE VALUES ===\n");
  printf("Frame expects 1024 bytes output\n");
  printf("Literals: %u bytes\n", lit_regen);
  printf("From sequences: %u bytes\n", 1024 - lit_regen);

  // With 1 sequence and 256 literals:
  // Sequence: LL=256, OF=?, ML=768
  // LL=256 uses code 27 (base 128 + 128 extra = 256)
  // ML=768 uses code 43 (base 259 + 509 extra)
  // OF for copy from 256 back = OF code 8 (1<<8 = 256)

  printf("\nExpected single sequence:\n");
  printf("  LL = 256 (code 27 with extra bits)\n");
  printf("  ML = 768 (code 43: base 259, 9 extra bits = 509)\n");
  printf("  OF = 256 -> encoded as 256+3=259 for Zstd encoding\n");

  // Verify with CPU decompress
  std::vector<uint8_t> decompressed(1024);
  size_t decomp_size =
      ZSTD_decompress(decompressed.data(), 1024, compressed.data(), comp_size);
  printf("\nCPU Decompressed: %zu bytes ", decomp_size);
  if (decomp_size == 1024 &&
      memcmp(decompressed.data(), input.data(), 1024) == 0) {
    printf("(VERIFIED ✓)\n");
  } else {
    printf("(ERROR)\n");
  }

  return 0;
}
