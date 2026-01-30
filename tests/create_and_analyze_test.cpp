#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" {
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
}

int main() {
  // Create 1024-byte test data (same as test_rfc8878_integration)
  std::vector<uint8_t> input(1024);
  for (size_t i = 0; i < 1024; i++) {
    input[i] = (uint8_t)(i % 256);
  }

  // Compress with CPU zstd
  size_t max_comp = ZSTD_compressBound(1024);
  std::vector<uint8_t> compressed(max_comp);
  size_t comp_size =
      ZSTD_compress(compressed.data(), max_comp, input.data(), 1024, 1);

  if (ZSTD_isError(comp_size)) {
    printf("Compression failed: %s\n", ZSTD_getErrorName(comp_size));
    return 1;
  }

  printf("Compressed 1024 bytes to %zu bytes\n", comp_size);

  // Save to file
  FILE *f = fopen("test_1kb.zst", "wb");
  if (f) {
    fwrite(compressed.data(), 1, comp_size, f);
    fclose(f);
    printf("Saved to test_1kb.zst\n\n");
  }

  // Verify with CPU
  std::vector<uint8_t> decompressed(1024);
  size_t decomp_size =
      ZSTD_decompress(decompressed.data(), 1024, compressed.data(), comp_size);

  if (ZSTD_isError(decomp_size)) {
    printf("Decompression failed: %s\n", ZSTD_getErrorName(decomp_size));
    return 1;
  }

  printf("CPU decompressed to %zu bytes - ", decomp_size);
  bool ok = (decomp_size == 1024);
  for (size_t i = 0; i < 1024 && ok; i++) {
    if (decompressed[i] != input[i])
      ok = false;
  }
  printf("%s\n\n", ok ? "VERIFIED ✓" : "MISMATCH ✗");

  // Now parse and dump the structure
  printf("=== ZSTD FILE STRUCTURE ===\n\n");

  size_t pos = 0;
  uint32_t magic = compressed[0] | (compressed[1] << 8) |
                   (compressed[2] << 16) | (compressed[3] << 24);
  printf("[0x%04zx] Magic: 0x%08X %s\n", pos, magic,
         magic == 0xFD2FB528 ? "✓" : "✗");
  pos += 4;

  uint8_t fhd = compressed[pos];
  printf("[0x%04zx] Frame Header Descriptor: 0x%02X\n", pos, fhd);
  printf(
      "  FCS_Field_Size: %u, Single_Segment: %u, Dict_ID: %u, Checksum: %u\n",
      (fhd >> 6) & 3, (fhd >> 5) & 1, (fhd >> 3) & 3, fhd & 1);
  pos++;

  // Parse frame content size based on FCS field
  uint8_t fcs_flag = (fhd >> 6) & 3;
  size_t frame_size = 0;
  if (fcs_flag == 1) {
    frame_size = compressed[pos];
    printf("[0x%04zx] Frame Content Size: %zu (1 byte)\n", pos, frame_size);
    pos++;
  } else if (fcs_flag == 2) {
    frame_size = compressed[pos] | (compressed[pos + 1] << 8);
    printf("[0x%04zx] Frame Content Size: %zu (2 bytes)\n", pos, frame_size);
    pos += 2;
  }

  printf("\n--- BLOCK 0 @ 0x%04zx ---\n", pos);

  uint32_t block_header = compressed[pos] | (compressed[pos + 1] << 8) |
                          (compressed[pos + 2] << 16);
  bool last = block_header & 1;
  uint8_t type = (block_header >> 1) & 3;
  uint32_t size = (block_header >> 3) & 0x1FFFFF;

  printf("Block Header: 0x%06X\n", block_header);
  printf("  Last: %u, Type: %u (%s), Size: %u\n", last, type,
         type == 0 ? "Raw" : (type == 1 ? "RLE" : "Compressed"), size);
  pos += 3;

  size_t block_start = pos;

  // Parse literals section
  uint8_t lit_hdr = compressed[pos];
  uint8_t lit_type = lit_hdr & 3;
  uint8_t size_fmt = (lit_hdr >> 2) & 3;

  printf("\n[+0x%04zx] Literals Section Header: 0x%02X\n", pos - block_start,
         lit_hdr);
  printf("  Type: %u (%s), Size_Format: %u\n", lit_type,
         lit_type == 0 ? "Raw" : (lit_type == 1 ? "RLE" : "Compressed"),
         size_fmt);

  size_t lit_hdr_size, regen_size, comp_size_lit;
  if (lit_type <= 1) {
    if (size_fmt == 0) {
      lit_hdr_size = 1;
      regen_size = (lit_hdr >> 3) & 0x1F;
    } else if (size_fmt == 1) {
      lit_hdr_size = 2;
      regen_size = ((lit_hdr >> 4) & 0xF) | (compressed[pos + 1] << 4);
    } else {
      lit_hdr_size = 3;
      regen_size = ((lit_hdr >> 4) & 0xF) | (compressed[pos + 1] << 4) |
                   (compressed[pos + 2] << 12);
    }
    comp_size_lit = (lit_type == 1) ? 1 : regen_size;

    printf(" Regenerated_Size: %zu, Compressed_Size: %zu, Header: %zu bytes\n",
           regen_size, comp_size_lit, lit_hdr_size);

    if (lit_type == 1) {
      printf("  RLE Byte: 0x%02X\n", compressed[pos + lit_hdr_size]);
    }
  }

  size_t seq_start = pos + lit_hdr_size + comp_size_lit;
  printf("\n[+0x%04zx] Sequences Section\n", seq_start - block_start);
  printf("  (Sequences data starts here, %zu bytes remaining in block)\n",
         block_start + size - seq_start);

  printf("\n=== KEY INSIGHT ===\n");
  printf("Literals regenerate to: %zu bytes\n", regen_size);
  printf("Sequences should produce: %zu - %zu = %zu bytes\n", frame_size,
         regen_size, frame_size - regen_size);
  printf("Total expected output: %zu bytes\n", frame_size);

  return 0;
}
