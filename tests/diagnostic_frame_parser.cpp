#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <zstd.h>

// Focused diagnostic: Compare GPU vs CPU sequence interpretation
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
  printf("Compressed %zu bytes -> %zu bytes\n\n", input.size(), comp_size);

  // Parse frame manually
  size_t pos = 0;

  // Magic number (4 bytes)
  uint32_t magic = compressed[0] | (compressed[1] << 8) |
                   (compressed[2] << 16) | (compressed[3] << 24);
  printf("Magic: 0x%08X\n", magic);
  pos = 4;

  // Frame header descriptor
  uint8_t fhd = compressed[pos++];
  uint8_t fcs_flag = (fhd >> 6) & 3;
  bool single_seg = (fhd >> 5) & 1;
  printf("FHD: 0x%02X (FCS=%u, SingleSeg=%u)\n", fhd, fcs_flag, single_seg);

  // Frame content size
  size_t frame_size = 0;
  if (single_seg && fcs_flag == 0) {
    // Single byte FCS
    frame_size = compressed[pos++];
    if (frame_size == 0)
      frame_size = 256; // 0 means 256 per spec
    printf("Frame Content Size: %zu (1 byte)\n", frame_size);
  } else if (fcs_flag == 1) {
    frame_size = compressed[pos] | (compressed[pos + 1] << 8);
    frame_size += 256; // RFC offset
    printf("Frame Content Size: %zu (2 bytes, +256)\n", frame_size);
    pos += 2;
  } else {
    printf("FCS field size other: %u\n", fcs_flag);
  }

  printf("\n=== BLOCK @ offset %zu ===\n", pos);

  // Block header (3 bytes)
  uint32_t bhdr = compressed[pos] | (compressed[pos + 1] << 8) |
                  (compressed[pos + 2] << 16);
  bool last = bhdr & 1;
  uint8_t btype = (bhdr >> 1) & 3;
  uint32_t bsize = (bhdr >> 3) & 0x1FFFFF;
  printf("Block: Last=%u, Type=%u, Size=%u\n", last, btype, bsize);
  pos += 3;

  // Literals header
  printf("\n=== LITERALS @ offset %zu ===\n", pos);
  uint8_t lhdr0 = compressed[pos];
  uint8_t ltype = lhdr0 & 3;
  uint8_t lfmt = (lhdr0 >> 2) & 3;
  printf("Literal Header Byte 0: 0x%02X\n", lhdr0);
  printf("  Type: %u (%s)\n", ltype,
         ltype == 0 ? "Raw" : (ltype == 1 ? "RLE" : "Compressed"));
  printf("  Size Format: %u\n", lfmt);

  uint32_t regen_size = 0;
  uint32_t lit_hdr_size = 0;
  uint32_t comp_size_lit = 0;

  if (ltype <= 1) { // Raw or RLE
    if (lfmt == 0) {
      lit_hdr_size = 1;
      regen_size = (lhdr0 >> 3) & 0x1F;
    } else if (lfmt == 1) {
      lit_hdr_size = 2;
      regen_size = (lhdr0 >> 4) | (compressed[pos + 1] << 4);
    } else if (lfmt == 2) {
      lit_hdr_size = 3;
      regen_size = (lhdr0 >> 4) | (compressed[pos + 1] << 4) |
                   (compressed[pos + 2] << 12);
    } else {
      lit_hdr_size = 4;
      regen_size = (lhdr0 >> 4) | (compressed[pos + 1] << 4) |
                   (compressed[pos + 2] << 12) | (compressed[pos + 3] << 20);
    }
    comp_size_lit = (ltype == 1) ? 1 : regen_size;

    printf("  Header Size: %u\n", lit_hdr_size);
    printf("  Regen Size: %u\n", regen_size);
    if (ltype == 1) {
      printf("  RLE Byte: 0x%02X\n", compressed[pos + lit_hdr_size]);
    }
  }

  pos += lit_hdr_size + comp_size_lit;

  // Sequences header
  printf("\n=== SEQUENCES @ offset %zu ===\n", pos);
  uint8_t seq_hdr0 = compressed[pos];
  uint32_t num_seq = 0;
  uint32_t seq_hdr_size = 0;

  if (seq_hdr0 == 0) {
    num_seq = 0;
    seq_hdr_size = 1;
  } else if (seq_hdr0 < 128) {
    num_seq = seq_hdr0;
    seq_hdr_size = 1;
  } else if (seq_hdr0 < 255) {
    num_seq = ((seq_hdr0 - 128) << 8) | compressed[pos + 1];
    seq_hdr_size = 2;
  } else {
    num_seq = compressed[pos + 1] | (compressed[pos + 2] << 8);
    num_seq += 0x7F00;
    seq_hdr_size = 3;
  }
  printf("Num Sequences: %u (header size: %u)\n", num_seq, seq_hdr_size);
  pos += seq_hdr_size;

  if (num_seq > 0) {
    uint8_t mode_byte = compressed[pos++];
    uint8_t ll_mode = (mode_byte >> 6) & 3;
    uint8_t of_mode = (mode_byte >> 4) & 3;
    uint8_t ml_mode = (mode_byte >> 2) & 3;
    printf("Mode Byte: 0x%02X\n", mode_byte);
    printf("  LL Mode: %u, OF Mode: %u, ML Mode: %u\n", ll_mode, of_mode,
           ml_mode);
  }

  printf("\n=== SUMMARY ===\n");
  printf("Frame expects: %zu bytes output\n", frame_size);
  printf("Literals: %u bytes (type: %s)\n", regen_size,
         ltype == 0 ? "Raw" : (ltype == 1 ? "RLE" : "Compressed"));
  printf("Sequences: %u\n", num_seq);
  printf("Remaining output must come from matches: %zu bytes\n",
         frame_size - regen_size);

  // Verify CPU decompress
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
