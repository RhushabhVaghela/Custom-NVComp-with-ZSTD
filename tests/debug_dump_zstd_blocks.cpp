#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Read CPU-generated zstd file and dump every detail
int main() {
  // Read the debug file created by test_rfc8878_integration
  FILE *f = fopen("build/debug_1mb_cpu.zst", "rb");
  if (!f) {
    printf("Error: Could not open debug_1mb_cpu.zst\n");
    printf("Run test_rfc8878_integration first to generate it.\n");
    return 1;
  }

  fseek(f, 0, SEEK_END);
  size_t file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  std::vector<uint8_t> data(file_size);
  fread(data.data(), 1, file_size, f);
  fclose(f);

  printf("=== ZSTD File Analysis ===\n");
  printf("Total size: %zu bytes\n\n", file_size);

  size_t pos = 0;

  // Magic number
  if (file_size < 4) {
    printf("Error: File too small\n");
    return 1;
  }

  uint32_t magic = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16) |
                   (data[pos + 3] << 24);
  printf("[0x%04zx] Magic Number: 0x%08X ", pos, magic);
  if (magic == 0xFD2FB528)
    printf("(Valid ZSTD)\n");
  else
    printf("(INVALID!)\n");
  pos += 4;

  // Frame Header Descriptor
  uint8_t fhd = data[pos];
  printf("[0x%04zx] Frame Header Descriptor: 0x%02X\n", pos, fhd);
  uint8_t fcs_size = (fhd >> 6) & 3;
  bool single_segment = (fhd >> 5) & 1;
  uint8_t dict_size = (fhd >> 3) & 3;
  bool checksum = fhd & 1;
  printf("  FCS: %u (%u bytes), Single Segment: %s, Dict: %u, Checksum: %s\n",
         fcs_size,
         (fcs_size == 0 ? 0 : (fcs_size == 1 ? 1 : (fcs_size == 2 ? 2 : 8))),
         single_segment ? "Yes" : "No", dict_size, checksum ? "Yes" : "No");
  pos++;

  // Window descriptor (if not single segment)
  if (!single_segment) {
    uint8_t wd = data[pos];
    printf("[0x%04zx] Window Descriptor: 0x%02X\n", pos, wd);
    pos++;
  }

  // Frame Content Size
  size_t frame_content_size = 0;
  if (fcs_size == 1) {
    frame_content_size = data[pos];
    printf("[0x%04zx] Frame Content Size: %zu (1 byte)\n", pos,
           frame_content_size);
    pos += 1;
  } else if (fcs_size == 2) {
    frame_content_size = data[pos] | (data[pos + 1] << 8);
    printf("[0x%04zx] Frame Content Size: %zu (2 bytes)\n", pos,
           frame_content_size);
    pos += 2;
  } else if (fcs_size == 3) {
    frame_content_size =
        data[pos] | (data[pos + 1] << 8) | ((uint64_t)data[pos + 2] << 16) |
        ((uint64_t)data[pos + 3] << 24) | ((uint64_t)data[pos + 4] << 32) |
        ((uint64_t)data[pos + 5] << 40) | ((uint64_t)data[pos + 6] << 48) |
        ((uint64_t)data[pos + 7] << 56);
    printf("[0x%04zx] Frame Content Size: %zu (8 bytes)\n", pos,
           frame_content_size);
    pos += 8;
  }

  printf("\n=== BLOCKS ===\n");

  int block_num = 0;
  while (pos < file_size) {
    if (file_size - pos < 3) {
      if (checksum && file_size - pos == 4) {
        uint32_t csum = data[pos] | (data[pos + 1] << 8) |
                        (data[pos + 2] << 16) | (data[pos + 3] << 24);
        printf("\n[0x%04zx] Checksum: 0x%08X\n", pos, csum);
        break;
      }
      printf("\nWarning: %zu bytes remaining (too small for block header)\n",
             file_size - pos);
      break;
    }

    // Block header (3 bytes)
    uint32_t block_header =
        data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16);
    bool last_block = block_header & 1;
    uint8_t block_type = (block_header >> 1) & 3;
    uint32_t block_size = (block_header >> 3) & 0x1FFFFF;

    printf("\n--- BLOCK %d @ 0x%04zx ---\n", block_num++, pos);
    printf("Header: 0x%06X\n", block_header);
    printf("  Last: %s\n", last_block ? "Yes" : "No");
    printf("  Type: %u (%s)\n", block_type,
           block_type == 0
               ? "Raw"
               : (block_type == 1
                      ? "RLE"
                      : (block_type == 2 ? "Compressed" : "Reserved")));
    printf("  Size: %u bytes\n", block_size);
    pos += 3;

    if (pos + block_size > file_size) {
      printf("ERROR: Block extends past file end!\n");
      break;
    }

    // Dump block content
    if (block_type == 2) { // Compressed block
      printf("\nCompressed Block Content:\n");

      // Parse literals section
      uint8_t lit_header = data[pos];
      uint8_t lit_type = lit_header & 3;
      uint8_t size_format = (lit_header >> 2) & 3;

      printf("[+0x%04zx] Literals Header: 0x%02X\n", pos, lit_header);
      printf("  Type: %u (%s)\n", lit_type,
             lit_type == 0 ? "Raw"
                           : (lit_type == 1 ? "RLE"
                                            : (lit_type == 2 ? "Compressed"
                                                             : "Treeless")));
      printf("  Size Format: %u\n", size_format);

      size_t lit_header_size = 0;
      size_t regenerated_size = 0;
      size_t compressed_size = 0;

      if (lit_type <= 1) { // Raw or RLE
        if (size_format == 0) {
          lit_header_size = 1;
          regenerated_size = (lit_header >> 3) & 0x1F;
        } else if (size_format == 1) {
          lit_header_size = 2;
          regenerated_size = ((lit_header >> 4) & 0xF) | (data[pos + 1] << 4);
        } else if (size_format == 2) {
          lit_header_size = 3;
          regenerated_size = ((lit_header >> 4) & 0xF) | (data[pos + 1] << 4) |
                             (data[pos + 2] << 12);
        } else {
          lit_header_size = 4;
          regenerated_size = ((lit_header >> 4) & 0xF) | (data[pos + 1] << 4) |
                             (data[pos + 2] << 12) | (data[pos + 3] << 20);
        }
        compressed_size = (lit_type == 1) ? 1 : regenerated_size;

        printf("  Regenerated Size: %zu bytes\n", regenerated_size);
        printf("  Compressed Size: %zu bytes\n", compressed_size);
        printf("  Header Size: %zu bytes\n", lit_header_size);

        if (lit_type == 1) {
          uint8_t rle_byte = data[pos + lit_header_size];
          printf("  RLE Byte: 0x%02X ('%c')\n", rle_byte,
                 (rle_byte >= 32 && rle_byte < 127) ? rle_byte : '.');
        }
      } else { // Compressed or Treeless
        printf("  (Compressed literals - parsing not fully implemented)\n");
      }

      printf("\nFirst 64 bytes of block data:\n");
      for (size_t i = 0; i < 64 && i < block_size; i++) {
        if (i % 16 == 0)
          printf("  %04zx: ", i);
        printf("%02X ", data[pos + i]);
        if ((i + 1) % 16 == 0)
          printf("\n");
      }
      if (block_size % 16 != 0)
        printf("\n");
    }

    pos += block_size;

    if (last_block) {
      printf("\n(Last block reached)\n");
      break;
    }
  }

  printf("\n=== END OF FILE @ 0x%04zx ===\n", pos);
  if (pos < file_size) {
    printf("Remaining bytes: %zu\n", file_size - pos);
  }

  return 0;
}
