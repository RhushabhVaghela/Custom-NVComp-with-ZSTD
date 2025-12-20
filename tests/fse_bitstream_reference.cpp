// Minimal FSE Bitstream Reference Implementation
// Purpose: Verify correct bitstream layout for backward reading
// Based on Zstandard FSE specification

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// CRITICAL: FSE bitstreams are read BACKWARDS (from end to beginning)
// Encoder strategy: Write bits forward, then REVERSE the entire byte array

class FSEBitstreamWriter {
public:
  std::vector<u8> buffer;
  u64 bit_accumulator;
  u32 bits_in_accumulator;

  FSEBitstreamWriter() : bit_accumulator(0), bits_in_accumulator(0) {
    buffer.reserve(1024);
  }

  // Write bits (LSB-first within each byte)
  void write_bits(u32 value, u32 num_bits) {
    bit_accumulator |= (u64)value << bits_in_accumulator;
    bits_in_accumulator += num_bits;

    // Flush complete bytes
    while (bits_in_accumulator >= 8) {
      buffer.push_back((u8)(bit_accumulator & 0xFF));
      bit_accumulator >>= 8;
      bits_in_accumulator -= 8;
    }
  }

  // Finalize: write terminator, flush remaining bits, then REVERSE
  void finalize() {
    // Write terminator bit '1'
    write_bits(1, 1);

    // Flush remaining bits (will pad with zeros automatically)
    if (bits_in_accumulator > 0) {
      buffer.push_back((u8)(bit_accumulator & 0xFF));
    }

    // CRITICAL: REVERSE the entire buffer for backward reading
    std::reverse(buffer.begin(), buffer.end());

    printf("[ENCODER] Finalized bitstream: %zu bytes\n", buffer.size());
    printf("[ENCODER] First 10 bytes: ");
    for (size_t i = 0; i < 10 && i < buffer.size(); i++) {
      printf("%02X ", buffer[i]);
    }
    printf("\n");
  }

  const u8 *data() const { return buffer.data(); }
  size_t size() const { return buffer.size(); }
};

class FSEBitstreamReader {
public:
  const u8 *bitstream;
  size_t bitstream_size;
  u32 bit_position; // Current bit position (reading backwards)

  FSEBitstreamReader(const u8 *data, size_t size)
      : bitstream(data), bitstream_size(size) {

    // Find terminator bit (scan backwards)
    int byte_idx = (int)bitstream_size - 1;
    while (byte_idx >= 0 && bitstream[byte_idx] == 0) {
      byte_idx--;
    }

    if (byte_idx >= 0) {
      u8 b = bitstream[byte_idx];
      int bit_idx = 7;
      while (bit_idx >= 0 && ((b >> bit_idx) & 1) == 0) {
        bit_idx--;
      }
      bit_position = byte_idx * 8 + bit_idx;
      printf("[DECODER] Found terminator at bit %u (byte %d, bit %d)\n",
             bit_position, byte_idx, bit_idx);
    } else {
      printf("[DECODER] ERROR: No terminator found!\n");
      bit_position = bitstream_size * 8;
    }
  }

  // Read bits backwards
  u32 read_bits(u32 num_bits) {
    if (num_bits == 0)
      return 0;

    u32 result = 0;
    for (u32 i = 0; i < num_bits; i++) {
      if (bit_position == 0) {
        printf("[DECODER] ERROR: Ran out of bits!\n");
        return result;
      }

      bit_position--;
      u32 byte_idx = bit_position / 8;
      u32 bit_idx = bit_position % 8;
      u32 bit_val = (bitstream[byte_idx] >> bit_idx) & 1;
      result |= bit_val << i; // LSB-first reconstruction
    }

    return result;
  }
};

int main() {
  printf("=== FSE Bitstream Reference Test ===\n\n");

  // Test 1: Write and read a simple sequence
  FSEBitstreamWriter writer;

  // Encode symbsequence (each symbol writes variable bits)
  printf("[TEST] Encoding sequence: 5 bits [10011], 3 bits [101], 9 bits "
         "[101010101]\n");
  writer.write_bits(0b10011, 5);     // Binary: 10011
  writer.write_bits(0b101, 3);       // Binary: 101
  writer.write_bits(0b101010101, 9); // Binary: 101010101

  writer.finalize();

  // Decode
  FSEBitstreamReader reader(writer.data(), writer.size());

  // NOTE: Decoder reads BACKWARDS, so it should read in REVERSE order!
  // It encounters: [Terminator] [101010101] [101] [10011]
  u32 val3 = reader.read_bits(9);
  u32 val2 = reader.read_bits(3);
  u32 val1 = reader.read_bits(5);

  printf("\n[DECODER] Read values (backward order):\n");
  printf("  9 bits: %u (0b%09b) - expected 341 (0b101010101)\n", val3, val3);
  printf("  3 bits: %u (0b%03b) - expected 5 (0b101)\n", val2, val2);
  printf("  5 bits: %u (0b%05b) - expected 19 (0b10011)\n", val1, val1);

  if (val3 == 0b101010101 && val2 == 0b101 && val1 == 0b10011) {
    printf("\n✅ Bitstream format verified!\n");
    return 0;
  } else {
    printf("\n❌ Bitstream format INCORRECT!\n");
    return 1;
  }
}
