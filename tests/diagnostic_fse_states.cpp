#include <cstdint>
#include <cstdio>

// Manually decode the FSE bitstream for our 1KB test
// Bitstream: 00 FD 06 AA 35 05 (6 bytes)
// Last byte 05 = 0b00000101, sentinel at bit 2 (from LSB)
// Sentinel position: 5*8 + 2 = 42

int main() {
  uint8_t bs[] = {0x00, 0xFD, 0x06, 0xAA, 0x35, 0x05};
  size_t bs_size = 6;

  // Find sentinel
  uint8_t last = bs[bs_size - 1]; // 0x05 = 0b00000101
  int hb = 7;
  while (hb >= 0 && !((last >> hb) & 1))
    hb--;

  printf("Last byte: 0x%02X = 0b", last);
  for (int i = 7; i >= 0; i--)
    printf("%d", (last >> i) & 1);
  printf("\n");
  printf("Sentinel bit (from LSB, 0-indexed): %d\n", hb);

  size_t sentinel_pos = (bs_size - 1) * 8 + hb;
  printf("Sentinel position (total bits): %zu\n\n", sentinel_pos);

  // Read bits from sentinel position backwards
  // Per RFC 8878, initial states are read in order: LL, OF, ML
  // LL table_log = 6, OF table_log = 5, ML table_log = 6
  // Total = 17 bits

  printf("=== Reading Initial States ===\n");
  printf("Position before reading: %zu\n", sentinel_pos);

  // Helper to read N bits from position (reading backwards in bitstream)
  auto read_bits_backward = [&](size_t &pos, int num_bits) -> uint32_t {
    uint32_t val = 0;
    for (int i = 0; i < num_bits; i++) {
      if (pos == 0)
        break;
      pos--;
      size_t byte_idx = pos / 8;
      size_t bit_idx = pos % 8;
      uint32_t bit = (bs[byte_idx] >> bit_idx) & 1;
      val |= (bit << i);
    }
    return val;
  };

  size_t pos = sentinel_pos;

  // Read LL state (6 bits)
  uint32_t state_ll = read_bits_backward(pos, 6);
  printf("Read LL state: %u (6 bits), pos now=%zu\n", state_ll, pos);

  // Read OF state (5 bits)
  uint32_t state_of = read_bits_backward(pos, 5);
  printf("Read OF state: %u (5 bits), pos now=%zu\n", state_of, pos);

  // Read ML state (6 bits)
  uint32_t state_ml = read_bits_backward(pos, 6);
  printf("Read ML state: %u (6 bits), pos now=%zu\n", state_ml, pos);

  printf("\n=== Summary ===\n");
  printf("LL initial state: %u\n", state_ll);
  printf("OF initial state: %u\n", state_of);
  printf("ML initial state: %u\n", state_ml);

  // The GPU shows: InitStates: LL=19, OF=11, ML=21
  // Let's see if our reading matches

  printf("\n=== GPU showed ===\n");
  printf("LL=19, OF=11, ML=21\n");

  return 0;
}
