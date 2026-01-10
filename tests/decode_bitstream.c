#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Norms and Tables
static const short LL_defaultNorm[36] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,  1,  2,  2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
static const short OF_defaultNorm[29] = {1, 1, 1, 1, 1,  1,  2,  2,  2, 1,
                                         1, 1, 1, 1, 1,  1,  1,  1,  1, 1,
                                         1, 1, 1, 1, -1, -1, -1, -1, -1};
static const short ML_defaultNorm[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};

#define LL_LOG 6
#define OF_LOG 5
#define ML_LOG 6

uint8_t ll_symbol[64], of_symbol[64], ml_symbol[64];

void build_table(const short *norm, int maxSymbol, int tableLog,
                 uint8_t *symbol) {
  uint32_t tableSize = 1 << tableLog;
  uint32_t tableMask = tableSize - 1;
  uint32_t step = (tableSize >> 1) + (tableSize >> 3) + 3;
  uint32_t highThreshold = tableSize - 1;
  memset(symbol, 0, tableSize);
  for (int s = 0; s <= maxSymbol; s++)
    if (norm[s] == -1)
      symbol[highThreshold--] = s;
  uint32_t position = 0;
  for (int s = 0; s <= maxSymbol; s++) {
    if (norm[s] <= 0)
      continue;
    for (int i = 0; i < norm[s]; i++) {
      symbol[position] = s;
      position = (position + step) & tableMask;
      while (position > highThreshold)
        position = (position + step) & tableMask;
    }
  }
}

static const int LL_bits[] = {0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,
                              0, 0, 0, 0, 1, 1, 1, 1, 2,  2,  3,  3,
                              4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const int LL_base[] = {0,   1,   2,   3,   4,   5,    6,    7,    8,
                              9,   10,  11,  12,  13,  14,   15,   16,   18,
                              20,  22,  24,  28,  32,  40,   48,   64,   80,
                              112, 144, 208, 336, 592, 1104, 2128, 4176, 8272};
static const int ML_bits[] = {0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0,
                              0, 0, 0, 0, 1, 1, 1,  1,  2,  2,  3, 3, 4, 4,
                              5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16};
static const int ML_base[] = {
    3,  4,   5,   6,   7,   8,   9,   10,   11,   12,   13,   14, 15, 16,
    17, 18,  19,  20,  21,  22,  23,  24,   25,   26,   27,   28, 29, 30,
    31, 32,  33,  34,  35,  37,  39,  41,   43,   47,   51,   59, 67, 83,
    99, 131, 163, 227, 355, 483, 739, 1251, 2275, 4323, 65539};

typedef struct {
  const uint8_t *data;
  int bit_pos;
  int size;
} BitReader;
uint32_t read_bits(BitReader *r, int n) {
  if (n == 0)
    return 0;
  r->bit_pos -= n;
  int byte_off = r->bit_pos / 8;
  int bit_off = r->bit_pos % 8;
  if (r->bit_pos < 0)
    return 0;
  uint64_t val = 0;
  for (int i = 0; i < 8 && byte_off + i < r->size; i++)
    val |= ((uint64_t)r->data[byte_off + i]) << (i * 8);
  return (val >> bit_off) & ((1ULL << n) - 1);
}

void try_decode(const uint8_t *input, int size, int shift,
                int skip_header_bytes) {
  uint8_t buffer[16];
  memset(buffer, 0, 16);
  for (int i = 0; i < size; i++) {
    uint16_t val = input[i] << shift;
    buffer[i] |= (val & 0xFF);
    if (i + 1 < 16)
      buffer[i + 1] |= (val >> 8);
  }

  int eff_size = size + (shift > 0 ? 1 : 0);
  int start_pos = (eff_size * 8) - 1;
  while (start_pos >= 0 && !((buffer[start_pos / 8] >> (start_pos % 8)) & 1))
    start_pos--;

  if (start_pos < 0)
    return;

  BitReader reader = {buffer, start_pos, eff_size};

  uint32_t ll_state = read_bits(&reader, LL_LOG);
  uint32_t of_state = read_bits(&reader, OF_LOG);
  uint32_t ml_state = read_bits(&reader, ML_LOG);

  uint32_t ll_sym = ll_symbol[ll_state];
  uint32_t of_sym = of_symbol[of_state];
  uint32_t ml_sym = ml_symbol[ml_state];

  int of_bits_cnt = of_sym;
  uint32_t of_extra = read_bits(&reader, of_bits_cnt);
  int ml_bits_cnt = ML_bits[ml_sym];
  uint32_t ml_extra = read_bits(&reader, ml_bits_cnt);
  int ll_bits_cnt = LL_bits[ll_sym];
  uint32_t ll_extra = read_bits(&reader, ll_bits_cnt);

  uint32_t ll_val = LL_base[ll_sym] + ll_extra;
  uint32_t ml_val = ML_base[ml_sym] + ml_extra;
  uint32_t of_val = (of_sym <= 2) ? of_sym + 1 : (1 << of_sym) + of_extra;

  // Check for interesting values (120=Standard, 256=Goal, 768=Goal)
  if ((ll_val >= 110 && ll_val <= 130) || (ll_val >= 250 && ll_val <= 265) ||
      (ml_val >= 700 && ml_val <= 800)) {
    printf("CANDIDATE: Offset=%d Shift=%d | LL=%u ML=%u OF=%u | States: LL%u "
           "OF%u ML%u | Syms: L%u O%u M%u\n",
           skip_header_bytes, shift, ll_val, ml_val, of_val, ll_state, of_state,
           ml_state, ll_sym, of_sym, ml_sym);
  }
}

int main() {
  build_table(LL_defaultNorm, 35, LL_LOG, ll_symbol);
  build_table(OF_defaultNorm, 28, OF_LOG, of_symbol);
  build_table(ML_defaultNorm, 52, ML_LOG, ml_symbol);

  uint8_t raw[] = {0x00, 0xFD, 0x06, 0xAA, 0x35, 0x05};

  for (int offset = 0; offset < 4; offset++) {
    for (int len = 3; len <= 6 - offset; len++) {
      for (int shift = 0; shift < 8; shift++) {
        try_decode(raw + offset, len, shift, offset);
      }
    }
  }
  printf("Done.\n");
  return 0;
}
