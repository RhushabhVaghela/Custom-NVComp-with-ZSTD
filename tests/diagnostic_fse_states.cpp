#include <cstdio>
#include <vector>
#include <cstdint>

// From libzstd or RFC
const int16_t default_ll_norm[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};

void build_table() {
    const int tableSize = 64;
    const int tableLog = 6;
    const int maxSymbol = 35;
    const uint32_t step = (tableSize >> 1) + (tableSize >> 3) + 3; // 32 + 8 + 3 = 43
    const uint32_t mask = tableSize - 1;

    uint8_t spread_symbol[tableSize];
    int highThreshold = tableSize - 1;

    // Place low-prob symbols
    for (int s = 0; s <= maxSymbol; s++) {
        if (default_ll_norm[s] == -1) {
            spread_symbol[highThreshold--] = (uint8_t)s;
        }
    }

    // Spread regular symbols
    uint32_t pos = 0;
    for (int s = 0; s <= maxSymbol; s++) {
        int count = default_ll_norm[s];
        if (count <= 0) continue;
        for (int i = 0; i < count; i++) {
            spread_symbol[pos] = (uint8_t)s;
            pos = (pos + step) & mask;
            while (pos > highThreshold) pos = (pos + step) & mask;
        }
    }

    printf("State | Symbol | nbBits | nextState\n");
    printf("------|--------|--------|----------\n");

    uint32_t symbolNext[maxSymbol + 1];
    for (int s = 0; s <= maxSymbol; s++) {
        symbolNext[s] = (default_ll_norm[s] == -1) ? 1 : default_ll_norm[s];
    }

    for (int state = 0; state < tableSize; state++) {
        int symbol = spread_symbol[state];
        uint32_t n = symbolNext[symbol]++;
        int nbBits, newState;
        if (default_ll_norm[symbol] == -1) {
            nbBits = tableLog;
            newState = 0;
        } else {
            int highBit = 0;
            uint32_t tmp = n;
            while (tmp >>= 1) highBit++;
            nbBits = tableLog - highBit;
            newState = (n << nbBits) - tableSize;
        }
        if (state < 32 || symbol == 27 || state == 19)
            printf("%5d | %6d | %6d | %9d\n", state, symbol, nbBits, newState);
    }
}

int main() {
    build_table();
    
    // Add bitstream decoding check too
    uint8_t bs[] = {0x00, 0xFD, 0x06, 0xAA, 0x35, 0x05};
    size_t bs_size = 6;
    size_t sentinel_pos = (bs_size - 1) * 8 + 2; // Last byte 0x05 has sentinel at bit 2

    auto read_bits_backward = [&](size_t &pos, int num_bits) -> uint32_t {
        uint32_t val = 0;
        for (int i = 0; i < num_bits; i++) {
            if (pos == 0) break;
            pos--;
            size_t byte_idx = pos / 8;
            size_t bit_idx = pos % 8;
            uint32_t bit = (bs[byte_idx] >> bit_idx) & 1;
            val |= (bit << i);
        }
        return val;
    };

    size_t pos = sentinel_pos;
    printf("\nBitstream decoding (order ML, OF, LL):\n");
    uint32_t sml = read_bits_backward(pos, 6);
    uint32_t sof = read_bits_backward(pos, 5);
    uint32_t sll = read_bits_backward(pos, 6);
    printf("Init states: ML=%u, OF=%u, LL=%u\n", sml, sof, sll);
    
    return 0;
}
