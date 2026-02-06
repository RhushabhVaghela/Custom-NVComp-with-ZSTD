#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

const int16_t default_ll_norm[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
const int16_t default_ml_norm[53] = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
const int16_t default_of_norm[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};

struct Table {
    uint8_t symbol[64];
};

void build_fse_table(const int16_t* norm, int maxSym, int log, Table& t) {
    int size = 1 << log;
    uint32_t step = (size >> 1) + (size >> 3) + 3;
    uint32_t mask = size - 1;
    uint8_t spread[64];
    int high = size - 1;
    for (int s = 0; s <= maxSym; s++) if (norm[s] == -1) spread[high--] = s;
    uint32_t pos = 0;
    for (int s = 0; s <= maxSym; s++) {
        if (norm[s] <= 0) continue;
        for (int i = 0; i < norm[s]; i++) {
            spread[pos] = s;
            pos = (pos + step) & mask;
            while (pos > high) pos = (pos + step) & mask;
        }
    }
    for (int i = 0; i < size; i++) t.symbol[i] = spread[i];
}

uint32_t get_ll_base(uint32_t c) {
    static const uint32_t b[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,28,32,40,48,64,128,256,512,1024,2048,4096,8192,16384,32768,65536};
    return (c<36)?b[c]:0;
}
uint32_t get_ll_bits(uint32_t c) {
    static const uint8_t b[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,4,6,7,8,9,10,11,12,13,14,15,16};
    return (c<36)?b[c]:0;
}
uint32_t get_ml_base(uint32_t c) {
    static const uint32_t b[] = {3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,39,41,43,47,51,59,67,83,99,131,163,227,291,419,547,803,1059,1571,2083};
    return (c<53)?b[c]:0;
}
uint32_t get_ml_bits(uint32_t c) {
    static const uint8_t b[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,16};
    return (c<53)?b[c]:0;
}

int main() {
    Table tLL, tML, tOF;
    build_fse_table(default_ll_norm, 35, 6, tLL);
    build_fse_table(default_ml_norm, 52, 6, tML);
    build_fse_table(default_of_norm, 28, 5, tOF);

    uint8_t bs[] = {0x00, 0xFD, 0x06, 0xAA, 0x35, 0x05};
    int sentinel = 42;

    auto read_bits = [&](int& pos, int n) -> uint32_t {
        if (n == 0) return 0;
        uint64_t data = 0;
        int start = pos - n;
        for (int i = 0; i < 8; i++) {
            int byte = (start + i*8) / 8;
            if (byte >= 0 && byte < 6) data |= ((uint64_t)bs[byte] << (i*8));
        }
        uint32_t val = (data >> (start % 8)) & ((1ULL << n) - 1);
        pos -= n;
        return val;
    };

    const char* names[] = {"LL", "OF", "ML"};
    int p[] = {0, 1, 2};
    do {
        int ep[] = {0, 1, 2};
        do {
            int pos = sentinel;
            uint32_t s[3];
            int logs[] = {6, 5, 6};
            for (int i = 0; i < 3; i++) s[p[i]] = read_bits(pos, logs[p[i]]);
            
            uint32_t syms[] = {tLL.symbol[s[0]], tOF.symbol[s[1]], tML.symbol[s[2]]};
            uint32_t ex_bits[] = {get_ll_bits(syms[0]), (uint32_t)syms[1], get_ml_bits(syms[2])};
            
            uint32_t ex[3];
            for (int i = 0; i < 3; i++) ex[ep[i]] = read_bits(pos, ex_bits[ep[i]]);
            
            uint32_t val_ll = get_ll_base(syms[0]) + ex[0];
            uint32_t val_of = (syms[1] <= 2) ? (syms[1] + 1) : ((1u << syms[1]) + ex[1]);
            uint32_t val_ml = get_ml_base(syms[2]) + ex[2];
            uint32_t offset = val_of - 3;

            if (val_ml == 768) {
                printf("Init: %s %s %s | Extra: %s %s %s | LL=%u, OF=%u, ML=%u, RemBits=%d\n",
                    names[p[0]], names[p[1]], names[p[2]],
                    names[ep[0]], names[ep[1]], names[ep[2]],
                    val_ll, offset, val_ml, pos);
            }
        } while (std::next_permutation(ep, ep + 3));
    } while (std::next_permutation(p, p + 3));

    return 0;
}
