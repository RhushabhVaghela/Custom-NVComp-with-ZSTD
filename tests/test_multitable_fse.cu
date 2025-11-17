// ============================================================================
// test_multitable_fse.cu - Verify MultiTableFSE default initialization and clear
// ============================================================================

#include "cuda_zstd_fse.h"
#include <cstdio>

int main() {
    cuda_zstd::fse::MultiTableFSE mt;

    if (mt.active_tables != 0) {
        std::fprintf(stderr, "MultiTableFSE.active_tables expected 0, got %u\n", mt.active_tables);
        return 1;
    }

    for (int i = 0; i < 4; ++i) {
        if (mt.tables[i].d_symbol_table != nullptr || mt.tables[i].table_log != 0 || mt.tables[i].table_size != 0 || mt.tables[i].max_symbol != 0) {
            std::fprintf(stderr, "MultiTableFSE.tables[%d] not zero-initialized\n", i);
            return 1;
        }
    }

    mt.active_tables = 0xF;
    mt.clear();

    if (mt.active_tables != 0) {
        std::fprintf(stderr, "MultiTableFSE.clear() failed to reset active_tables (%u)\n", mt.active_tables);
        return 1;
    }

    printf("MultiTableFSE default initialization and clear() -> OK\n");
    return 0;
}
