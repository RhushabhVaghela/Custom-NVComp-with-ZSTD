# RFC 8878 Zstandard Decompression Fixes

- [x] Fix RLE value conversions (OF, ML, LL) <!-- id: 0 -->
    - [x] Offset RLE (1 << code) <!-- id: 1 -->
    - [x] Match Length RLE (switch-case) <!-- id: 2 -->
    - [x] Lit Length RLE (RFC formula) <!-- id: 3 -->
- [x] Fix `size_format` parsing for Raw/RLE literals <!-- id: 4 -->
- [/] Fix Block Type Handling for Decompression <!-- id: 5 -->
    - [/] Trace [decompress](file:///d:/Research%20Experiments/TDPE_and_GPU_loading/NVComp%20with%20ZSTD/include/cuda_zstd_manager.h#337-340) frame parsing logic <!-- id: 6 -->
    - [ ] Implement correct handling for Raw Blocks (Type 0) <!-- id: 7 -->
    - [ ] Verify RLE Block (Type 1) handling <!-- id: 8 -->
- [ ] Verify fixes with [test_rfc8878_integration](file:///d:/Research%20Experiments/TDPE_and_GPU_loading/NVComp%20with%20ZSTD/build/test_rfc8878_integration) <!-- id: 9 -->
- [ ] Clean up debug prints <!-- id: 10 -->
