# Huffman Decoding Fixes & RFC 8878 Compliance

- [ ] Analyze failure in `test_libzstd_interop` <!-- id: 1 -->
    - [ ] Run `test_libzstd_interop` and capture debug output (weight dump)
    - [ ] Run `test_huffman_roundtrip` to establish baseline failure
- [ ] Compare FSE implementations <!-- id: 2 -->
    - [ ] Compare [src/cuda_zstd_huffman.cu](file:///d:/Research%20Experiments/TDPE_and_GPU_loading/NVComp%20with%20ZSTD/src/cuda_zstd_huffman.cu) FSE logic vs [src/cuda_zstd_fse.cu](file:///d:/Research%20Experiments/TDPE_and_GPU_loading/NVComp%20with%20ZSTD/src/cuda_zstd_fse.cu)
    - [ ] Identify discrepancies in `weights_to_code_lengths` or FSE table building
- [ ] Fix Weight Calculation / FSE Logic <!-- id: 3 -->
    - [ ] Fix `weights_to_code_lengths` (power of 2 sum, last weight)
    - [ ] Fix any FSE decoding issues identified
- [ ] Verification <!-- id: 4 -->
    - [ ] Run `test_huffman_roundtrip` (should pass)
    - [ ] Run `test_libzstd_interop` (should pass)
