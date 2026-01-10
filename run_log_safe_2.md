FAILED: uncompressed_size == input_size at /mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/tests/test_rfc8878_integration.cu:157
========================================
RFC 8878 Integration Verification
========================================

[INT] Testing GPU Compress -> CPU Decompress (Size: 1024, Level: 1)
  [PASS] GPU Compress -> CPU Decompress successful
[INT] Testing CPU Compress -> GPU Decompress (Size: 1024, Level: 1)
  [INFO] CPU Generated Data Verified Valid (Size: 276)
  [FAIL] Size mismatch: expected 1024, got 290
[INT] Testing GPU Compress -> CPU Decompress (Size: 131072, Level: 3)
  [PASS] GPU Compress -> CPU DecomprFAILED: uncompressed_size == input_size at /mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/tests/test_rfc8878_integration.cu:157
ess successful
[INT] Testing CPU Compress -> GPU Decompress (Size: 131072, Level: 3)
  [INFO] CPU Generated Data Verified Valid (Size: 279)
  [FAIL] Size mismatch: expected 131072, got 314
[INT] Testing GPU Compress -> CPU Decompress (Size: 1048576, Level: 5)
[DEBUG] Memory: Total 109051904, Global Used 72877568, Available 36174336
  [FAIL] CPU Decompression failed: Data corruption detected
[INT] Testing CPU Compress -> GPU Decompress (Size: 1048576, Level: 5)
  [INFO] CPU Generated Data Verified Valid (Size: 363)
  [FAIL] GPU Decompression failed with status 6
[INT] Testing GPU Compress -> CPU Decompress (Size: 2097152, Level: 3)
[WARNING] initialize_context: Pre-existing CUDA error: invalid argument
[DEBUG] Memory: Total 218103808, Global Used 145229568, Available 72874240
  [FAIL] CPU Decompression failed: Data corruption detected
[INT] Testing CPU Compress -> GPU Decompress (Size: 2097152, Level: 3)
  [INFO] CPU Generated Data Verified Valid (Size: 459)
  [FAIL] GPU Decompression failed with status 6

❌ SOME Integration Tests FAILED
