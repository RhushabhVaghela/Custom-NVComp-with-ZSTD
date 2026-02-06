# Fix Literal Header Parsing for RFC8878 Test

## Goal
Decode RFC 8878 literal headers exactly per Table 7 so GPU decoder sees correct literal limits and `test_rfc8878_integration` passes in both directions.

## Tasks
- [x] Inspect reference literal header spec vs current `decompress_literals` (raw/compressed/regenerated) interpretation to locate bit-field mistakes → Verify by documenting expected bit positions for each size_format.
- [ ] Update parsing code in `src/cuda_zstd_manager.cu` to correctly extract RegenSize/CompressedSize for compressed literal types, handling shared vs dedicated bits without double-counting → Verify with unit-style printf on sample header to match CPU reference sizes (e.g. regen=256, comp=?? from CPU block).
- [ ] Ensure `h_decompressed_size` and `h_compressed_size` propagate to Huffman decode and instrumentation only once per header (remove conflicting overwrites) → Verify via logging that limit equals literal section size (no fallback to 16).
- [ ] Rerun `./build/bin/test_rfc8878_integration` to confirm both GPU→CPU and CPU→GPU paths complete; capture any remaining failures for follow-up.

## Done When
- [ ] RFC 8878 integration test reports success in both directions without literal overflow errors.
