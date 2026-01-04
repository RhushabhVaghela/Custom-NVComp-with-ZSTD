import zstandard as zstd

with open("build/test_interop.zst", "rb") as f:
    compressed = f.read()

dctx = zstd.ZstdDecompressor()
decompressed = dctx.decompress(compressed)

print(f"Decompressed size: {len(decompressed)}")
print(f"First 100 bytes: {decompressed[:100]}")
