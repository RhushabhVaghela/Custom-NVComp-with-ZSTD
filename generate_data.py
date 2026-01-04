import zstandard as zstd
import os

data = b"Hello World! This is a test string that should be at least 512 bytes long to ensure it triggers various block sizes and sequences. " * 10
# Make it exactly 475 bytes if that was the reference, or just let it be.
# The previous log said Expected: 475.
# Let's just make consistent data.

with open("build/test_interop.txt", "wb") as f:
    f.write(data)

cctx = zstd.ZstdCompressor(level=3)
compressed = cctx.compress(data)

with open("build/test_interop.zst", "wb") as f:
    f.write(compressed)

print(f"Generated test_interop.txt ({len(data)} bytes) and test_interop.zst ({len(compressed)} bytes)")
