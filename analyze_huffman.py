#!/usr/bin/env python3
"""Analyze Huffman weights from zstd compressed file using libzstd internals."""

import zstandard as zstd

# Read the compressed file
with open("build/test_interop.zst", "rb") as f:
    data = f.read()

print(f"Compressed file size: {len(data)} bytes")

# Show header bytes around literals block
print(f"Header bytes (first 40): {data[:40].hex()}")

# Look at the Huffman header area (after frame header, block header)
# Frame header: 4 magic + 1 descriptor + optional
# Block header: 3 bytes
# Literals header: starts after

# For compressed literals with Huffman, the Huffman tree description
# starts at the literals header
print(f"\nBytes at offset 10-35 (literals area): {data[10:35].hex()}")

# Try to decompress to verify it works
dctx = zstd.ZstdDecompressor()
decompressed = dctx.decompress(data)
print(f"\nDecompressed size: {len(decompressed)} bytes")
print(f"First 50 bytes: {decompressed[:50]}")

# Count character frequencies to understand expected Huffman distribution
from collections import Counter
char_counts = Counter(decompressed)
print(f"\nCharacter frequency (top 20):")
for char, count in char_counts.most_common(20):
    print(f"  {repr(chr(char)):5s} ({char:3d}): {count:4d}")
