import os
import sys
import torch

# --- Force loader path ---
# This MUST come BEFORE importing the kernel
# Get the directory of the current script (tests/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (V4.0 (Production-Ready)/)
parent_dir = os.path.join(script_dir, '..')
# Get the kernel directory (V4.0 (Production-Ready)/jit_kernel/)
kernel_dir = os.path.join(parent_dir, 'jit_kernel')


# Add the parent and kernel directories to Python's search path
sys.path.append(parent_dir)
sys.path.append(kernel_dir)

# import kernel library
import jit_kernel_cuda

# Test compression with REPETITIVE data
# data = torch.randint(0, 256, (1024*1024,), dtype=torch.uint8).cuda()
data = torch.zeros(1024*1024, dtype=torch.uint8).cuda() # <-- Use this

print(f"Original: {data.nbytes} bytes")
compressed = jit_kernel_cuda.jit_compress_zstd_v1(data)
print(f"Compressed: {compressed.nbytes} bytes") # <-- This will be TINY

# Test decompression
decompressed = jit_kernel_cuda.jit_decompress_zstd_v1(compressed, data.nbytes)
print(f"Decompressed: {decompressed.nbytes} bytes")
print(f"Match: {torch.all(data == decompressed)}\n")

# Summary
print(f"\nOriginal: {data.nbytes} bytes")
print(f"Compressed: {compressed.nbytes} bytes") 
print(f"Decompressed: {decompressed.nbytes} bytes")
print(f"Match: {torch.all(data == decompressed)}")