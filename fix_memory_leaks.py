#!/usr/bin/env python3
"""
Fix memory leaks by adding device_workspace cleanup to all error returns in compress function.
The device_workspace is allocated at line ~1213 and needs to be freed on all error paths.
"""

import re

# Read the file
with open('src/cuda_zstd_manager.cu', 'r') as f:
    lines = f.readlines()

# Find the compress function (starts around line 1002, ends around line 2370)
# We need to add cleanup before all returns between allocation and cleanup_and_fail label

# Lines that already have cleanup (don't modify):
# - Line 1217: early return before allocation
# - Line 1491: already fixed 
# - Line 1542: already fixed  
# - Line 2364: cleanup_and_fail label

# Lines that need cleanup (returns between 1213 and 2364):
fixes_needed = [
    # Format: (line_number_0_indexed, return_line_content_to_match)
    # We'll insert cleanup BEFORE these returns
]

# Scan for returns between lines 1220-2364
in_compress = False
device_workspace_allocated = False
for i, line in enumerate(lines):
    line_num = i + 1
    
    # Track if we're in the compress function
    if 'virtual Status compress' in line and 'override' in line:
        in_compress = True
        print(f"Found compress function start at line {line_num}")
    
    # Track allocation point
    if in_compress and 'cudaMalloc(&device_workspace' in line:
        device_workspace_allocated = True
        print(f"Found device_workspace allocation at line {line_num}")
   
    # Track end of compress function
    if in_compress and line_num > 2370:
        in_compress = False
        device_workspace_allocated = False
   
    # Find returns that need fixing
    if in_compress and device_workspace_allocated and line_num > 1220 and line_num < 2364:
        if 'return Status::' in line and 'if (device_workspace)' not in lines[i-1]:
            # Check if this return already has cleanup in previous line
            prev_line = lines[i-1] if i > 0 else ""
            if 'cudaFree(device_workspace)' not in prev_line:
                fixes_needed.append((i, line.strip(), line_num))
                print(f"Line {line_num} needs fix: {line.strip()[:60]}")

print(f"\nTotal lines needing fixes: {len(fixes_needed)}")

# Apply fixes in reverse order (so line numbers don't shift)
for line_idx, return_stmt, line_num in reversed(fixes_needed):
    # Get indentation of the return statement
    indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    cleanup_line = ' ' * indent + 'if (device_workspace) cudaFree(device_workspace);\n'
    
    # Insert cleanup before the return
    lines.insert(line_idx, cleanup_line)
    print(f"Added cleanup before line {line_num}: {return_stmt[:60]}")

# Write back
with open('src/cuda_zstd_manager.cu', 'w') as f:
    f.writelines(lines)

print(f"\nFixed {len(fixes_needed)} memory leak locations!")
