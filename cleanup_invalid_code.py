#!/usr/bin/env python3
import re

# Read the file
with open('src/cuda_zstd_manager.cu', 'r') as f:
    content = f.read()

# Pattern to match the invalid cleanup code
# This matches the entire block from "// Early return OK" through "return Status::ERROR_BUFFER_TOO_SMALL;"
pattern = r'      return Status::ERROR_BUFFER_TOO_SMALL; // Early return OK\s*\n\s*// \(device_workspace handled if\s*\n\s*// allocated\?\)\s*\n\s*// Wait, device_workspace allocated above needs free!\s*\n\s*// Changing strictly to use a cleanup routine at end is safer\.\s*\n\s*// But for this specific leak \(device_workspace\), let\'s track it\.\s*\n\s*if \(device_workspace\)\s*\n\s*cudaFree\(device_workspace\);\s*\n\s*return Status::ERROR_BUFFER_TOO_SMALL;'

# Replace with single return
replacement = '      return Status::ERROR_BUFFER_TOO_SMALL;'

# Perform replacement
new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Also handle variations with different indentation
patterns_to_clean = [
    (r'        return Status::ERROR_BUFFER_TOO_SMALL; // Early return OK\s*\n\s*// \(device_workspace handled if\s*\n\s*// allocated\?\)\s*\n\s*// Wait, device_workspace allocated above needs free!\s*\n\s*// Changing strictly to use a cleanup routine at end is safer\.\s*\n\s*// But for this specific leak \(device_workspace\), let\'s track it\.\s*\n\s*if \(device_workspace\)\s*\n\s*cudaFree\(device_workspace\);\s*\n\s*return Status::ERROR_BUFFER_TOO_SMALL;', '        return Status::ERROR_BUFFER_TOO_SMALL;'),
    (r'          return Status::ERROR_BUFFER_TOO_SMALL; // Early return OK\s*\n\s*// \(device_workspace handled if\s*\n\s*// allocated\?\)\s*\n\s*// Wait, device_workspace allocated above needs free!\s*\n\s*// Changing strictly to use a cleanup routine at end is safer\.\s*\n\s*// But for this specific leak \(device_workspace\), let\'s track it\.\s*\n\s*if \(device_workspace\)\s*\n\s*cudaFree\(device_workspace\);\s*\n\s*return Status::ERROR_BUFFER_TOO_SMALL;', '          return Status::ERROR_BUFFER_TOO_SMALL;'),
]

for pattern, repl in patterns_to_clean:
    new_content = re.sub(pattern, repl, new_content, flags=re.MULTILINE)

# Write back
with open('src/cuda_zstd_manager.cu', 'w') as f:
    f.write(new_content)

print("Cleanup complete!")
