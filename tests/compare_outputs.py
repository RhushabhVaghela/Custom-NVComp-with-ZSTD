#!/usr/bin/env python3
# Compare GPU payload with CPU bitstream

# Read CPU reference output
with open('cpu_fse_output.bin', 'rb') as f:
    cpu_output = f.read()

print('=== CPU Reference (Zstandard FSE) ===')
print(f'Size: {len(cpu_output)} bytes')
print(f'First 40 bytes: {" ".join(f"{b:02x}" for b in cpu_output[:40])}')
print(f'Last 40 bytes: {" ".join(f"{b:02x}" for b in cpu_output[-40:])}')
print()

# The GPU test writes to stderr, need to capture
import subprocess
result = subprocess.run(['./test_single_chunk'], capture_output=True)
output = result.stdout.decode('utf-8', errors='ignore')

# Find encoded size
for line in output.split('\n'):
    if 'Encoded successfully:' in line:
        gpu_size = int(line.split(':')[1].split()[0])
        print(f'=== GPU Encoder ===')
        print(f'Total size: {gpu_size} bytes')
        print(f'Header: 524 bytes')
        print(f'Payload: {gpu_size - 524} bytes')
        print()
        break

print('=== Comparison ===')
print(f'GPU payload size: {gpu_size - 524} bytes')
print(f'CPU bitstream size: {len(cpu_output)} bytes')
diff = abs((gpu_size - 524) - len(cpu_output))
print(f'Difference: {diff} bytes')

if diff == 0:
    print('\n✓ Sizes match! Need to compare actual bytes.')
else:
    print(f'\n✗ Size mismatch by {diff} bytes')
    print('This suggests different compression algorithms or table formats')
