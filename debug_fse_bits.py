#!/usr/bin/env python3
"""Debug FSE bitstream reading - matching EXACT C++ offsets."""

# From C++ debug:
# h_input bytes (first 24): 90 29 6d a8 46 36 df 24 b3 b0 1b 4f fb e6 f0 57 6f 6a 79 d2 63 a7 39
# bit_pos_header=43, bitstream_start=6, compressed_size=23

# h_input is passed as (original_buffer + 1), so it starts AFTER the 0x17 header byte
# original FSE header: 17 90 29 6d a8 46 36 df 24 b3 b0 1b 4f fb e6 f0 57 6f 6a 79 d2 63 a7 39
# h_input = original + 1 = 90 29 6d a8 46 36 df 24 ...

# Bitstream starts at h_input[6] = df

# Let's use the exact h_input bytes from C++
h_input_hex = "90296da84636df24b3b01b4ffbe6f0576f6a79d263a739"
h_input = bytes.fromhex(h_input_hex)
print(f"h_input ({len(h_input)} bytes): {h_input.hex()}")

# Bitstream starts at offset 6
bitstream_start = 6
bitstream = h_input[bitstream_start:]
bitstream_size = len(h_input) - bitstream_start  # 17 bytes
print(f"bitstream ({len(bitstream)} bytes): {bitstream.hex()}")

# Read backward from end
bit_data = int.from_bytes(bitstream, 'little')
total_bits = len(bitstream) * 8  # 136
print(f"Total bits: {total_bits}")

# Find sentinel bit (highest '1' bit)
sentinel_pos = total_bits - 1
while sentinel_pos >= 0:
    if (bit_data >> sentinel_pos) & 1:
        break
    sentinel_pos -= 1

print(f"Sentinel bit at position: {sentinel_pos}")
print(f"Bits available after sentinel: {sentinel_pos}")

# Read backward from just before sentinel
accuracy_log = 6
bit_pos = sentinel_pos

def read_bits_backward(n):
    """Read n bits backward, MSB first (like Zstd FSE)"""
    global bit_pos
    result = 0
    for i in range(n):
        bit_pos -= 1
        if (bit_data >> bit_pos) & 1:
            result |= (1 << (n - 1 - i))
    return result

# Read initial states: state2 first, then state1
state2_raw = read_bits_backward(accuracy_log)
state1_raw = read_bits_backward(accuracy_log)
table_size = 1 << accuracy_log

state2 = state2_raw + table_size
state1 = state1_raw + table_size

print()
print(f"=== Python Results ===")
print(f"state2_raw={state2_raw} (0b{state2_raw:06b}), state2={state2}")
print(f"state1_raw={state1_raw} (0b{state1_raw:06b}), state1={state1}")
print(f"bit_pos after reading states: {bit_pos}")

# What C++ reported:
print()
print("=== C++ Reported ===")
print("S1=83, S2=115, bit_pos after states=121")
print(f"Expected state1_raw = 83-64 = {83-64} = 0b{83-64:06b}")
print(f"Expected state2_raw = 115-64 = {115-64} = 0b{115-64:06b}")

print()
print("=== Comparison ===")
if state1 == 83 and state2 == 115:
    print("MATCH! Python and C++ agree on initial states.")
else:
    print(f"MISMATCH! Python: S1={state1}, S2={state2}  C++: S1=83, S2=115")
