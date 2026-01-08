kernel_code = """

/**
 * @brief Forward-reading Huffman decoder kernel for single-stream mode.
 * Reads the bitstream forward (as written by the encoder).
 * Canonical codes are written LSB-first, so we read LSB-first without reversal.
 */
__global__ void huffman_decode_forward_kernel(
    const byte_t *input, u32 input_size, const u32 *d_first_code,
    const u16 *d_symbol_index, const u8 *d_symbols, byte_t *output,
    u32 total_regen_size, u32 bitstream_start_bits) {

  u32 max_len = d_symbol_index[0];
  u32 bit_pos = bitstream_start_bits;
  const u32 end_bit_pos = input_size * 8;
  u32 num_decoded = 0;

  // Bit container for forward reading
  u64 bit_container = 0;
  u32 bits_available = 0;

  while (num_decoded < total_regen_size) {
    // Refill bit container when low
    while (bits_available <= 56 && bit_pos < end_bit_pos) {
      u32 byte_idx = bit_pos >> 3;
      if (byte_idx >= input_size)
        break;
      u32 bits_in_byte = min(8u, end_bit_pos - bit_pos);
      u8 next_byte = input[byte_idx] & ((1u << bits_in_byte) - 1);
      bit_container |= ((u64)next_byte) << bits_available;
      bits_available += bits_in_byte;
      bit_pos += bits_in_byte;
    }

    if (bits_available == 0)
      break;

    // Decode symbol - read LSB-first (codes are stored LSB-first)
    u32 code = 0;
    u32 len = 0;
    for (u32 l = 1; l <= max_len; l++) {
      if (l > bits_available)
        break;
      // Extract bottom 'l' bits (LSB-first reading)
      code = (u32)(bit_container & ((1U << l) - 1));

      u32 count_at_len = d_symbol_index[l + 1] - d_symbol_index[l];
      if (count_at_len > 0 && code >= d_first_code[l] &&
          code < d_first_code[l] + count_at_len) {
        len = l;
        break;
      }
    }

    if (len == 0)
      break; // Failed to decode

    u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];
    output[num_decoded] = symbol;

    num_decoded++;
    bit_container >>= len;
    bits_available -= len;
  }
}

"""

# Read the file
with open('src/cuda_zstd_huffman.cu', 'r') as f:
    content = f.read()

# Find the position to insert (before the decode_huffman_rfc8878 function)
marker = 'Status decode_huffman_rfc8878(const byte_t *d_input, size_t input_size,'
pos = content.find(marker)

if pos == -1:
    print("Could not find insertion point")
    exit(1)

# Insert the kernel code before the function
new_content = content[:pos] + kernel_code + content[pos:]

# Update single-stream path to use forward decoder
old_call = """  } else {
    huffman_decode_rfc8878_kernel<<<1, 1, 0, stream>>>(
        d_bitstream_base, input_size, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size, 0, bitstream_size_base * 8, 0, 0,
        decompressed_size);
  }"""

new_call = """  } else {
    // Use forward-reading decoder for single-stream mode
    huffman_decode_forward_kernel<<<1, 1, 0, stream>>>(
        d_bitstream_base, bitstream_size_base, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size, 0);
  }"""

new_content = new_content.replace(old_call, new_call)

# Write the file back
with open('src/cuda_zstd_huffman.cu', 'w') as f:
    f.write(new_content)

print("Successfully added forward decoder kernel")
