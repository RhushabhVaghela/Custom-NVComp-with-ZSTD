# Read the file
with open('src/cuda_zstd_huffman.cu', 'r') as f:
    content = f.read()

# Fix: codes are stored LSB-first in bitstream, so read MSB-first for canonical lookup
old_kernel = '''/**
 * @brief Forward-reading Huffman decoder for 4-stream interleaved format.
 * Reads 4 interleaved streams and outputs symbols in proper order.
 */
__global__ void huffman_decode_4stream_forward_kernel(
    const byte_t *input, u32 input_size, const u32 *d_first_code,
    const u16 *d_symbol_index, const u8 *d_symbols, byte_t *output,
    u32 total_regen_size) {

  u32 max_len = d_symbol_index[0];

  // Calculate stream sizes (same as encoder)
  u32 N1 = (total_regen_size + 3) / 4;
  u32 N2 = (total_regen_size + 2) / 4;
  u32 N3 = (total_regen_size + 1) / 4;
  u32 N4 = total_regen_size / 4;

  u32 L1 = N1 * 3;  // Estimate: avg 3 bits per symbol
  u32 L2 = N2 * 3;
  u32 L3 = N3 * 3;
  u32 L4 = input_size - L1 - L2 - L3;

  if (threadIdx.x < 4) {
    u32 stream_id = threadIdx.x;
    u32 num_symbols = (stream_id == 0) ? N1 :
                      (stream_id == 1) ? N2 :
                      (stream_id == 2) ? N3 : N4;
    u32 stream_len = (stream_id == 0) ? L1 :
                     (stream_id == 1) ? L2 :
                     (stream_id == 2) ? L3 : L4;

    u32 stream_offset = (stream_id == 0) ? 0 :
                        (stream_id == 1) ? L1 :
                        (stream_id == 2) ? L1 + L1 : L1 + L2 + L3;

    const byte_t *stream_data = input + stream_offset;
    u32 stream_size = stream_len;

    u64 bit_container = 0;
    u32 bits_available = 0;
    u32 bit_pos = 0;

    for (u32 i = 0; i < num_symbols; i++) {
      // Refill
      while (bits_available <= 56 && bit_pos < stream_size * 8) {
        u32 byte_idx = bit_pos >> 3;
        if (byte_idx >= stream_size)
          break;
        u32 bits_in_byte = min(8u, stream_size * 8 - bit_pos);
        u8 next_byte = stream_data[byte_idx] & ((1u << bits_in_byte) - 1);
        bit_container |= ((u64)next_byte) << bits_available;
        bits_available += bits_in_byte;
        bit_pos += bits_in_byte;
      }

      if (bits_available == 0)
        break;

      // Decode
      u32 code = 0;
      u32 len = 0;
      for (u32 l = 1; l <= max_len; l++) {
        if (l > bits_available)
          break;
        code = (u32)(bit_container & ((1U << l) - 1));
        u32 count_at_len = d_symbol_index[l + 1] - d_symbol_index[l];
        if (count_at_len > 0 && code >= d_first_code[l] &&
            code < d_first_code[l] + count_at_len) {
          len = l;
          break;
        }
      }

      if (len == 0)
        break;

      u8 symbol = d_symbols[d_symbol_index[len] + (code - d_first_code[len])];

      // Output in interleaved order
      u32 output_idx = stream_id + i * 4;
      if (output_idx < total_regen_size) {
        output[output_idx] = symbol;
      }

      bit_container >>= len;
      bits_available -= len;
    }
  }
}'''

new_kernel = '''/**
 * @brief Forward-reading Huffman decoder kernel.
 * Reads the bitstream forward (as written by the encoder).
 * Canonical codes are stored LSB-first, so we read MSB-first from the bitstream.
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

    // Decode symbol - read MSB-first from bitstream
    // Codes are stored LSB-first, so we read MSB-first to match canonical ordering
    u32 code = 0;
    u32 len = 0;
    for (u32 l = 1; l <= max_len; l++) {
      if (l > bits_available)
        break;
      // Extract top 'l' bits (MSB-first reading)
      code = (u32)(bit_container >> (bits_available - l)) & ((1U << l) - 1);

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
}'''

content = content.replace(old_kernel, new_kernel)

# Update the call site
old_call = '''  } else {
    // Use forward-reading decoder with 4-stream interleaving
    huffman_decode_4stream_forward_kernel<<<1, 4, 0, stream>>>(
        d_bitstream_base, bitstream_size_base, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size);
  }'''

new_call = '''  } else {
    // Use forward-reading decoder for single-stream mode
    huffman_decode_forward_kernel<<<1, 1, 0, stream>>>(
        d_bitstream_base, bitstream_size_base, d_first_code, d_symbol_index, d_symbols,
        d_output, decompressed_size, 0);
  }'''

content = content.replace(old_call, new_call)

# Write back
with open('src/cuda_zstd_huffman.cu', 'w') as f:
    f.write(content)

print("Fixed forward decoder to read MSB-first")
