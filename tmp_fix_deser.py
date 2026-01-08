import re

# Read the file
with open('src/cuda_zstd_huffman.cu', 'r') as f:
    content = f.read()

# Find and replace the deserialize_huffman_table_rfc8878 function
old_func = '''__host__ Status deserialize_huffman_table_rfc8878(const byte_t *h_input,
                                                  u32 input_size,
                                                  u8 *h_code_lengths,
                                                  u32 *header_size) {
  if (input_size < 1) {
    return Status::ERROR_CORRUPT_DATA;
  }

  u8 header_byte = h_input[0];
  u8 h_weights[MAX_HUFFMAN_SYMBOLS] = {0};
  u32 num_symbols = 0;
  Status status;

  if (header_byte >= 128) {
    // Direct representation (4-bit weights)
    *header_size = 1 + ((header_byte - 127 + 1) / 2); // header + weight bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_direct(h_input + 1, header_byte, h_weights,
                                           &num_symbols);
  } else {
    // FSE-compressed representation
    *header_size = 1 + header_byte; // header + compressed bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_fse(h_input + 1, header_byte, h_weights,
                                        &num_symbols);
  }'''

new_func = '''__host__ Status deserialize_huffman_table_rfc8878(const byte_t *h_input,
                                                  u32 input_size,
                                                  u8 *h_code_lengths,
                                                  u32 *header_size) {
  if (input_size < 1) {
    return Status::ERROR_CORRUPT_DATA;
  }

  u8 header_byte = h_input[0];

  // Check for custom format: [MaxBits(1)][CodeLengths(256)]
  // serialize_huffman_table writes MAX_HUFFMAN_BITS (=24) as first byte
  if (header_byte == MAX_HUFFMAN_BITS && input_size >= 1 + MAX_HUFFMAN_SYMBOLS) {
    memcpy(h_code_lengths, h_input + 1, MAX_HUFFMAN_SYMBOLS);
    *header_size = 1 + MAX_HUFFMAN_SYMBOLS;
    return Status::SUCCESS;
  }

  u8 h_weights[MAX_HUFFMAN_SYMBOLS] = {0};
  u32 num_symbols = 0;
  Status status;

  if (header_byte >= 128) {
    // Direct representation (4-bit weights)
    *header_size = 1 + ((header_byte - 127 + 1) / 2); // header + weight bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_direct(h_input + 1, header_byte, h_weights,
                                           &num_symbols);
  } else {
    // FSE-compressed representation
    *header_size = 1 + header_byte; // header + compressed bytes

    if (input_size < *header_size) {
      return Status::ERROR_CORRUPT_DATA;
    }

    status = decode_huffman_weights_fse(h_input + 1, header_byte, h_weights,
                                        &num_symbols);
  }'''

content = content.replace(old_func, new_func)

# Write back
with open('src/cuda_zstd_huffman.cu', 'w') as f:
    f.write(content)

print("Fixed deserialize_huffman_table_rfc8878")
