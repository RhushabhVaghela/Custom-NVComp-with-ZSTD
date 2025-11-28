// ============================================================================
// Preparation Kernel: Calculate codes and extra bits from raw values
// ============================================================================

__global__ void fse_prepare_sequences_kernel(
    const u32* d_literal_lengths,
    const u32* d_match_lengths,
    const u32* d_offsets,
    u32 num_sequences,
    u8* d_ll_codes,
    u8* d_ml_codes,
    u8* d_of_codes,
    u32* d_ll_extras,
    u32* d_ml_extras,
    u32* d_of_extras,
    u8* d_ll_num_bits,
    u8* d_ml_num_bits,
    u8* d_of_num_bits
) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sequences) return;
    
    u32 ll = d_literal_lengths[idx];
    u32 ml = d_match_lengths[idx];
    u32 of = d_offsets[idx];
    
    // Calculate literal length code and extra bits
    u32 ll_code = sequence::ZstdSequence::get_lit_len_code(ll);
    u32 ll_extra = sequence::ZstdSequence::get_lit_len_extra_bits(ll);
    u32 ll_num_bits = sequence::ZstdSequence::get_lit_len_num_extra_bits(ll_code);
    
    // Calculate match length code and extra bits
    u32 ml_code = sequence::ZstdSequence::get_match_len_code(ml);
    u32 ml_extra = sequence::ZstdSequence::get_match_len_extra_bits(ml);
    u32 ml_num_bits = sequence::ZstdSequence::get_match_len_num_extra_bits(ml_code);
    
    // Calculate offset code and extra bits
    u32 of_code = sequence::ZstdSequence::get_offset_code(of);
    u32 of_extra = sequence::ZstdSequence::get_offset_code_extra_bits(of_code);
    u32 of_num_bits = sequence::ZstdSequence::get_offset_code_num_extra_bits(of_code);
    
    // Store results
    d_ll_codes[idx] = (u8)ll_code;
    d_ml_codes[idx] = (u8)ml_code;
    d_of_codes[idx] = (u8)of_code;
    
    d_ll_extras[idx] = ll_extra;
    d_ml_extras[idx] = ml_extra;
    d_of_extras[idx] = of_extra;
    
    d_ll_num_bits[idx] = (u8)ll_num_bits;
    d_ml_num_bits[idx] = (u8)ml_num_bits;
    d_of_num_bits[idx] = (u8)of_num_bits;
    
    // Debug first sequence
    if (idx == 0) {
        printf("[PREPARE_CODES] Seq 0: LL=%u→code=%u(extra=%u,%ubits) ML=%u→code=%u(extra=%u,%ubits) OF=%u→code=%u(extra=%u,%ubits)\n",
               ll, ll_code, ll_extra, ll_num_bits,
               ml, ml_code, ml_extra, ml_num_bits,
               of, of_code, of_extra, of_num_bits);
    }
}

