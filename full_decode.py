#!/usr/bin/env python3
"""Full Python Reference Decoder for Zstd Sequences (Predefined Only)."""

import sys

# --- Constants & Tables ---

# RFC 8878 Dist tables
ML_DEFAULT = [1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

LL_DEFAULT = [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
              2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
              2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1]

OF_DEFAULT = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, -1, -1, -1, -1, -1]

# Base & Bits Tables
def get_ml_val(code):
    if code < 32: return code + 3, 0
    ML_BASE = {32:35, 33:37, 34:39, 35:41, 36:43, 37:47, 38:51, 39:59,
               40:67, 41:83, 42:99, 43:131, 44:163, 45:227, 46:291, 47:419,
               48:547, 49:803, 50:1059, 51:1571, 52:2083}
    ML_BITS = {32:1, 33:1, 34:1, 35:1, 36:2, 37:2, 38:3, 39:3,
               40:4, 41:4, 42:5, 43:5, 44:6, 45:6, 46:7, 47:7,
               48:8, 49:8, 50:9, 51:9, 52:16}
    return ML_BASE[code], ML_BITS[code]

def get_ll_val(code):
    if code < 16: return code, 0
    LL_BASE = {16:16, 17:18, 18:20, 19:22, 20:24, 21:28, 22:32, 23:40,
               24:48, 25:64, 26:128, 27:256, 28:512, 29:1024, 30:2048,
               31:4096, 32:8192, 33:16384, 34:32768, 35:65536}
    LL_BITS = {16:1, 17:1, 18:1, 19:1, 20:2, 21:2, 22:3, 23:3,
               24:4, 25:6, 26:7, 27:8, 28:9, 29:10, 30:11,
               31:12, 32:13, 33:14, 34:15, 35:16}
    return LL_BASE[code], LL_BITS[code]

def get_of_val(code):
    # RFC 8878: offset = 2^code + bits
    if code == 0: return 1, 0 # Should not happen? Or special.
    return 1 << code, code

# --- FSE Table Builder ---
def build_table(norm, max_sym, log):
    size = 1 << log
    table = [{'sym':0, 'bits':0, 'base':0} for _ in range(size)]
    high_thresh = size - 1
    symbol_next = [0]*(max_sym+1)
    
    # Phase 1: Prob -1 (Reverse Index Order? Or Forward?)
    # RFC: "assigned to the last positions in the table, in reverse order of their index"
    # Interpretation 1: Iterate s=0..max. Assign to high--. 
    # (Matches C++ implementation that we suspect is wrong? Or Right?)
    # Let's stick to C++ interpretation for now to verify.
    for s in range(max_sym + 1):
        if norm[s] == -1:
            table[high_thresh]['sym'] = s
            high_thresh -= 1
            symbol_next[s] = 1
        else:
            symbol_next[s] = norm[s]
            
    # Phase 2: Spread
    pos = 0
    step = (size >> 1) + (size >> 3) + 3
    mask = size - 1
    for s in range(max_sym + 1):
        if norm[s] <= 0: continue
        for _ in range(norm[s]):
            table[pos]['sym'] = s
            pos = (pos + step) & mask
            while pos > high_thresh:
                pos = (pos + step) & mask
                
    # Phase 3: Bits & Base
    # Group by symbol to calculate bits/base correctly
    # But for State Update simulation, we need newState.
    # newState calculation:
    # nbBits = log - highBit(next_state)
    # newState = (next_state << nbBits) - size
    for s in range(max_sym+1):
        # We need to know which states map to this symbol to assign next_state
        # Iterate table to find state indices for s
        s_states = [i for i, entry in enumerate(table) if entry['sym'] == s]
        # In FSE, the order of occurrence in loop determines assignment?
        # No, 'symbol_next' tracks the count.
        # But here 'symbol_next[s]' is already filled (initial count).
        # We need to reset it to 1 (for -1) or value?
        # RFC: "We scan the table... for each state..."
        pass

    # Re-scan to compute bits/base
    # Reset symbol_next to 1 (or count for -1?)
    # Wait, next_state for -1 sym starts at 1. for normal starts at count.
    curr_next = [0]*(max_sym+1)
    for s in range(max_sym+1):
        if norm[s] == -1: curr_next[s] = 1
        elif norm[s] > 0: curr_next[s] = norm[s]
        
    for i in range(size):
        s = table[i]['sym']
        ns = curr_next[s]
        curr_next[s] += 1
        
        # Calc bits
        # highBit = floor(log2(ns))? No.
        # nbBits = log - floor(log2(ns))? -> highBit calculation
        # C++: 31 - clz(ns).
        high_bit = 0
        if ns > 0:
            high_bit = ns.bit_length() - 1
        
        nb = log - high_bit
        base = (ns << nb) - size
        
        table[i]['bits'] = nb
        table[i]['base'] = base
        
    return table

# --- Decoder ---
class BitReader:
    def __init__(self, data):
        self.bits = 0
        for i, b in enumerate(data):
            self.bits |= (b << (i*8))
        # Find sentinel
        self.num_bits = self.bits.bit_length() - 1
        # Remove sentinel bit? No, pos starts below it.
        # self.pos points to next bit to read (MSB index)
        # Sentinel is at bit_length()-1.
        # Next bit is bit_length()-2.
        self.pos = self.bits.bit_length() - 1
        
    def read(self, n):
        if n == 0: return 0
        self.pos -= n
        val = (self.bits >> self.pos) & ((1 << n) - 1)
        return val

def decode(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Sequence section starts at 104+2 = 106 
    # (From previous analysis: Seq Section 10 bytes. header 2. Data 8)
    byte_seq = data[106:114]
    
    reader = BitReader(byte_seq)
    
    # Build tables
    ll_tab = build_table(LL_DEFAULT, 35, 6)
    of_tab = build_table(OF_DEFAULT, 28, 5)
    ml_tab = build_table(ML_DEFAULT, 52, 6)
    
    # HACK: Force State 63 -> Sym 50 to test hypothesis
    # Sym 50 has Prob -1. nbBits 6. newState 0 (base).
    # Current State 63 has Sym 46. nbBits 6. newState 0.
    # We just swap the symbol.
    print("APPLYING HACK: ML State 63 -> Sym 50")
    ml_tab[63]['sym'] = 50
    
    # Read initial states
    ll_state = reader.read(6)
    of_state = reader.read(5)
    ml_state = reader.read(6)
    
    print(f"Initial: LL={ll_state}, OF={of_state}, ML={ml_state}")
    
    num_seq = 2
    total_len = 0
    
    for i in range(num_seq):
        # Decode symbol from state
        ll_entry = ll_tab[ll_state]
        of_entry = of_tab[of_state]
        ml_entry = ml_tab[ml_state]
        
        ll_sym = ll_entry['sym']
        of_sym = of_entry['sym']
        ml_sym = ml_entry['sym']
        
        # Read extra bits
        # Order: OF, ML, LL
        of_base_val, of_n = get_of_val(of_sym)
        of_extra = reader.read(of_n)
        offset = of_base_val + of_extra
        
        ml_base_val, ml_n = get_ml_val(ml_sym)
        ml_extra = reader.read(ml_n)
        match_len = ml_base_val + ml_extra
        
        ll_base_val, ll_n = get_ll_val(ll_sym)
        ll_extra = reader.read(ll_n)
        lit_len = ll_base_val + ll_extra
        
        print(f"Seq {i}: LL={lit_len} (Sym {ll_sym}), ML={match_len} (Sym {ml_sym}), OF={offset} (Sym {of_sym})")
        
        total_len += lit_len + match_len
        
        # Update states (if not last)
        # Loop i=0 is first (decoded). i=1 is last.
        # Wait. Reader reads BACKWARDS.
        # The FIRST sequence decoded is the LAST in the file.
        # My loop i=0 decodes the LAST sequence.
        # Loop i=1 decodes the FIRST sequence.
        
        if i < num_seq - 1:
            # Update LL, ML, OF
            ll_state = ll_entry['base'] + reader.read(ll_entry['bits'])
            ml_state = ml_entry['base'] + reader.read(ml_entry['bits'])
            of_state = of_entry['base'] + reader.read(of_entry['bits'])
            print(f"  Updated States: LL={ll_state}, ML={ml_state}, OF={of_state}")

    print(f"Total Decoded Size: {total_len + 127 - (total_len - 127)}") 
    # Wait, LitLen contributes to literals size.
    # Total Output = Lit_Section_Size + Match_Lens?
    # No. Lit_Len sums to Lit_Section_Size.
    # Output size = Sum(Lit_Len) + Sum(Match_Len).
    print(f"Total Output Size: {total_len}")

if __name__ == "__main__":
    decode(sys.argv[1])
