#!/usr/bin/env python3
import sys

def inspect(filename):
    with open(filename, 'rb') as f:
        data = f.read()

    # Seq section from test_interop.zst (Offset 106, 8 bytes)
    seq_data = data[106:114] 
    val = int.from_bytes(seq_data, 'little')
    bit_len = val.bit_length()
    pos = bit_len - 1
    
    print(f"Total Bits: {bit_len}. Start Pos: {pos}")
    
    def read(n, name):
        nonlocal pos, val
        if n == 0:
            print(f"{name}: 0 bits")
            return 0
        res = (val >> (pos - n + 1)) & ((1 << n) - 1)
        print(f"{name} ({n}): {res} (0b{res:0{n}b})")
        pos -= n
        return res

    # 1. Initialization
    ll_state = read(6, "LL Init")
    of_state = read(5, "OF Init")
    ml_state = read(6, "ML Init")
    
    print("-" * 20)
    print("STEP 1 (Decoded First - Seq Index 1)")
    
    # Expected Symbols: LL=25, OF=5, ML=1
    
    # 1. OF Extra (Sym 5)
    # Code 5: 1 bit extra (RFC Table 10)
    of_extra = read(1, "OF Extra (Sym 5)")
    
    # 2. ML Extra (Sym 1)
    # Code 1: 0 bits extra (RFC Table 9)
    ml_extra = read(0, "ML Extra (Sym 1)")
    
    # 3. LL Extra (Sym 25)
    # Code 25: 2 bits extra (RFC Table 8)
    ll_extra = read(2, "LL Extra (Sym 25)")
    
    print(f"Pos after Extras: {pos}")
    
    # 4. State Updates
    # LL State 39 (Sym 25) -> nbBits 4 (Ref)
    ll_u = read(4, "LL Update (State 39)")
    
    # ML State 44 (Sym 1) -> nbBits 4 (Ref)
    ml_u = read(4, "ML Update (State 44)")
    
    # OF State 10 (Sym 5) -> nbBits 5 (Ref/Assumed)
    of_u = read(5, "OF Update (State 10)")
    
    print("-" * 20)
    print("STEP 2 (Decoded Last - Seq Index 0)")
    
    # Expected Symbols: LL=24, OF=7, ML=50 (Patched)
    
    # 1. OF Extra (Sym 7)
    # Code 7: 2 bits extra (RFC Table 10)
    of_extra_2 = read(2, "OF Extra (Sym 7)")
    
    # 2. ML Extra (Sym 50)
    # Code 50: 16 bits extra (RFC Table 9 for Code 50: Extra 16? Wait)
    # Code 50 is Match Length 2079. 
    # Table 9: Code 50 -> 16 bits.
    # Code 52 -> 16 bits.
    ml_extra_2 = read(16, "ML Extra (Sym 50)")
    
    # 3. LL Extra (Sym 24)
    # Code 24: 2 bits extra (RFC Table 8)
    ll_extra_2 = read(2, "LL Extra (Sym 24)")

if __name__ == "__main__":
    inspect(sys.argv[1])
