
# Default FSE distribution table reconstruction
# Based on ZSTD_seqSymbol { nextState, nbAddBits, nbBits, baseVal }

# ML Table Data (From grep)
ML_data = [
    # ML1
    (0, 0, 6, 3), (0, 0, 4, 4),
    (32, 0, 5, 5), (0, 0, 5, 6),
    (0, 0, 5, 8), (0, 0, 5, 9),
    (0, 0, 5, 11), (0, 0, 6, 13),
    (0, 0, 6, 16), (0, 0, 6, 19),
    (0, 0, 6, 22), (0, 0, 6, 25),
    (0, 0, 6, 28), (0, 0, 6, 31),
    (0, 0, 6, 34), (0, 1, 6, 37),
    (0, 1, 6, 41), (0, 2, 6, 47),
    (0, 3, 6, 59), (0, 4, 6, 83),
    (0, 7, 6, 131), (0, 9, 6, 515),
    (16, 0, 4, 4), (0, 0, 4, 5),
    # ML2
    (32, 0, 5, 6), (0, 0, 5, 7),
    (32, 0, 5, 9), (0, 0, 5, 10),
    (0, 0, 6, 12), (0, 0, 6, 15),
    (0, 0, 6, 18), (0, 0, 6, 21),
    (0, 0, 6, 24), (0, 0, 6, 27),
    (0, 0, 6, 30), (0, 0, 6, 33),
    (0, 1, 6, 35), (0, 1, 6, 39),
    (0, 2, 6, 43), (0, 3, 6, 51),
    (0, 4, 6, 67), (0, 5, 6, 99),
    (0, 8, 6, 259), (32, 0, 4, 4),
    (48, 0, 4, 4), (16, 0, 4, 5),
    (32, 0, 5, 7), (32, 0, 5, 8),
    (32, 0, 5, 10), (32, 0, 5, 11),
    (0, 0, 6, 14), (0, 0, 6, 17),
    (0, 0, 6, 20), (0, 0, 6, 23),
    (0, 0, 6, 26), (0, 0, 6, 29),
    (0, 0, 6, 32), (0, 16, 6, 65539),
    (0, 15, 6, 32771), (0, 14, 6, 16387),
    (0, 13, 6, 8195), (0, 12, 6, 4099),
    (0, 11, 6, 2051), (0, 10, 6, 1027),
]

# OF Table Data
OF_data = [
    (0, 0, 5, 0), (0, 6, 4, 61),
    (0, 9, 5, 509), (0, 15, 5, 32765),
    (0, 21, 5, 2097149), (0, 3, 5, 5),
    (0, 7, 4, 125), (0, 12, 5, 4093),
    (0, 18, 5, 262141), (0, 23, 5, 8388605),
    (0, 5, 5, 29), (0, 8, 4, 253),
    (0, 14, 5, 16381), (0, 20, 5, 1048573),
    (0, 2, 5, 1), (16, 7, 4, 125),
    (0, 11, 5, 2045), (0, 17, 5, 131069),
    (0, 22, 5, 4194301), (0, 4, 5, 13),
    (16, 8, 4, 253), (0, 13, 5, 8189),
    (0, 19, 5, 524285), (0, 1, 5, 1),
    (16, 6, 4, 61), (0, 10, 5, 1021),
    (0, 16, 5, 65533), (0, 28, 5, 268435453),
    (0, 27, 5, 134217725), (0, 26, 5, 67108861),
    (0, 25, 5, 33554429), (0, 24, 5, 16777213),
]

# LL Table Data
LL_data = [
    # LL1
    (0, 0, 4, 0), (16, 0, 4, 0),
    (32, 0, 5, 1), (0, 0, 5, 3),
    (0, 0, 5, 4), (0, 0, 5, 6),
    (0, 0, 5, 7), (0, 0, 5, 9),
    (0, 0, 5, 10), (0, 0, 5, 12),
    (0, 0, 6, 14), (0, 1, 5, 16),
    (0, 1, 5, 20), (0, 1, 5, 22),
    (0, 2, 5, 28), (0, 3, 5, 32),
    (0, 4, 5, 48), (32, 6, 5, 64),
    (0, 7, 5, 128), (0, 8, 6, 256),
    (0, 10, 6, 1024), (0, 12, 6, 4096),
    (32, 0, 4, 0), (0, 0, 4, 1),
    (0, 0, 5, 2), (32, 0, 5, 4),
    (0, 0, 5, 5), (32, 0, 5, 7),
    (0, 0, 5, 8), (32, 0, 5, 10),
    # LL2
    (0, 0, 5, 11), (0, 0, 6, 13),
    (32, 1, 5, 16), (0, 1, 5, 18),
    (32, 1, 5, 22), (0, 2, 5, 24),
    (32, 3, 5, 32), (0, 3, 5, 40),
    (0, 6, 4, 64), (16, 6, 4, 64),
    (32, 7, 5, 128), (0, 9, 6, 512),
    (0, 11, 6, 2048), (48, 0, 4, 0),
    (16, 0, 4, 1), (32, 0, 5, 2),
    (32, 0, 5, 3), (32, 0, 5, 5),
    (32, 0, 5, 6), (32, 0, 5, 8),
    (32, 0, 5, 9), (32, 0, 5, 11),
    (32, 0, 5, 12), (0, 0, 6, 15),
    (32, 1, 5, 18), (32, 1, 5, 20),
    (32, 2, 5, 24), (32, 2, 5, 28),
    (32, 3, 5, 40), (32, 4, 5, 48),
    (0, 16, 6, 65536), (0, 15, 6, 32768),
    (0, 14, 6, 16384), (0, 13, 6, 8192),
]

def reconstruct(name, data, table_log, max_sym, verify_sum):
    counts = {}
    for entry in data:
        # Entry: (nextState, nbAddBits, nbBits, baseVal)
        nbAddBits = entry[1]
        baseVal = entry[3]
        
        # Deduce Symbol from (nbAddBits, baseVal)
        # Using RFC logic: baseVal encodes the 'Base' for the symbol
        # For OF, baseVal might match directly?
        # For ML, baseVal seems to be Base + 3?
        # For LL, baseVal matches Base directly?
        
        # Simple heursitic: Value = baseVal. Bits = nbAddBits.
        # Find Symbol S such that Table[S].bits == Bits AND Table[S].base == Value (or approx).
        
        # We need RFC Constants
        # Define mappings based on observation
        
        symbol = -1
        
        # Determine symbol
        if name == "ML":
            # ML: Val matches RFC Base + 3.
            # Bits match RFC defined bits.
            # Need RFC ML Table
            # (Bits, Base)
            ML_RFC = [
                (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), 
                (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18),
                (1, 19), (1, 21), (1, 23), (1, 25), (1, 27), (1, 29), (1, 31), (1, 33),
                (2, 35), (2, 39), (2, 43), (2, 47),
                (3, 51), (3, 59), (3, 67), (3, 75),
                (4, 83), (4, 99), (4, 115), (4, 131), # Wait, 131 is next?
                (5, 131), (5, 163), (5, 195), (5, 227),
                (6, 259), (6, 323),
                (7, 515), (7, 643),
                (8, 1027), (8, 1283),
                (9, 2051), (9, 2563),
                (10, 4099), (10, 5123),
                (11, 8195), (11, 10243),
                (12, 16387), (12, 20483),
                (13, 32771), (13, 40963),
                (14, 65539), (14, 81923),
                (15, 131075), (15, 163843),
                (16, 262147), (16, 327683)
            ]
            # My manual table above is approximate.
            # Let's just key off nbAddBits and baseVal from input.
            # Sym 0: Bits 0, Base 3.
            # Sym 52: Bits 16, Base 262147?
            
            # Search logical match
            # Actually, baseVal in Zstd is likely monotonic.
            # 3, 4, 5, 6...
            # 19, 21, 23...
            # 35...
            # The list of bases is fixed.
            # I will just PRINT the (bits, base) pairs and then manually map them if specific values are odd.
            # But better: Just count unique (bits, base) pairs?
            # No, (0, 4) appears for multiple symbols? No.
            # ML/LL/OF symbols have unique (bits, base) pairs.
            # So unique (bits, base) => Unique Symbol.
            # 
            # We can just Dump the list of Symbols identified by (bits, base).
            # But we need to output Array of Counts for S=0..Max.
            # So we MUST map to S.
            pass
        
        # ... logic to find S ...
        # Since I don't have perfect table, I will Output "bits:base" and Count.
        # Then I will map them to S manually in result.
        
        key = f"{nbAddBits}_{baseVal}"
        counts[key] = counts.get(key, 0) + 1
        
    # Sort keys by Base (approx symbol order)
    sorted_keys = sorted(counts.keys(), key=lambda x: int(x.split('_')[1]))
    
    print(f"--- {name} Table ---")
    for k in sorted_keys:
        print(f"Bits/Base {k}: Count {counts[k]}")
        
def generate_cpp_arrays(name, data):
    # Filter unique bases, keep the one with max bits? Or min?
    # Usually they should be consistent.
    # Use dictionary to deduplicate by base.
    
    unique_map = {}
    for entry in data:
        base = entry[3]
        bits = entry[1]
        
        if base in unique_map:
            if unique_map[base] != bits:
                print(f"// WARNING: Conflict for Base {base}: {unique_map[base]} vs {bits}")
        unique_map[base] = bits
        
    sorted_bases = sorted(unique_map.keys())
    
    bits_arr = []
    base_arr = []
    
    for base in sorted_bases:
        bits_arr.append(unique_map[base])
        base_arr.append(base)
        
    print(f"// {name}_bits (Size {len(bits_arr)})")
    print(f"static const u8 {name}_bits[{len(bits_arr)}] = {{")
    print(", ".join(map(str, bits_arr)))
    print("};")
    
    print(f"// {name}_base (Size {len(base_arr)})")
    print(f"static const u32 {name}_base[{len(base_arr)}] = {{")
    print(", ".join(map(str, base_arr)))
    print("};")

generate_cpp_arrays("LL", LL_data)
generate_cpp_arrays("ML", ML_data)
generate_cpp_arrays("OF", OF_data)
