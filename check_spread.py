
def build_dtable(normalized_counts, table_log, max_symbol):
    table_size = 1 << table_log
    symbol = [0] * table_size
    
    # Spread
    high_threshold = table_size - 1
    # Phase 1: -1 -> High
    for s, c in enumerate(normalized_counts):
        if c == -1:
            symbol[high_threshold] = s
            high_threshold -= 1
            
    # Phase 2: Spread
    step = (table_size >> 1) + (table_size >> 3) + 3
    mask = table_size - 1
    pos = 0
    
    for s, c in enumerate(normalized_counts):
        if c <= 0: continue
        for _ in range(c):
            symbol[pos] = s
            pos = (pos + step) & mask
            while pos > high_threshold:
                pos = (pos + step) & mask
                
    return symbol

# Default LL (Total 64)
# 0:4, 1:3, 2-12:2, 13-15:1, 16-24:2, 25:3, 26:2, 27-35:1
default_ll_norm = [4, 3] + [2]*11 + [1]*3 + [2]*9 + [3, 2] + [1]*9
# Check sum
print(f"LL Sum: {sum(default_ll_norm)}") # Should be 64
ll_table = build_dtable(default_ll_norm, 6, 35)
print(f"LL State 62: Sym={ll_table[62]}")
print(f"LL State 19: Sym={ll_table[19]}")

# Default ML (Total 64)
# 0:1, 1:4, 2:3, 3-8:2, 9-52:1
default_ml_norm = [1, 4, 3] + [2]*6 + [1]*(52-9+1)
print(f"ML Sum: {sum(default_ml_norm)}") # Should be 64
ml_table = build_dtable(default_ml_norm, 6, 52)
print(f"ML State 62: Sym={ml_table[62]}")
print(f"ML State 19: Sym={ml_table[19]}")

print("\n--- LL Table Map ---")
for s in [0, 24, 25, 26, 27, 28, 29, 30]:
    states = [i for i, sym in enumerate(ll_table) if sym == s]
    print(f"Sym {s}: States {states}")

print("\n--- ML Table Map ---")
for s in [0, 3, 30, 43, 46, 52]:
    states = [i for i, sym in enumerate(ml_table) if sym == s]
    print(f"Sym {s}: States {states}")
