# Complete ParseCost Migration Guide

## Current Status

### ✅ V2 Implementation Complete
- **V2 Kernel**: `src/cuda_zstd_lz77.cu` (lines 1371-1406) ✅
- **V2 Wrapper**: `src/lz77_parallel.cu` (lines 334-372) ✅  
- **Header Declarations**: Both `.h` files updated ✅
- **Standalone Demo**: `tests/standalone_v2_demo.cu` compiles and runs ✅

### ❌ Blocker: ParseCost Format Incompatibility

Old format (used in 500+ lines of legacy code):
```cpp
struct ParseCost {
    u32 cost;
    u32 len;  
    u32 offset;
    bool is_match;
};
```

New format (required for V2):
```cpp
struct ParseCost {
    unsigned long long data;
    void set(u32 cost, u32 parent);
    u32 cost() const;
    u32 parent() const;  
};
```

## Migration Steps

### Step 1: Understand the Change

**Old approach**: Store match metadata directly
- `.cost` = cumulative cost
- `.len` = match/literal length
- `.offset` = match offset
- `.is_match` = type flag

**New approach**: Store only DP state (cost + parent pointer)
- `.cost()` = cumulative cost (high 32 bits)
- `.parent()` = parent position (low 32 bits)
- Backtracking reconstructs matches by walking parents

### Step 2: Update All Kernels

**Files to modify**: `src/cuda_zstd_lz77.cu`

**Functions to update** (lines 795-1360):
1. `count_sequences_kernel` (lines 795-838)
2. `backtrack_parallel_kernel` (lines 840-920)
3. `backtrack_sequences_host` (lines 1100-1200)
4. `lz77_compress_offload_optimal_parse_to_cpu` (lines 1209-1360)

**Migration pattern for each function**:

```cpp
// OLD CODE (delete this):
while (pos > 0) {
    ParseCost entry = costs[pos];
    if (entry.is_match) {
        match_lengths[i] = entry.len;
        offsets[i] = entry.offset;
        pos -= entry.len;
    } else {
        literal_count++;
        pos--;
    }
}

// NEW CODE (replace with this):
// Build parent chain first
std::vector<u32> path;
u32 pos = input_size;
while (pos > 0) {
    path.push_back(pos);
    u32 parent = costs[pos].parent();
    if (parent >= pos) break;  // Safety
    pos = parent;
}

// Walk forward through path to reconstruct sequences
// At each position, check if we skipped ahead (match) or moved by 1 (literal)
for (int i = path.size() - 1; i > 0; i--) {
    u32 from = path[i];
    u32 to = path[i-1];
    Match& match = matches[from];
    
    if (to - from > 1 && match.length >= 3) {
        // Match
        match_lengths[seq_idx] = to - from;
        offsets[seq_idx] = match.offset;
        literal_lengths[seq_idx] = 0;
    } else {
        // Literal(s)
        literal_lengths[seq_idx] = to - from;
        match_lengths[seq_idx] = 0;
    }
    seq_idx++;
}
```

### Step 3: Update Initialization

**Old**:
```cpp
h_costs[0] = {0, 0, 0, false};
h_costs[i].cost = 1000000000;
```

**New**:
```cpp
h_costs[0].set(0, 0);
cudaMemset(h_costs + 1, 0xFF, input_size * sizeof(ParseCost));
```

### Step 4: Update Comparisons

**Old**:
```cpp
if (new_cost < h_costs[pos].cost) {
    h_costs[pos].cost = new_cost;
    h_costs[pos].len = length;
    h_costs[pos].offset = offset;
    h_costs[pos].is_match = true;
}
```

**New**:
```cpp
ParseCost new_val;
new_val.set(new_cost, parent_pos);
atomicMin(&h_costs[pos].data, new_val.data);
```

### Step 5: Test Each Function

After updating each function:
1. Compile
2. Run test
3. Verify correctness
4. Fix any issues
5. Move to next function

## Estimated Timeline

- **Step 1** (Understanding): 30 mins
- **Step 2** (Update 4 functions): 2-3 hours
- **Step 3-4** (Init & comparisons): 30 mins
- **Step 5** (Testing): 1 hour

**Total**: 4-5 hours for complete migration

## Quick Win Alternative

If you want V2 working NOW:

1. Create NEW test file using ONLY V2 code (no legacy)
2. Test V2 in isolation
3. Migrate legacy code later when needed

This gets you V2 performance immediately while deferring the full migration.
