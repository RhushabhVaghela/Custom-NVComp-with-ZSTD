# Legacy ParseCost Usage Inventory

## Summary

**Total locations found**: 27 uses of old ParseCost format

**Files affected**:
- `src/lz77_parallel.cu`: 8 locations
- `src/cuda_zstd_lz77.cu`: 19 locations

---

## src/lz77_parallel.cu (8 locations)

### Function: `compute_costs_kernel` (Lines 50-104)
- Line 61: `best_match.offset = offset;`
- Line 86: `literal_cost.len = 1;`
- Line 87: `literal_cost.is_match = false;`
- Line 97: `best_cost.len = m.length;`
- Line 98: `best_cost.offset = m.offset;`
- Line 99: `best_cost.is_match = true;`

**Priority**: HIGH (used in compute_optimal_parse)
**Effort**: MEDIUM (needs rewrite for parent-based DP)

### Function: `compute_optimal_parse` (Lines 260-280)
- Line 271: `initial_cost.len = 0;`
- Line 272: `initial_cost.is_match = false;`

**Priority**: HIGH (main entry point)
**Effort**: LOW (just initialization, replace with `.set()`)

---

## src/cuda_zstd_lz77.cu (19 locations)

### Match Finding Functions (Lines 461, 469, 587)
- Line 461, 469, 587: `m.offset = ...` (Match struct, NOT ParseCost)

**Priority**: NONE (These are Match structs, not ParseCost - OK)

### Function: `optimal_parse_kernel` (Lines 700-750)
- Line 713-715: Literal assignment
- Line 741-743: Match assignment

**Priority**: HIGH (V1 kernel - can be REPLACED entirely with V2)
**Effort**: ZERO (delete old, use V2)

### Function: `count_sequences_kernel` (Lines ~810-840)
- Line 819, 824: Read `.len` field

**Priority**: MEDIUM (used for sequence counting)
**Effort**: HIGH (needs complete rewrite with parent-walking)

### Function: CPU Optimal Parse (Lines ~1250-1300)
- Lines 1266-1268: Literal assignment  
- Lines 1291-1293: Match assignment

**Priority**: LOW (CPU fallback, rarely used)
**Effort**: MEDIUM (can comment out or rewrite)

---

## Migration Strategy

### Phase 1: Replace V1 Kernel with V2 ✅
Delete `optimal_parse_kernel` (lines 700-750), it's replaced by V2
**Effort**: 5 minutes
**Test**: Code compiles

### Phase 2: Update Main Functions (HIGH priority)
1. `compute_optimal_parse` - initialization only
2. `compute_costs_kernel` - rewrite for parent-based DP

**Effort**: 1-2 hours
**Test**: V2 test still works

### Phase 3: Update Helper Functions (MEDIUM priority)
1. `count_sequences_kernel` - rewrite with parent-walking

**Effort**: 1 hour
**Test**: Sequence counting correct

### Phase 4: CPU Fallback (LOW priority - OPTIONAL)
1. Comment out or rewrite CPU optimal parse

**Effort**: 30 mins - 1 hour
**Test**: Build succeeds

---

## Total Effort Estimate

- Phase 1: 5 mins
- Phase 2: 1-2 hours
- Phase 3: 1 hour
- Phase 4: 30 mins (optional)

**Total**: 2.5-4 hours for complete migration
