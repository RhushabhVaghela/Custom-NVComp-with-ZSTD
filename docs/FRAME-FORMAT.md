# CUDA-ZSTD Frame Format Reference

## Overview

CUDA-ZSTD produces RFC 8878 compliant ZSTD frames, ensuring interoperability with CPU ZSTD implementations.

## Frame Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         ZSTD Frame                               │
├─────────────────────────────────────────────────────────────────┤
│ Magic Number (4 bytes): 0xFD2FB528                              │
├─────────────────────────────────────────────────────────────────┤
│ Frame Header (2-14 bytes)                                        │
│   - Frame Header Descriptor (1 byte)                             │
│   - Window Descriptor (0-1 bytes)                                │
│   - Dictionary ID (0-4 bytes)                                    │
│   - Frame Content Size (0-8 bytes)                               │
├─────────────────────────────────────────────────────────────────┤
│ Block 1                                                          │
│   - Block Header (3 bytes)                                       │
│   - Block Content (variable)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Block 2...N                                                      │
├─────────────────────────────────────────────────────────────────┤
│ Content Checksum (0-4 bytes) - XXHash64 truncated               │
└─────────────────────────────────────────────────────────────────┘
```

## Frame Header Descriptor

```
Byte 0: Frame Header Descriptor
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │
├───┴───┼───┴───┼───┴───┼───┴───┤
│ FCS   │ SWS   │ Dict  │ Cflag │
└───────┴───────┴───────┴───────┘

FCS (bits 7-6): Frame Content Size field size
  00: 0 bytes (unknown size)
  01: 1 byte
  10: 2 bytes  
  11: 8 bytes

SWS (bit 5): Single Segment (no window descriptor)
Dict (bits 4-3): Dictionary ID field size
Cflag (bit 0): Content checksum present
```

## Block Types

| Type | Value | Description |
|:-----|:-----:|:------------|
| Raw | 0 | Uncompressed data |
| RLE | 1 | Single byte repeated |
| Compressed | 2 | ZSTD compressed |
| Reserved | 3 | Not used |

## Block Header

```
3-byte block header:
┌────────────────────┬───────────────────────┐
│ Last Block (1 bit) │ Block Type (2 bits)   │
├────────────────────┴───────────────────────┤
│ Block Size (21 bits)                        │
└─────────────────────────────────────────────┘
```

## Compressed Block Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ Literals Section                                                 │
│   - Literals Header (1-5 bytes)                                  │
│   - Huffman Tree (optional)                                      │
│   - Compressed Literals                                          │
├─────────────────────────────────────────────────────────────────┤
│ Sequences Section                                                │
│   - Sequence Header (1-3 bytes)                                  │
│   - FSE Tables (Literals Length, Offset, Match Length)           │
│   - Compressed Sequences                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Magic Numbers

| Magic | Hex | Type |
|:------|:----|:-----|
| ZSTD Frame | `0xFD2FB528` | Standard frame |
| Skippable | `0x184D2A5?` | Metadata frame |
| Dictionary | `0xEC30A437` | Dictionary |

## Source Files

| File | Description |
|:-----|:------------|
| `include/cuda_zstd_types.h` | Frame constants |
| `src/cuda_zstd_manager.cu` | Frame assembly |

## Related Documentation
- [ARCHITECTURE-OVERVIEW.md](ARCHITECTURE-OVERVIEW.md)
- [MANAGER-IMPLEMENTATION.md](MANAGER-IMPLEMENTATION.md)
