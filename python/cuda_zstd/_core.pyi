"""Type stubs for the compiled _core extension module.

These stubs provide IDE auto-complete and type-checking for the native
pybind11 extension.  They are NOT used at runtime.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Union, overload

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_LEVEL: int
MAX_LEVEL: int
DEFAULT_LEVEL: int
__version__: str

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Status(IntEnum):
    SUCCESS: int
    ERROR_GENERIC: int
    ERROR_INVALID_PARAMETER: int
    ERROR_OUT_OF_MEMORY: int
    ERROR_CUDA_ERROR: int
    ERROR_INVALID_MAGIC: int
    ERROR_CORRUPT_DATA: int
    ERROR_BUFFER_TOO_SMALL: int
    ERROR_UNSUPPORTED_VERSION: int
    ERROR_COMPRESSION: int
    ERROR_DECOMPRESSION: int

class Strategy(IntEnum):
    FAST: int
    DFAST: int
    GREEDY: int
    LAZY: int
    LAZY2: int
    BTLAZY2: int
    BTOPT: int
    BTULTRA: int

class ChecksumPolicy(IntEnum):
    NONE: int
    COMPUTE: int
    COMPUTE_AND_VERIFY: int

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CompressionConfig:
    level: int
    strategy: Strategy
    window_log: int
    hash_log: int
    chain_log: int
    search_log: int
    min_match: int
    block_size: int
    enable_ldm: bool
    checksum: ChecksumPolicy

    def __init__(self) -> None: ...
    @staticmethod
    def from_level(level: int) -> CompressionConfig: ...
    @staticmethod
    def optimal(input_size: int) -> CompressionConfig: ...
    @staticmethod
    def get_default() -> CompressionConfig: ...

class CompressionStats:
    input_bytes: int
    output_bytes: int
    num_blocks: int
    compression_time_ms: float
    decompression_time_ms: float

    def get_ratio(self) -> float: ...
    def get_compression_throughput_gbps(self) -> float: ...

# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

_BufferLike = Union[bytes, bytearray, memoryview, Any]

class Manager:
    level: int

    @overload
    def __init__(self, level: int = 3) -> None: ...
    @overload
    def __init__(self, config: CompressionConfig) -> None: ...
    def compress(self, data: _BufferLike) -> bytes: ...
    def decompress(self, data: _BufferLike) -> bytes: ...
    def compress_batch(self, inputs: list[_BufferLike]) -> list[bytes]: ...
    def decompress_batch(self, inputs: list[_BufferLike]) -> list[bytes]: ...
    def get_stats(self) -> CompressionStats: ...
    def reset_stats(self) -> None: ...
    def get_config(self) -> CompressionConfig: ...
    def __enter__(self) -> Manager: ...
    def __exit__(self, *args: object) -> None: ...

# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def compress(data: _BufferLike, level: int = 3) -> bytes: ...
def decompress(data: _BufferLike) -> bytes: ...
def compress_batch(inputs: list[_BufferLike], level: int = 3) -> list[bytes]: ...
def decompress_batch(inputs: list[_BufferLike]) -> list[bytes]: ...
def validate_compressed_data(data: _BufferLike, check_checksum: bool = True) -> bool: ...
def estimate_compressed_size(uncompressed_size: int, level: int = 3) -> int: ...
def is_cuda_available() -> bool: ...
def get_cuda_device_info() -> dict[str, object]: ...
