"""Shared pytest fixtures and skip markers for cuda_zstd tests."""

from __future__ import annotations

import os
import random

import pytest

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

try:
    import cuda_zstd

    _CUDA_OK = cuda_zstd.is_cuda_available()
except Exception:
    _CUDA_OK = False

requires_cuda = pytest.mark.skipif(
    not _CUDA_OK,
    reason="No CUDA-capable GPU detected or cuda_zstd._core not built",
)
"""Apply ``@requires_cuda`` to tests that need a working GPU."""

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

requires_numpy = pytest.mark.skipif(
    not _HAS_NUMPY,
    reason="NumPy is not installed",
)

try:
    import cupy as cp  # noqa: F401

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

requires_cupy = pytest.mark.skipif(
    not _HAS_CUPY,
    reason="CuPy is not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_data() -> bytes:
    """A small, repetitive payload (~1 KB)."""
    return b"hello world! " * 80  # 1040 bytes


@pytest.fixture()
def medium_data() -> bytes:
    """A moderately sized payload (~100 KB) with mixed content."""
    rng = random.Random(42)
    chunks: list[bytes] = []
    for _ in range(100):
        # Mix of repetitive and random segments
        chunks.append(b"ABCDEFGH" * 64)  # 512 bytes repetitive
        chunks.append(bytes(rng.getrandbits(8) for _ in range(512)))  # 512 random
    return b"".join(chunks)  # ~100 KB


@pytest.fixture()
def large_data() -> bytes:
    """A larger payload (~1 MB) for throughput tests."""
    rng = random.Random(123)
    block = bytes(rng.getrandbits(8) for _ in range(1024))
    # Repeat + vary to create realistic compressible data
    chunks = []
    for i in range(1024):
        if i % 4 == 0:
            chunks.append(block)
        else:
            chunks.append(block[:512] + bytes(rng.getrandbits(8) for _ in range(512)))
    return b"".join(chunks)  # ~1 MB


@pytest.fixture()
def batch_data() -> list[bytes]:
    """A list of 8 buffers of varying sizes for batch tests."""
    rng = random.Random(999)
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    return [bytes(rng.getrandbits(8) for _ in range(s)) for s in sizes]
