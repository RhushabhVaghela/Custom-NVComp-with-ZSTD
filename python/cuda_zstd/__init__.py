"""
cuda_zstd — GPU-accelerated Zstandard compression using CUDA
=============================================================

Provides blazing-fast ZSTD compression and decompression on NVIDIA GPUs.

Quick start::

    import cuda_zstd

    compressed = cuda_zstd.compress(b"hello world" * 1000)
    original   = cuda_zstd.decompress(compressed)

    # Reusable manager (recommended for repeated calls)
    with cuda_zstd.Manager(level=3) as mgr:
        c = mgr.compress(data)
        d = mgr.decompress(c)

    # Batch API — compress many buffers in a single GPU launch
    results = cuda_zstd.compress_batch([buf1, buf2, buf3])

Requirements:
    - NVIDIA GPU with CUDA support
    - CUDA Toolkit installed and on PATH
    - (Optional) NumPy for array support
    - (Optional) CuPy for zero-copy GPU arrays
"""

from __future__ import annotations

__all__ = [
    # Core functions
    "compress",
    "decompress",
    "compress_batch",
    "decompress_batch",
    # Validation & estimation
    "validate_compressed_data",
    "estimate_compressed_size",
    # Manager class
    "Manager",
    # Configuration
    "CompressionConfig",
    "CompressionStats",
    # Enums
    "Status",
    "Strategy",
    "ChecksumPolicy",
    # CUDA utilities
    "is_cuda_available",
    "get_cuda_device_info",
    # Constants
    "MIN_LEVEL",
    "MAX_LEVEL",
    "DEFAULT_LEVEL",
    # Version
    "__version__",
]

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy as np

    from cuda_zstd._core import (
        ChecksumPolicy as ChecksumPolicy,
        CompressionConfig as CompressionConfig,
        CompressionStats as CompressionStats,
        Manager as _CManager,
        Status as Status,
        Strategy as Strategy,
    )

# ---------------------------------------------------------------------------
# Import the compiled C++ extension module.
# ---------------------------------------------------------------------------

_CORE_AVAILABLE: bool = False
_IMPORT_ERROR: ImportError | None = None

try:
    from cuda_zstd._core import (
        ChecksumPolicy,  # noqa: F811
        CompressionConfig,  # noqa: F811
        CompressionStats,  # noqa: F811
        Manager as _CManager,  # noqa: F811
        Status,  # noqa: F811
        Strategy,  # noqa: F811
        compress as _compress,
        compress_batch as _compress_batch,
        decompress as _decompress,
        decompress_batch as _decompress_batch,
        estimate_compressed_size,
        get_cuda_device_info,
        is_cuda_available,
        validate_compressed_data,
        DEFAULT_LEVEL,
        MAX_LEVEL,
        MIN_LEVEL,
        __version__,
    )

    _CORE_AVAILABLE = True
except ImportError as _exc:
    _IMPORT_ERROR = _exc

    # Fallback constants so the module is still importable without the
    # native extension (e.g. for documentation builds, type checking).
    MIN_LEVEL = 1  # type: ignore[assignment]
    MAX_LEVEL = 22  # type: ignore[assignment]
    DEFAULT_LEVEL = 3  # type: ignore[assignment]
    __version__ = "1.0.0"  # type: ignore[assignment]  # keep in sync with root pyproject.toml

    def validate_compressed_data(*args: object, **kwargs: object) -> bool:  # type: ignore[misc]
        """Stub — native extension not available."""
        _raise_not_built()
        return False  # unreachable

    def estimate_compressed_size(*args: object, **kwargs: object) -> int:  # type: ignore[misc]
        """Stub — native extension not available."""
        _raise_not_built()
        return 0  # unreachable


def _raise_not_built() -> None:
    """Raise a helpful ImportError when _core is not available."""
    raise ImportError(
        "cuda_zstd native extension (_core) is not available. "
        "This usually means:\n"
        "  1. The package was not built (run `pip install .` from python/)\n"
        "  2. CUDA Toolkit is not installed or not found by CMake\n"
        "  3. No NVIDIA GPU is present\n"
        f"\nOriginal error: {_IMPORT_ERROR}"
    ) from _IMPORT_ERROR


def _ensure_core() -> None:
    """Guard that raises if the native extension is missing."""
    if not _CORE_AVAILABLE:
        _raise_not_built()


# ---------------------------------------------------------------------------
# Type alias for accepted input data
# ---------------------------------------------------------------------------
BufferLike = Union[bytes, bytearray, memoryview, "np.ndarray", Any]
"""Any object that supports the Python buffer protocol (or CuPy array)."""


# ===========================================================================
# High-level Python wrapper
# ===========================================================================


class Manager:
    """GPU-accelerated ZSTD compression/decompression manager.

    A ``Manager`` holds GPU resources (workspace buffers, CUDA streams) and
    can be reused across many compress/decompress calls.  For one-shot usage,
    the module-level :func:`compress` and :func:`decompress` functions are
    more convenient (they create a temporary manager internally).

    Parameters
    ----------
    level : int, default 3
        Compression level (1–22).  Higher levels yield better compression
        ratios at the cost of speed.
    config : CompressionConfig, optional
        Fine-grained configuration.  If provided, ``level`` is ignored.

    Examples
    --------
    >>> with cuda_zstd.Manager(level=5) as mgr:
    ...     compressed = mgr.compress(b"hello " * 10000)
    ...     original = mgr.decompress(compressed)
    ...     print(f"ratio: {len(original) / len(compressed):.1f}x")
    """

    _mgr: _CManager | None

    def __init__(
        self,
        level: int = 3,
        *,
        config: CompressionConfig | None = None,
    ) -> None:
        _ensure_core()

        if config is not None:
            self._mgr = _CManager(config)  # type: ignore[misc]
        else:
            self._mgr = _CManager(level)  # type: ignore[misc]

    # -- Internal helper ----------------------------------------------------

    def _require_open(self) -> _CManager:
        """Return the underlying C++ manager, or raise if closed."""
        if self._mgr is None:
            raise RuntimeError("Manager has been closed")
        return self._mgr  # type: ignore[return-value]

    # -- Compression --------------------------------------------------------

    def compress(self, data: BufferLike) -> bytes:
        """Compress *data* on the GPU and return compressed bytes.

        Parameters
        ----------
        data : bytes, bytearray, numpy.ndarray, or CuPy array
            The uncompressed input.  If a CuPy array is passed, the data is
            used directly on the GPU without an extra host→device copy.

        Returns
        -------
        bytes
            The compressed payload.
        """
        return self._require_open().compress(data)

    def decompress(self, data: BufferLike) -> bytes:
        """Decompress *data* on the GPU and return the original bytes.

        Parameters
        ----------
        data : bytes or buffer
            ZSTD-compressed payload produced by :meth:`compress`.

        Returns
        -------
        bytes
            The decompressed output.
        """
        return self._require_open().decompress(data)

    # -- Batch operations ---------------------------------------------------

    def compress_batch(self, inputs: list[BufferLike]) -> list[bytes]:
        """Compress multiple buffers in a single GPU launch.

        Parameters
        ----------
        inputs : list of buffer-like objects
            Each element is an independent chunk to compress.

        Returns
        -------
        list[bytes]
            One compressed buffer per input.
        """
        return self._require_open().compress_batch(inputs)

    def decompress_batch(self, inputs: list[BufferLike]) -> list[bytes]:
        """Decompress multiple compressed buffers in a single GPU launch.

        Parameters
        ----------
        inputs : list of buffer-like objects
            Each element is a ZSTD-compressed chunk.

        Returns
        -------
        list[bytes]
            One decompressed buffer per input.
        """
        return self._require_open().decompress_batch(inputs)

    # -- Properties ---------------------------------------------------------

    @property
    def level(self) -> int:
        """Current compression level (1–22)."""
        return self._require_open().level

    @level.setter
    def level(self, value: int) -> None:
        self._require_open().level = value

    @property
    def stats(self) -> CompressionStats:
        """Compression statistics since last :meth:`reset_stats` call."""
        return self._require_open().get_stats()

    def reset_stats(self) -> None:
        """Reset accumulated compression/decompression statistics."""
        self._require_open().reset_stats()

    @property
    def config(self) -> CompressionConfig:
        """Current compression configuration (read-only snapshot)."""
        return self._require_open().get_config()

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> Manager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release GPU resources held by this manager.

        After calling ``close()``, the manager must not be used.  Prefer
        using the manager as a context manager instead::

            with cuda_zstd.Manager() as mgr:
                ...
        """
        # The C++ destructor frees GPU memory.  Setting _mgr to None ensures
        # a clear error if the user tries to call compress() after close().
        self._mgr = None

    # -- Dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        if self._mgr is None:
            return "<cuda_zstd.Manager [closed]>"
        return f"<cuda_zstd.Manager level={self.level}>"


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


def compress(data: BufferLike, *, level: int = 3) -> bytes:
    """Compress *data* on the GPU.

    This is a convenience wrapper that creates a temporary
    :class:`Manager` for each call.  For repeated compression,
    create a :class:`Manager` and reuse it.

    Parameters
    ----------
    data : bytes, bytearray, numpy.ndarray, or CuPy array
        Uncompressed input data.
    level : int, default 3
        Compression level (1–22).

    Returns
    -------
    bytes
        Compressed output.

    Examples
    --------
    >>> import cuda_zstd
    >>> compressed = cuda_zstd.compress(b"hello world" * 1000, level=5)
    >>> cuda_zstd.decompress(compressed) == b"hello world" * 1000
    True
    """
    _ensure_core()
    return _compress(data, level)  # type: ignore[name-defined]


def decompress(data: BufferLike) -> bytes:
    """Decompress GPU-ZSTD compressed data.

    Parameters
    ----------
    data : bytes or buffer
        Compressed payload.

    Returns
    -------
    bytes
        Decompressed output.
    """
    _ensure_core()
    return _decompress(data)  # type: ignore[name-defined]


def compress_batch(
    inputs: list[BufferLike], *, level: int = 3
) -> list[bytes]:
    """Compress multiple buffers in a single GPU batch launch.

    Parameters
    ----------
    inputs : list of buffer-like objects
        Independent chunks to compress.
    level : int, default 3
        Compression level.

    Returns
    -------
    list[bytes]
        One compressed buffer per input.
    """
    _ensure_core()
    return _compress_batch(inputs, level)  # type: ignore[name-defined]


def decompress_batch(inputs: list[BufferLike]) -> list[bytes]:
    """Decompress multiple compressed buffers in a single GPU batch.

    Parameters
    ----------
    inputs : list of buffer-like objects
        Compressed chunks.

    Returns
    -------
    list[bytes]
        One decompressed buffer per input.
    """
    _ensure_core()
    return _decompress_batch(inputs)  # type: ignore[name-defined]
