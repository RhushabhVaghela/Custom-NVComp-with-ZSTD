"""Basic round-trip and API tests for cuda_zstd.

Every test in this file requires a CUDA GPU.  Tests are skipped
automatically when no GPU is available (see conftest.py).
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_cuda, requires_cupy, requires_numpy

# Guard the import so collection works even without a GPU.
cuda_zstd = pytest.importorskip("cuda_zstd")


# ===========================================================================
# Round-trip: compress → decompress
# ===========================================================================


@requires_cuda
class TestRoundTrip:
    """Compress then decompress and verify data integrity."""

    def test_small_data(self, small_data: bytes) -> None:
        compressed = cuda_zstd.compress(small_data)
        assert compressed != small_data  # should actually be different
        result = cuda_zstd.decompress(compressed)
        assert result == small_data

    def test_medium_data(self, medium_data: bytes) -> None:
        compressed = cuda_zstd.compress(medium_data)
        assert len(compressed) < len(medium_data)  # should compress
        result = cuda_zstd.decompress(compressed)
        assert result == medium_data

    def test_large_data(self, large_data: bytes) -> None:
        compressed = cuda_zstd.compress(large_data)
        assert len(compressed) < len(large_data)
        result = cuda_zstd.decompress(compressed)
        assert result == large_data

    def test_bytearray_input(self, small_data: bytes) -> None:
        """bytearray should be accepted as input."""
        ba = bytearray(small_data)
        compressed = cuda_zstd.compress(ba)
        result = cuda_zstd.decompress(compressed)
        assert result == small_data

    def test_memoryview_input(self, small_data: bytes) -> None:
        """memoryview should be accepted as input."""
        mv = memoryview(small_data)
        compressed = cuda_zstd.compress(mv)
        result = cuda_zstd.decompress(compressed)
        assert result == small_data


# ===========================================================================
# Compression levels
# ===========================================================================


@requires_cuda
class TestCompressionLevels:
    """Test different compression levels."""

    @pytest.mark.parametrize("level", [1, 3, 5, 9, 15, 22])
    def test_level_roundtrip(self, small_data: bytes, level: int) -> None:
        compressed = cuda_zstd.compress(small_data, level=level)
        result = cuda_zstd.decompress(compressed)
        assert result == small_data

    def test_higher_level_better_ratio(self, medium_data: bytes) -> None:
        """Higher compression levels should generally give equal or better ratios."""
        c1 = cuda_zstd.compress(medium_data, level=1)
        c9 = cuda_zstd.compress(medium_data, level=9)
        # Level 9 should be <= level 1 in size (or very close)
        # Allow a small tolerance since this isn't always strictly monotonic
        assert len(c9) <= len(c1) * 1.05


# ===========================================================================
# Manager class
# ===========================================================================


@requires_cuda
class TestManager:
    """Test the Manager class lifecycle and methods."""

    def test_basic_usage(self, small_data: bytes) -> None:
        mgr = cuda_zstd.Manager(level=3)
        compressed = mgr.compress(small_data)
        result = mgr.decompress(compressed)
        assert result == small_data
        mgr.close()

    def test_context_manager(self, small_data: bytes) -> None:
        with cuda_zstd.Manager(level=3) as mgr:
            compressed = mgr.compress(small_data)
            result = mgr.decompress(compressed)
            assert result == small_data
        # After exiting, repr should indicate closed
        assert "closed" in repr(mgr)

    def test_reuse(self, small_data: bytes, medium_data: bytes) -> None:
        """Manager should be reusable across multiple calls."""
        with cuda_zstd.Manager(level=3) as mgr:
            for data in [small_data, medium_data, small_data]:
                compressed = mgr.compress(data)
                result = mgr.decompress(compressed)
                assert result == data

    def test_level_property(self) -> None:
        with cuda_zstd.Manager(level=5) as mgr:
            assert mgr.level == 5
            mgr.level = 10
            assert mgr.level == 10

    def test_repr(self) -> None:
        mgr = cuda_zstd.Manager(level=7)
        assert "level=7" in repr(mgr)
        mgr.close()
        assert "closed" in repr(mgr)

    def test_config_property(self) -> None:
        with cuda_zstd.Manager(level=3) as mgr:
            cfg = mgr.config
            assert hasattr(cfg, "level")
            assert hasattr(cfg, "strategy")
            assert hasattr(cfg, "block_size")

    def test_stats_property(self, small_data: bytes) -> None:
        with cuda_zstd.Manager(level=3) as mgr:
            mgr.compress(small_data)
            stats = mgr.stats
            assert hasattr(stats, "input_bytes")
            assert hasattr(stats, "output_bytes")
            assert stats.input_bytes > 0

    def test_reset_stats(self, small_data: bytes) -> None:
        with cuda_zstd.Manager(level=3) as mgr:
            mgr.compress(small_data)
            mgr.reset_stats()
            stats = mgr.stats
            assert stats.input_bytes == 0

    def test_from_config(self, small_data: bytes) -> None:
        """Manager should accept a CompressionConfig object."""
        cfg = cuda_zstd.CompressionConfig.from_level(5)
        with cuda_zstd.Manager(config=cfg) as mgr:
            compressed = mgr.compress(small_data)
            result = mgr.decompress(compressed)
            assert result == small_data


# ===========================================================================
# Batch operations
# ===========================================================================


@requires_cuda
class TestBatch:
    """Test batch compress/decompress APIs."""

    def test_compress_batch(self, batch_data: list[bytes]) -> None:
        compressed = cuda_zstd.compress_batch(batch_data)
        assert len(compressed) == len(batch_data)
        for c in compressed:
            assert isinstance(c, bytes)
            assert len(c) > 0

    def test_decompress_batch(self, batch_data: list[bytes]) -> None:
        compressed = cuda_zstd.compress_batch(batch_data)
        decompressed = cuda_zstd.decompress_batch(compressed)
        assert len(decompressed) == len(batch_data)
        for orig, dec in zip(batch_data, decompressed):
            assert dec == orig

    def test_batch_via_manager(self, batch_data: list[bytes]) -> None:
        with cuda_zstd.Manager(level=3) as mgr:
            compressed = mgr.compress_batch(batch_data)
            decompressed = mgr.decompress_batch(compressed)
            for orig, dec in zip(batch_data, decompressed):
                assert dec == orig

    def test_batch_empty_list(self) -> None:
        """Empty input list should return empty output list."""
        assert cuda_zstd.compress_batch([]) == []
        assert cuda_zstd.decompress_batch([]) == []

    def test_batch_single_item(self, small_data: bytes) -> None:
        compressed = cuda_zstd.compress_batch([small_data])
        assert len(compressed) == 1
        decompressed = cuda_zstd.decompress_batch(compressed)
        assert len(decompressed) == 1
        assert decompressed[0] == small_data

    def test_batch_with_level(self, batch_data: list[bytes]) -> None:
        compressed = cuda_zstd.compress_batch(batch_data, level=9)
        decompressed = cuda_zstd.decompress_batch(compressed)
        for orig, dec in zip(batch_data, decompressed):
            assert dec == orig


# ===========================================================================
# CompressionConfig
# ===========================================================================


@requires_cuda
class TestCompressionConfig:
    """Test CompressionConfig creation and factory methods."""

    def test_default(self) -> None:
        cfg = cuda_zstd.CompressionConfig.get_default()
        assert isinstance(cfg.level, int)
        assert isinstance(cfg.block_size, int)

    def test_from_level(self) -> None:
        cfg = cuda_zstd.CompressionConfig.from_level(10)
        assert cfg.level == 10

    def test_optimal(self) -> None:
        cfg = cuda_zstd.CompressionConfig.optimal(1_000_000)
        assert isinstance(cfg.level, int)
        assert isinstance(cfg.block_size, int)

    def test_repr(self) -> None:
        cfg = cuda_zstd.CompressionConfig.from_level(5)
        r = repr(cfg)
        assert "CompressionConfig" in r
        assert "level=" in r


# ===========================================================================
# CUDA utilities
# ===========================================================================


@requires_cuda
class TestCudaUtils:
    """Test CUDA utility functions."""

    def test_is_cuda_available(self) -> None:
        # If we got here, CUDA is available (requires_cuda guard)
        assert cuda_zstd.is_cuda_available() is True

    def test_get_cuda_device_info(self) -> None:
        info = cuda_zstd.get_cuda_device_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "compute_capability" in info
        assert "total_memory_mb" in info
        assert "multiprocessor_count" in info
        assert "device_index" in info
        assert isinstance(info["name"], str)
        assert info["total_memory_mb"] > 0


# ===========================================================================
# Constants and enums
# ===========================================================================


class TestConstants:
    """Test module-level constants (no GPU needed)."""

    def test_level_constants(self) -> None:
        assert isinstance(cuda_zstd.MIN_LEVEL, int)
        assert isinstance(cuda_zstd.MAX_LEVEL, int)
        assert isinstance(cuda_zstd.DEFAULT_LEVEL, int)
        assert cuda_zstd.MIN_LEVEL <= cuda_zstd.DEFAULT_LEVEL <= cuda_zstd.MAX_LEVEL

    def test_version(self) -> None:
        assert isinstance(cuda_zstd.__version__, str)
        parts = cuda_zstd.__version__.split(".")
        assert len(parts) >= 2  # at least major.minor


@requires_cuda
class TestEnums:
    """Test that enums are properly exposed."""

    def test_status_enum(self) -> None:
        assert cuda_zstd.Status.SUCCESS is not None
        assert cuda_zstd.Status.ERROR_GENERIC is not None

    def test_strategy_enum(self) -> None:
        assert cuda_zstd.Strategy.FAST is not None
        assert cuda_zstd.Strategy.BTULTRA is not None

    def test_checksum_policy_enum(self) -> None:
        assert cuda_zstd.ChecksumPolicy.NONE is not None
        assert cuda_zstd.ChecksumPolicy.COMPUTE_AND_VERIFY is not None


# ===========================================================================
# NumPy integration
# ===========================================================================


@requires_cuda
@requires_numpy
class TestNumPy:
    """Test NumPy array input support."""

    def test_compress_numpy_array(self, small_data: bytes) -> None:
        import numpy as np

        arr = np.frombuffer(small_data, dtype=np.uint8)
        compressed = cuda_zstd.compress(arr)
        result = cuda_zstd.decompress(compressed)
        assert result == small_data

    def test_compress_numpy_via_manager(self, small_data: bytes) -> None:
        import numpy as np

        arr = np.frombuffer(small_data, dtype=np.uint8)
        with cuda_zstd.Manager(level=3) as mgr:
            compressed = mgr.compress(arr)
            result = mgr.decompress(compressed)
            assert result == small_data

    def test_batch_numpy(self, small_data: bytes) -> None:
        import numpy as np

        arrays = [
            np.frombuffer(small_data, dtype=np.uint8),
            np.frombuffer(small_data[:512], dtype=np.uint8),
        ]
        compressed = cuda_zstd.compress_batch(arrays)
        decompressed = cuda_zstd.decompress_batch(compressed)
        assert decompressed[0] == small_data
        assert decompressed[1] == small_data[:512]


# ===========================================================================
# CuPy integration (zero-copy GPU path)
# ===========================================================================


@requires_cuda
@requires_cupy
class TestCuPy:
    """Test CuPy array zero-copy support."""

    def test_compress_cupy_array(self, small_data: bytes) -> None:
        import cupy as cp

        gpu_arr = cp.frombuffer(small_data, dtype=cp.uint8)
        compressed = cuda_zstd.compress(gpu_arr)
        result = cuda_zstd.decompress(compressed)
        assert result == small_data

    def test_compress_cupy_via_manager(self, small_data: bytes) -> None:
        import cupy as cp

        gpu_arr = cp.frombuffer(small_data, dtype=cp.uint8)
        with cuda_zstd.Manager(level=3) as mgr:
            compressed = mgr.compress(gpu_arr)
            result = mgr.decompress(compressed)
            assert result == small_data


# ===========================================================================
# Error handling
# ===========================================================================


@requires_cuda
class TestErrorHandling:
    """Test that errors are raised correctly."""

    def test_decompress_corrupt_data(self) -> None:
        """Corrupted data should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            cuda_zstd.decompress(b"this is not compressed data!!!")

    def test_decompress_truncated_data(self, small_data: bytes) -> None:
        """Truncated compressed data should raise RuntimeError."""
        compressed = cuda_zstd.compress(small_data)
        truncated = compressed[: len(compressed) // 2]
        with pytest.raises(RuntimeError):
            cuda_zstd.decompress(truncated)

    def test_manager_after_close(self) -> None:
        """Using a closed Manager should raise (AttributeError or similar)."""
        mgr = cuda_zstd.Manager(level=3)
        mgr.close()
        with pytest.raises((AttributeError, RuntimeError)):
            mgr.compress(b"data")


# ===========================================================================
# Edge cases
# ===========================================================================


@requires_cuda
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_data(self) -> None:
        """Single-byte data should round-trip correctly."""
        data = b"x"
        compressed = cuda_zstd.compress(data)
        result = cuda_zstd.decompress(compressed)
        assert result == data

    def test_repeated_byte(self) -> None:
        """Highly repetitive data should compress very well."""
        data = b"\x00" * 100_000
        compressed = cuda_zstd.compress(data)
        assert len(compressed) < len(data) // 10  # expect >10x ratio
        result = cuda_zstd.decompress(compressed)
        assert result == data

    def test_all_byte_values(self) -> None:
        """Data with all 256 byte values should round-trip."""
        data = bytes(range(256)) * 100
        compressed = cuda_zstd.compress(data)
        result = cuda_zstd.decompress(compressed)
        assert result == data

    def test_compress_already_compressed(self, small_data: bytes) -> None:
        """Compressing already-compressed data should still round-trip."""
        c1 = cuda_zstd.compress(small_data)
        c2 = cuda_zstd.compress(c1)
        d2 = cuda_zstd.decompress(c2)
        d1 = cuda_zstd.decompress(d2)
        assert d1 == small_data


# ===========================================================================
# Validation & estimation utilities
# ===========================================================================


@requires_cuda
class TestValidation:
    """Test validate_compressed_data and estimate_compressed_size."""

    def test_validate_good_data(self, small_data: bytes) -> None:
        """Valid compressed data should pass validation."""
        compressed = cuda_zstd.compress(small_data)
        assert cuda_zstd.validate_compressed_data(compressed) is True

    def test_validate_good_data_with_checksum(self, small_data: bytes) -> None:
        """Validation with checksum should pass for valid data."""
        compressed = cuda_zstd.compress(small_data)
        assert cuda_zstd.validate_compressed_data(compressed, check_checksum=True) is True

    def test_validate_good_data_no_checksum(self, small_data: bytes) -> None:
        """Validation without checksum should pass for valid data."""
        compressed = cuda_zstd.compress(small_data)
        assert cuda_zstd.validate_compressed_data(compressed, check_checksum=False) is True

    def test_validate_corrupt_data(self) -> None:
        """Corrupt data should fail validation."""
        garbage = b"this is definitely not valid compressed data!!!"
        assert cuda_zstd.validate_compressed_data(garbage) is False

    def test_estimate_compressed_size(self) -> None:
        """Estimated size should be >= actual compressed size."""
        data_size = 100_000
        estimated = cuda_zstd.estimate_compressed_size(data_size, level=3)
        assert isinstance(estimated, int)
        assert estimated > 0

    def test_estimate_vs_actual(self, medium_data: bytes) -> None:
        """Estimate should be an upper bound on actual compressed size."""
        estimated = cuda_zstd.estimate_compressed_size(len(medium_data), level=3)
        compressed = cuda_zstd.compress(medium_data, level=3)
        assert estimated >= len(compressed)

    @pytest.mark.parametrize("level", [1, 3, 9, 22])
    def test_estimate_at_different_levels(self, level: int) -> None:
        """Estimate should work at all compression levels."""
        estimated = cuda_zstd.estimate_compressed_size(1_000_000, level=level)
        assert isinstance(estimated, int)
        assert estimated > 0


# ===========================================================================
# Hybrid Engine
# ===========================================================================


@requires_cuda
class TestHybridEngine:
    """Test the HybridEngine class lifecycle and methods."""

    def test_roundtrip(self, small_data: bytes) -> None:
        """Compress then decompress via HybridEngine."""
        eng = cuda_zstd.HybridEngine(level=3)
        compressed = eng.compress(small_data)
        result = eng.decompress(compressed)
        assert result == small_data
        eng.close()

    def test_context_manager(self, small_data: bytes) -> None:
        """HybridEngine should work as a context manager."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            compressed = eng.compress(small_data)
            result = eng.decompress(compressed)
            assert result == small_data
        # After exiting, repr should indicate closed
        assert "closed" in repr(eng)

    def test_level_property(self) -> None:
        """Level property should be readable and writable."""
        with cuda_zstd.HybridEngine(level=5) as eng:
            assert eng.level == 5
            eng.level = 10
            assert eng.level == 10

    def test_query_routing(self) -> None:
        """query_routing should return an ExecutionBackend enum value."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            backend = eng.query_routing(100)
            assert isinstance(backend, cuda_zstd.ExecutionBackend)
            # Small data should route to CPU
            assert backend == cuda_zstd.ExecutionBackend.CPU_LIBZSTD

    def test_repr(self) -> None:
        """repr should show level when open and [closed] when closed."""
        eng = cuda_zstd.HybridEngine(level=7)
        assert "level=7" in repr(eng)
        assert "HybridEngine" in repr(eng)
        eng.close()
        assert "closed" in repr(eng)

    def test_stats(self, small_data: bytes) -> None:
        """Stats should track compression operations."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            eng.compress(small_data)
            stats = eng.stats
            assert hasattr(stats, "input_bytes")
            assert hasattr(stats, "output_bytes")
            assert stats.input_bytes > 0

    def test_reset_stats(self, small_data: bytes) -> None:
        """reset_stats should zero out statistics."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            eng.compress(small_data)
            eng.reset_stats()
            stats = eng.stats
            assert stats.input_bytes == 0

    def test_config(self) -> None:
        """config property should return a HybridConfig."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            cfg = eng.config
            assert isinstance(cfg, cuda_zstd.HybridConfig)
            assert cfg.compression_level == 3

    def test_medium_data_roundtrip(self, medium_data: bytes) -> None:
        """Medium-sized data should round-trip correctly."""
        with cuda_zstd.HybridEngine(level=3) as eng:
            compressed = eng.compress(medium_data)
            assert len(compressed) < len(medium_data)
            result = eng.decompress(compressed)
            assert result == medium_data

    def test_after_close(self) -> None:
        """Using a closed HybridEngine should raise RuntimeError."""
        eng = cuda_zstd.HybridEngine(level=3)
        eng.close()
        with pytest.raises(RuntimeError):
            eng.compress(b"data")


# ===========================================================================
# Hybrid convenience functions
# ===========================================================================


@requires_cuda
class TestHybridConvenience:
    """Test module-level hybrid_compress and hybrid_decompress."""

    def test_roundtrip(self, small_data: bytes) -> None:
        """hybrid_compress/hybrid_decompress should round-trip."""
        compressed = cuda_zstd.hybrid_compress(small_data)
        result = cuda_zstd.hybrid_decompress(compressed)
        assert result == small_data

    def test_with_level(self, small_data: bytes) -> None:
        """hybrid_compress should accept a level parameter."""
        compressed = cuda_zstd.hybrid_compress(small_data, level=9)
        result = cuda_zstd.hybrid_decompress(compressed)
        assert result == small_data

    def test_medium_data(self, medium_data: bytes) -> None:
        """Medium data should compress smaller and round-trip."""
        compressed = cuda_zstd.hybrid_compress(medium_data)
        assert len(compressed) < len(medium_data)
        result = cuda_zstd.hybrid_decompress(compressed)
        assert result == medium_data


# ===========================================================================
# Hybrid enums
# ===========================================================================


@requires_cuda
class TestHybridEnums:
    """Test that hybrid-related enums are properly exposed."""

    def test_hybrid_mode(self) -> None:
        """HybridMode enum should have expected values."""
        assert cuda_zstd.HybridMode.AUTO is not None
        assert cuda_zstd.HybridMode.PREFER_CPU is not None
        assert cuda_zstd.HybridMode.PREFER_GPU is not None
        assert cuda_zstd.HybridMode.FORCE_CPU is not None
        assert cuda_zstd.HybridMode.FORCE_GPU is not None
        assert cuda_zstd.HybridMode.ADAPTIVE is not None

    def test_data_location(self) -> None:
        """DataLocation enum should have expected values."""
        assert cuda_zstd.DataLocation.HOST is not None
        assert cuda_zstd.DataLocation.DEVICE is not None
        assert cuda_zstd.DataLocation.MANAGED is not None
        assert cuda_zstd.DataLocation.UNKNOWN is not None

    def test_execution_backend(self) -> None:
        """ExecutionBackend enum should have expected values."""
        assert cuda_zstd.ExecutionBackend.CPU_LIBZSTD is not None
        assert cuda_zstd.ExecutionBackend.GPU_KERNELS is not None
        assert cuda_zstd.ExecutionBackend.CPU_PARALLEL is not None
        assert cuda_zstd.ExecutionBackend.GPU_BATCH is not None
