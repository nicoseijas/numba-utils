"""numba-utils: high-performance building blocks for Numba."""

from numba_utils.decorators import boundscheck, cached_njit, njit_fast, parallel
from numba_utils.profiling import (
    BenchmarkResult,
    ComparisonResult,
    TimingStats,
    benchmark,
    compare,
    compile_time,
    warmup,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "TimingStats",
    "benchmark",
    "boundscheck",
    "cached_njit",
    "compare",
    "compile_time",
    "njit_fast",
    "parallel",
    "warmup",
]
