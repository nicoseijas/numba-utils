"""Timing and comparison helpers for JIT-compiled code."""

from numba_utils.profiling._compare import ComparisonResult, TimingStats, compare
from numba_utils.profiling._timer import (
    BenchmarkResult,
    benchmark,
    compile_time,
    warmup,
)

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "TimingStats",
    "benchmark",
    "compare",
    "compile_time",
    "warmup",
]
