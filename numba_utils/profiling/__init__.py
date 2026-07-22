"""Timing and comparison helpers for JIT-compiled code."""

from numba_utils.profiling._compare import ComparisonResult, TimingStats, compare
from numba_utils.profiling._timer import (
    BenchmarkResult,
    benchmark,
    compile_stats,
    compile_time,
    warmup,
    warmup_signatures,
)

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "TimingStats",
    "benchmark",
    "compare",
    "compile_stats",
    "compile_time",
    "warmup",
    "warmup_signatures",
]
