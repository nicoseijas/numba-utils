"""numba-utils: high-performance building blocks for Numba."""

from numba_utils import diagnostics
from numba_utils._config import config, configure
from numba_utils.algorithms import (
    argmax2,
    counting_sort,
    fast_argpartition,
    insertion_sort,
    nth_element,
    partial_sort,
    quickselect,
    radix_sort,
    topk,
)
from numba_utils.arrays import (
    bincount,
    binary_search,
    cumulative_sum,
    fast_clip,
    histogram,
    lower_bound,
    normalize,
    rolling_mean,
    rolling_sum,
    unique_sorted,
    upper_bound,
)
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
    "argmax2",
    "benchmark",
    "bincount",
    "binary_search",
    "boundscheck",
    "cached_njit",
    "compare",
    "compile_time",
    "config",
    "configure",
    "counting_sort",
    "cumulative_sum",
    "diagnostics",
    "fast_argpartition",
    "fast_clip",
    "histogram",
    "insertion_sort",
    "lower_bound",
    "njit_fast",
    "normalize",
    "nth_element",
    "parallel",
    "partial_sort",
    "quickselect",
    "radix_sort",
    "rolling_mean",
    "rolling_sum",
    "topk",
    "unique_sorted",
    "upper_bound",
    "warmup",
]
