"""Array utilities: searching, transforms, rolling windows, histograms."""

from numba_utils.arrays._hist import bincount, histogram
from numba_utils.arrays._rolling import rolling_mean, rolling_sum
from numba_utils.arrays._search import binary_search, lower_bound, upper_bound
from numba_utils.arrays._transform import cumulative_sum, fast_clip, normalize
from numba_utils.arrays._unique import unique_sorted

__all__ = [
    "bincount",
    "binary_search",
    "cumulative_sum",
    "fast_clip",
    "histogram",
    "lower_bound",
    "normalize",
    "rolling_mean",
    "rolling_sum",
    "unique_sorted",
    "upper_bound",
]
