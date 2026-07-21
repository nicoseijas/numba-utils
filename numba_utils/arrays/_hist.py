"""Integer-optimized counting: bincount and fixed-range histogram."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def bincount(arr, minlength=0):
    """Count occurrences of each value in a 1-D array of nonnegative ints.

    ``result[v]`` is how many times ``v`` appears. Output length is
    ``max(arr.max() + 1, minlength)``. Raises ``ValueError`` on negatives.

    Complexity: O(n + max). Memory: O(max).
    """
    if minlength < 0:
        raise ValueError("bincount: minlength must be >= 0")
    mx = minlength - 1
    for i in range(arr.shape[0]):
        v = arr[i]
        if v < 0:
            raise ValueError("bincount: negative values are not allowed")
        if v > mx:
            mx = v
    counts = np.zeros(mx + 1, np.int64)
    for i in range(arr.shape[0]):
        counts[arr[i]] += 1
    return counts


@cached_njit
def histogram(arr, bins, lo, hi):
    """Histogram of 1-D ``arr`` over ``bins`` equal-width bins in ``[lo, hi]``.

    Single pass, no edge array: bin index is computed by scaling, which is
    why this beats ``np.histogram``. Values outside ``[lo, hi]`` and NaN
    are ignored; ``hi`` itself lands in the last bin (NumPy convention).

    Complexity: O(n + bins). Memory: O(bins).
    """
    if bins < 1:
        raise ValueError("histogram: bins must be >= 1")
    if not lo < hi:
        raise ValueError("histogram: lo must be < hi")
    counts = np.zeros(bins, np.int64)
    scale = bins / (hi - lo)
    for i in range(arr.shape[0]):
        x = arr[i]
        # Inverted-range test so NaN (which fails every comparison) is
        # skipped: int(NaN) is INT64_MIN, an out-of-bounds index.
        if not (lo <= x <= hi):
            continue
        idx = int((x - lo) * scale)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts
