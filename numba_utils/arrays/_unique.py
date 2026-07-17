"""Deduplication of sorted arrays."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def unique_sorted(arr):
    """Unique values of an already-sorted 1-D array.

    Skips the sort that dominates ``np.unique`` — that's the whole
    speedup. ``arr`` must be sorted ascending (NOT checked). Two passes:
    count uniques, then fill an exactly-sized output.

    Complexity: O(n). Memory: O(u) for u unique values.
    """
    n = arr.shape[0]
    if n == 0:
        return arr.copy()
    count = 1
    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            count += 1
    out = np.empty(count, arr.dtype)
    out[0] = arr[0]
    j = 1
    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            out[j] = arr[i]
            j += 1
    return out
