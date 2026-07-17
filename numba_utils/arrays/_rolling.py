"""Sliding-window aggregations."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def rolling_sum(arr, window, out=None):
    """Sliding-window sum of 1-D ``arr``; output length ``n - window + 1``.

    Accumulates in float64 with the add/subtract sliding trick, so each
    step is O(1) but long float streams accumulate rounding drift — fine
    for windowed features, not for exact accounting.

    Complexity: O(n). Memory: O(n - window + 1), O(1) with ``out=``.
    """
    n = arr.shape[0]
    if window < 1 or window > n:
        raise ValueError("rolling_sum: window must be in [1, len(arr)]")
    m = n - window + 1
    if out is None:
        out = np.empty(m, np.float64)
    if out.shape[0] != m:
        raise ValueError("rolling_sum: out has wrong length")
    acc = 0.0
    for i in range(window):
        acc += arr[i]
    out[0] = acc
    for i in range(1, m):
        acc += arr[i + window - 1] - arr[i - 1]
        out[i] = acc
    return out


@cached_njit
def rolling_mean(arr, window, out=None):
    """Sliding-window mean of 1-D ``arr``; output length ``n - window + 1``.

    Same accumulation caveats as :func:`rolling_sum`.

    Complexity: O(n). Memory: O(n - window + 1), O(1) with ``out=``.
    """
    result = rolling_sum(arr, window, out)
    inv = 1.0 / window
    for i in range(result.shape[0]):
        result[i] *= inv
    return result
