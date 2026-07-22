"""Elementwise transforms with optional ``out=`` buffer reuse."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def fast_clip(arr, lo, hi, out=None):
    """Clamp every element of 1-D ``arr`` into ``[lo, hi]``.

    Returns a new array, or writes into ``out`` (same length and dtype as
    ``arr``) and returns it. Pass ``lo``/``hi`` matching ``arr``'s dtype;
    mixed int/float promotes the comparison, not the output — which
    DIVERGES from ``np.clip``: an int array with float bounds keeps the
    int dtype here, so a clamped-to-bound element is truncated
    (``fast_clip([0,1,2,3,4], 0.5, 3.5)`` gives ``[0 1 2 3 3]``;
    ``np.clip`` promotes to float64 and gives ``[0.5 1. 2. 3. 3.5]``).

    Complexity: O(n). Memory: O(n), O(1) with ``out=``.
    """
    if lo > hi:
        raise ValueError("fast_clip: lo must be <= hi")
    if out is None:
        out = np.empty_like(arr)
    if out.shape[0] != arr.shape[0]:
        raise ValueError("fast_clip: out has wrong length")
    for i in range(arr.shape[0]):
        x = arr[i]
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        out[i] = x
    return out


@cached_njit
def normalize(arr, out=None):
    """Min-max scale 1-D ``arr`` into ``[0, 1]`` as float64.

    A constant array maps to all zeros. ``out`` must be a float array of
    the same length. Raises ``ValueError`` on empty input.

    Any NaN in the input makes the WHOLE output NaN — min and max are
    undefined, exactly as ``(arr - arr.min()) / (arr.max() - arr.min())``
    behaves in NumPy. (Earlier releases let the result depend on WHERE
    the NaN sat: at position 0 it contaminated everything, elsewhere
    only its own cell.)

    Complexity: O(n), two passes. Memory: O(n), O(1) with ``out=``.
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("normalize: empty array")
    if out is None:
        out = np.empty(n, np.float64)
    if out.shape[0] != n:
        raise ValueError("normalize: out has wrong length")
    mn = arr[0]
    mx = arr[0]
    for i in range(n):
        x = arr[i]
        # NaN poisons min/max wherever it sits; without this check the
        # comparisons below silently SKIP a NaN that isn't at index 0.
        if x != x:
            for j in range(n):
                out[j] = np.nan
            return out
        if x < mn:
            mn = x
        elif x > mx:
            mx = x
    if mx == mn:
        for i in range(n):
            out[i] = 0.0
    else:
        inv = 1.0 / (mx - mn)
        for i in range(n):
            out[i] = (arr[i] - mn) * inv
    return out


@cached_njit
def cumulative_sum(arr, out=None):
    """Inclusive prefix sum of 1-D ``arr``, preserving its dtype.

    Accumulates in ``arr``'s dtype — UNLIKE ``np.cumsum``, which
    promotes lower-precision integers to the platform int:
    ``np.cumsum`` of an int8 array of hundreds sums in int64;
    this keeps int8 and WRAPS (``[100, -56, 44]`` where NumPy gives
    ``[100, 200, 300]``). Cast up first if the sums can exceed the
    dtype. Float32 inputs likewise accumulate float32 error.

    Complexity: O(n). Memory: O(n), O(1) with ``out=``.
    """
    n = arr.shape[0]
    if out is None:
        out = np.empty_like(arr)
    if out.shape[0] != n:
        raise ValueError("cumulative_sum: out has wrong length")
    if n == 0:
        return out
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = out[i - 1] + arr[i]
    return out
