"""Elementwise transforms with optional ``out=`` buffer reuse."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def fast_clip(arr, lo, hi, out=None):
    """Clamp every element of 1-D ``arr`` into ``[lo, hi]``.

    Returns a new array, or writes into ``out`` (same length and dtype as
    ``arr``) and returns it. Pass ``lo``/``hi`` matching ``arr``'s dtype;
    mixed int/float promotes the comparison, not the output.

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
    for i in range(1, n):
        x = arr[i]
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

    Accumulates in ``arr``'s dtype (like ``np.cumsum``): integer inputs
    can overflow, float32 inputs accumulate float32 error.

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
