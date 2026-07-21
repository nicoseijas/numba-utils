"""Core RNG helpers: seeding, shuffling, uniform choice."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def seed(value):
    """Seed Numba's nopython RNG for reproducible jitted sampling.

    Numba keeps its own random state per thread — seeding NumPy from
    Python does NOT affect jitted code. Call this instead.
    """
    np.random.seed(value)


@cached_njit
def shuffle(arr):
    """Uniform in-place Fisher–Yates shuffle of 1-D ``arr``; returns it.

    ``arr`` must be 1-D: on a 2-D array the element swap works on row
    VIEWS, which alias — rows get silently duplicated instead of
    exchanged. Raises ``ValueError`` for non-1-D input.

    Complexity: O(n). Memory: O(1).
    """
    if arr.ndim != 1:
        raise ValueError("shuffle: arr must be 1-D (row swaps alias)")
    for i in range(arr.shape[0] - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


@cached_njit
def permutation(n):
    """Random permutation of ``0..n-1`` as an int64 array.

    Complexity: O(n). Memory: O(n).
    """
    if n < 0:
        raise ValueError("permutation: n must be >= 0")
    return shuffle(np.arange(n))


@cached_njit
def choice(arr, size):
    """``size`` elements of ``arr`` sampled uniformly WITH replacement.

    For sampling without replacement use :func:`reservoir_sampling`.

    Complexity: O(size). Memory: O(size).
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("choice: empty array")
    if size < 0:
        raise ValueError("choice: size must be >= 0")
    out = np.empty(size, arr.dtype)
    for i in range(size):
        out[i] = arr[np.random.randint(0, n)]
    return out
