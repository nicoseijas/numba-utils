"""Selection: quickselect, nth_element and argpartition.

Mutation contract, stated per function and consistent across the module:
``nth_element`` partitions IN PLACE (the zero-allocation building block);
``quickselect`` and ``fast_argpartition`` never touch their input.
"""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def _median3(a, b, c):
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    else:
        if a < c:
            return a
        elif b < c:
            return c
        else:
            return b


@cached_njit
def nth_element(arr, k):
    """Partition ``arr`` IN PLACE so ``arr[k]`` is its k-th smallest value.

    After the call: ``arr[:k] <= arr[k] <= arr[k+1:]`` (both sides
    unordered). Returns ``arr[k]``. Iterative Hoare quickselect with
    median-of-three pivots. This is the C++ ``std::nth_element``.
    NaN breaks the partition comparisons: results are undefined if
    ``arr`` contains NaN — filter first.

    Complexity: average O(n), worst O(n²). Memory: O(1).
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("nth_element: empty array")
    if k < 0 or k >= n:
        raise ValueError("nth_element: k out of range")
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        pivot = _median3(arr[lo], arr[mid], arr[hi])
        i = lo
        j = hi
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break
    return arr[k]


@cached_njit
def quickselect(arr, k):
    """k-th smallest value of ``arr`` (0-indexed). Input is NOT modified.

    Convenience wrapper: copies, then runs :func:`nth_element`. Call
    ``nth_element`` directly in hot loops to skip the copy.

    Complexity: average O(n), worst O(n²). Memory: O(n) for the copy.
    """
    tmp = arr.copy()
    return nth_element(tmp, k)


@cached_njit
def fast_argpartition(arr, k):
    """Indices of the k smallest elements of ``arr``, in no particular order.

    Like ``np.argpartition(arr, k)[:k]`` but returns exactly k indices and
    skips NumPy's full output. Input is NOT modified; the quickselect runs
    on an index array compared through ``arr``. Results are undefined if
    ``arr`` contains NaN.

    Complexity: average O(n), worst O(n²). Memory: O(n) for the indices.
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("fast_argpartition: k must be in [1, len(arr)]")
    idx = np.arange(n)
    target = k - 1
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        pivot = _median3(arr[idx[lo]], arr[idx[mid]], arr[idx[hi]])
        i = lo
        j = hi
        while i <= j:
            while arr[idx[i]] < pivot:
                i += 1
            while arr[idx[j]] > pivot:
                j -= 1
            if i <= j:
                idx[i], idx[j] = idx[j], idx[i]
                i += 1
                j -= 1
        if target <= j:
            hi = j
        elif target >= i:
            lo = i
        else:
            break
    return idx[:k].copy()
