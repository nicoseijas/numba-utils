"""Binary search primitives over sorted 1-D arrays.

All functions require ``arr`` sorted ascending — this is NOT checked
(checking would be O(n) and defeat the point). All are plain ``@njit``
dispatchers, callable from other jitted code.
"""

from __future__ import annotations

from numba_utils.decorators import cached_njit


@cached_njit
def lower_bound(arr, value):
    """First index ``i`` in sorted ``arr`` with ``arr[i] >= value``.

    Returns ``len(arr)`` if every element is smaller than ``value``.

    Complexity: O(log n). Memory: O(1).
    """
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cached_njit
def upper_bound(arr, value):
    """First index ``i`` in sorted ``arr`` with ``arr[i] > value``.

    Returns ``len(arr)`` if every element is ``<= value``.
    ``upper_bound - lower_bound`` counts occurrences of ``value``.

    Complexity: O(log n). Memory: O(1).
    """
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cached_njit
def binary_search(arr, value):
    """Index of ``value`` in sorted ``arr``, or -1 if absent.

    With duplicates, returns the first occurrence. Uses exact equality —
    for floats, prefer :func:`lower_bound` plus a tolerance check.

    Complexity: O(log n). Memory: O(1).
    """
    idx = lower_bound(arr, value)
    if idx < arr.shape[0] and arr[idx] == value:
        return idx
    return -1
