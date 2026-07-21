"""Sorting: insertion sort, partial sort, counting sort, radix sort.

Mutation contract: ``insertion_sort`` and ``partial_sort`` work IN PLACE
(zero allocation — that's their reason to exist) and return the array for
chaining. ``counting_sort`` and ``radix_sort`` allocate scratch space
inherently, so they return a NEW array and leave the input untouched.
"""

from __future__ import annotations

import numpy as np

from numba_utils.algorithms._select import nth_element
from numba_utils.decorators import cached_njit

_COUNTING_SORT_MAX_RANGE = 2**27


@cached_njit
def insertion_sort(arr):
    """Sort ``arr`` ascending IN PLACE; returns it.

    O(n²), but with tiny constants and O(n) on nearly-sorted input —
    the right tool below ~64 elements or as a small-partition finisher.
    Results are undefined if ``arr`` contains NaN.

    Complexity: O(n²) worst, O(n) nearly-sorted. Memory: O(1).
    """
    for i in range(1, arr.shape[0]):
        x = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > x:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = x
    return arr


@cached_njit
def partial_sort(arr, k):
    """Rearrange ``arr`` IN PLACE so ``arr[:k]`` holds the k smallest, sorted.

    ``arr[k:]`` contains the rest in arbitrary order. Returns ``arr``.
    The C++ ``std::partial_sort``: select with :func:`nth_element`, then
    sort only the front. Results are undefined if ``arr`` contains NaN.

    Complexity: average O(n + k log k). Memory: O(1).
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("partial_sort: k must be in [1, len(arr)]")
    if k < n:
        nth_element(arr, k - 1)
    arr[:k].sort()
    return arr


@cached_njit
def counting_sort(arr):
    """Return a NEW sorted array of integers via counting sort.

    Non-comparison sort: O(n + range) where range = max - min + 1. Wins
    when the value range is small relative to n (codes, buckets, bytes).
    Raises ``ValueError`` if range exceeds 2**27 (a 1 GiB counts array) —
    use :func:`radix_sort` there. Integer dtypes only.

    Complexity: O(n + range). Memory: O(n + range).
    """
    n = arr.shape[0]
    out = np.empty_like(arr)
    if n == 0:
        return out
    mn = arr[0]
    mx = arr[0]
    for i in range(1, n):
        v = arr[i]
        if v < mn:
            mn = v
        elif v > mx:
            mx = v
    # Modular difference (see _radix_key): the uint64 distance is exact
    # even when mx - mn overflows int64 (e.g. INT64_MIN and INT64_MAX in
    # the same array), so the range check cannot be fooled by wraparound.
    dist = np.uint64(np.int64(mx) - np.int64(mn))
    if dist >= np.uint64(_COUNTING_SORT_MAX_RANGE):
        raise ValueError(
            "counting_sort: value range too large (> 2**27), use radix_sort"
        )
    value_range = np.int64(dist) + 1
    counts = np.zeros(value_range, np.int64)
    for i in range(n):
        counts[np.int64(arr[i]) - np.int64(mn)] += 1
    j = 0
    for v in range(value_range):
        value = mn + v
        for _ in range(counts[v]):
            out[j] = value
            j += 1
    return out


@cached_njit
def _radix_key(x, mn):
    # Modular difference (x - mn) mod 2**64 equals the true nonnegative
    # distance for every integer dtype, so one biased uint64 key handles
    # signed and unsigned inputs alike.
    return np.uint64(np.int64(x) - np.int64(mn))


@cached_njit
def radix_sort(arr):
    """Return a NEW sorted array of integers via LSD radix sort.

    8-bit digits over min-biased uint64 keys; handles all signed and
    unsigned integer dtypes, negatives included. All digit histograms are
    built in ONE pass, then each byte gets a scatter pass — and bytes
    that are constant across the array (including the high bytes the
    value range doesn't reach) are skipped entirely. Integer dtypes only.

    When it wins: values spanning <= 3-4 bytes (each byte is a full
    scatter pass). For full-range 64-bit keys, NumPy's SIMD introsort
    is faster — see BENCHMARKS.md for both cases.

    Complexity: O(n · (1 + bytes_used)). Memory: O(n) scratch + counters.
    """
    n = arr.shape[0]
    a = arr.copy()
    if n <= 1:
        return a
    b = np.empty_like(arr)
    mn = arr[0]
    mx = arr[0]
    for i in range(1, n):
        v = arr[i]
        if v < mn:
            mn = v
        elif v > mx:
            mx = v
    max_key = _radix_key(mx, mn)
    eight = np.uint64(8)
    mask = np.uint64(0xFF)
    passes = 1
    while passes < 8 and (max_key >> np.uint64(8 * passes)) != np.uint64(0):
        passes += 1
    counts = np.zeros((passes, 256), np.int64)
    for i in range(n):
        key = _radix_key(a[i], mn)
        for p in range(passes):
            counts[p, key & mask] += 1
            key = key >> eight
    for p in range(passes):
        shift = np.uint64(8 * p)
        cp = counts[p]
        constant_byte = False
        for d in range(256):
            if cp[d] == n:
                constant_byte = True
                break
        if constant_byte:
            continue
        total = 0
        for d in range(256):
            c = cp[d]
            cp[d] = total
            total += c
        for i in range(n):
            digit = (_radix_key(a[i], mn) >> shift) & mask
            b[cp[digit]] = a[i]
            cp[digit] += 1
        a, b = b, a
    return a
