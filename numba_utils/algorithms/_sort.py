"""Sorting: insertion/partial/counting/radix sort, stable argsort, lexsort.

Mutation contract: ``insertion_sort`` and ``partial_sort`` work IN PLACE
(zero allocation — that's their reason to exist) and return the array for
chaining. ``counting_sort`` and ``radix_sort`` allocate scratch space
inherently, so they return a NEW array and leave the input untouched.
``stable_argsort`` and ``lexsort`` return index arrays; the input is
never modified.
"""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types
from numba.core.errors import TypingError
from numba.extending import overload

from numba_utils.algorithms._select import nth_element
from numba_utils.decorators import cached_njit

_COUNTING_SORT_MAX_RANGE = 2**27


def _require_integer(arr):
    # Interpreted fallback (kernels are always jitted; kept for parity).
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("integer dtypes only")


@overload(_require_integer)
def _ol_require_integer(arr):
    # Compile-time dtype gate: float inputs would silently truncate
    # through the int64 key conversions and FABRICATE values in the
    # output. Rejecting at typing time is the only check nopython can
    # express (dtypes are not runtime-comparable). Boolean is allowed
    # — it sorted fine before the gate existed, and np.int64(bool) is
    # exact (a 0.3.2 regression rejected it).
    if not isinstance(arr.dtype, (nb_types.Integer, nb_types.Boolean)):
        raise TypingError(
            "integer dtypes only — float input would be silently "
            "truncated; sort floats with np.sort or stable_argsort"
        )

    def impl(arr):
        return None

    return impl


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
    Raises ``ValueError`` when the range exceeds 2**27, or when it is
    large both absolutely (> 2**20, an 8 MiB counts array) AND relative
    to n (``range > 64n``) — there :func:`radix_sort` is strictly
    better and the O(range) memory and scan would dominate. Integer
    dtypes only (enforced at compile time).

    Complexity: O(n + range). Memory: O(n + range).
    """
    _require_integer(arr)
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
    # Counting sort is O(n + range): a range far beyond n allocates and
    # scans range entries to sort n elements. Reject only when the
    # range is large BOTH absolutely (the counts array itself is big —
    # 2**20 int64 = 8 MiB) AND relatively (range >> n, where radix_sort
    # wins). This admits the canonical regimes (uint16 codes over
    # n~1000, a few hundred values spanning 10k) while still rejecting
    # the pathological 2-elements-spanning-2**27 = 1 GiB case.
    if value_range > (1 << 20) and value_range > 64 * n:
        raise ValueError(
            "counting_sort: value range too large relative to n, "
            "use radix_sort"
        )
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
    _require_integer(arr)
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


@cached_njit
def stable_argsort(arr):
    """Indices that sort 1-D ``arr`` ascending, ties keeping input order.

    Honest positioning: this is ``np.argsort(arr, kind="mergesort")``,
    which IS supported and stable inside nopython code — but the
    ``kind="stable"`` spelling is not, and nothing in Numba's docs says
    which kinds are. This function does that remembering for you and is
    the building block of :func:`lexsort`. NaN sorts last, like NumPy.

    Complexity: O(n log n). Memory: O(n).
    """
    return np.argsort(arr, kind="mergesort")


@cached_njit
def lexsort(keys):
    """Indices sorting by multiple keys — ``np.lexsort`` for nopython code.

    Numba does not implement ``np.lexsort``; this composes stable
    argsorts, one pass per key. ``keys`` is a 2-D array where each ROW
    is a key and the LAST row is the primary key (NumPy's convention),
    so results match ``np.lexsort(keys)`` exactly. Unlike NumPy's, it
    takes one uniform-dtype array, not a tuple — stack heterogeneous
    keys yourself, or chain :func:`stable_argsort` passes.

    Complexity: O(k · n log n) for k keys of length n. Memory: O(n).
    """
    n_keys = keys.shape[0]
    n = keys.shape[1]
    if n_keys == 0:
        raise ValueError("lexsort: need at least one key")
    order = np.arange(n)
    permuted = np.empty(n, keys.dtype)
    for k in range(n_keys):
        for i in range(n):
            permuted[i] = keys[k, order[i]]
        sub = np.argsort(permuted, kind="mergesort")
        reordered = np.empty(n, np.int64)
        for i in range(n):
            reordered[i] = order[sub[i]]
        order = reordered
    return order
