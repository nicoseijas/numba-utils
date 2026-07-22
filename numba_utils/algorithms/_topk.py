"""Top-k and extremum queries."""

from __future__ import annotations

from numba_utils.algorithms._select import nth_element
from numba_utils.decorators import cached_njit

# Below n/16, a size-k min-heap beats quickselect: one read-only pass,
# no O(n) copy, and most elements fail the `x > heap[0]` test immediately.
_HEAP_PATH_FACTOR = 16


@cached_njit
def _sift_down(heap, start, size):
    root = start
    while True:
        child = 2 * root + 1
        if child >= size:
            return
        if child + 1 < size and heap[child + 1] < heap[child]:
            child += 1
        if heap[child] < heap[root]:
            heap[root], heap[child] = heap[child], heap[root]
            root = child
        else:
            return


@cached_njit
def _topk_heap(arr, k):
    n = arr.shape[0]
    heap = arr[:k].copy()
    for start in range(k // 2 - 1, -1, -1):
        _sift_down(heap, start, k)
    for i in range(k, n):
        x = arr[i]
        if x > heap[0]:
            heap[0] = x
            _sift_down(heap, 0, k)
    heap.sort()
    return heap[::-1].copy()


@cached_njit
def topk(arr, k):
    """The k LARGEST values of ``arr``, sorted descending. Input untouched.

    Small k (``k * 16 <= n``): single read-only pass with a size-k
    min-heap — every element that beats the heap root costs an
    O(log k) sift, so the pass is O(n) on random input (few beats) but
    O(n log k) worst case (ascending input beats the root every time).
    Large k: quickselect via :func:`nth_element` on a copy, then sort
    only the k winners. Either way the full sort never happens. For
    the k smallest, use :func:`fast_argpartition`. Results are
    undefined if ``arr`` contains NaN.

    Complexity: O(n + k log k) average on random input; O(n log k)
    worst case. Memory: O(k) small k, O(n) large k.
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("topk: k must be in [1, len(arr)]")
    if k * _HEAP_PATH_FACTOR <= n:
        return _topk_heap(arr, k)
    tmp = arr.copy()
    if k < n:
        nth_element(tmp, n - k)
    result = tmp[n - k :]
    result.sort()
    return result[::-1].copy()


@cached_njit
def argmax2(arr):
    """Index AND value of the maximum, in one pass: ``(idx, value)``.

    Saves the second scan of the ``arr[np.argmax(arr)]`` idiom. First
    occurrence wins on ties. Raises ``ValueError`` on empty input.
    Results are undefined if ``arr`` contains NaN — and they DIVERGE
    from NumPy, which propagates NaN as the max (``np.argmax([1, nan,
    3])`` is 1, the NaN; this returns 2, the largest real value; a NaN
    at position 0 is returned as the max because no comparison can
    displace it).

    Complexity: O(n). Memory: O(1).
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("argmax2: empty array")
    best = arr[0]
    best_idx = 0
    for i in range(1, n):
        if arr[i] > best:
            best = arr[i]
            best_idx = i
    return best_idx, best
