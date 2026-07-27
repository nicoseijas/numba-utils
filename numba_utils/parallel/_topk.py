"""Parallel top-k: per-chunk heaps merged serially."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, prange

from numba_utils.algorithms import topk
from numba_utils.decorators import cached_njit, njit_parallel

_SERIAL_THRESHOLD = 1 << 16
# x86-64 and Apple silicon both use 64-byte lines; a wrong guess here
# costs padding bytes, never correctness.
_CACHE_LINE_BYTES = 64


@cached_njit
def _sift_down(heap, start, size):
    # Co-located with its prange driver on purpose (docs/parallelism.md).
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


# cache=False explicitly: the parfor-transformed function trips Numba's
# "dynamic globals" cache limitation and would warn on every compile.
@njit_parallel(cache=False)
def parallel_topk(arr, k):
    """The k LARGEST values of ``arr``, sorted descending, in parallel.

    Each thread keeps a size-k min-heap of its chunk's largest values in
    its own PRIVATE row, padded to a 64-byte boundary so two threads
    never write the same cache line (chunks shorter than k contribute
    everything they have — no padding with sentinel VALUES, which would
    corrupt duplicates; the row padding is untouched memory, not data);
    the ``threads·k`` candidates are then merged with the serial
    :func:`numba_utils.topk`. Falls back to serial below the size
    threshold or when chunks would be smaller than k.

    The padding is why the adversarial case stays fast. Heap writes are
    gated by ``x > heap[0]``, so random input writes rarely and shares
    little; an ASCENDING chunk writes on every element, and every write
    lands on ``heap[0]`` — the slot that, unpadded, sits on the same
    cache line as the previous thread's tail. Measured against the
    unpadded layout (float64, n=2**22, 24 threads): ascending input
    gains 10-17%, and 60-85% at 2-12 threads; random input is a wash,
    with repeated runs straddling zero — there is little sharing to
    avoid when writes are rare. Two controls pin the mechanism to the
    padding rather than the layout: a 2-D but UNPADDED variant performs
    like the flat one, and at k=8 a float64 row is already exactly one
    64-byte line, where the difference collapses to +0.05%.

    Complexity: O(n + threads·k·log k). Memory: O(threads·k), plus at
    most 63 bytes of padding per thread.
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("parallel_topk: k must be in [1, len(arr)]")
    n_threads = get_num_threads()
    chunk = (n + n_threads - 1) // n_threads
    if n < _SERIAL_THRESHOLD or chunk < k or n_threads == 1:
        return topk(arr, k)
    # Round each row up to a whole 64-byte cache line. Computed from
    # itemsize rather than hardcoded to 8 elements: a fixed element
    # count only lands on a line boundary for 8-byte dtypes, and this
    # kernel is dtype-generic (float32 rows would still straddle).
    per_line = _CACHE_LINE_BYTES // arr.itemsize
    if per_line < 1:
        per_line = 1
    padded = ((k + per_line - 1) // per_line) * per_line
    private = np.empty((n_threads, padded), arr.dtype)
    counts = np.empty(n_threads, np.int64)
    for t in prange(n_threads):
        # Ceil-division chunks overshoot n when the thread count is high
        # relative to n (threads >= ~sqrt(n)): clamp so trailing threads
        # get an empty range instead of a negative m.
        start = min(t * chunk, n)
        end = min(start + chunk, n)
        m = end - start
        if m <= k:
            for j in range(m):
                private[t, j] = arr[start + j]
            counts[t] = m
        else:
            heap = private[t, :k]
            for j in range(k):
                heap[j] = arr[start + j]
            for s in range(k // 2 - 1, -1, -1):
                _sift_down(heap, s, k)
            for i in range(start + k, end):
                x = arr[i]
                if x > heap[0]:
                    heap[0] = x
                    _sift_down(heap, 0, k)
            counts[t] = k
    total = 0
    for t in range(n_threads):
        total += counts[t]
    merged = np.empty(total, arr.dtype)
    position = 0
    for t in range(n_threads):
        for j in range(counts[t]):
            merged[position] = private[t, j]
            position += 1
    return topk(merged, k)
