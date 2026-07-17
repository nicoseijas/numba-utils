"""Parallel top-k: per-chunk heaps merged serially."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, prange

from numba_utils.algorithms import topk
from numba_utils.decorators import cached_njit, njit_parallel

_SERIAL_THRESHOLD = 1 << 16


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

    Each thread keeps a size-k min-heap of its chunk's largest values
    (chunks shorter than k contribute everything they have — no padding
    with sentinel values, which would corrupt duplicates); the
    ``threads·k`` candidates are then merged with the serial
    :func:`numba_utils.topk`. Falls back to serial below the size
    threshold or when chunks would be smaller than k.

    Complexity: O(n + threads·k·log k). Memory: O(threads·k).
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("parallel_topk: k must be in [1, len(arr)]")
    n_threads = get_num_threads()
    chunk = (n + n_threads - 1) // n_threads
    if n < _SERIAL_THRESHOLD or chunk < k or n_threads == 1:
        return topk(arr, k)
    candidates = np.empty(n_threads * k, arr.dtype)
    counts = np.empty(n_threads, np.int64)
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        m = end - start
        base = t * k
        if m <= k:
            for j in range(m):
                candidates[base + j] = arr[start + j]
            counts[t] = m
        else:
            heap = candidates[base : base + k]
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
        base = t * k
        for j in range(counts[t]):
            merged[position] = candidates[base + j]
            position += 1
    return topk(merged, k)
