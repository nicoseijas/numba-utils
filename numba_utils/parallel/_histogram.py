"""Parallel histogram with per-thread private counts."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, prange

from numba_utils.arrays import histogram
from numba_utils.decorators import njit_parallel

_SERIAL_THRESHOLD = 1 << 16


# cache=False explicitly: get_num_threads makes the parfor-transformed
# function trip Numba's "dynamic globals" cache limitation (would warn).
@njit_parallel(cache=False)
def parallel_histogram(arr, bins, lo, hi):
    """Histogram of 1-D ``arr`` over ``bins`` equal-width bins in
    ``[lo, hi]``, counted in parallel.

    The pattern that makes it correct AND fast: each thread fills its own
    PRIVATE row of counts (rows padded to 64-byte boundaries so threads
    never write the same cache line — no atomics, no false sharing), and
    rows are merged serially at the end. Bit-exact with the serial
    :func:`numba_utils.histogram`, which it falls back to below the size
    threshold.

    Complexity: O(n + threads·bins). Memory: O(threads·bins).
    """
    n = arr.shape[0]
    if n < _SERIAL_THRESHOLD:
        return histogram(arr, bins, lo, hi)
    if bins < 1:
        raise ValueError("parallel_histogram: bins must be >= 1")
    if not lo < hi:
        raise ValueError("parallel_histogram: lo must be < hi")
    n_threads = get_num_threads()
    # Pad each thread's row to a multiple of 8 int64 (= 64 bytes) so two
    # threads never share a cache line.
    padded = ((bins + 7) // 8) * 8
    private = np.zeros((n_threads, padded), np.int64)
    scale = bins / (hi - lo)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            x = arr[i]
            if x < lo or x > hi:
                continue
            idx = int((x - lo) * scale)
            if idx >= bins:
                idx = bins - 1
            private[t, idx] += 1
    counts = np.zeros(bins, np.int64)
    for t in range(n_threads):
        for b in range(bins):
            counts[b] += private[t, b]
    return counts
