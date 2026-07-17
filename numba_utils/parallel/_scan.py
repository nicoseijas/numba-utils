"""Parallel inclusive prefix sum (two-phase blocked scan)."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, prange

from numba_utils.decorators import njit_parallel

_SERIAL_THRESHOLD = 1 << 16


# cache=False explicitly: get_num_threads makes the parfor-transformed
# function trip Numba's "dynamic globals" cache limitation (would warn).
@njit_parallel(cache=False)
def parallel_prefix_sum(arr, out=None):
    """Inclusive prefix sum of 1-D ``arr`` in float64, computed in parallel.

    The classic two-phase blocked scan: each thread scans its own chunk
    and records the chunk total; chunk totals are scanned serially (tiny);
    each thread then adds its chunk's offset. Two prange launches total.

    Unlike :func:`numba_utils.cumulative_sum` (dtype-preserving, serial),
    output is always float64 and parallel reassociation can differ from
    the serial scan in the last bits. Serial fallback below the size
    threshold.

    Complexity: O(n). Memory: O(n) output + O(threads).
    """
    n = arr.shape[0]
    if out is None:
        out = np.empty(n, np.float64)
    if out.shape[0] != n:
        raise ValueError("parallel_prefix_sum: out has wrong length")
    if n < _SERIAL_THRESHOLD:
        if n > 0:
            acc = 0.0
            for i in range(n):
                acc += arr[i]
                out[i] = acc
        return out
    n_threads = get_num_threads()
    chunk = (n + n_threads - 1) // n_threads
    chunk_sums = np.zeros(n_threads, np.float64)
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        acc = 0.0
        for i in range(start, end):
            acc += arr[i]
            out[i] = acc
        chunk_sums[t] = acc
    offsets = np.empty(n_threads, np.float64)
    running = 0.0
    for t in range(n_threads):
        offsets[t] = running
        running += chunk_sums[t]
    for t in prange(n_threads):
        offset = offsets[t]
        if offset != 0.0:
            start = t * chunk
            end = min(start + chunk, n)
            for i in range(start, end):
                out[i] += offset
    return out
