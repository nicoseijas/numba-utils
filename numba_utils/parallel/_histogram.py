"""Parallel histogram with per-thread private counts."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, prange

from numba_utils.arrays import histogram
from numba_utils.arrays._hist import MAX_HISTOGRAM_BINS
from numba_utils.decorators import njit_parallel

_SERIAL_THRESHOLD = 1 << 16
# Budget for the per-thread private tables, in int64 slots (2**27 =
# 1 GiB total). The library-wide bins cap (2**30) bounds ONE row at
# 8 GiB — multiplied by n_threads that is a working set the cap alone
# does not see (256 GiB at 32 threads). Past this budget the O(threads
# · bins) merge dominates the parallel gain anyway, so the serial
# fallback loses little.
_MAX_PRIVATE_SLOTS = 1 << 27


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

    Complexity: O(n + threads·bins). Memory: O(threads·bins), bounded:
    past a 1 GiB private-table budget (``threads·bins`` over 2**27
    slots) it delegates to the serial kernel — the bins cap alone
    bounds one row, not the per-thread multiplication.
    """
    n = arr.shape[0]
    # Check the bin cap BEFORE the serial delegation so the same bins
    # is accepted or rejected regardless of array length (the cap is a
    # library-wide contract, not a property of one path). Near 2**63
    # the (bins + 7) padding arithmetic below would overflow int64 into
    # a negative dimension, which parallel lowering turns into an
    # out-of-bounds write instead of a clean allocation error.
    if bins < 1:
        raise ValueError("parallel_histogram: bins must be >= 1")
    if bins > MAX_HISTOGRAM_BINS:
        raise ValueError("parallel_histogram: bins too large (> 2**30)")
    if n < _SERIAL_THRESHOLD:
        return histogram(arr, bins, lo, hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("parallel_histogram: lo and hi must be finite")
    if not lo < hi:
        raise ValueError("parallel_histogram: lo must be < hi")
    span = hi - lo
    scale = bins / span
    # See histogram: an overflowing span (scale 0) or a subnormal span
    # (scale inf) cannot bin by scaling — every count would silently
    # land in bin 0. Fail loudly instead.
    if not (np.isfinite(span) and np.isfinite(scale)):
        raise ValueError(
            "parallel_histogram: hi - lo overflows or is too small "
            "for this many bins"
        )
    n_threads = get_num_threads()
    # Pad each thread's row to a multiple of 8 int64 (= 64 bytes) so two
    # threads never share a cache line.
    padded = ((bins + 7) // 8) * 8
    # The acceptance surface (bins cap) is library-wide and unchanged;
    # what the budget bounds is the PRIVATE working set, which scales
    # with n_threads and is invisible to the caller. Same result either
    # way: the serial kernel is bit-exact with the merge.
    if n_threads * padded > _MAX_PRIVATE_SLOTS:
        return histogram(arr, bins, lo, hi)
    private = np.zeros((n_threads, padded), np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for i in range(start, end):
            x = arr[i]
            # Inverted-range test so NaN is skipped, as in the serial
            # histogram: int(NaN) is INT64_MIN, an out-of-bounds index.
            if not (lo <= x <= hi):
                continue
            idx = int((x - lo) * scale)
            if idx >= bins:
                idx = bins - 1
            # Symmetric clamp, as in the serial histogram: a subnormal
            # hi - lo degenerates scale and 0 * inf = NaN -> INT64_MIN
            # even for in-range x.
            if idx < 0:
                idx = 0
            private[t, idx] += 1
    counts = np.zeros(bins, np.int64)
    for t in range(n_threads):
        for b in range(bins):
            counts[b] += private[t, b]
    return counts
