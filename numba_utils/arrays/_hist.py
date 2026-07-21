"""Integer-optimized counting: bincount and fixed-range histogram."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit

# Shared bin cap: 2**30 int64 counts is already 8 GiB, and near 2**63
# the (bins + 7) padding arithmetic in the parallel path overflows.
# Library-wide so serial and parallel draw the same line.
MAX_HISTOGRAM_BINS = 1 << 30


@cached_njit
def bincount(arr, minlength=0):
    """Count occurrences of each value in a 1-D array of nonnegative ints.

    ``result[v]`` is how many times ``v`` appears. Output length is
    ``max(arr.max() + 1, minlength)``. Raises ``ValueError`` on negatives.

    Complexity: O(n + max). Memory: O(max).
    """
    if minlength < 0:
        raise ValueError("bincount: minlength must be >= 0")
    mx = minlength - 1
    for i in range(arr.shape[0]):
        v = arr[i]
        if v < 0:
            raise ValueError("bincount: negative values are not allowed")
        if v > mx:
            mx = v
    counts = np.zeros(mx + 1, np.int64)
    for i in range(arr.shape[0]):
        counts[arr[i]] += 1
    return counts


@cached_njit
def histogram(arr, bins, lo, hi):
    """Histogram of 1-D ``arr`` over ``bins`` equal-width bins in ``[lo, hi]``.

    Single pass, no edge array: bin index is computed by scaling, which is
    why this beats ``np.histogram``. Values outside ``[lo, hi]`` and NaN
    are ignored; ``hi`` itself lands in the last bin (NumPy convention).

    Two boundary caveats of the scaling approach (vs NumPy's edge array):

    - Edge-adjacent values can land one bin away from
      ``np.histogram``'s assignment in narrow ranges — the per-bin
      counts may differ by a small amount while the total is identical.
      Use ``np.histogram`` when exact edge placement matters.
    - A degenerate ``hi - lo`` (overflowing to inf, or subnormal so
      ``bins / (hi - lo)`` is inf) cannot be binned by scaling and
      raises rather than silently placing every value in bin 0. The
      rejected surface is ``hi - lo < bins / 1.8e308`` — NumPy handles
      part of it via bin edges, so prefer ``np.histogram`` for
      extreme-magnitude ranges.

    Complexity: O(n + bins). Memory: O(bins).
    """
    if bins < 1:
        raise ValueError("histogram: bins must be >= 1")
    if bins > MAX_HISTOGRAM_BINS:
        raise ValueError("histogram: bins too large (> 2**30)")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("histogram: lo and hi must be finite")
    if not lo < hi:
        raise ValueError("histogram: lo must be < hi")
    span = hi - lo
    scale = bins / span
    # Two degenerate regimes produce silently FALSE counts if allowed
    # through: a span that overflows to inf (finite lo, hi near
    # +/-1e308) makes scale 0, and a subnormal span makes scale inf —
    # in both, every value lands in bin 0 and counts.sum() still
    # matches n, so nothing downstream can tell. NumPy refuses the
    # first regime too. Fail loudly; the realistic trigger is
    # histogram(a, bins, a.min(), a.max()) on nearly-constant data.
    if not (np.isfinite(span) and np.isfinite(scale)):
        raise ValueError(
            "histogram: hi - lo overflows or is too small for this "
            "many bins"
        )
    counts = np.zeros(bins, np.int64)
    for i in range(arr.shape[0]):
        x = arr[i]
        # Inverted-range test so NaN (which fails every comparison) is
        # skipped: int(NaN) is INT64_MIN, an out-of-bounds index.
        if not (lo <= x <= hi):
            continue
        idx = int((x - lo) * scale)
        if idx >= bins:
            idx = bins - 1
        # Degenerate scales are rejected above; the symmetric clamp
        # stays as the last line of memory safety (e.g. a global
        # fastmath override weakening the NaN filter above).
        if idx < 0:
            idx = 0
        counts[idx] += 1
    return counts
