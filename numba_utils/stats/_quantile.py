"""Weighted quantile (inverted CDF convention)."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def weighted_quantile(values, weights, q):
    """``q``-quantile of ``values`` weighted by ``weights``.

    Inverted CDF convention — the smallest value whose cumulative
    weight reaches ``q`` of the total — chosen because it is the one
    convention NumPy itself supports with weights: results match
    ``np.quantile(values, q, weights=weights, method="inverted_cdf")``
    exactly. Returns an element of ``values`` (no interpolation).

    Like NumPy's, ``q=0`` returns the smallest value even if its weight
    is zero; for ``q > 0`` zero-weight values are never the crossing
    point. Validation fails fast:
    ``q`` outside ``[0, 1]``, empty input, NaN values, and — as in
    :func:`numba_utils.weighted_sampling` — negative or non-finite
    weights or an all-zero total raise ``ValueError`` (NaN would pass a
    plain ``w < 0`` check and silently corrupt the cumulative sums).

    Complexity: O(n log n) for the sort. Memory: O(n).
    """
    n = values.shape[0]
    if n == 0:
        raise ValueError("weighted_quantile: empty values")
    if weights.shape[0] != n:
        raise ValueError("weighted_quantile: values and weights lengths differ")
    if not (0.0 <= q <= 1.0):
        raise ValueError("weighted_quantile: q must be in [0, 1]")
    total = 0.0
    for i in range(n):
        v = np.float64(values[i])
        if np.isnan(v):
            raise ValueError("weighted_quantile: NaN value")
        w = np.float64(weights[i])
        if not np.isfinite(w):
            raise ValueError("weighted_quantile: weight is not finite")
        if w < 0:
            raise ValueError("weighted_quantile: negative weight")
        total += w
    if total <= 0.0:
        raise ValueError("weighted_quantile: weights sum to zero")
    # An unstable argsort is enough: among tied values the crossing
    # element has the same VALUE regardless of their relative order.
    order = np.argsort(values)
    target = q * total
    acc = 0.0
    for i in range(n):
        idx = order[i]
        acc += np.float64(weights[idx])
        if acc >= target:
            return values[idx]
    # Float rounding can leave acc a hair below q * total at the end;
    # the answer is then the largest value, as in NumPy.
    return values[order[n - 1]]
