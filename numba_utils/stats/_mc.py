"""Weighted Monte Carlo estimation without the reach² bug."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit
from numba_utils.random._sampling import philox_partial_shuffle


@cached_njit
def weighted_mc_mean(values, weights, n_sub, key, counter):
    """Weighted mean ``Σ w·v / Σ w`` estimated on a UNIFORM subsample
    of the support, weighted by the true weights — never both.

    The bug this exists to prevent: subsampling proportional to the
    weights and THEN weighting by them again gives effective
    ``weight²``. It is invisible at near-uniform weights and explosive
    at concentrated ones (a production solver shipped it twice:
    equities above 1, a negative rake). The correct pattern is fixed
    here: the subsample is drawn uniformly over the support
    (``weight > 0``), and the true weight enters exactly once, as the
    weight. Guard your own estimators with
    :func:`numba_utils.testing.assert_no_reweight_bias`.

    Driven by the stateless Philox stream ``key`` (consumes at most
    ``n_sub`` counters starting at ``counter``), so the estimate is
    reproducible regardless of threads or call order. If the support
    has at most ``n_sub`` entries, returns the EXACT weighted mean.
    This is the standard ratio estimator: consistent, with O(1/n_sub)
    bias — the SE-vs-truth check in ``assert_no_reweight_bias`` is the
    honest way to bound it for your sizes.

    Validation as in :func:`numba_utils.weighted_quantile`: NaN values,
    NaN/negative/non-finite weights and an all-zero total raise
    ``ValueError``.

    Complexity: O(n + n_sub). Memory: O(n_support).
    """
    n = values.shape[0]
    if n == 0:
        raise ValueError("weighted_mc_mean: empty values")
    if weights.shape[0] != n:
        raise ValueError("weighted_mc_mean: values and weights lengths differ")
    if n_sub < 1:
        raise ValueError("weighted_mc_mean: n_sub must be >= 1")
    n_support = 0
    for i in range(n):
        v = np.float64(values[i])
        if np.isnan(v):
            raise ValueError("weighted_mc_mean: NaN value")
        w = np.float64(weights[i])
        if not np.isfinite(w):
            raise ValueError("weighted_mc_mean: weight is not finite")
        if w < 0:
            raise ValueError("weighted_mc_mean: negative weight")
        if w > 0:
            n_support += 1
    if n_support == 0:
        raise ValueError("weighted_mc_mean: weights sum to zero")
    support = np.empty(n_support, np.int64)
    j = 0
    for i in range(n):
        if weights[i] > 0:
            support[j] = i
            j += 1
    if n_support > n_sub:
        philox_partial_shuffle(support, n_sub, key, counter)
        m = n_sub
    else:
        m = n_support
    acc_wv = 0.0
    acc_w = 0.0
    for s in range(m):
        i = support[s]
        w = np.float64(weights[i])
        acc_wv += w * np.float64(values[i])
        acc_w += w
    return acc_wv / acc_w
