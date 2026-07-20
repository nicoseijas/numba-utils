"""Weighted and without-replacement sampling."""

from __future__ import annotations

import numpy as np

from numba_utils.arrays import upper_bound
from numba_utils.decorators import cached_njit


@cached_njit
def reservoir_sampling(arr, k):
    """``k`` elements of ``arr`` sampled uniformly WITHOUT replacement.

    Algorithm R: single pass, works when n is large or only known at the
    end. Output order is arbitrary. Input untouched.

    Complexity: O(n). Memory: O(k).
    """
    n = arr.shape[0]
    if k < 1 or k > n:
        raise ValueError("reservoir_sampling: k must be in [1, len(arr)]")
    out = arr[:k].copy()
    for i in range(k, n):
        j = np.random.randint(0, i + 1)
        if j < k:
            out[j] = arr[i]
    return out


@cached_njit
def weighted_sampling(weights, size):
    """``size`` indices in ``[0, n)`` sampled WITH replacement, with
    probability proportional to ``weights``.

    Cumulative sums + binary search: O(n + size·log n). For many draws
    from the same fixed weights, :func:`alias_setup` + :func:`alias_sample`
    is O(n) setup + O(1) per draw. Zero weights are never sampled;
    negative, non-finite (NaN/inf) or all-zero weights raise
    ``ValueError``, as does a sum that overflows to infinity.

    Complexity: O(n + size·log n). Memory: O(n + size).
    """
    n = weights.shape[0]
    if n == 0:
        raise ValueError("weighted_sampling: empty weights")
    if size < 0:
        raise ValueError("weighted_sampling: size must be >= 0")
    cum = np.empty(n, np.float64)
    total = 0.0
    for i in range(n):
        w = np.float64(weights[i])
        # NaN fails every comparison, so test it explicitly: `w < 0` alone
        # would let NaN through and silently produce a degenerate table.
        if not np.isfinite(w):
            raise ValueError("weighted_sampling: weight is not finite")
        if w < 0:
            raise ValueError("weighted_sampling: negative weight")
        total += w
        cum[i] = total
    if not np.isfinite(total):
        raise ValueError("weighted_sampling: weights sum is not finite")
    if total <= 0.0:
        raise ValueError("weighted_sampling: weights sum to zero")
    out = np.empty(size, np.int64)
    for i in range(size):
        u = np.random.random() * total
        out[i] = upper_bound(cum, u)
    return out


@cached_njit
def alias_setup(weights):
    """Walker alias tables ``(prob, alias)`` for O(1) weighted draws.

    Pay O(n) once, then every :func:`alias_draw` costs one uniform, one
    comparison and at most one table lookup — the fastest known method
    for many draws from fixed weights. Validates like
    :func:`weighted_sampling`.

    Complexity: O(n). Memory: O(n).
    """
    n = weights.shape[0]
    if n == 0:
        raise ValueError("alias_setup: empty weights")
    total = 0.0
    for i in range(n):
        w = np.float64(weights[i])
        # See weighted_sampling: NaN passes `w < 0`, so reject it up front.
        if not np.isfinite(w):
            raise ValueError("alias_setup: weight is not finite")
        if w < 0:
            raise ValueError("alias_setup: negative weight")
        total += w
    if not np.isfinite(total):
        raise ValueError("alias_setup: weights sum is not finite")
    if total <= 0.0:
        raise ValueError("alias_setup: weights sum to zero")
    prob = np.empty(n, np.float64)
    alias = np.zeros(n, np.int64)
    scaled = np.empty(n, np.float64)
    for i in range(n):
        scaled[i] = weights[i] * n / total
    small = np.empty(n, np.int64)
    large = np.empty(n, np.int64)
    n_small = 0
    n_large = 0
    for i in range(n):
        if scaled[i] < 1.0:
            small[n_small] = i
            n_small += 1
        else:
            large[n_large] = i
            n_large += 1
    while n_small > 0 and n_large > 0:
        n_small -= 1
        s = small[n_small]
        n_large -= 1
        g = large[n_large]
        prob[s] = scaled[s]
        alias[s] = g
        scaled[g] -= 1.0 - scaled[s]
        if scaled[g] < 1.0:
            small[n_small] = g
            n_small += 1
        else:
            large[n_large] = g
            n_large += 1
    while n_large > 0:
        n_large -= 1
        prob[large[n_large]] = 1.0
    while n_small > 0:
        n_small -= 1
        prob[small[n_small]] = 1.0
    return prob, alias


@cached_njit
def alias_draw(prob, alias):
    """One weighted index from tables built by :func:`alias_setup`. O(1)."""
    i = np.random.randint(0, prob.shape[0])
    if np.random.random() < prob[i]:
        return i
    return alias[i]


@cached_njit
def alias_sample(prob, alias, size):
    """``size`` weighted indices from :func:`alias_setup` tables. O(size)."""
    if size < 0:
        raise ValueError("alias_sample: size must be >= 0")
    out = np.empty(size, np.int64)
    for i in range(size):
        out[i] = alias_draw(prob, alias)
    return out
