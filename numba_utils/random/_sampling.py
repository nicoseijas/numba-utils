"""Weighted and without-replacement sampling."""

from __future__ import annotations

import numpy as np

from numba_utils.arrays import upper_bound
from numba_utils.decorators import cached_njit
from numba_utils.random._philox import philox_randint


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
def partial_shuffle(arr, k):
    """Rearrange ``arr`` IN PLACE so ``arr[:k]`` is a uniform sample of
    its elements WITHOUT replacement; returns ``arr``.

    Partial Fisher–Yates: exactly ``k`` swaps and ``k`` random draws —
    not a full shuffle. This is the zero-allocation repeated-draw
    Monte Carlo primitive: keep one scratch deck and call this per
    iteration instead of copying or reshuffling the whole array.
    ``arr[k:]`` holds the rest in arbitrary order. Uses Numba's
    per-thread global RNG (seed with :func:`numba_utils.seed`); for
    thread-count-independent reproducibility drive indices with the
    ``philox_*`` functions instead.

    ``arr`` must be 1-D: on a 2-D array the swap works on row VIEWS,
    which alias — the "sample" silently contains duplicated rows
    (dealing the same card twice). Raises ``ValueError`` instead.

    Complexity: O(k). Memory: O(1).
    """
    if arr.ndim != 1:
        raise ValueError("partial_shuffle: arr must be 1-D (row swaps alias)")
    n = arr.shape[0]
    if k < 0 or k > n:
        raise ValueError("partial_shuffle: k must be in [0, len(arr)]")
    for i in range(k):
        j = np.random.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


@cached_njit
def sample_without_replacement(arr, k):
    """``k`` elements of ``arr`` sampled uniformly WITHOUT replacement,
    as a new array. Input untouched. ``k = 0`` returns an empty array
    (matching :func:`partial_shuffle`, which accepts it too).

    Copies the pool once, partial Fisher–Yates (``k`` swaps), then
    copies the ``k`` winners out (so the size-k result doesn't pin the
    size-n scratch buffer alive) — O(n + k) copied elements total.
    Compared to :func:`reservoir_sampling` (one pass over all n, n
    random draws), this wins when ``k << n`` and the array is
    materialized — the common "deal k cards from a deck" case. In a
    hot loop, skip the per-call copies: keep a scratch array and use
    :func:`partial_shuffle` directly.

    Complexity: O(n + k) copies + O(k) swaps. Memory: O(n).
    """
    n = arr.shape[0]
    if k < 0 or k > n:
        raise ValueError("sample_without_replacement: k must be in [0, len(arr)]")
    tmp = arr.copy()
    partial_shuffle(tmp, k)
    return tmp[:k].copy()


@cached_njit
def philox_partial_shuffle(arr, k, key, counter):
    """Counter-based :func:`partial_shuffle`: same in-place partial
    Fisher–Yates, but driven by the stateless Philox stream ``key`` —
    reproducible regardless of threads, processes or call order.

    Consumes counters ``counter .. counter + k - 1``: give each work
    unit a disjoint counter range (e.g.
    ``counter = iteration * k_per_iteration``) and the whole run is
    reproducible by construction. Returns ``arr``.

    ``arr`` must be 1-D (see :func:`partial_shuffle`: row swaps alias).

    Complexity: O(k). Memory: O(1).
    """
    if arr.ndim != 1:
        raise ValueError(
            "philox_partial_shuffle: arr must be 1-D (row swaps alias)"
        )
    n = arr.shape[0]
    if k < 0 or k > n:
        raise ValueError("philox_partial_shuffle: k must be in [0, len(arr)]")
    c = np.uint64(counter)
    for i in range(k):
        j = i + philox_randint(key, c, n - i)
        arr[i], arr[j] = arr[j], arr[i]
        c += np.uint64(1)
    return arr


@cached_njit
def philox_sample_without_replacement(arr, k, key, counter):
    """Counter-based :func:`sample_without_replacement`: ``k`` elements
    without replacement from the stateless Philox stream ``key``.
    Input untouched; consumes counters ``counter .. counter + k - 1``.
    ``k = 0`` returns an empty array and consumes no counters.

    Copies the pool once plus the ``k`` winners (see
    :func:`sample_without_replacement`). In a hot loop, skip the
    per-call copies: keep a scratch array and use
    :func:`philox_partial_shuffle` directly.

    Complexity: O(n + k) copies + O(k) swaps. Memory: O(n).
    """
    n = arr.shape[0]
    if k < 0 or k > n:
        raise ValueError(
            "philox_sample_without_replacement: k must be in [0, len(arr)]"
        )
    tmp = arr.copy()
    philox_partial_shuffle(tmp, k, key, counter)
    return tmp[:k].copy()


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
    n_last = n - 1
    out = np.empty(size, np.int64)
    for i in range(size):
        u = np.random.random() * total
        idx = upper_bound(cum, u)
        # Defense in depth: with round-to-nearest and random()'s 2**-53
        # granularity, u < total always holds (verified analytically
        # and empirically), so idx <= n-1. The clamp guards against a
        # future fastmath/rounding-mode/RNG-granularity change turning
        # this into an out-of-range index.
        if idx > n_last:
            idx = n_last
        out[i] = idx
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
    """One weighted index from tables built by :func:`alias_setup`. O(1).

    Raises ``ValueError`` if the two tables' lengths differ (mixed-up
    tables would otherwise index out of range — silent corruption in
    nopython).
    """
    if prob.shape[0] != alias.shape[0]:
        raise ValueError("alias_draw: prob and alias lengths differ")
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
