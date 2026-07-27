"""Test-case generation and reproducible RNG setup."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from numba_utils.random import seed as numba_seed


def random_arrays(
    n_cases: int,
    size: int,
    *,
    dtype: type = np.float64,
    seed: int = 0,
    include_edges: bool = True,
) -> Iterator[np.ndarray]:
    """Yield 1-D test arrays: random cases plus the edge cases that
    actually break kernels.

    Edge cases (when ``include_edges``), six of them in this order:
    constant array, sorted ascending, sorted descending, few distinct
    values (duplicates), a single-element array, and the dtype's
    extremes. Then ``n_cases`` random arrays of ``size``
    (normal-distributed for float dtypes, mid-range integers for int
    dtypes). Deterministic for a given ``seed``.

    The extremes case is deliberate, not a side effect of a cast: the
    random integers are generated INSIDE the dtype's own range (earlier
    releases drew from ``[-1000, 1000)`` and cast, so the negative half
    wrapped for unsigned and narrow dtypes — ``np.uint32(-1000)`` is
    4294966296), and the dtype's true ``min``/``max`` sentinels arrive
    in their own case instead. For float dtypes the extremes are signed
    zeros, the smallest normal and ``+-1`` — finite on purpose: seeding
    a generic generator with ``inf``/``NaN`` breaks reference
    implementations more often than it catches kernel bugs, so pass
    those explicitly when they are what you mean to test.
    """
    if n_cases < 0:
        raise ValueError("random_arrays: n_cases must be >= 0")
    if size < 1:
        raise ValueError("random_arrays: size must be >= 1")
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    is_float = np.issubdtype(dt, np.floating)

    def _random(n: int) -> np.ndarray:
        if is_float:
            return rng.normal(0.0, 100.0, n).astype(dtype)
        info = np.iinfo(dt)
        lo = max(-1000, info.min)
        hi = min(lo + 2000, info.max)
        return rng.integers(lo, hi, n).astype(dtype)

    def _extremes(n: int) -> np.ndarray:
        if is_float:
            info = np.finfo(dt)
            vals = np.array([0.0, -0.0, info.tiny, -info.tiny, 1.0, -1.0], dt)
        else:
            iinfo = np.iinfo(dt)
            vals = np.array([iinfo.min, iinfo.max, 0, 1], dt)
        return np.resize(vals, n)

    if include_edges:
        yield np.full(size, 3).astype(dtype)
        base = _random(size)
        yield np.sort(base)
        yield np.sort(base)[::-1].copy()
        yield rng.integers(0, 3, size).astype(dtype)
        yield _random(1)
        yield _extremes(size)
    for _ in range(n_cases):
        yield _random(size)


def deterministic_rng(seed: int = 0) -> np.random.Generator:
    """Make ALL THREE random worlds reproducible in one call.

    Seeds NumPy's legacy global state (``np.random.*``), Numba's
    nopython RNG (which is separate — see ``numba_utils.random.seed``),
    and returns a seeded ``np.random.Generator`` for modern NumPy code.
    Call at the top of a test or benchmark and every source of
    randomness is pinned.
    """
    np.random.seed(seed)
    numba_seed(seed)
    return np.random.default_rng(seed)
