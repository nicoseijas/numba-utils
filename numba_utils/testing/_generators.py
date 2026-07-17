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

    Edge cases (when ``include_edges``): constant array, sorted
    ascending, sorted descending, few distinct values (duplicates), and
    a single-element array. Then ``n_cases`` random arrays of ``size``
    (normal-distributed for float dtypes, wide-range integers for int
    dtypes). Deterministic for a given ``seed``.
    """
    if n_cases < 0:
        raise ValueError("random_arrays: n_cases must be >= 0")
    if size < 1:
        raise ValueError("random_arrays: size must be >= 1")
    rng = np.random.default_rng(seed)
    is_float = np.issubdtype(np.dtype(dtype), np.floating)

    def _random(n: int) -> np.ndarray:
        if is_float:
            return rng.normal(0.0, 100.0, n).astype(dtype)
        return rng.integers(-1000, 1000, n).astype(dtype)

    if include_edges:
        yield np.full(size, 3).astype(dtype)
        base = _random(size)
        yield np.sort(base)
        yield np.sort(base)[::-1].copy()
        yield rng.integers(0, 3, size).astype(dtype)
        yield _random(1)
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
