"""Assertions for stochastic (Monte Carlo) functions.

:func:`assert_equivalent` covers deterministic case-by-case equality.
Stochastic code needs two different guarantees, and mixing them up
produces tests that are either flaky or vacuous:

- **Reproducibility** — same seed, bit-identical result
  (:func:`assert_reproducible`).
- **Convergence** — different seeds, statistically close to the truth
  (:func:`assert_converges`).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from numba_utils.testing._generators import deterministic_rng


def _assert_exactly_equal(a: Any, b: Any, context: str) -> None:
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not (isinstance(a, tuple) and isinstance(b, tuple)) or len(a) != len(b):
            raise AssertionError(f"{context}: result structures differ")
        for j, (x, y) in enumerate(zip(a, b)):
            _assert_exactly_equal(x, y, f"{context}, tuple element {j}")
        return
    # assert_array_equal is exact (and treats NaN as equal to NaN),
    # for scalars and arrays alike.
    np.testing.assert_array_equal(a, b, err_msg=context)


def assert_reproducible(
    fn: Callable[..., Any], args: tuple = (), *, seed: int = 0, runs: int = 2
) -> Any:
    """Assert ``fn(*args)`` is bit-identical across ``runs`` runs under
    the same seed.

    Before each run, :func:`deterministic_rng` pins all three random
    worlds (NumPy legacy, NumPy ``Generator``, Numba's separate
    nopython state). A failure means ``fn`` draws from an unseeded
    source — thread-scheduling-dependent RNG, wall clock, iteration
    order of an unordered container. Returns the (verified) result of
    the last run.
    """
    if not callable(fn):
        raise TypeError("assert_reproducible expects a callable")
    if runs < 2:
        raise ValueError("assert_reproducible: runs must be >= 2")
    deterministic_rng(seed)
    reference = fn(*args)
    for r in range(1, runs):
        deterministic_rng(seed)
        result = fn(*args)
        _assert_exactly_equal(
            reference,
            result,
            f"assert_reproducible: run {r} differs from run 0 "
            f"under identical seed {seed}",
        )
    return reference


def assert_converges(
    fn: Callable[..., Any],
    truth: float,
    args: tuple = (),
    *,
    n_runs: int = 30,
    sigma: float = 3.0,
    seed: int = 0,
) -> tuple[float, float]:
    """Assert the mean of ``fn(*args)`` over ``n_runs`` differently-
    seeded runs lies within ``sigma`` standard errors of ``truth``.

    Runs ``fn`` under seeds ``seed .. seed + n_runs - 1`` (each pinned
    via :func:`deterministic_rng`), estimates the standard error from
    the runs themselves (``std(ddof=1) / sqrt(n_runs)``), and asserts
    ``|mean - truth| <= sigma * SE``. Returns ``(mean, se)``.

    This is a statistical test: at the default ``sigma=3`` a CORRECT
    implementation still fails about 0.27% of the time. Don't tighten
    ``sigma`` to 2 in CI (4.6% false positives); if the test flakes at
    3, suspect the code before the tolerance. A zero-variance result
    (every seed identical) is compared to ``truth`` exactly and fails
    with its own message — it usually means the seeds are not reaching
    the sampler.
    """
    if not callable(fn):
        raise TypeError("assert_converges expects a callable")
    if n_runs < 2:
        raise ValueError("assert_converges: n_runs must be >= 2")
    if not (sigma > 0):
        raise ValueError("assert_converges: sigma must be > 0")
    if not math.isfinite(truth):
        raise ValueError("assert_converges: truth must be finite")
    values = np.empty(n_runs, np.float64)
    for r in range(n_runs):
        deterministic_rng(seed + r)
        values[r] = float(fn(*args))
    if not np.all(np.isfinite(values)):
        raise AssertionError(
            "assert_converges: fn returned a non-finite value"
        )
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n_runs))
    if se == 0.0:
        if mean != truth:
            raise AssertionError(
                f"assert_converges: all {n_runs} seeds produced the "
                f"identical value {mean!r} != truth {truth!r} — the "
                "seeds are likely not reaching the sampler"
            )
        return mean, se
    deviation = abs(mean - truth) / se
    if deviation > sigma:
        raise AssertionError(
            f"assert_converges: mean {mean!r} is {deviation:.2f} SE from "
            f"truth {truth!r} (limit {sigma}; SE {se:.3g}, "
            f"{n_runs} runs)"
        )
    return mean, se
