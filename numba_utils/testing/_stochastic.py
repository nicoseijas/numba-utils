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
    fn: Callable[..., Any],
    args: tuple = (),
    *,
    seed: int = 0,
    runs: int = 2,
    pass_seed: bool = False,
) -> Any:
    """Assert ``fn`` is bit-identical across ``runs`` runs under the
    same seed.

    Before each run, :func:`deterministic_rng` pins all three random
    worlds (NumPy legacy, NumPy ``Generator``, Numba's separate
    nopython state). With ``pass_seed=True``, ``fn`` is called as
    ``fn(seed, *args)`` — the mode for counter-based kernels
    (:func:`numba_utils.philox_uniform`), whose stream comes from an
    argument rather than global state. A failure means ``fn`` draws
    from an unseeded source — thread-scheduling-dependent RNG, wall
    clock, iteration order of an unordered container. Returns the
    (verified) result of the last run.
    """
    if not callable(fn):
        raise TypeError("assert_reproducible expects a callable")
    if runs < 2:
        raise ValueError("assert_reproducible: runs must be >= 2")
    call_args = (seed, *args) if pass_seed else args
    deterministic_rng(seed)
    reference = fn(*call_args)
    for r in range(1, runs):
        deterministic_rng(seed)
        result = fn(*call_args)
        _assert_exactly_equal(
            reference,
            result,
            f"assert_reproducible: run {r} differs from run 0 "
            f"under identical seed {seed}",
        )
    return reference


def assert_within_se(
    samples, target: float, *, k: float = 3.0
) -> tuple[float, float]:
    """Assert the mean of iid ``samples`` lies within ``k`` standard
    errors of ``target``; returns ``(mean, se)``.

    The SE is MEASURED from the samples (``std(ddof=1) / sqrt(n)``),
    never assumed — a residual below an unmeasured noise floor is not
    a pass. This is the one-sample-set primitive under
    :func:`assert_converges`, which also documents the Student-t
    false-positive rates the measured-SE statistic carries; the same
    ``n >= 5`` floor applies. Contributed pattern from a production
    CFR solver's certification harness.
    """
    arr = np.asarray(samples, np.float64)
    if arr.ndim != 1:
        raise ValueError("assert_within_se: samples must be 1-D")
    n = arr.shape[0]
    if n < 5:
        raise ValueError(
            "assert_within_se: need at least 5 samples (the measured-SE "
            "statistic is Student-t; its false-positive rate explodes "
            "below that)"
        )
    if not (k > 0):
        raise ValueError("assert_within_se: k must be > 0")
    if not math.isfinite(target):
        raise ValueError("assert_within_se: target must be finite")
    if not np.all(np.isfinite(arr)):
        raise AssertionError("assert_within_se: samples contain a non-finite value")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    if se == 0.0:
        if mean != target:
            raise AssertionError(
                f"assert_within_se: all {n} samples are the identical "
                f"value {mean!r} != target {target!r} — if they come "
                "from seeded runs, the seeds are likely not reaching "
                "the sampler"
            )
        return mean, se
    deviation = abs(mean - target) / se
    if deviation > k:
        raise AssertionError(
            f"assert_within_se: mean {mean!r} is {deviation:.2f} SE from "
            f"target {target!r} (limit {k}; SE {se:.3g}, {n} samples)"
        )
    return mean, se


def assert_converges(
    fn: Callable[..., Any],
    truth: float,
    args: tuple = (),
    *,
    n_runs: int = 30,
    sigma: float = 3.0,
    seed: int = 0,
    pass_seed: bool = False,
) -> tuple[float, float]:
    """Assert the mean of ``fn`` over ``n_runs`` differently-seeded
    runs lies within ``sigma`` standard errors of ``truth``.

    Runs ``fn`` under seeds ``seed .. seed + n_runs - 1`` (each pinned
    via :func:`deterministic_rng`), estimates the standard error from
    the runs themselves (``std(ddof=1) / sqrt(n_runs)``), and asserts
    ``|mean - truth| <= sigma * SE``. Returns ``(mean, se)``. With
    ``pass_seed=True``, each run calls ``fn(run_seed, *args)`` — the
    mode for counter-based kernels (:func:`numba_utils.philox_uniform`),
    which are pure functions of their key: vary the key per run or the
    variance is zero and the test cannot work.

    This is a statistical test, and because the SE is ESTIMATED from
    the runs, the statistic is Student-t with ``n_runs - 1`` degrees of
    freedom — the false-positive rate for a CORRECT implementation is
    higher than the normal-distribution intuition suggests:

    ========  ==============================
    n_runs    false positives at ``sigma=3``
    ========  ==============================
    30        ~0.55%
    10        ~1.5%
    5         ~4%
    ========  ==============================

    ``n_runs`` below 5 is rejected (at ``n_runs=2`` the rate would be
    ~20%). Don't tighten ``sigma`` to 2 in CI; if the test flakes at 3,
    suspect the code before the tolerance. A zero-variance result
    (every seed identical) is compared to ``truth`` exactly and fails
    with its own message — it usually means the seeds are not reaching
    the sampler.
    """
    if not callable(fn):
        raise TypeError("assert_converges expects a callable")
    if n_runs < 5:
        raise ValueError(
            "assert_converges: n_runs must be >= 5 (the t-statistic's "
            "false-positive rate explodes below that; ~20% at n_runs=2)"
        )
    if not (sigma > 0):
        raise ValueError("assert_converges: sigma must be > 0")
    if not math.isfinite(truth):
        raise ValueError("assert_converges: truth must be finite")
    values = np.empty(n_runs, np.float64)
    for r in range(n_runs):
        deterministic_rng(seed + r)
        if pass_seed:
            values[r] = float(fn(seed + r, *args))
        else:
            values[r] = float(fn(*args))
    try:
        return assert_within_se(values, truth, k=sigma)
    except AssertionError as exc:
        raise AssertionError(f"assert_converges: {exc}") from None
