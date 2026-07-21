"""Certification helpers: make the checks themselves falsifiable.

A green reference test only certifies something if breaking the kernel
turns it red. These helpers come from a production CFR solver's
L0→L3 certification pyramid, where "plausible numbers from an
uncertified kernel" voided real conclusions more than once.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from numba_utils.testing._stochastic import assert_within_se


def _max_abs_diff(a: Any, b: Any) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.shape != y.shape:
        raise AssertionError(
            f"mutation_screams: intact and broken results have different "
            f"shapes ({x.shape} vs {y.shape}) — that IS a scream, but an "
            "uninformative one; make the mutant return comparable output"
        )
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x - y)))


def mutation_screams(
    fn: Callable[[bool], Any],
    *,
    threshold: float,
    metric: Callable[[Any, Any], float] | None = None,
) -> float:
    """Assert that a deliberately broken kernel produces output that
    deviates from the intact one by MORE than ``threshold``.

    ``fn(broken)`` runs the kernel: ``fn(False)`` intact, ``fn(True)``
    with a deliberate bug injected (the production kernel takes a
    ``broken=`` kwarg, or the test closure flips whatever the mutation
    is). The deviation is ``metric(intact, broken)`` — by default the
    max absolute elementwise difference.

    Why this exists: **a check that cannot fail certifies nothing.**
    A reference test that stays green when the kernel is broken is
    measuring the wrong thing — this is the red half that proves the
    green half is alive. Pick ``threshold`` at the size of the error
    the surrounding checks are supposed to catch, not at float noise.

    Returns the measured deviation. A non-finite deviation (the mutant
    produced NaN/inf) counts as a scream.
    """
    if not callable(fn):
        raise TypeError("mutation_screams expects a callable fn(broken)")
    if not (threshold > 0) or not math.isfinite(threshold):
        raise ValueError("mutation_screams: threshold must be finite and > 0")
    intact = fn(False)
    broken = fn(True)
    deviation = float(
        metric(intact, broken) if metric is not None else _max_abs_diff(intact, broken)
    )
    if math.isfinite(deviation) and not (deviation > threshold):
        raise AssertionError(
            f"mutation_screams: the mutation does NOT scream — deviation "
            f"{deviation!r} <= threshold {threshold!r}. The check this "
            "mutation protects cannot fail, so it certifies nothing."
        )
    return deviation


def assert_no_reweight_bias(
    estimator: Callable[[np.ndarray, np.ndarray, int], float],
    *,
    n: int = 2000,
    n_runs: int = 20,
    seed: int = 0,
    k: float = 5.0,
) -> tuple[float, float]:
    """Assert a weighted-mean estimator is free of the reach² bug:
    subsampling proportional to the weights and then weighting again
    (effective ``weight²``).

    ``estimator(values, weights, run_seed) -> float`` estimates
    ``Σ w·v / Σ w``. The harness feeds it values CORRELATED with
    strongly skewed weights — the regime where double-weighting biases
    hardest. Both properties matter: with near-uniform weights the bug
    is invisible (that is exactly why it survives in production), and
    with values independent of the weights both estimators agree in
    expectation, hiding it again. Asserts the ``n_runs`` estimates
    stay within ``k`` measured standard errors of the exact weighted
    mean (:func:`assert_within_se`); returns ``(mean, se)``.

    From a production solver where this bug shipped twice: invisible
    at 0/1 reach, and at strictly mixed frequencies it produced
    equities above 1 and an impossible negative rake.
    """
    if not callable(estimator):
        raise TypeError("assert_no_reweight_bias expects a callable estimator")
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    weights = u**6  # a few entries carry most of the mass
    values = u  # correlated with the weights, see docstring
    exact = float(np.sum(weights * values) / np.sum(weights))
    estimates = [
        float(estimator(values, weights, seed + 1000 + r)) for r in range(n_runs)
    ]
    try:
        return assert_within_se(estimates, exact, k=k)
    except AssertionError as exc:
        raise AssertionError(f"assert_no_reweight_bias: {exc}") from None
