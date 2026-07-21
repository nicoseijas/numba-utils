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
    # Positions where both sides agree — including NaN==NaN and
    # inf==inf — contribute zero. Without this, an intact output that
    # legitimately contains NaN made ``NaN - NaN = NaN`` count as a
    # "scream" even when the mutation changed NOTHING. A non-finite
    # deviation can now only come from a position where the two runs
    # actually differ (NaN appeared/disappeared, inf changed), which is
    # a real scream.
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    if bool(np.all(same)):
        return 0.0
    # inf - inf at agreeing positions is discarded by the mask; keep
    # NumPy from warning about it.
    with np.errstate(invalid="ignore"):
        return float(np.max(np.where(same, 0.0, np.abs(x - y))))


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
    CHANGED a value to NaN/inf, or vice versa) counts as a scream —
    positions where intact and broken agree, including agreeing on
    NaN/inf, contribute nothing. ``fn`` returning ``None`` is rejected:
    an in-place kernel must return the buffer it mutated, otherwise
    there is literally nothing to compare and the certification is
    vacuous. A custom ``metric`` must follow the same contract: 0 for
    identical outputs, finite/non-finite growth only with real change.
    """
    if not callable(fn):
        raise TypeError("mutation_screams expects a callable fn(broken)")
    if not (threshold > 0) or not math.isfinite(threshold):
        raise ValueError("mutation_screams: threshold must be finite and > 0")
    intact = fn(False)
    broken = fn(True)
    if intact is None or broken is None:
        which = "intact" if intact is None else "broken"
        if intact is None and broken is None:
            which = "both"
        raise TypeError(
            f"mutation_screams: fn returned None ({which} run) — an "
            "in-place kernel must RETURN the buffer it mutated; with "
            "None there is nothing to compare, so the check would "
            "certify a mutation that is not wired to anything"
        )
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

    A pass additionally requires the run to be CONCLUSIVE: ``k·SE``
    must be smaller than half the distance between the exact weighted
    mean and the double-weighted mean ``Σ w²·v / Σ w²`` (what the bug
    converges to on this fixture). The k·SE criterion alone
    self-hides: a bug that also inflates the estimator's variance
    widens its own tolerance band — a high-variance estimator could
    pass while too noisy to distinguish correct from broken. When the
    resolution is insufficient the assert fails as inconclusive
    instead of certifying nothing; raise ``n_runs`` or the estimator's
    internal sample size.

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
    double_weighted = float(
        np.sum(weights**2 * values) / np.sum(weights**2)
    )
    estimates = [
        float(estimator(values, weights, seed + 1000 + r)) for r in range(n_runs)
    ]
    try:
        mean, se = assert_within_se(estimates, exact, k=k)
    except AssertionError as exc:
        raise AssertionError(f"assert_no_reweight_bias: {exc}") from None
    resolution = k * se
    gap = abs(exact - double_weighted)
    if resolution > gap / 2:
        raise AssertionError(
            f"assert_no_reweight_bias: INCONCLUSIVE — the run's "
            f"resolution (k·SE = {resolution:.3g}) cannot distinguish "
            f"the exact weighted mean ({exact:.6g}) from the "
            f"double-weighted one ({double_weighted:.6g}, gap "
            f"{gap:.3g}). A pass at this noise level certifies "
            "nothing; increase n_runs or the estimator's sample size."
        )
    return mean, se
