# Testing strategy

## Why we don't publish a line-coverage number

Traditional line coverage is **intentionally not reported** for this
project. In a Numba codebase there are three different things one could
mean by "coverage":

1. coverage of the Python wrapper code,
2. coverage of Python-level logic (config, diagnostics, generators),
3. coverage of the nopython kernels that users actually execute.

`coverage.py` can only see the first two: Numba compiles nopython
functions to native code, and the Python bytecode those tools trace
never runs. Measured naively, this repository reports ~44% — with every
exhaustively-tested kernel counted as uncovered. Publishing that number
would transmit exactly the wrong message; running the suite with
`NUMBA_DISABLE_JIT=1` to inflate it would measure the Python fallback,
not the product that ships.

This is the testing counterpart of the project's Benchmark Honesty
policy: never show a metric known to be systematically misleading.

## What we guarantee instead

For numerical software, semantic correctness against an independent
reference is a **stronger** guarantee than Python line coverage.

| Guarantee | Status |
| --- | --- |
| Unit tests (200+, all modules) | ✅ |
| Independent reference validation (NumPy, heapq, set) | ✅ |
| Randomized inputs (fixed seeds) | ✅ |
| Edge cases (empty, single, constant, sorted, duplicates) | ✅ |
| Error paths (every documented `ValueError`/`IndexError`) | ✅ |
| Callable-from-`@njit` verified per module | ✅ |
| Bit-exactness where promised (e.g. `parallel_histogram`) | ✅ |
| Benchmarked (reproducible, losses included) | ✅ |
| Deterministic RNG across all three random states | ✅ |
| Python line coverage | Intentionally omitted |

## Why reference validation matters more here

Nopython code has **no bounds checking**: a sizing bug writes out of
bounds silently, corrupting memory instead of crashing. A line can
execute — and count as covered — while producing garbage. The only
trustworthy signal is comparing kernel output against an independent
implementation on inputs designed to break it. Every kernel test in
`tests/` does exactly that.

## Testing your own kernels

The same machinery ships in `numba_utils.testing`:

```python
from numba_utils.testing import assert_equivalent, random_arrays

assert_equivalent(
    python_impl,
    njit_impl,
    random_arrays(n_cases=20, size=10_000),
)
```

`assert_equivalent` copies every array per call (mutating kernels can't
contaminate the comparison), names the failing case, and fails on an
empty generator instead of passing vacuously. `random_arrays` includes
the edge cases that actually break kernels. `deterministic_rng` pins
NumPy's legacy state, a NumPy `Generator`, and Numba's separate
nopython RNG in one call. During development, `@boundscheck` with
`NUMBA_UTILS_DEV=1` turns silent corruption into an `IndexError`.

## Testing stochastic kernels

Monte Carlo code needs two guarantees that `assert_equivalent` cannot
give, and conflating them produces tests that are flaky or vacuous:

- `assert_reproducible(fn, seed=...)` — same seed, bit-identical
  result across runs (via `deterministic_rng`). Fails when `fn` draws
  from anything unseeded: thread-scheduling-dependent RNG, wall clock.
- `assert_converges(fn, truth, n_runs=..., sigma=3)` — differently
  seeded runs must land within `sigma` standard errors of the truth.
  This is a statistical test, and because the SE is estimated from the
  runs the statistic is Student-t with `n_runs - 1` degrees of
  freedom: at `sigma=3` a correct implementation still fails ~0.55% of
  the time with the default 30 runs (~4% at the minimum of 5). Those
  rates are part of the contract; tightening `sigma` to 2 is how CI
  gets flaky.

Counter-based (Philox) kernels are pure functions of their key —
varying global seeds gives zero variance. Both asserts take
`pass_seed=True`, which passes each run's seed to `fn` so the key can
vary per run.

## Certification: make the checks falsifiable

A green reference test only certifies something if breaking the kernel
turns it red. Two helpers (contributed from a production CFR solver's
certification pyramid, where plausible numbers from an uncertified
kernel voided real conclusions):

- `mutation_screams(fn, threshold=...)` — `fn(broken)` runs the kernel
  intact or with a deliberate bug injected; the deviation between the
  two must EXCEED the threshold. A check that cannot fail certifies
  nothing; this is the red half that proves the green half is alive.
- `assert_within_se(samples, target, k=3)` — the one-sample-set
  primitive under `assert_converges`: the SE is measured from the
  samples, never assumed. A residual below an unmeasured noise floor
  is not a pass.
- `assert_no_reweight_bias(estimator)` — screams on the reach² bug
  (subsampling proportional to the weights and then weighting again):
  invisible at near-uniform weights, explosive at concentrated ones.
  The correct pattern ships as `numba_utils.weighted_mc_mean`
  (uniform-subsample-then-weight, Philox-driven).

Two disciplines the same harness taught: derive the estimator FROM the
estimand where possible (return both sides from one symmetric
normalizer and the zero-sum identity cannot fail without a NaN — an
algebraic consequence beats a test that can pass by accident), and
prefer an independent reference that shares no code path with the
kernel under test — `assert_equivalent` is built for exactly that.
