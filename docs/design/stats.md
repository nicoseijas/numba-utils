# Design: stats

## Why only three functions

The original roadmap sketched a `math/` grab-bag (`clamp`, `lerp`,
`mean`, `std`, ...). The identity filter killed it: NumPy already does
those well, and a wrapper adds surface without value. What survived is
the set where the *naive implementation is wrong*, not just slow:

- `logsumexp` / `softmax` — the direct formulas overflow `exp` for any
  input beyond ~709. The max-shift identity is the whole point; both
  benchmarks list the naive baseline's speed AND note it is broken.
- `weighted_quantile` — weighted quantiles have several incompatible
  conventions, and hand-rolled versions routinely mix them up or let
  NaN corrupt the cumulative sums silently.

## Why the inverted CDF convention

Of the many weighted-quantile definitions, inverted CDF is the one
NumPy itself supports with weights
(`np.quantile(..., weights=..., method="inverted_cdf")`), which gives
this implementation an exact, testable reference — every test asserts
equality, not closeness. It returns an element of the input (no
interpolation), matching NumPy edge for edge, including `q=0` on a
zero-weight minimum.

An unstable argsort is enough: among tied values the crossing element
has the same value regardless of their relative order.

## Why validation is strict

Same lesson as `weighted_sampling` (0.1.1) and `histogram` (0.1.2):
NaN passes a plain `w < 0` check and corrupts silently downstream.
`weighted_quantile` rejects NaN values, NaN/negative/non-finite
weights and an all-zero total up front. `logsumexp`/`softmax` instead
follow the mathematical expression (NaN propagates, ±inf documented) —
they are transforms, not estimators, and garbage-in/garbage-out is the
expected contract there.
