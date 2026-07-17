# Benchmark honesty

This is the policy that separates a serious technical library from a
marketing collection. It is enforced in
[GUIDELINES.md](https://github.com/nicoseijas/numba-utils/blob/main/GUIDELINES.md)
and lived out in
[BENCHMARKS.md](https://github.com/nicoseijas/numba-utils/blob/main/BENCHMARKS.md).

## Every function answers one of three questions

1. **Faster** than the standard alternative — shown with numbers.
2. **Similar performance, better ergonomics** — stated, with the reason
   the ergonomics matter (usually: it runs inside `@njit` where the
   standard tool cannot reach).
3. **Slower, but solves a problem unavailable elsewhere** — the loss AND
   the unique capability, both in writing.

## Three prohibitions

- **Never hide unfavorable results.** Losing rows stay in the table.
  `radix_sort` on full-range keys and `counter` versus `np.unique` are
  published as losses on purpose — they tell you when *not* to reach for
  the function, which is worth more than a win.
- **Never benchmark unrealistic scenarios.** The measured call pattern is
  the one a real user would run. See [Benchmarking](benchmarking.md) for
  what "realistic" means here (warmup, sizes, buffer reuse).
- **Never exclude competitive implementations.** The baseline is the best
  standard tool for the job, not a strawman. When NumPy's SIMD sort or a
  bandwidth-bound sweep wins, that is what the table shows.

## The testing counterpart

Honesty about speed has a mirror in honesty about correctness. This
library ships **no line-coverage badge** — coverage.py cannot see inside
compiled kernels, so the number it reports is a misleading artifact. The
real guarantee is validation against independent reference
implementations. The full reasoning is in [Testing](testing.md).

## Why this is a differentiator

Anyone can publish the runs where they win. Publishing the runs where
NumPy wins — next to the wins, in the same table — is a costly signal
that the numbers can be trusted. That trust is the product.
