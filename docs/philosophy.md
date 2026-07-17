# Philosophy

Not guidelines — identity. Every design decision in numba-utils traces
back to one of these.

## Performance First

No function enters the library without a performance or ergonomics
justification, benchmarked. If NumPy is already optimal for a job, we
say so and don't ship a worse version with our name on it.

## No Hidden Magic

Every decorator is a thin, documented alias over `njit`. Every kernel is
readable source you can step into. Nothing rewrites your code, nothing
guesses your intent, and global configuration overrides are explicit
policy tools — never silent behavior changes.

## Numba Compatible

Everything works with `@njit`, without hacks. Kernels are dispatchers
callable from your own jitted code; containers are jitclasses usable
inside nopython functions. If a helper only works from Python, its
docstring says so.

## Minimal APIs

`topk(arr, 10)` — not a function with twenty keyword parameters. One
obvious way to call each thing; buffer reuse via a single optional
`out=`; policy via one global config, not per-call flags.

## Benchmark Honesty

Every algorithm answers one of three questions, publicly:

1. **Faster** than the standard alternative — shown with numbers.
2. **Similar performance, better ergonomics** — stated, with the reason
   the ergonomics matter.
3. **Slower, but solves a problem unavailable elsewhere** — the loss AND
   the unique capability, both in writing.

Never hide unfavorable results. Never benchmark unrealistic scenarios.
Never exclude competitive implementations to look better.
[BENCHMARKS.md](https://github.com/nicoseijas/numba-utils/blob/main/BENCHMARKS.md)
contains losing rows on purpose: they tell you when NOT to use a
function, which is worth more than the wins.

## Knowledge Is Part of the Library

Production Numba behaves differently than tutorial Numba. The lessons —
cache crashes on multi-process farms, prange barrier costs, silent
out-of-bounds corruption — ship as code (serial fallbacks, padded
per-thread state), as tooling (`diagnostics.check`), and as
documentation ([performance](performance.md), [cache](numba-cache.md),
[parallelism](parallelism.md), [benchmarking](benchmarking.md)).

---

Enforcement lives in
[GUIDELINES.md](https://github.com/nicoseijas/numba-utils/blob/main/GUIDELINES.md)
— these principles are why those rules exist. Individual trade-offs are
recorded in [design/](design/README.md), and the testing counterpart of
Benchmark Honesty — why no line-coverage badge exists — in
[testing.md](testing.md).
