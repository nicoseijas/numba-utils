# Performance Notes

Practical knowledge about how Numba behaves in production. Most libraries
document their API; the problems users actually hit live here.

- [Numba's on-disk cache](numba-cache.md) — what it buys you, and the
  environments where it crashes.
- [Parallelism](parallelism.md) — when `prange` pays, when it silently
  loses, and how to structure multi-process work.
- [Benchmarking](benchmarking.md) — measuring JIT-compiled code without
  fooling yourself.

## When Numba actually helps

Numba shines on **explicit loops over typed numeric data**: element-wise
kernels NumPy would materialize intermediates for, algorithms with
data-dependent control flow (searches, selection, sorting), and code
called from other jitted code without boxing.

NumPy stays competitive — sometimes unbeatable — when the operation is a
single memory-bandwidth-bound sweep (`np.clip`, `np.sum`) or backed by
SIMD-optimized kernels (`np.sort` on integers). See BENCHMARKS.md for
honest numbers on both kinds; `fast_clip` at 1.01x is there on purpose.

## Compilation cost and how to amortize it

Every function pays a one-time JIT cost per process per signature
(milliseconds to minutes for large kernels). Strategies:

- **Long-lived processes.** A worker farm that spawns a process per task
  re-pays the full JIT every time; one long-lived process pays it once.
- **`warmup()`** before timing or serving traffic; `compile_time()`
  tells you what the first call will cost.
- **The on-disk cache** (`cached_njit`) eliminates recompiles across
  runs — read [numba-cache.md](numba-cache.md) before relying on it in
  multi-process setups.

## Keep signatures stable

Each new argument dtype combination triggers a full recompile. Mixing
`float32`/`float64` arrays, or passing Python ints where the kernel
usually sees floats, silently multiplies compile time and code size.
`diagnostics.check()` flags functions with signature churn.

## Common mistakes

- **Falling back to object mode** (or forcing `forceobj`): you get
  Python speed with extra overhead. numba-utils decorators are all
  nopython.
- **Python lists in hot code**: use NumPy arrays or `numba.typed.List`.
- **Allocating inside hot loops**: pass an `out=` buffer (every
  numba-utils array function that returns an array accepts one).
- **Trusting `fastmath=True` for reproducible results**: it relaxes
  IEEE 754. Fine for throughput, wrong for bit-exact comparisons.
- **Assuming out-of-bounds access will crash**: nopython mode does not
  bounds-check; a sizing bug corrupts memory silently. Validate kernel
  output against an independent reference (all numba-utils tests compare
  against NumPy), and use `@boundscheck` with `NUMBA_UTILS_DEV=1` during
  development.
