# Benchmarking JIT-compiled code

Numba makes it easy to measure the wrong thing. These are the rules the
project's own benchmarks (BENCHMARKS.md) follow.

## Never time the first call

The first call compiles. `numba_utils.profiling` has the tools:

```python
from numba_utils import warmup, compile_time, compare, benchmark

warmup(fn, *args)                 # trigger compilation, get elapsed
compile_time(fn, *args)           # estimate pure compile cost
compare(np_fn, fn, args=..., n=50, warmup_runs=2)   # stats + speedup
```

`compare()` runs uncounted warmup rounds before measuring, so JIT never
pollutes the numbers.

## Micro-benchmarks over-report

A tight benchmark loop around a jitted function invites the compiler to
hoist loop-invariant work, fold constants, or eliminate results nobody
reads. Real workloads have observed order-of-magnitude gaps between a
micro-benchmark ("millions of evals per second") and in-application
throughput (tens of thousands).

Rules:

- Benchmark the **realistic call pattern** — real input sizes, real
  dtypes, results actually consumed.
- Prefer **end-to-end throughput** of the application loop (items/s over
  a full pass) as the number you trust; treat micro-benchmarks as
  relative indicators only.
- If a number looks too good, assume the compiler deleted your work
  until proven otherwise.

## Reproducibility

A benchmark that can't be replicated is an anecdote. Pin, and publish
with the results:

- the RNG **seed**,
- **input sizes** and dtypes,
- **iteration counts** (and warmup rounds),
- **software versions** (Python, numba, numpy) and the hardware when it
  matters.

`benchmarks/bench_arrays_algorithms.py` is the template: fixed seed,
fixed sizes, emits a Markdown table with the environment header. Rerun
it with `python benchmarks/bench_arrays_algorithms.py > BENCHMARKS.md`.

## Report losses too

A benchmark suite that only contains wins is marketing. Keep the cases
where NumPy is faster (see `radix_sort` full-range in BENCHMARKS.md) —
they tell users when NOT to use a function, which saves them more time
than the wins do.

## Variance

Mean alone hides instability. `compare()` reports mean, median, variance
and min/max per side; if variance is large relative to the mean, the
measurement is noise-dominated — increase `n`, isolate the machine, or
pin sizes up.

## Timer overhead and drift

Two details `benchmark()`/`compare()` handle that hand-rolled loops
usually don't:

- **Timer overhead.** Two `perf_counter` reads per call cost
  ~100-200 ns — on a ns-scale kernel that inflates the measured MEAN
  by 10-40% depending on the machine (the median resists; the mean
  doesn't). Fast functions are therefore timed in auto-sized batches
  (`TimingStats.inner` calls per sample, sized so one sample lasts
  ~100 µs); functions at or above ~100 µs per call keep per-call
  timing. Force either mode with `inner=`.
- **Drift.** `compare()` interleaves the two sides — each round times
  one sample of each, alternating who goes first — so thermal drift,
  frequency scaling and cache state land on both, not on whichever
  ran second.
