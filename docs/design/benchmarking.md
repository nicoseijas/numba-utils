# Design: benchmarking tooling

(For *how to benchmark correctly*, see [../benchmarking.md](../benchmarking.md).
This page records why the tooling and policy are designed the way they are.)

## Why function-mode `benchmark()` warms up by default

The single most common error in Numba benchmarks found in the wild is
timing the first call — which includes compilation — or timing from
Python without warmup. The correct behavior must be the DEFAULT, not an
option the user has to know to enable: `benchmark(fn, args=...)` runs
uncounted warmup calls first, and including compilation requires the
explicit `warmup_runs=0`. Defaults are the API's opinion; this one is
the library's whole thesis.

## Why `benchmark` is one hybrid function

`with benchmark():` (block timing) and `benchmark(fn)` (function stats)
could have been two names. One name won because they are one concept —
"measure this" — and the argument type (callable vs label) makes the
mode unambiguous. Two names would double the API for zero expressive
gain, against the Minimal APIs principle.

## Why baselines are the best standard tool, not a strawman

`topk` is measured against `np.partition`-based selection, not a full
`np.sort`; containers are measured as jitted workload vs the idiomatic
Python equivalent (heapq, set) — not as boxed per-call invocations that
would flatter us. A benchmark answers "when does this library add
value?", never "how do we win a chart?". When the honest baseline wins
(radix_sort full-range vs NumPy's SIMD sort, counter vs np.unique), the
row stays.

## Why CI runs benchmarks but publishes no numbers

Shared CI runners have noisy, variable performance; publishing their
numbers would launder noise into claims. CI's job is invariants —
suites compile, run, and exit cleanly on every commit. Publishable
numbers come from a controlled, documented environment: BENCHMARKS.md
is regenerated locally with fixed seeds, fixed sizes, and the machine
stated in its header.

## Why results are frozen dataclasses

`TimingStats`, `ComparisonResult`, `BenchmarkResult` and
`FunctionReport` are immutable. Measurement results are facts; letting
code mutate a stats object after the fact is a category error, and
frozen dataclasses make it a `AttributeError` instead of a silent one.

## Why samples are batched and `compare()` interleaves

Two `perf_counter` reads per call cost ~100-200 ns. For ms-scale
workloads that is noise; for ns-scale kernels it inflated the measured
MEAN by 10-40% machine-dependent — in the audit's repro the "overhead"
exceeded the kernel (164.5 vs 72.7 ns/call). The median resists the
inflation; the mean does not, and the mean is what feeds `speedup`.
So fast functions are timed in auto-sized batches (`TimingStats.inner`
calls per sample, ~100 µs per sample) with the timer outside the
batch. The threshold matters in the other direction too: at or above
~100 µs per call, batching would only hide per-call variance that
per-call timing reports for free, so `inner` stays 1 there.

`compare()` used to measure `first` completely, then `second` —
whichever ran second absorbed the accumulated thermal drift, frequency
scaling and cache-state changes. Rounds now interleave, alternating
which side goes first each round, so slow drift lands symmetrically on
both. Auto-calibration is per side: with a 100x speed gap, one shared
batch size would leave the fast side exposed to exactly the overhead
the batching exists to remove.
