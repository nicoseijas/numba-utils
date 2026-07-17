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
