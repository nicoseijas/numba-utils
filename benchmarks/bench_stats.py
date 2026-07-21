"""Benchmarks for the stats/ module.

Emits a Markdown section on stdout; BENCHMARKS.md is regenerated with:

    python benchmarks/bench_arrays_algorithms.py >  BENCHMARKS.md
    python benchmarks/bench_random_collections.py >> BENCHMARKS.md
    python benchmarks/bench_parallel.py >> BENCHMARKS.md
    python benchmarks/bench_graph.py >> BENCHMARKS.md
    python benchmarks/bench_stats.py >> BENCHMARKS.md

NumPy baselines are the idiomatic expressions (note the naive
logsumexp/softmax baselines are also numerically UNSTABLE — they
overflow beyond ~709; the comparison is about speed, the reason these
functions exist is correctness). See docs/benchmarking.md.
"""

from __future__ import annotations

import numpy as np

from numba_utils import compare, logsumexp, softmax, weighted_quantile

SEED = 42
RUNS = 11
WARMUP = 2
N = 2_000_000
N_QUANTILE = 1_000_000


def np_logsumexp(arr):
    return np.log(np.sum(np.exp(arr)))


def np_softmax(arr):
    e = np.exp(arr - arr.max())
    return e / e.sum()


def np_weighted_quantile(values, weights, q):
    return np.quantile(values, q, weights=weights, method="inverted_cdf")


def main() -> None:
    rng = np.random.default_rng(SEED)
    floats = rng.normal(0.0, 3.0, N)
    values = rng.normal(0.0, 10.0, N_QUANTILE)
    weights = rng.uniform(0.0, 5.0, N_QUANTILE)

    cases = [
        (
            f"logsumexp ({N:,} f64) (vs naive NumPy)",
            np_logsumexp, logsumexp, (floats,),
        ),
        (
            f"softmax ({N:,} f64) (vs max-shifted NumPy)",
            np_softmax, softmax, (floats,),
        ),
        (
            f"weighted_quantile ({N_QUANTILE:,} f64) (vs np.quantile weighted)",
            np_weighted_quantile, weighted_quantile, (values, weights, 0.75),
        ),
    ]

    print("\n## Stats\n")
    print("| case | baseline | numba-utils | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, np_fn, nu_fn, args in cases:
        result = compare(np_fn, nu_fn, args=args, n=RUNS, warmup_runs=WARMUP)
        print(
            f"| {name} | {result.first.mean * 1e3:.2f} ms "
            f"| {result.second.mean * 1e3:.2f} ms "
            f"| {result.speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
