"""Benchmarks for the parallel/ module vs the best serial alternative.

Emits a Markdown section on stdout (appended to BENCHMARKS.md by the
regeneration recipe in bench_random_collections.py's docstring, plus
this script). Parallel gains depend on core count and memory bandwidth;
the honest baseline is the best serial tool, not a strawman.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads

from numba_utils import (
    chunked_reduce,
    compare,
    cumulative_sum,
    histogram,
    parallel_histogram,
    parallel_prefix_sum,
    parallel_sum,
    parallel_topk,
    philox_uniform,
    topk,
)

N_CHUNKS = 256


@chunked_reduce
def _mc_philox_sum(chunk_id, start, end):
    acc = 0.0
    for i in range(start, end):
        acc += philox_uniform(42, i)
    return acc

SEED = 42
N = 20_000_000
RUNS = 11
WARMUP = 2
K = 100
BINS = 64


def main() -> None:
    rng = np.random.default_rng(SEED)
    floats = rng.normal(0.0, 1.0, N)

    def serial_hist(arr):
        return histogram(arr, BINS, -4.0, 4.0)

    def par_hist(arr):
        return parallel_histogram(arr, BINS, -4.0, 4.0)

    cases = [
        (f"parallel_sum ({N:,} f64) vs np.sum", np.sum, parallel_sum, (floats,)),
        (
            f"parallel_histogram {BINS} bins ({N:,} f64) vs serial histogram",
            serial_hist, par_hist, (floats,),
        ),
        (
            f"parallel_prefix_sum ({N:,} f64) vs serial cumulative_sum",
            cumulative_sum, parallel_prefix_sum, (floats,),
        ),
        (
            f"parallel_topk k={K} ({N:,} f64) vs serial topk",
            topk, parallel_topk, (floats, K),
        ),
        (
            f"chunked_reduce MC ({N:,} philox draws, {N_CHUNKS} chunks) "
            "vs its serial driver",
            _mc_philox_sum.serial, _mc_philox_sum.parallel, (N, N_CHUNKS),
        ),
    ]

    print(f"\n## Parallel ({get_num_threads()} threads)\n")
    print("| case | serial baseline | parallel | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, base_fn, par_fn, args in cases:
        result = compare(base_fn, par_fn, args=args, n=RUNS, warmup_runs=WARMUP)
        print(
            f"| {name} | {result.first.mean * 1e3:.2f} ms "
            f"| {result.second.mean * 1e3:.2f} ms "
            f"| {result.speedup:.2f}x |"
        )
    print(
        "\nParallel float reductions reorder operations; results can "
        "differ from serial in the last bits (parallel_histogram is "
        "bit-exact). Gains depend on core count and memory bandwidth."
    )


if __name__ == "__main__":
    main()
