"""Benchmarks for the random/ and collections/ modules.

Emits a Markdown section on stdout; BENCHMARKS.md is regenerated with:

    python benchmarks/bench_arrays_algorithms.py >  BENCHMARKS.md
    python benchmarks/bench_random_collections.py >> BENCHMARKS.md

Container benchmarks drive the jitted structures from @njit code and
compare against the idiomatic pure-Python equivalent (heapq, set) —
that is the realistic usage on both sides. See docs/benchmarking.md.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit

from numba_utils import (
    PriorityQueue,
    SparseSet,
    alias_sample,
    alias_setup,
    compare,
    counter,
    seed,
    shuffle,
    weighted_sampling,
)

SEED = 42
RUNS = 11
WARMUP = 2
N_SHUFFLE = 1_000_000
N_WEIGHTS = 10_000
N_DRAWS = 100_000
N_COUNT = 1_000_000
N_HEAP = 50_000
N_CHURN = 200_000


@njit
def _pq_workload(values):
    pq = PriorityQueue(values.shape[0])
    for v in values:
        pq.push(v)
    total = 0.0
    while not pq.is_empty():
        total += pq.pop_min()
    return total


def _heapq_workload(values):
    heap = []
    for v in values:
        heapq.heappush(heap, v)
    total = 0.0
    while heap:
        total += heapq.heappop(heap)
    return total


@njit
def _sparseset_workload(ops, universe):
    sset = SparseSet(universe)
    hits = 0
    for v in ops:
        if sset.contains(v):
            sset.discard(v)
        else:
            sset.add(v)
            hits += 1
    return hits


def _pyset_workload(ops, universe):
    sset = set()
    hits = 0
    for v in ops:
        if v in sset:
            sset.discard(v)
        else:
            sset.add(v)
            hits += 1
    return hits


def main() -> None:
    rng = np.random.default_rng(SEED)
    seed(SEED)
    shuffle_arr = rng.normal(0.0, 1.0, N_SHUFFLE)
    weights = rng.random(N_WEIGHTS)
    prob, alias = alias_setup(weights)
    count_arr = rng.integers(0, 1000, N_COUNT)
    heap_values = rng.normal(0.0, 1.0, N_HEAP)
    churn_ops = rng.integers(0, 1000, N_CHURN)
    churn_ops_list = [int(v) for v in churn_ops]

    def np_weighted(w, size):
        return rng.choice(w.shape[0], size=size, p=w / w.sum())

    def np_counter(arr):
        return np.unique(arr, return_counts=True)

    cases = [
        (
            f"shuffle ({N_SHUFFLE:,} f64)",
            np.random.shuffle, shuffle, (shuffle_arr,),
        ),
        (
            f"weighted_sampling ({N_WEIGHTS:,} w, {N_DRAWS:,} draws)",
            np_weighted, weighted_sampling, (weights, N_DRAWS),
        ),
        (
            f"alias_sample (setup amortized, {N_DRAWS:,} draws)",
            np_weighted,
            lambda w, size: alias_sample(prob, alias, size),
            (weights, N_DRAWS),
        ),
        (
            f"counter ({N_COUNT:,} i64, 1k distinct)",
            np_counter, counter, (count_arr,),
        ),
    ]

    print("\n## Random & collections\n")
    print("| case | baseline | numba-utils | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, base_fn, nu_fn, args in cases:
        result = compare(base_fn, nu_fn, args=args, n=RUNS, warmup_runs=WARMUP)
        print(
            f"| {name} (vs NumPy) | {result.first.mean * 1e3:.2f} ms "
            f"| {result.second.mean * 1e3:.2f} ms "
            f"| {result.speedup:.2f}x |"
        )

    pq = compare(
        _heapq_workload, _pq_workload, args=(heap_values,),
        n=RUNS, warmup_runs=WARMUP,
    )
    print(
        f"| PriorityQueue push+pop ({N_HEAP:,}) (vs heapq) "
        f"| {pq.first.mean * 1e3:.2f} ms | {pq.second.mean * 1e3:.2f} ms "
        f"| {pq.speedup:.2f}x |"
    )

    ss_theirs = compare(
        _pyset_workload, _pyset_workload, args=(churn_ops_list, 1000),
        n=RUNS, warmup_runs=WARMUP,
    )
    ss_ours = compare(
        _sparseset_workload, _sparseset_workload, args=(churn_ops, 1000),
        n=RUNS, warmup_runs=WARMUP,
    )
    speedup = ss_theirs.second.mean / ss_ours.second.mean
    print(
        f"| SparseSet churn ({N_CHURN:,} ops) (vs Python set) "
        f"| {ss_theirs.second.mean * 1e3:.2f} ms "
        f"| {ss_ours.second.mean * 1e3:.2f} ms | {speedup:.2f}x |"
    )
    print(
        "\n`counter` loses to sort-based `np.unique` on one-shot counting "
        "by design: its use case is incremental counting inside jitted "
        "loops, where materializing an array first is the expensive part."
    )


if __name__ == "__main__":
    main()
