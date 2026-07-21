"""Benchmarks for the arrays/ and algorithms/ modules vs their NumPy
equivalents. Emits a Markdown table on stdout:

    python benchmarks/bench_arrays_algorithms.py > BENCHMARKS.md

Reproducibility: seed, sizes and iteration counts are fixed below.
"""

from __future__ import annotations

import platform
import sys

import numba
import numpy as np

from numba_utils import (
    compare,
    counting_sort,
    cumulative_sum,
    fast_clip,
    histogram,
    lexsort,
    radix_sort,
    rolling_mean,
    stable_argsort,
    topk,
    unique_sorted,
)

SEED = 42
N_FLOAT = 2_000_000
N_INT = 5_000_000
RUNS = 11
WARMUP = 2
K = 100
WINDOW = 50
BINS = 64


def np_rolling_mean(arr, window):
    c = np.cumsum(arr)
    result = np.empty(arr.shape[0] - window + 1)
    result[0] = c[window - 1]
    result[1:] = c[window:] - c[:-window]
    return result / window


def np_topk(arr, k):
    n = arr.shape[0]
    return np.sort(np.partition(arr, n - k)[n - k :])[::-1]


def np_histogram_counts(arr, bins, lo, hi):
    return np.histogram(arr, bins=bins, range=(lo, hi))[0]


def np_stable_argsort(arr):
    return np.argsort(arr, kind="stable")


def main() -> None:
    rng = np.random.default_rng(SEED)
    floats = rng.normal(0.0, 1.0, N_FLOAT)
    ints = rng.integers(-(2**40), 2**40, N_INT)
    ints_narrow = rng.integers(0, 2**24, N_INT)
    ints_tiny_range = rng.integers(0, 1000, N_INT)
    sorted_ints = np.sort(rng.integers(0, 10_000, N_INT))
    lex_keys = rng.integers(0, 10, (3, 1_000_000))

    cases = [
        (
            f"fast_clip ({N_FLOAT:,} f64)",
            np.clip, fast_clip, (floats, -1.0, 1.0),
        ),
        (
            f"cumulative_sum ({N_FLOAT:,} f64)",
            np.cumsum, cumulative_sum, (floats,),
        ),
        (
            f"rolling_mean w={WINDOW} ({N_FLOAT:,} f64)",
            np_rolling_mean, rolling_mean, (floats, WINDOW),
        ),
        (
            f"topk k={K} ({N_FLOAT:,} f64)",
            np_topk, topk, (floats, K),
        ),
        (
            f"histogram {BINS} bins ({N_FLOAT:,} f64)",
            np_histogram_counts, histogram, (floats, BINS, -4.0, 4.0),
        ),
        (
            f"radix_sort full-range ({N_INT:,} i64)",
            np.sort, radix_sort, (ints,),
        ),
        (
            f"radix_sort range<2^24 ({N_INT:,} i64)",
            np.sort, radix_sort, (ints_narrow,),
        ),
        (
            f"counting_sort range<1000 ({N_INT:,} i64)",
            np.sort, counting_sort, (ints_tiny_range,),
        ),
        (
            f"unique_sorted ({N_INT:,} i64, sorted)",
            np.unique, unique_sorted, (sorted_ints,),
        ),
        (
            f"stable_argsort ({N_INT:,} i64, many ties)",
            np_stable_argsort, stable_argsort, (ints_tiny_range,),
        ),
        (
            "lexsort 3 keys (1,000,000 i64)",
            np.lexsort, lexsort, (lex_keys,),
        ),
    ]

    print(f"# Benchmarks\n")
    print(
        f"Python {sys.version.split()[0]}, numba {numba.__version__}, "
        f"numpy {np.__version__}, {platform.processor() or platform.machine()}.\n"
    )
    print(f"Seed {SEED}, {RUNS} runs after {WARMUP} warmup runs, mean times.\n")
    print("| case | NumPy | numba-utils | speedup |")
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
