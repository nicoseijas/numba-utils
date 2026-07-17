"""Diagnostics and kernel validation: trust, then verify.

Run:  python examples/03_diagnostics_and_testing.py
"""

import numpy as np

from numba_utils import benchmark, compile_stats, diagnostics, njit_fast
from numba_utils.testing import assert_equivalent, random_arrays


@njit_fast
def normalize_l2(arr: np.ndarray) -> np.ndarray:
    total = 0.0
    for x in arr:
        total += x * x
    norm = np.sqrt(total)
    out = np.empty_like(arr)
    if norm == 0.0:
        for i in range(arr.shape[0]):
            out[i] = 0.0
        return out
    for i in range(arr.shape[0]):
        out[i] = arr[i] / norm
    return out


def numpy_normalize_l2(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return np.zeros_like(arr)
    return arr / norm


def main() -> None:
    # 1. Validate the kernel against an independent reference, on
    #    generated cases including the edges that break kernels
    #    (constant, sorted, duplicates, single element).
    #    fastmath reorders float ops, so tolerances are explicit.
    cases = assert_equivalent(
        numpy_normalize_l2,
        normalize_l2,
        random_arrays(n_cases=20, size=10_000, seed=3),
        rtol=1e-9,
        atol=1e-12,
    )
    print(f"equivalent on {cases} generated cases\n")

    # 2. Benchmark correctly: warmup runs are the default, so JIT
    #    compilation is excluded without having to know about it.
    arr = np.random.default_rng(0).normal(0.0, 1.0, 1_000_000)
    benchmark(normalize_l2, args=(arr,), n=21)
    print()

    # 3. What did Numba actually build, and what should I know about it?
    print(diagnostics.show(normalize_l2, verbose=False))
    print()
    diagnostics.check(normalize_l2)
    print("\ncompile time:", f"{sum(compile_stats(normalize_l2).compile_times_s):.2f}s")


if __name__ == "__main__":
    main()
