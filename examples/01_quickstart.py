"""Quickstart: write a kernel, benchmark it honestly against NumPy.

Run:  python examples/01_quickstart.py
"""

import numpy as np

from numba_utils import compare, njit_fast, topk


# One fused pass. The NumPy version below allocates two temporary
# arrays (the clip result and its square) and sweeps memory three
# times — this is exactly where a jitted kernel wins.
@njit_fast
def clipped_energy(values: np.ndarray, lo: float, hi: float) -> float:
    total = 0.0
    for i in range(values.shape[0]):
        x = values[i]
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        total += x * x
    return total


def numpy_clipped_energy(values: np.ndarray, lo: float, hi: float) -> float:
    return float(np.sum(np.clip(values, lo, hi) ** 2))


def main() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 2_000_000)

    # compare() runs warmup rounds first: JIT compilation never
    # pollutes the numbers.
    result = compare(
        numpy_clipped_energy, clipped_energy, args=(values, -1.0, 1.0), n=21
    )
    print(result.summary())

    # Selection without a full sort: the 5 largest values, descending.
    print("\ntop 5:", topk(values, 5))


if __name__ == "__main__":
    main()
