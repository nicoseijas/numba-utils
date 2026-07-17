"""Parallel reductions: sum and the per-index reduce decorator."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import prange

from numba_utils.decorators import cached_njit, njit_parallel

# Below this, the prange launch barrier costs more than the parallelism
# pays (measured ~0.4 ms per launch regardless of thread count).
_SERIAL_THRESHOLD = 1 << 16


@njit_parallel
def parallel_sum(arr):
    """Sum of 1-D ``arr``, accumulated in float64 across threads.

    Serial loop below the size threshold (the launch barrier would
    dominate); prange scalar reduction above it. Parallel reassociation
    means the result can differ from the serial sum in the last bits.

    Honesty note: ``np.sum`` is memory-bandwidth-bound — expect parallel
    gains only when the array doesn't fit cache and cores add bandwidth.
    See BENCHMARKS.md.

    Complexity: O(n). Memory: O(1).
    """
    n = arr.shape[0]
    if n < _SERIAL_THRESHOLD:
        acc = 0.0
        for i in range(n):
            acc += arr[i]
        return acc
    total = 0.0
    for i in prange(n):
        total += arr[i]
    return total


def parallel_reduce(
    func: Callable[..., Any] | None = None, /
) -> Callable[..., Any]:
    """Decorator: turn a per-index kernel into a parallel summing driver.

    ::

        @parallel_reduce
        def score(i):
            return some_math(i)

        total = score(10_000_000)   # sum of score(i) for i in range(n)

    The kernel is jitted and summed over ``prange`` — serial below the
    size threshold. The generated driver captures the kernel in a
    closure, so it is not cached on disk (recompiled per process).

    Complexity: O(n) work, one prange launch. Memory: O(1).
    """

    def build(kernel_func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(kernel_func):
            raise TypeError("parallel_reduce expects a callable to decorate")
        kernel = cached_njit(kernel_func)

        @njit_parallel(cache=False)
        def driver(n):
            if n < 0:
                raise ValueError("parallel_reduce: n must be >= 0")
            if n < _SERIAL_THRESHOLD:
                acc = 0.0
                for i in range(n):
                    acc += kernel(i)
                return acc
            total = 0.0
            for i in prange(n):
                total += kernel(i)
            return total

        return driver

    if func is None:
        return build
    return build(func)
