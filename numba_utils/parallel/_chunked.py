"""Chunked reduction with a serial/parallel bit-exactness guarantee."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import prange

from numba_utils.decorators import cached_njit, njit_parallel


def chunked_reduce(
    func: Callable[..., Any] | None = None, /
) -> Callable[..., Any]:
    """Decorator: one per-chunk kernel, serial AND parallel drivers,
    **bit-identical results across both**.

    ::

        @chunked_reduce
        def simulate(chunk_id, start, end):
            acc = 0.0
            for i in range(start, end):
                acc += philox_uniform(SEED, i)   # counter-seeded MC
            return acc

        total = simulate(10_000_000, n_chunks=64)                 # parallel
        same  = simulate(10_000_000, n_chunks=64, parallel=False) # bit-equal

    The guarantee rests on two rules the driver enforces:

    - **Chunk boundaries depend only on ``(n_items, n_chunks)``** —
      never on the thread count. The same chunks exist on a laptop and
      a 128-core box.
    - **Partials merge serially, in chunk order.** Threads write only
      their own slot; the reduction order is fixed.

    So ``parallel=True`` and ``parallel=False`` return the SAME float,
    and an A/B harness never needs to re-validate when switching
    modes. Changing ``n_chunks`` legitimately changes rounding (a
    different, documented partition), so pin it per experiment. Pair
    the ``chunk_id``/index range with counter-based RNG
    (:func:`numba_utils.philox_uniform`) and results are also
    independent of how chunks are scheduled.

    The kernel is jitted with the library defaults; the drivers are
    compiled per process (the closure over the kernel is not
    disk-cacheable). The returned callable is a plain Python function;
    the jitted drivers are exposed as ``.serial`` / ``.parallel``
    attributes for calls from other jitted code.

    Complexity: O(n_items) work + O(n_chunks) merge.
    Memory: O(n_chunks).
    """

    def build(kernel_func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(kernel_func):
            raise TypeError("chunked_reduce expects a callable to decorate")
        kernel = cached_njit(kernel_func)

        @cached_njit
        def run_serial(n_items, n_chunks):
            chunk = (n_items + n_chunks - 1) // n_chunks
            partials = np.zeros(n_chunks, np.float64)
            for c in range(n_chunks):
                start = min(c * chunk, n_items)
                end = min(start + chunk, n_items)
                if start < end:
                    partials[c] = kernel(c, start, end)
            total = 0.0
            for c in range(n_chunks):
                total += partials[c]
            return total

        @njit_parallel(cache=False)
        def run_parallel(n_items, n_chunks):
            chunk = (n_items + n_chunks - 1) // n_chunks
            partials = np.zeros(n_chunks, np.float64)
            for c in prange(n_chunks):
                start = min(c * chunk, n_items)
                end = min(start + chunk, n_items)
                if start < end:
                    partials[c] = kernel(c, start, end)
            total = 0.0
            for c in range(n_chunks):
                total += partials[c]
            return total

        def driver(n_items, n_chunks, *, parallel=True):
            if n_items < 0:
                raise ValueError("chunked_reduce: n_items must be >= 0")
            if n_chunks < 1:
                raise ValueError("chunked_reduce: n_chunks must be >= 1")
            if parallel:
                return run_parallel(n_items, n_chunks)
            return run_serial(n_items, n_chunks)

        driver.serial = run_serial
        driver.parallel = run_parallel
        driver.__name__ = getattr(kernel_func, "__name__", "chunked_reduce")
        return driver

    if func is None:
        return build
    return build(func)
