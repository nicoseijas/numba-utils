"""Battle-tested parallel building blocks.

Not prange wrappers: COMPLETE parallel operations with the known
pitfalls engineered around (see docs/parallelism.md):

- Every operation falls back to a serial path below a size threshold,
  where the prange launch barrier costs more than the parallelism pays.
- Per-thread private state is padded to cache-line boundaries (no false
  sharing) and merged serially at the end (no atomics, no races).
- Kernels are co-located with their prange drivers in this package.

Parallel float reductions reorder operations: results can differ from
the serial version in the last bits. Where that matters, compare against
the serial counterpart (``numba_utils.testing.assert_equivalent``).
"""

from numba_utils.parallel._chunked import chunked_reduce
from numba_utils.parallel._histogram import parallel_histogram
from numba_utils.parallel._reduce import parallel_reduce, parallel_sum
from numba_utils.parallel._scan import parallel_prefix_sum
from numba_utils.parallel._topk import parallel_topk

SERIAL_THRESHOLD = 1 << 16

__all__ = [
    "SERIAL_THRESHOLD",
    "chunked_reduce",
    "parallel_histogram",
    "parallel_prefix_sum",
    "parallel_reduce",
    "parallel_sum",
    "parallel_topk",
]
