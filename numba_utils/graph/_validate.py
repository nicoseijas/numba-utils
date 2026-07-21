"""Shared CSR structure validation for the graph algorithms."""

from __future__ import annotations

from numba_utils.decorators import cached_njit


@cached_njit
def check_csr(indptr, indices):
    """Validate CSR structure; returns the node count.

    Enforces ``indptr[0] == 0``, ``indptr[-1] == len(indices)`` and
    monotonic non-decreasing ``indptr`` — together these bound every
    neighbor position inside ``indices``, so a malformed ``indptr``
    raises ``ValueError`` instead of driving out-of-bounds reads and
    writes (nopython has no bounds checking; a wild ``indptr`` entry
    is a segfault, not an exception). O(n), free next to the O(n + m)
    algorithms that call it.
    """
    if indptr.shape[0] < 1:
        raise ValueError("graph: indptr must have at least one entry")
    n = indptr.shape[0] - 1
    m = indices.shape[0]
    if indptr[0] != 0 or indptr[n] != m:
        raise ValueError(
            "graph: malformed indptr (must start at 0 and end at "
            "len(indices))"
        )
    prev = indptr[0]
    for i in range(1, n + 1):
        v = indptr[i]
        if v < prev:
            raise ValueError("graph: indptr must be non-decreasing")
        prev = v
    return n
