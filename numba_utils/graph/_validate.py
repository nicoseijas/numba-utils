"""Shared CSR structure validation for the graph algorithms."""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types
from numba.core.errors import TypingError
from numba.extending import overload

from numba_utils.decorators import cached_njit


def _require_integer_csr(indptr, indices):
    # Interpreted fallback (the callers are always jitted).
    if not (
        np.issubdtype(indptr.dtype, np.integer)
        and np.issubdtype(indices.dtype, np.integer)
    ):
        raise TypeError("graph: indptr and indices must be integer arrays")


@overload(_require_integer_csr)
def _ol_require_integer_csr(indptr, indices):
    # Compile-time dtype gate: a float indptr admits NaN, which fails
    # every comparison and slides through naive monotonicity checks —
    # and np.zeros(n + 1) is float64 by default, so arriving here by
    # accident is easy. Reject the whole dtype class at typing time.
    if not isinstance(indptr.dtype, nb_types.Integer) or not isinstance(
        indices.dtype, nb_types.Integer
    ):
        raise TypingError(
            "graph: indptr and indices must be integer arrays "
            "(np.zeros default is float64 — build indptr with an "
            "integer dtype)"
        )

    def impl(indptr, indices):
        return None

    return impl


@cached_njit
def check_csr(indptr, indices):
    """Validate CSR structure; returns the node count.

    Enforces integer dtypes (compile time), ``indptr[0] == 0``,
    ``indptr[-1] == len(indices)`` and monotonic non-decreasing
    ``indptr`` — together these bound every neighbor position inside
    ``indices``, so a malformed ``indptr`` raises ``ValueError``
    instead of driving out-of-bounds reads and writes (nopython has no
    bounds checking; a wild ``indptr`` entry is a segfault, not an
    exception). O(n), free next to the O(n + m) algorithms that call
    it.
    """
    _require_integer_csr(indptr, indices)
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
        # Inverted test (the histogram lesson, applied laterally): a
        # NaN-like value that fails every comparison must FAIL this
        # check, not pass it. With the integer gate above NaN cannot
        # occur, but the check must not depend on that.
        if not (v >= prev):
            raise ValueError("graph: indptr must be non-decreasing")
        prev = v
    return n
