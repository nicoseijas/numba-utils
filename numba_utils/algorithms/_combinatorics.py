"""Combination index tables for evaluator-style kernels."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit

_MAX_TABLE_ROWS = 1 << 27


@cached_njit
def combination_table(n, k):
    """All C(n, k) k-combinations of ``range(n)`` as an int64 table of
    shape ``(C(n, k), k)``, in lexicographic order.

    The point is the shape: loop ``for row in range(table.shape[0])``
    and the count is always right. Hand evaluators that hardcode combo
    counts (``range(10)`` for C(5, 2)) break the day the board has 4
    cards — deriving the length from the table kills that bug class at
    the root. Build once per (n, k) outside the hot loop and pass the
    table into kernels.

    ``k = 0`` gives one empty combination (shape ``(1, 0)``). Raises
    ``ValueError`` if ``k`` is outside ``[0, n]`` or the table would
    exceed 2**27 rows.

    Complexity: O(C(n, k) · k). Memory: O(C(n, k) · k).
    """
    if n < 0:
        raise ValueError("combination_table: n must be >= 0")
    if k < 0 or k > n:
        raise ValueError("combination_table: k must be in [0, n]")
    # C(n, k) via the multiplicative formula over min(k, n - k): the
    # partial products are then C(n, 1..kk), monotonically increasing,
    # so the row cap fires iff the FINAL count is too large (a plain
    # loop over k would false-reject e.g. C(60, 58) at its C(60, 30)
    # peak) and every intermediate stays far below int64 overflow.
    kk = min(k, n - k)
    count = 1
    for i in range(kk):
        count = count * (n - i) // (i + 1)
        if count > _MAX_TABLE_ROWS:
            raise ValueError(
                "combination_table: C(n, k) exceeds 2**27 rows"
            )
    table = np.empty((count, k), np.int64)
    idx = np.arange(k)
    row = 0
    while True:
        for j in range(k):
            table[row, j] = idx[j]
        row += 1
        # advance the odometer: rightmost digit that can still move
        i = k - 1
        while i >= 0 and idx[i] == i + n - k:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1
    return table
