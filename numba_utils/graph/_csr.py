"""Edge-list to CSR conversion."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def edges_to_csr(n_nodes, src, dst):
    """Convert directed edge lists to CSR adjacency:
    ``(indptr, indices, order)``.

    ``src[e] -> dst[e]`` for each edge. ``indptr`` has length
    ``n_nodes + 1``; node ``u``'s neighbors are
    ``indices[indptr[u]:indptr[u + 1]]``, preserving input edge order
    per node (stable). ``order[p]`` is the ORIGINAL edge index stored
    at CSR position ``p`` — use it to align any per-edge payload:
    ``weights_csr = weights[order]``. For an undirected graph, append
    both directions to the edge lists first.

    Raises ``ValueError`` on out-of-range node ids or mismatched
    ``src``/``dst`` lengths.

    Complexity: O(n + m). Memory: O(n + m).
    """
    m = src.shape[0]
    if dst.shape[0] != m:
        raise ValueError("edges_to_csr: src and dst lengths differ")
    if n_nodes < 0:
        raise ValueError("edges_to_csr: n_nodes must be >= 0")
    indptr = np.zeros(n_nodes + 1, np.int64)
    for e in range(m):
        u = src[e]
        v = dst[e]
        if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
            raise ValueError("edges_to_csr: node id out of range")
        indptr[u + 1] += 1
    for i in range(n_nodes):
        indptr[i + 1] += indptr[i]
    indices = np.empty(m, np.int64)
    order = np.empty(m, np.int64)
    fill = indptr[:n_nodes].copy()
    for e in range(m):
        u = src[e]
        pos = fill[u]
        indices[pos] = dst[e]
        order[pos] = e
        fill[u] += 1
    return indptr, indices, order
