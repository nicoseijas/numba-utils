"""Graph algorithms over CSR adjacency, usable inside ``@njit``.

Graphs are plain arrays, the native currency of nopython code — no
Graph class. A directed graph is ``(indptr, indices)`` in CSR form:
node ``u``'s neighbors are ``indices[indptr[u]:indptr[u + 1]]``
(the same layout as ``scipy.sparse.csr_matrix``). Build it from edge
lists with :func:`edges_to_csr`; for an undirected graph, add each
edge in both directions.

``indices`` entries are bounds-checked as they are visited (a malformed
CSR raises ``ValueError`` instead of corrupting memory); ``indptr`` is
trusted to be monotonically non-decreasing, as ``edges_to_csr``
produces.
"""

from numba_utils.graph._csr import edges_to_csr
from numba_utils.graph._shortest_path import dijkstra
from numba_utils.graph._traversal import bfs, dfs_preorder, topological_sort
from numba_utils.graph._unionfind import UnionFind

__all__ = [
    "UnionFind",
    "bfs",
    "dfs_preorder",
    "dijkstra",
    "edges_to_csr",
    "topological_sort",
]
