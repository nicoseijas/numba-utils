"""Traversals over CSR adjacency: BFS, DFS preorder, topological sort.

All iterative with preallocated array queues/stacks — no recursion
(deep native recursion is a crash, not an exception; see
docs/parallelism.md for the sibling rule).
"""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit
from numba_utils.graph._validate import check_csr


@cached_njit
def bfs(indptr, indices, source):
    """Breadth-first hop distances from ``source``.

    Returns an int64 array: ``result[v]`` is the number of edges on a
    shortest path from ``source`` to ``v``, or ``-1`` if unreachable.

    Complexity: O(n + m). Memory: O(n).
    """
    n = check_csr(indptr, indices)
    if source < 0 or source >= n:
        raise ValueError("bfs: source out of range")
    dist = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int64)
    dist[source] = 0
    queue[0] = source
    head = 0
    tail = 1
    while head < tail:
        u = queue[head]
        head += 1
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if v < 0 or v >= n:
                raise ValueError("bfs: indices contains an out-of-range node")
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue[tail] = v
                tail += 1
    return dist


@cached_njit
def dfs_preorder(indptr, indices, source):
    """Nodes reachable from ``source`` in depth-first preorder.

    Matches the recursive DFS that visits neighbors in CSR order, but
    runs on an explicit stack. Returns an int64 array of the visited
    nodes (length = reachable count).

    Complexity: O(n + m). Memory: O(n + m) for the stack.
    """
    n = check_csr(indptr, indices)
    if source < 0 or source >= n:
        raise ValueError("dfs_preorder: source out of range")
    visited = np.zeros(n, np.bool_)
    # Each edge pushes its head at most once, plus the source.
    stack = np.empty(indices.shape[0] + 1, np.int64)
    out = np.empty(n, np.int64)
    stack[0] = source
    top = 1
    count = 0
    while top > 0:
        top -= 1
        u = stack[top]
        if visited[u]:
            continue
        visited[u] = True
        out[count] = u
        count += 1
        # Push neighbors in reverse so the lowest CSR position pops
        # first — preorder then matches the recursive formulation.
        p = indptr[u + 1] - 1
        while p >= indptr[u]:
            v = indices[p]
            if v < 0 or v >= n:
                raise ValueError(
                    "dfs_preorder: indices contains an out-of-range node"
                )
            if not visited[v]:
                stack[top] = v
                top += 1
            p -= 1
    return out[:count].copy()


@cached_njit
def topological_sort(indptr, indices):
    """Topological order of a DAG (Kahn's algorithm).

    Returns an int64 array of all ``n`` nodes such that every edge
    ``u -> v`` has ``u`` before ``v``. Deterministic: among ready nodes,
    lower indices come first (zero in-degree seeds are scanned in
    ascending order through a FIFO). Raises ``ValueError`` if the graph
    contains a cycle.

    Complexity: O(n + m). Memory: O(n).
    """
    n = check_csr(indptr, indices)
    m = indices.shape[0]
    indegree = np.zeros(n, np.int64)
    for p in range(m):
        v = indices[p]
        if v < 0 or v >= n:
            raise ValueError(
                "topological_sort: indices contains an out-of-range node"
            )
        indegree[v] += 1
    queue = np.empty(n, np.int64)
    tail = 0
    for i in range(n):
        if indegree[i] == 0:
            queue[tail] = i
            tail += 1
    out = np.empty(n, np.int64)
    head = 0
    count = 0
    while head < tail:
        u = queue[head]
        head += 1
        out[count] = u
        count += 1
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            # Same in-loop guard as bfs/dijkstra (defense in depth on
            # top of the upfront passes): an unchecked v here would be
            # an arbitrary write through indegree/queue.
            if v < 0 or v >= n:
                raise ValueError(
                    "topological_sort: indices contains an out-of-range node"
                )
            indegree[v] -= 1
            if indegree[v] == 0:
                queue[tail] = v
                tail += 1
    if count < n:
        raise ValueError("topological_sort: graph contains a cycle")
    return out
