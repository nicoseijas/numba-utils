"""Single-source shortest paths (Dijkstra) over weighted CSR adjacency."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def _heap_push(heap_d, heap_v, size, d, v):
    heap_d[size] = d
    heap_v[size] = v
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        if heap_d[i] < heap_d[parent]:
            heap_d[i], heap_d[parent] = heap_d[parent], heap_d[i]
            heap_v[i], heap_v[parent] = heap_v[parent], heap_v[i]
            i = parent
        else:
            break
    return size + 1


@cached_njit
def _heap_pop(heap_d, heap_v, size):
    top_v = heap_v[0]
    size -= 1
    heap_d[0] = heap_d[size]
    heap_v[0] = heap_v[size]
    i = 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and heap_d[child + 1] < heap_d[child]:
            child += 1
        if heap_d[child] < heap_d[i]:
            heap_d[i], heap_d[child] = heap_d[child], heap_d[i]
            heap_v[i], heap_v[child] = heap_v[child], heap_v[i]
            i = child
        else:
            break
    return top_v, size


@cached_njit
def dijkstra(indptr, indices, weights, source):
    """Shortest-path distances from ``source`` over nonnegative weights.

    ``weights`` is aligned with CSR positions — build it with the
    ``order`` array from :func:`edges_to_csr`
    (``weights_csr = edge_weights[order]``). Returns a float64 array;
    unreachable nodes get ``inf``. ``+inf`` weights are allowed (the
    edge is effectively absent); NaN or negative weights raise
    ``ValueError`` up front — NaN would otherwise pass a plain ``w < 0``
    check and silently corrupt the distances.

    Binary heap with lazy deletion (no decrease-key): each successful
    relaxation pushes, stale entries are skipped on pop, so the heap is
    sized m + 1 once — zero allocation in the loop.

    Complexity: O((n + m) log n). Memory: O(n + m).
    """
    n = indptr.shape[0] - 1
    m = indices.shape[0]
    if weights.shape[0] != m:
        raise ValueError("dijkstra: weights length must match indices")
    if source < 0 or source >= n:
        raise ValueError("dijkstra: source out of range")
    for p in range(m):
        w = weights[p]
        if np.isnan(w):
            raise ValueError("dijkstra: NaN weight")
        if w < 0:
            raise ValueError("dijkstra: negative weight")
    dist = np.full(n, np.inf)
    done = np.zeros(n, np.bool_)
    heap_d = np.empty(m + 1, np.float64)
    heap_v = np.empty(m + 1, np.int64)
    dist[source] = 0.0
    size = _heap_push(heap_d, heap_v, 0, 0.0, source)
    while size > 0:
        u, size = _heap_pop(heap_d, heap_v, size)
        if done[u]:
            continue
        done[u] = True
        du = dist[u]
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if v < 0 or v >= n:
                raise ValueError(
                    "dijkstra: indices contains an out-of-range node"
                )
            nd = du + weights[p]
            if nd < dist[v]:
                dist[v] = nd
                size = _heap_push(heap_d, heap_v, size, nd, v)
    return dist
