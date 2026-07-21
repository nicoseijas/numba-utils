"""Benchmarks for the graph/ module.

Emits a Markdown section on stdout; BENCHMARKS.md is regenerated with:

    python benchmarks/bench_arrays_algorithms.py >  BENCHMARKS.md
    python benchmarks/bench_random_collections.py >> BENCHMARKS.md
    python benchmarks/bench_parallel.py >> BENCHMARKS.md
    python benchmarks/bench_graph.py >> BENCHMARKS.md

Baselines are the idiomatic pure-Python equivalents (deque BFS, heapq
Dijkstra, a plain DSU class) over prebuilt adjacency lists — the
realistic representation on each side. See docs/benchmarking.md.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
from numba import njit

from numba_utils import UnionFind, bfs, compare, dijkstra, edges_to_csr

SEED = 42
RUNS = 11
WARMUP = 2
N_NODES = 50_000
N_EDGES = 200_000


def py_bfs(adj, n, source):
    dist = [-1] * n
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def py_dijkstra(adj_w, n, source):
    dist = [float("inf")] * n
    dist[source] = 0.0
    done = [False] * n
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in adj_w[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


class PyDSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True


def py_union_churn(src, dst, n):
    dsu = PyDSU(n)
    for a, b in zip(src.tolist(), dst.tolist()):
        dsu.union(a, b)
    return dsu.components


@njit
def nu_union_churn(src, dst, n):
    uf = UnionFind(n)
    for e in range(src.shape[0]):
        uf.union(src[e], dst[e])
    return uf.n_components()


def main() -> None:
    rng = np.random.default_rng(SEED)
    src = rng.integers(0, N_NODES, N_EDGES)
    dst = rng.integers(0, N_NODES, N_EDGES)
    weights = rng.uniform(0.0, 10.0, N_EDGES)

    indptr, indices, order = edges_to_csr(N_NODES, src, dst)
    w_csr = weights[order]
    adj = [[] for _ in range(N_NODES)]
    adj_w = [[] for _ in range(N_NODES)]
    for u, v, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
        adj[u].append(v)
        adj_w[u].append((v, w))

    def py_bfs_case(_indptr, _indices, source):
        return py_bfs(adj, N_NODES, source)

    def py_dijkstra_case(_indptr, _indices, _w, source):
        return py_dijkstra(adj_w, N_NODES, source)

    cases = [
        (
            f"bfs ({N_NODES:,} nodes, {N_EDGES:,} edges) (vs Python deque)",
            py_bfs_case, bfs, (indptr, indices, 0),
        ),
        (
            f"dijkstra ({N_NODES:,} nodes, {N_EDGES:,} edges) (vs heapq)",
            py_dijkstra_case, dijkstra, (indptr, indices, w_csr, 0),
        ),
        (
            f"UnionFind churn ({N_EDGES:,} unions) (vs Python DSU)",
            py_union_churn, nu_union_churn, (src, dst, N_NODES),
        ),
    ]

    print("\n## Graph\n")
    print("| case | baseline | numba-utils | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, py_fn, nu_fn, args in cases:
        result = compare(py_fn, nu_fn, args=args, n=RUNS, warmup_runs=WARMUP)
        print(
            f"| {name} | {result.first.mean * 1e3:.2f} ms "
            f"| {result.second.mean * 1e3:.2f} ms "
            f"| {result.speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
