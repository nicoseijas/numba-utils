# Design: graph

## Why CSR arrays, not a Graph class

A `Graph` jitclass would add construction ceremony and force one
blessed representation. CSR `(indptr, indices)` is already the lingua
franca (`scipy.sparse.csr_matrix` uses the same layout), costs two
plain arrays, and passes across the `@njit` boundary for free. Every
algorithm takes the arrays directly; `edges_to_csr` builds them from
the edge lists users actually start with, and returns an `order` array
so any per-edge payload (weights, labels) can be aligned with one
fancy-index — no zipped structs needed in nopython.

## Why traversal bounds-checks `indices`

Kernels compile without bounds checking, so a malformed CSR (an
`indices` entry outside `[0, n)`) would write through a wild pointer —
exactly the silent-corruption class the 0.1.2 audit fixed in
`histogram`. The per-edge check is one predictable branch; the
algorithms raise `ValueError` instead of corrupting memory. `indptr`
monotonicity is trusted (checking it per node would duplicate work the
construction already guarantees).

## Why Dijkstra carries its own heap

`collections.PriorityQueue` stores bare priorities — no payload — and
Dijkstra needs `(distance, node)` pairs. Rather than a pair-encoding
hack (packing node ids into float mantissas loses exactness), the
kernel keeps two parallel arrays as a binary heap with **lazy
deletion**: each successful relaxation pushes, stale entries are
skipped on pop via the `done` mask. Each edge relaxes at most once, so
the heap is allocated once at `m + 1` — zero allocation inside the
loop, no decrease-key machinery.

NaN weights are rejected up front — the same lesson as
`weighted_sampling`: NaN passes a plain `w < 0` check and would
silently corrupt distances. `+inf` is legal (the edge is effectively
absent).

## Why iterative DFS with a preorder-preserving stack

Recursion in nopython is a native-stack crash risk, not a catchable
exception. The explicit stack pushes neighbors in reverse CSR order so
the produced preorder is identical to the recursive formulation —
reference implementations in tests stay trivially comparable.

## Why Kahn (not DFS) for topological_sort

Kahn's algorithm gives cycle detection for free (processed count < n)
and a deterministic, explainable order: ready nodes leave in ascending
index. DFS-based toposort would need the same explicit-stack care for
a less useful ordering guarantee.

## Why `union` returns a bool

`UnionFind.union` returning "did a merge happen" makes the common
Kruskal/cycle-detection loop a one-liner and costs nothing. Union by
size (not rank) keeps a real `component_size` query for free.
