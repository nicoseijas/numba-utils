import heapq
from collections import deque

import numpy as np
import pytest
from numba import njit

from numba_utils.graph import (
    UnionFind,
    bfs,
    dfs_preorder,
    dijkstra,
    edges_to_csr,
    topological_sort,
)

RNG = np.random.default_rng(11)


def random_graph(n, m, *, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    return src, dst


def adjacency(n, src, dst):
    adj = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    return adj


class TestEdgesToCsr:
    def test_reconstructs_adjacency(self):
        n, m = 50, 400
        src, dst = random_graph(n, m)
        indptr, indices, order = edges_to_csr(n, src, dst)
        adj = adjacency(n, src, dst)
        assert indptr[0] == 0 and indptr[n] == m
        for u in range(n):
            assert list(indices[indptr[u] : indptr[u + 1]]) == adj[u]

    def test_order_aligns_edge_payloads(self):
        n = 4
        src = np.array([2, 0, 2, 1], np.int64)
        dst = np.array([3, 1, 0, 2], np.int64)
        weights = np.array([10.0, 20.0, 30.0, 40.0])
        indptr, indices, order = edges_to_csr(n, src, dst)
        w_csr = weights[order]
        for p in range(4):
            e = order[p]
            assert w_csr[p] == weights[e]
            assert indices[p] == dst[e]

    def test_empty_graph(self):
        indptr, indices, order = edges_to_csr(
            3, np.empty(0, np.int64), np.empty(0, np.int64)
        )
        assert list(indptr) == [0, 0, 0, 0]
        assert indices.shape == (0,)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            edges_to_csr(2, np.array([0]), np.array([2]))
        with pytest.raises(ValueError):
            edges_to_csr(2, np.array([-1]), np.array([0]))
        with pytest.raises(ValueError):
            edges_to_csr(2, np.array([0, 1]), np.array([0]))


class TestBfs:
    def test_matches_reference(self):
        n, m = 60, 300
        src, dst = random_graph(n, m, seed=1)
        indptr, indices, _ = edges_to_csr(n, src, dst)
        adj = adjacency(n, src, dst)

        expected = np.full(n, -1, np.int64)
        expected[0] = 0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if expected[v] < 0:
                    expected[v] = expected[u] + 1
                    queue.append(v)

        np.testing.assert_array_equal(bfs(indptr, indices, 0), expected)

    def test_unreachable_is_minus_one(self):
        indptr, indices, _ = edges_to_csr(
            3, np.array([0], np.int64), np.array([1], np.int64)
        )
        np.testing.assert_array_equal(bfs(indptr, indices, 0), [0, 1, -1])

    def test_source_out_of_range_raises(self):
        indptr, indices, _ = edges_to_csr(
            2, np.empty(0, np.int64), np.empty(0, np.int64)
        )
        with pytest.raises(ValueError):
            bfs(indptr, indices, 5)

    def test_malformed_indices_raise(self):
        indptr = np.array([0, 1], np.int64)
        indices = np.array([7], np.int64)  # node 7 in a 1-node graph
        with pytest.raises(ValueError):
            bfs(indptr, indices, 0)


class TestDfsPreorder:
    def test_matches_recursive_reference(self):
        n, m = 40, 160
        src, dst = random_graph(n, m, seed=2)
        indptr, indices, _ = edges_to_csr(n, src, dst)
        adj = adjacency(n, src, dst)

        visited = [False] * n
        expected = []

        def rec(u):
            visited[u] = True
            expected.append(u)
            for v in adj[u]:
                if not visited[v]:
                    rec(v)

        rec(0)
        np.testing.assert_array_equal(dfs_preorder(indptr, indices, 0), expected)

    def test_only_reachable_nodes(self):
        indptr, indices, _ = edges_to_csr(
            4, np.array([0, 1], np.int64), np.array([1, 0], np.int64)
        )
        np.testing.assert_array_equal(dfs_preorder(indptr, indices, 0), [0, 1])


class TestTopologicalSort:
    def test_property_on_random_dag(self):
        # random DAG: edges only from lower to strictly higher ids
        n, m = 80, 400
        rng = np.random.default_rng(3)
        src = rng.integers(0, n - 1, m)
        span = rng.integers(1, n, m)
        dst = np.minimum(src + span, n - 1)
        keep = src < dst
        indptr, indices, _ = edges_to_csr(n, src[keep], dst[keep])
        order = topological_sort(indptr, indices)
        assert sorted(order.tolist()) == list(range(n))
        position = np.empty(n, np.int64)
        position[order] = np.arange(n)
        for u in range(n):
            for p in range(indptr[u], indptr[u + 1]):
                assert position[u] < position[indices[p]]

    def test_deterministic_lowest_index_first(self):
        # two independent chains: 0->2, 1->3
        indptr, indices, _ = edges_to_csr(
            4, np.array([0, 1], np.int64), np.array([2, 3], np.int64)
        )
        np.testing.assert_array_equal(
            topological_sort(indptr, indices), [0, 1, 2, 3]
        )

    def test_cycle_raises(self):
        indptr, indices, _ = edges_to_csr(
            2, np.array([0, 1], np.int64), np.array([1, 0], np.int64)
        )
        with pytest.raises(ValueError):
            topological_sort(indptr, indices)


class TestDijkstra:
    def _reference(self, n, adj_w, source):
        dist = [np.inf] * n
        dist[source] = 0.0
        heap = [(0.0, source)]
        done = [False] * n
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
        return np.array(dist)

    def test_matches_reference(self):
        n, m = 60, 400
        src, dst = random_graph(n, m, seed=4)
        weights = np.round(np.random.default_rng(5).uniform(0.0, 10.0, m), 3)
        indptr, indices, order = edges_to_csr(n, src, dst)
        w_csr = weights[order]

        adj_w = [[] for _ in range(n)]
        for u, v, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
            adj_w[u].append((v, w))

        np.testing.assert_allclose(
            dijkstra(indptr, indices, w_csr, 0),
            self._reference(n, adj_w, 0),
            rtol=1e-12,
        )

    def test_unreachable_is_inf(self):
        indptr, indices, order = edges_to_csr(
            3, np.array([0], np.int64), np.array([1], np.int64)
        )
        dist = dijkstra(indptr, indices, np.array([2.5]), 0)
        np.testing.assert_array_equal(dist, [0.0, 2.5, np.inf])

    def test_inf_weight_edge_is_effectively_absent(self):
        indptr, indices, order = edges_to_csr(
            2, np.array([0], np.int64), np.array([1], np.int64)
        )
        dist = dijkstra(indptr, indices, np.array([np.inf]), 0)
        np.testing.assert_array_equal(dist, [0.0, np.inf])

    def test_invalid_weights_raise(self):
        indptr, indices, order = edges_to_csr(
            2, np.array([0], np.int64), np.array([1], np.int64)
        )
        with pytest.raises(ValueError):
            dijkstra(indptr, indices, np.array([-1.0]), 0)
        with pytest.raises(ValueError):
            dijkstra(indptr, indices, np.array([np.nan]), 0)
        with pytest.raises(ValueError):
            dijkstra(indptr, indices, np.array([1.0, 2.0]), 0)


class TestUnionFind:
    def test_matches_reference_dsu(self):
        n, ops = 200, 500
        rng = np.random.default_rng(6)
        pairs = rng.integers(0, n, (ops, 2))

        parent = list(range(n))

        def ref_find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        uf = UnionFind(n)
        components = n
        for a, b in pairs.tolist():
            ra, rb = ref_find(a), ref_find(b)
            merged_ref = ra != rb
            if merged_ref:
                parent[rb] = ra
                components -= 1
            assert uf.union(a, b) == merged_ref
        assert uf.n_components() == components
        for a, b in pairs.tolist():
            assert uf.connected(a, b) == (ref_find(a) == ref_find(b))

    def test_component_size(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.component_size(2) == 3
        assert uf.component_size(3) == 1

    def test_union_returns_false_on_cycle_edge(self):
        uf = UnionFind(3)
        assert uf.union(0, 1) is True
        assert uf.union(1, 0) is False

    def test_out_of_range_raises(self):
        uf = UnionFind(2)
        with pytest.raises(IndexError):
            uf.find(2)
        with pytest.raises(ValueError):
            UnionFind(0)


class TestInsideNjit:
    def test_pipeline_inside_njit(self):
        @njit
        def reach_and_dist(n, src, dst, weights, source):
            indptr, indices, order = edges_to_csr(n, src, dst)
            w_csr = weights[order]
            hops = bfs(indptr, indices, source)
            dist = dijkstra(indptr, indices, w_csr, source)
            return hops, dist

        src = np.array([0, 0, 1], np.int64)
        dst = np.array([1, 2, 2], np.int64)
        weights = np.array([1.0, 5.0, 1.0])
        hops, dist = reach_and_dist(3, src, dst, weights, 0)
        np.testing.assert_array_equal(hops, [0, 1, 1])
        np.testing.assert_array_equal(dist, [0.0, 1.0, 2.0])

    def test_unionfind_inside_njit(self):
        @njit
        def count_components(n, src, dst):
            uf = UnionFind(n)
            for e in range(src.shape[0]):
                uf.union(src[e], dst[e])
            return uf.n_components()

        src = np.array([0, 2], np.int64)
        dst = np.array([1, 3], np.int64)
        assert count_components(5, src, dst) == 3
