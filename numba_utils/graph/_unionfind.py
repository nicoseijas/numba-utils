"""Disjoint-set union (union-find) as a jitclass."""

from __future__ import annotations

import numpy as np
from numba import int64
from numba.experimental import jitclass


@jitclass(
    [
        ("_parent", int64[:]),
        ("_size", int64[:]),
        ("_n_components", int64),
    ]
)
class UnionFind:
    """Disjoint-set union over ``n`` elements with union by size and
    path compression — near-O(1) amortized ``find``/``union``.

    ``union`` returns whether a merge happened (``False`` if already
    connected), which doubles as the cycle test when adding edges.
    Out-of-range elements raise ``IndexError``.
    """

    def __init__(self, n):
        if n < 1:
            raise ValueError("UnionFind: n must be >= 1")
        self._parent = np.arange(n)
        self._size = np.ones(n, np.int64)
        self._n_components = n

    def find(self, x):
        """Representative (root) of ``x``'s component."""
        if x < 0 or x >= self._parent.shape[0]:
            raise IndexError("UnionFind: element out of range")
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            nxt = self._parent[x]
            self._parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        """Merge the components of ``a`` and ``b``; ``True`` if they
        were separate."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        self._n_components -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def component_size(self, x):
        return self._size[self.find(x)]

    def n_components(self):
        return self._n_components
