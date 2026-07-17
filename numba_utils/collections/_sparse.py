"""Integer-domain containers: sparse set and slot pool."""

from __future__ import annotations

import numpy as np
from numba import boolean, int64
from numba.experimental import jitclass


@jitclass([("_dense", int64[:]), ("_sparse", int64[:]), ("_size", int64)])
class SparseSet:
    """Set of ints in ``[0, universe)`` with O(1) add/discard/contains
    AND O(1) clear — the entity-set of choice for simulation loops that
    reset every iteration. Memory: 2 int64 per universe slot.
    """

    def __init__(self, universe):
        if universe < 1:
            raise ValueError("SparseSet: universe must be >= 1")
        self._dense = np.empty(universe, np.int64)
        self._sparse = np.zeros(universe, np.int64)
        self._size = 0

    def _check(self, x):
        if x < 0 or x >= self._dense.shape[0]:
            raise IndexError("SparseSet: value out of range")

    def contains(self, x):
        self._check(x)
        idx = self._sparse[x]
        return idx < self._size and self._dense[idx] == x

    def add(self, x):
        if self.contains(x):
            return
        self._dense[self._size] = x
        self._sparse[x] = self._size
        self._size += 1

    def discard(self, x):
        if not self.contains(x):
            return
        idx = self._sparse[x]
        last = self._dense[self._size - 1]
        self._dense[idx] = last
        self._sparse[last] = idx
        self._size -= 1

    def values(self):
        """Members as a fresh array, in insertion-ish (unspecified) order."""
        return self._dense[: self._size].copy()

    def clear(self):
        self._size = 0

    def size(self):
        return self._size


@jitclass(
    [
        ("_free", int64[:]),
        ("_n_free", int64),
        ("_in_use", boolean[:]),
    ]
)
class ObjectPool:
    """Slot allocator over ``capacity`` preallocated slots: ``acquire()``
    hands out a free slot index, ``release(slot)`` returns it. The slots
    index into whatever preallocated arrays the caller owns — this is the
    nopython version of an object pool, where "objects" are rows of
    reusable buffers. Double release raises ``ValueError``.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("ObjectPool: capacity must be >= 1")
        self._free = np.arange(capacity)
        self._n_free = capacity
        self._in_use = np.zeros(capacity, np.bool_)

    def acquire(self):
        if self._n_free == 0:
            raise ValueError("ObjectPool: exhausted")
        self._n_free -= 1
        slot = self._free[self._n_free]
        self._in_use[slot] = True
        return slot

    def release(self, slot):
        if slot < 0 or slot >= self._in_use.shape[0]:
            raise IndexError("ObjectPool: slot out of range")
        if not self._in_use[slot]:
            raise ValueError("ObjectPool: slot is not in use (double release?)")
        self._in_use[slot] = False
        self._free[self._n_free] = slot
        self._n_free += 1

    def in_use(self, slot):
        if slot < 0 or slot >= self._in_use.shape[0]:
            raise IndexError("ObjectPool: slot out of range")
        return self._in_use[slot]

    def n_available(self):
        return self._n_free
