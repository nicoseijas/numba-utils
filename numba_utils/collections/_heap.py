"""Fixed-capacity binary min-heap (float64)."""

from __future__ import annotations

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass


@jitclass([("_data", float64[:]), ("_size", int64)])
class PriorityQueue:
    """Fixed-capacity min-heap of float64 priorities.

    O(log n) push/pop_min, O(1) peek_min, zero allocation after
    construction. For a max-heap, push negated values. ``push`` on a
    full heap and ``pop_min``/``peek_min`` on an empty one raise
    ``ValueError``.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("PriorityQueue: capacity must be >= 1")
        self._data = np.empty(capacity, np.float64)
        self._size = 0

    def push(self, value):
        if self._size == self._data.shape[0]:
            raise ValueError("PriorityQueue: full")
        data = self._data
        i = self._size
        data[i] = value
        self._size += 1
        while i > 0:
            parent = (i - 1) >> 1
            if data[i] < data[parent]:
                data[i], data[parent] = data[parent], data[i]
                i = parent
            else:
                break

    def pop_min(self):
        if self._size == 0:
            raise ValueError("PriorityQueue: empty")
        data = self._data
        top = data[0]
        self._size -= 1
        size = self._size
        data[0] = data[size]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= size:
                break
            if child + 1 < size and data[child + 1] < data[child]:
                child += 1
            if data[child] < data[i]:
                data[i], data[child] = data[child], data[i]
                i = child
            else:
                break
        return top

    def peek_min(self):
        if self._size == 0:
            raise ValueError("PriorityQueue: empty")
        return self._data[0]

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0
