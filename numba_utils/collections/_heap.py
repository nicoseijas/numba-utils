"""Fixed-capacity binary min-heap: float64 default + dtype-generic factory."""

from __future__ import annotations

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from numba_utils.collections._dtypes import validate_value_type

_PRIORITY_QUEUE_CACHE: dict = {}


def priority_queue_type(value_type):
    """Fixed-capacity min-heap jitclass specialized to ``value_type``.

    O(log n) push/pop_min, O(1) peek_min, zero allocation after
    construction. ``value_type`` must be an ordered Numba scalar type
    (complex is rejected). For a max-heap, push negated values. Cached
    per type: ``priority_queue_type(float64) is PriorityQueue``.
    """
    cached = _PRIORITY_QUEUE_CACHE.get(value_type)
    if cached is not None:
        return cached
    np_dtype = validate_value_type(
        "priority_queue_type", value_type, ordered=True
    )

    @jitclass([("_data", value_type[:]), ("_size", int64)])
    class PriorityQueue:
        """Fixed-capacity min-heap of priorities.

        O(log n) push/pop_min, O(1) peek_min, zero allocation after
        construction. For a max-heap, push negated values. ``push`` on
        a full heap and ``pop_min``/``peek_min`` on an empty one raise
        ``ValueError``.
        """

        def __init__(self, capacity):
            if capacity < 1:
                raise ValueError("PriorityQueue: capacity must be >= 1")
            self._data = np.empty(capacity, np_dtype)
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

    _PRIORITY_QUEUE_CACHE[value_type] = PriorityQueue
    return PriorityQueue


# The ready-to-use float64 specialization — the same class the factory
# returns, so `priority_queue_type(float64) is PriorityQueue`.
PriorityQueue = priority_queue_type(float64)
