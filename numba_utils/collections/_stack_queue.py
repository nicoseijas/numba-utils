"""Fixed-capacity LIFO / FIFO / overwriting-ring containers (float64)."""

from __future__ import annotations

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass


@jitclass([("_data", float64[:]), ("_size", int64)])
class Stack:
    """Fixed-capacity LIFO of float64. O(1) push/pop/peek, zero allocation
    after construction. ``push`` on a full stack raises ``ValueError``."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Stack: capacity must be >= 1")
        self._data = np.empty(capacity, np.float64)
        self._size = 0

    def push(self, value):
        if self._size == self._data.shape[0]:
            raise ValueError("Stack: full")
        self._data[self._size] = value
        self._size += 1

    def pop(self):
        if self._size == 0:
            raise ValueError("Stack: empty")
        self._size -= 1
        return self._data[self._size]

    def peek(self):
        if self._size == 0:
            raise ValueError("Stack: empty")
        return self._data[self._size - 1]

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0


@jitclass([("_data", float64[:]), ("_head", int64), ("_size", int64)])
class FixedQueue:
    """Fixed-capacity FIFO of float64 over a circular buffer. O(1)
    push/pop, zero allocation after construction. ``push`` on a full
    queue raises ``ValueError`` — for overwrite-oldest semantics use
    :class:`RingBuffer`."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("FixedQueue: capacity must be >= 1")
        self._data = np.empty(capacity, np.float64)
        self._head = 0
        self._size = 0

    def push(self, value):
        capacity = self._data.shape[0]
        if self._size == capacity:
            raise ValueError("FixedQueue: full")
        self._data[(self._head + self._size) % capacity] = value
        self._size += 1

    def pop(self):
        if self._size == 0:
            raise ValueError("FixedQueue: empty")
        value = self._data[self._head]
        self._head = (self._head + 1) % self._data.shape[0]
        self._size -= 1
        return value

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == self._data.shape[0]


@jitclass([("_data", float64[:]), ("_next", int64), ("_size", int64)])
class RingBuffer:
    """Fixed-capacity float64 ring that OVERWRITES the oldest value when
    full — the last-N-observations container (rolling windows, telemetry).
    O(1) push and random access to recent values."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("RingBuffer: capacity must be >= 1")
        self._data = np.empty(capacity, np.float64)
        self._next = 0
        self._size = 0

    def push(self, value):
        capacity = self._data.shape[0]
        self._data[self._next] = value
        self._next = (self._next + 1) % capacity
        if self._size < capacity:
            self._size += 1

    def last(self, i):
        """i-th most recent value; ``last(0)`` is the newest."""
        if i < 0 or i >= self._size:
            raise IndexError("RingBuffer: index out of range")
        return self._data[(self._next - 1 - i) % self._data.shape[0]]

    def to_array(self):
        """Contents oldest-to-newest as a fresh array. O(size)."""
        out = np.empty(self._size, np.float64)
        capacity = self._data.shape[0]
        start = (self._next - self._size) % capacity
        for i in range(self._size):
            out[i] = self._data[(start + i) % capacity]
        return out

    def size(self):
        return self._size

    def is_full(self):
        return self._size == self._data.shape[0]
