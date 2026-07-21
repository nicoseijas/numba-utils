"""Fixed-capacity LIFO / FIFO / overwriting-ring containers.

Each container ships two ways:

- ``Stack``, ``FixedQueue``, ``RingBuffer`` — the ready-to-use float64
  specializations (scores, priorities, observations).
- ``stack_type(value_type)``, ``fixed_queue_type(value_type)``,
  ``ring_buffer_type(value_type)`` — factories returning the same
  container specialized to any Numba scalar type (``int64``,
  ``float32``, ...). Specializations are cached per type:
  ``stack_type(float64) is Stack``.

jitclass methods are not cached on disk, so the first use of each
specialization pays JIT compilation once per process.
"""

from __future__ import annotations

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from numba_utils.collections._dtypes import validate_value_type

_STACK_CACHE: dict = {}
_FIXED_QUEUE_CACHE: dict = {}
_RING_BUFFER_CACHE: dict = {}


def stack_type(value_type):
    """Fixed-capacity LIFO jitclass specialized to ``value_type``.

    O(1) push/pop/peek, zero allocation after construction. ``push`` on
    a full stack raises ``ValueError``. The returned class is cached:
    the same ``value_type`` always yields the same class, so instances
    from separate calls share one compiled specialization.
    """
    cached = _STACK_CACHE.get(value_type)
    if cached is not None:
        return cached
    np_dtype = validate_value_type("stack_type", value_type)

    @jitclass([("_data", value_type[:]), ("_size", int64)])
    class Stack:
        """Fixed-capacity LIFO. O(1) push/pop/peek, zero allocation
        after construction. ``push`` on a full stack raises
        ``ValueError``."""

        def __init__(self, capacity):
            if capacity < 1:
                raise ValueError("Stack: capacity must be >= 1")
            self._data = np.empty(capacity, np_dtype)
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

    _STACK_CACHE[value_type] = Stack
    return Stack


def fixed_queue_type(value_type):
    """Fixed-capacity FIFO jitclass specialized to ``value_type``.

    Circular buffer, O(1) push/pop, zero allocation after construction.
    ``push`` on a full queue raises ``ValueError`` — for
    overwrite-oldest semantics use :func:`ring_buffer_type`. Cached per
    type like :func:`stack_type`.
    """
    cached = _FIXED_QUEUE_CACHE.get(value_type)
    if cached is not None:
        return cached
    np_dtype = validate_value_type("fixed_queue_type", value_type)

    @jitclass([("_data", value_type[:]), ("_head", int64), ("_size", int64)])
    class FixedQueue:
        """Fixed-capacity FIFO over a circular buffer. O(1) push/pop,
        zero allocation after construction. ``push`` on a full queue
        raises ``ValueError``."""

        def __init__(self, capacity):
            if capacity < 1:
                raise ValueError("FixedQueue: capacity must be >= 1")
            self._data = np.empty(capacity, np_dtype)
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

    _FIXED_QUEUE_CACHE[value_type] = FixedQueue
    return FixedQueue


def ring_buffer_type(value_type):
    """Fixed-capacity overwriting ring jitclass specialized to
    ``value_type``.

    OVERWRITES the oldest value when full — the last-N-observations
    container (rolling windows, telemetry). O(1) push and random access
    to recent values. Cached per type like :func:`stack_type`.
    """
    cached = _RING_BUFFER_CACHE.get(value_type)
    if cached is not None:
        return cached
    np_dtype = validate_value_type("ring_buffer_type", value_type)

    @jitclass([("_data", value_type[:]), ("_next", int64), ("_size", int64)])
    class RingBuffer:
        """Fixed-capacity ring that OVERWRITES the oldest value when
        full. O(1) push and random access to recent values."""

        def __init__(self, capacity):
            if capacity < 1:
                raise ValueError("RingBuffer: capacity must be >= 1")
            self._data = np.empty(capacity, np_dtype)
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
            out = np.empty(self._size, np_dtype)
            capacity = self._data.shape[0]
            start = (self._next - self._size) % capacity
            for i in range(self._size):
                out[i] = self._data[(start + i) % capacity]
            return out

        def size(self):
            return self._size

        def is_full(self):
            return self._size == self._data.shape[0]

    _RING_BUFFER_CACHE[value_type] = RingBuffer
    return RingBuffer


# The ready-to-use float64 specializations — same classes the factories
# return, so `stack_type(float64) is Stack`.
Stack = stack_type(float64)
FixedQueue = fixed_queue_type(float64)
RingBuffer = ring_buffer_type(float64)
