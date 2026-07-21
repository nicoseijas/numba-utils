"""Data structures usable inside ``@njit`` code.

Stateful containers are ``jitclass``-based: construct them from Python
or from inside jitted code, methods run at native speed either way.
``Stack``, ``FixedQueue``, ``RingBuffer`` and ``PriorityQueue`` are the
ready-to-use float64 specializations; the ``*_type(value_type)``
factories return the same containers specialized to any Numba scalar
type (``stack_type(int64)``, ``priority_queue_type(float32)``, ...),
cached per type. Index-domain containers (``BitSet``, ``SparseSet``,
``ObjectPool``) stay int64 — indices, not values.

Note: jitclass methods are compiled per process and are not cached by
Numba; the first use of each specialization pays the JIT cost (see
docs/performance.md).
"""

from numba_utils.collections._bitset import BitSet
from numba_utils.collections._heap import PriorityQueue, priority_queue_type
from numba_utils.collections._sparse import ObjectPool, SparseSet
from numba_utils.collections._stack_queue import (
    FixedQueue,
    RingBuffer,
    Stack,
    fixed_queue_type,
    ring_buffer_type,
    stack_type,
)
from numba_utils.collections._typed import counter, typed_defaultdict

__all__ = [
    "BitSet",
    "FixedQueue",
    "ObjectPool",
    "PriorityQueue",
    "RingBuffer",
    "SparseSet",
    "Stack",
    "counter",
    "fixed_queue_type",
    "priority_queue_type",
    "ring_buffer_type",
    "stack_type",
    "typed_defaultdict",
]
