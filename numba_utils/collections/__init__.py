"""Data structures usable inside ``@njit`` code.

Stateful containers are ``jitclass``-based: construct them from Python
or from inside jitted code, methods run at native speed either way.
This first version fixes element dtypes (``float64`` values, ``int64``
indices) — dtype-generic factories are on the roadmap.

Note: jitclass methods are compiled per process and are not cached by
Numba; the first use pays the JIT cost (see docs/performance.md).
"""

from numba_utils.collections._bitset import BitSet
from numba_utils.collections._heap import PriorityQueue
from numba_utils.collections._sparse import ObjectPool, SparseSet
from numba_utils.collections._stack_queue import FixedQueue, RingBuffer, Stack
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
    "typed_defaultdict",
]
