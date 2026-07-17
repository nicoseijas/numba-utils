# Building a Priority Queue that works inside `@njit`

You reach for `heapq`, write `heapq.heappush(pq, item)` inside an `@njit`
function, and Numba refuses to compile it. This is not a gap in your
knowledge — `heapq`, `collections.deque`, `queue.PriorityQueue` and most
of the standard library's data structures simply do not exist in
nopython mode. The compiler has no Python objects to operate on.

So what *do* you do when a simulation loop, a Dijkstra shortest-path, or
an event-driven engine needs a heap, and the whole point was to keep the
hot loop in compiled code? You build the container out of the primitives
Numba *does* understand. This article walks through a fixed-capacity
binary min-heap that is constructible and usable entirely inside `@njit`,
and the design choices behind it.

## Why not just call `heapq`?

Nopython mode compiles a restricted subset of Python to native code. It
understands NumPy arrays, scalars, tuples, typed dicts and lists, and
**jitclasses** — purpose-built classes whose attributes have fixed,
declared types. It does *not* understand arbitrary Python objects, which
is exactly what `heapq` manipulates (a Python `list` of boxed objects).
Crossing into `@njit` and calling `heapq.heappush` is a type error, not
a missing feature.

There are three honest ways to get a heap into compiled code. Picking
between them is the actual design work.

### Option 1 — Arrays plus free functions

The C style: keep the heap as a bare array and a size, and thread both
through free functions.

```python
@njit
def heap_push(data, size, value):
    # ... sift up ...
    return size + 1     # caller must store this!
```

Zero overhead, and it caches to disk. But the caller manually carries
`size` across every call, and one stale `size` silently corrupts the
heap — nopython has no bounds checking to catch it. The discipline this
demands leaks into every call site.

### Option 2 — Generic factories

`make_priority_queue(dtype)` returns a class specialized to a dtype.
Flexible, but every instantiation is a fresh class compile, and the API
doubles: you maintain a factory *and* an instance type.

### Option 3 — A jitclass with fixed dtypes

A `jitclass` is a real object with typed fields, constructible inside
`@njit`, whose methods run at native speed. It is the framework-native
way to express a container, and the ergonomics compound in user code:

```python
from numba import njit
from numba_utils import PriorityQueue

@njit
def process(events):
    pq = PriorityQueue(len(events))   # built in nopython mode
    for e in events:
        pq.push(e)
    total = 0.0
    while not pq.is_empty():
        total += pq.pop_min()
    return total
```

This is the one the library ships. The trade-off it accepts — jitclass
methods are not cached to disk, so the first use pays JIT per process —
is documented, not hidden. For the common case (a heap used inside a
long-running compiled loop) that one-time cost is noise.

## The implementation

A binary min-heap is an array where each node at index `i` has children
at `2i+1` and `2i+2`. "Min-heap" means every parent is `≤` its children,
so the smallest element is always at index 0. Here is the whole thing.

```python
import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

@jitclass([("_data", float64[:]), ("_size", int64)])
class PriorityQueue:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("PriorityQueue: capacity must be >= 1")
        self._data = np.empty(capacity, np.float64)
        self._size = 0
```

The `jitclass` decorator takes a **spec**: each field with its Numba
type. `_data` is a 1-D `float64` array, `_size` an `int64`. Those types
are fixed at class-definition time — that is what lets Numba lay the
object out in memory and generate native method code.

### push — sift up

Append at the end, then swap the new value upward while it is smaller
than its parent:

```python
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
```

`(i - 1) >> 1` is the parent index. The loop stops the moment the
heap property holds, so a push is `O(log n)` worst case and often much
less. Note the explicit `full` check: with no bounds checking underneath,
writing past capacity would be silent corruption, so the guard is the
only thing standing between a full heap and a memory bug.

### pop_min — sift down

Take the root (the minimum), move the last element to the root, then
swap it downward toward its smaller child until the heap property is
restored:

```python
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
```

`child + 1 < size and data[child + 1] < data[child]` picks the *smaller*
of the two children to swap with — the step that keeps it a min-heap.
`peek_min` just returns `data[0]` without removing it, `O(1)`.

## Design notes worth stealing

- **Fixed capacity, zero allocation after construction.** The array is
  sized once in `__init__`. Inside the hot loop there is no allocation at
  all — the reason to be in compiled code in the first place.
- **Fixed dtype (`float64`) on purpose.** A generic version is planned as
  a separate, additive API. Shipping generics first would make every user
  pay compilation and API complexity for a minority need. `float64`
  priorities cover scores, costs, timestamps, and distances — most of
  what a heap actually orders.
- **Want a max-heap?** Push negated values. One obvious container, not a
  constructor flag.
- **The guards are load-bearing.** `push`-when-full and
  `pop`/`peek`-when-empty raise `ValueError`. In a language with no
  bounds checking, these are not politeness — they are the difference
  between an exception and silent corruption.

## When to use it (and when not)

Use a jitclass heap when the priority queue lives inside compiled code: a
discrete-event simulation, a graph traversal, an A\* frontier, a
streaming top-k. If you only need the k smallest of a materialized array
*once*, from Python, don't build a heap — a single selection pass
(`numba_utils.topk`, or `np.partition`) is faster and simpler. The heap
earns its keep when items arrive and leave incrementally, in order, and
you cannot afford to re-sort.

The full container ships in
[numba-utils](https://github.com/nicoseijas/numba-utils) as
`PriorityQueue`, alongside `SparseSet`, `RingBuffer`, `BitSet` and other
jitclasses built on exactly these principles. The design record for why
each one is a jitclass — and not a free-function or a factory — is in the
[collections design notes](../design/collections.md).

---

*Related: [Module reference](../modules.md) ·
[Why most Numba benchmarks are wrong](why-most-numba-benchmarks-are-wrong.md)*
