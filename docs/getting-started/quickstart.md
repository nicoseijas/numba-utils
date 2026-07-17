# Quickstart

Three things numba-utils does well, in one page: a kernel that beats
NumPy, an honest benchmark, and containers that live inside `@njit`.

## A fused kernel, benchmarked honestly

NumPy's `np.clip(values, lo, hi) ** 2` allocates two temporary arrays and
sweeps memory three times. A single fused pass in nopython mode avoids
all of that:

```python
import numpy as np
from numba_utils import compare, njit_fast, topk

@njit_fast
def clipped_energy(values, lo, hi):
    total = 0.0
    for i in range(values.shape[0]):
        x = values[i]
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        total += x * x
    return total

def numpy_clipped_energy(values, lo, hi):
    return float(np.sum(np.clip(values, lo, hi) ** 2))

values = np.random.default_rng(0).normal(0.0, 1.0, 2_000_000)

result = compare(numpy_clipped_energy, clipped_energy, args=(values, -1.0, 1.0), n=21)
print(result.summary())
```

`compare` runs warmup rounds first, so **JIT compilation never pollutes
the numbers** — a mistake that makes most Numba benchmarks wrong. See
[Benchmarking](../benchmarking.md).

## Selection without a full sort

```python
print(topk(values, 5))   # the 5 largest, descending — O(n), no full sort
```

`topk` is callable from inside your own jitted functions, not just from
Python.

## Containers inside `@njit`

`heapq` and `collections` can't cross into nopython mode. These can:

```python
from numba import njit
from numba_utils import PriorityQueue, SparseSet

@njit
def simulate(n_events):
    events = PriorityQueue(n_events)    # constructed in nopython mode
    active = SparseSet(100_000)         # O(1) add / discard / contains / clear
    ...
```

## Diagnose a compiled function

The part almost no library ships — a check for the production pitfalls:

```python
from numba_utils import diagnostics
diagnostics.check(clipped_energy)
# ⚠ fastmath=True relaxes IEEE 754 — not for exact/reproducible results
```

## Where to go next

- [Module reference](../modules.md) — the full API surface
- [Performance overview](../performance.md) and [Cache](../numba-cache.md)
- [Design records](../design/README.md) — why each trade-off was made
