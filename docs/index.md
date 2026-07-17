# numba-utils

**Battle-tested building blocks for production
[Numba](https://numba.pydata.org/) workloads.**

numba-utils is a standard library for numerical software built with
Numba. Every piece is callable from your own `@njit` code, benchmarked
against the standard alternative, and honest about when *not* to use it.

```python
from numba import njit
from numba_utils import topk

@njit
def winners(scores):
    return topk(scores, 10)     # O(n), no full sort — runs in nopython mode
```

## What's inside

- **Kernels** — search, selection, sorts, transforms, and rolling
  windows that run inside compiled code.
- **Containers** — `PriorityQueue`, `SparseSet`, `RingBuffer`, `BitSet`
  and more, as jitclasses you can construct inside `@njit`.
- **Parallel** — complete parallel operations (not `prange` wrappers)
  with serial fallbacks and bit-exact results where it matters.
- **Tooling** — honest benchmarking (`compare`), diagnostics for
  compiled functions (`diagnostics.check`), and a testing module built
  around reference validation.

## Why it exists

Production Numba behaves differently than tutorial Numba: caches that
crash across processes, `prange` barrier costs, silent out-of-bounds
corruption. numba-utils ships that hard-won knowledge as code,
diagnostics, and documentation — not just functions. Read the
[philosophy](philosophy.md) for the identity behind every decision.

## Next steps

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Module reference](modules.md)
- [Benchmark honesty](benchmark-honesty.md) — the policy that separates
  this from a marketing collection
