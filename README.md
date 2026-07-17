# numba-utils

**Battle-tested building blocks for numerical computing with
[Numba](https://numba.pydata.org/).** Built for engineers writing
serious numerical software.

## Why this exists

After enough numerical projects — long-running Monte Carlo engines,
solvers, simulation farms — you realize you've rewritten the same binary
search, the same typed collections, the same sampling algorithms and the
same benchmarking helpers again. And debugged the same Numba production
surprises again. This library extracts those building blocks into a
single, reusable, tested package.

It does not compete with Numba: it builds on top of it.

```
                    Numba
                      ▲
                      │
                 numba-utils
      ┌───────────────┼───────────────┐
   Arrays        Collections       Parallel
   Algorithms    Random            Profiling
   Decorators    Testing           Diagnostics
```

## Design principles

1. **Performance First** — nothing ships without a benchmarked justification.
2. **No Hidden Magic** — thin, readable layers over Numba; nothing rewrites your code.
3. **Numba Compatible** — everything callable from your own `@njit` code, no hacks.
4. **Minimal APIs** — `topk(arr, 10)`, not twenty keyword parameters.
5. **Benchmark Honesty** — losses are published next to the wins.

This is the project's identity, not a checklist: [docs/philosophy.md](docs/philosophy.md).

## Modules

**Core**

- [`decorators`](numba_utils/decorators) — `njit_fast`, `njit_parallel`, `cached_njit`, `boundscheck`
- [`arrays`](numba_utils/arrays) — search, transforms, rolling windows, histograms
- [`algorithms`](numba_utils/algorithms) — selection, top-k, specialized sorts

**Performance**

- [`parallel`](numba_utils/parallel) — complete parallel operations, not prange wrappers ([docs](docs/parallelism.md))
- [`profiling`](numba_utils/profiling) — benchmarking that excludes JIT compilation *by default* ([docs](docs/benchmarking.md))
- [`diagnostics`](numba_utils/diagnostics) — what Numba actually built, and which known issues apply

**Data structures**

- [`collections`](numba_utils/collections) — Stack, PriorityQueue, RingBuffer, BitSet, SparseSet, ObjectPool… usable inside `@njit`
- [`random`](numba_utils/random) — shuffling, reservoir and weighted sampling, Walker alias method

**Developer tools**

- [`testing`](numba_utils/testing) — validate kernels against independent references with generated edge cases
- [`config`](numba_utils/_config.py) — global policy (cache, fastmath, parallel, nogil) from code or environment

## Examples

Write a kernel, benchmark it honestly — warmup is the default, so JIT
compilation never pollutes the numbers:

```python
from numba_utils import njit_fast, compare

@njit_fast
def clipped_energy(values, lo, hi):
    total = 0.0
    for i in range(values.shape[0]):
        x = min(max(values[i], lo), hi)
        total += x * x
    return total

compare(numpy_version, clipped_energy, args=(values, -1.0, 1.0)).summary()
# fused single pass: 31x vs np.sum(np.clip(v, lo, hi) ** 2)
```

Containers that work *inside* jitted code:

```python
from numba import njit
from numba_utils import PriorityQueue, SparseSet

@njit
def simulate(n_events):
    events = PriorityQueue(n_events)   # constructed in nopython mode
    active = SparseSet(1000)           # O(1) add/discard/contains/clear
    ...
```

Trust, then verify — every kernel against an independent reference:

```python
from numba_utils import diagnostics
from numba_utils.testing import assert_equivalent, random_arrays

assert_equivalent(numpy_impl, njit_impl, random_arrays(n_cases=20, size=10_000))
diagnostics.check(njit_impl)   # known-issue warnings with recommendations
```

Runnable versions: [examples/](examples).

## Benchmark honesty

Every algorithm states whether it is **faster** than the standard
alternative, **similar but more ergonomic**, or **slower but solving a
problem unavailable elsewhere** — and unfavorable results are never
hidden. [BENCHMARKS.md](BENCHMARKS.md) contains losing rows on purpose:
they tell you when NOT to use a function.

The claims above are backed in-repo: reproducible
[benchmarks/](benchmarks) (fixed seeds, published environment), 200+
tests where every kernel is validated against an independent reference
(NumPy, heapq, set — nopython code fails *silently* out of bounds, so
reference validation is the only trustworthy coverage; the full
strategy, including why a line-coverage badge is intentionally absent,
is in [docs/testing.md](docs/testing.md)), runnable
[examples/](examples), technical [docs/](docs), and CI running all of it.

## Used in

The patterns here were extracted from real workloads:

- Monte Carlo equity engines
- game-theory solvers (CFR)
- quantitative research
- scientific simulations
- optimization loops

## Configuration

Environment-level policy without touching call sites — e.g. Numba's
on-disk cache crashes intermittently on some multi-process setups
([docs/numba-cache.md](docs/numba-cache.md)):

```python
import numba_utils as nu
nu.configure(cache=False)        # or NUMBA_UTILS_CACHE=0 in the environment
```

## Status

Phase 1 (see [ROADMAP.md](ROADMAP.md)):

- [x] decorators, profiling, diagnostics, config
- [x] arrays, algorithms
- [x] random, collections
- [x] parallel patterns, testing helpers
- [ ] dtype-generic collections, `stable_argsort`, `lexsort`
- [ ] PyPI release

## Development

```
python -m venv .venv
.venv/Scripts/pip install -e .[dev]
.venv/Scripts/python -m pytest
```

Contributions follow [GUIDELINES.md](GUIDELINES.md) — benchmarks are
mandatory, honesty is policy. Why things are built the way they are:
[docs/design/](docs/design). Project docs: [VISION.md](VISION.md),
[ROADMAP.md](ROADMAP.md), [docs/](docs).
