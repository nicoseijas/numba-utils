# numba-utils

**Battle-tested building blocks for production
[Numba](https://numba.pydata.org/) workloads.** Built for production
numerical software with Numba.

✓ Zero dependencies beyond NumPy + Numba &nbsp;·&nbsp; ✓ Callable inside
`@njit` &nbsp;·&nbsp; ✓ Honest benchmarks &nbsp;·&nbsp; ✓ Diagnostics
&nbsp;·&nbsp; ✓ CI &nbsp;·&nbsp; ✓ MIT

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

```python
from numba import njit
from numba_utils import topk

@njit
def winners(scores):
    return topk(scores, 10)     # O(n), no full sort — runs in nopython mode
```

```python
from numba import njit
from numba_utils import PriorityQueue, SparseSet

@njit
def simulate(n_events):
    events = PriorityQueue(n_events)    # constructed in nopython mode
    active = SparseSet(100_000)         # O(1) add/discard/contains/clear
    ...
```

```python
from numba_utils import compare

compare(numpy_impl, njit_impl, args=(values,))
# 31x on a fused kernel — JIT compilation excluded automatically
```

And the part that almost no library ships — diagnostics for compiled
code:

```python
>>> from numba_utils import diagnostics
>>> diagnostics.check(fn)
⚠ cache=True may crash when loaded across processes (farms, network FS)
  → NUMBA_UTILS_CACHE=0  or  configure(cache=False)
⚠ fastmath=True relaxes IEEE 754 — not for exact/reproducible results
```

## Why this exists

After enough numerical projects — long-running Monte Carlo engines,
solvers, simulation farms — you realize you've rewritten the same binary
search, the same typed collections, the same sampling algorithms and the
same benchmarking helpers again. And debugged the same Numba production
surprises again.

numba-utils ships more than code. It ships the production knowledge
that usually stays trapped inside numerical projects — as kernels with
the pitfalls engineered around, as diagnostics, and as
[documentation](docs). It does not compete with Numba: it builds on top
of it.

## Why not...

- **heapq / `collections`?** They can't be called from nopython mode.
  The containers here are jitclasses usable inside `@njit`.
- **NumPy?** Many helpers are built to run *inside* compiled kernels,
  where NumPy calls can't reach. Where NumPy is faster (bandwidth-bound
  sweeps, its SIMD sort), [BENCHMARKS.md](BENCHMARKS.md) says so.
- **SciPy?** A heavy dependency that isn't njit-callable; this stays at
  NumPy + Numba and works in the compiled path.
- **Numba itself?** Numba is a compiler. numba-utils is a standard
  library on top of it.

## Design principles

1. **Performance First** — nothing ships without a benchmarked justification.
2. **No Hidden Magic** — thin, readable layers over Numba; nothing rewrites your code.
3. **Numba Compatible** — everything callable from your own `@njit` code, no hacks.
4. **Minimal APIs** — `topk(arr, 10)`, not twenty keyword parameters.
5. **Benchmark Honesty** — losses are published next to the wins.

The identity behind these: [docs/philosophy.md](docs/philosophy.md).

## Modules

**Core** — decorators, arrays, algorithms ·
**Performance** — parallel (complete operations, not prange wrappers),
profiling (JIT excluded by default), diagnostics ·
**Data structures** — collections, random ·
**Developer tools** — testing, config

Full API: [docs/modules.md](docs/modules.md) · Runnable code:
[examples/](examples)

## Benchmark honesty

Every algorithm states whether it is **faster**, **similar but more
ergonomic**, or **slower but solving a problem unavailable elsewhere**.
[BENCHMARKS.md](BENCHMARKS.md) contains losing rows on purpose: they
tell you when NOT to use a function. Backed in-repo by reproducible
[benchmarks/](benchmarks), 200+ reference-validated tests
([why there's no coverage badge](docs/testing.md)), and CI running all
of it. Trade-off records: [docs/design/](docs/design).

## Used in

Patterns extracted from real workloads: Monte Carlo equity engines,
game-theory solvers (CFR), quantitative research, scientific
simulations, optimization loops.

## Status

Phase 1 (see [ROADMAP.md](ROADMAP.md)):

- [x] decorators, profiling, diagnostics, config
- [x] arrays, algorithms, random, collections
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
mandatory, honesty is policy.
