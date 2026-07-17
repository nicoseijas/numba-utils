# numba-utils

High-performance building blocks for [Numba](https://numba.pydata.org/).

## Why numba-utils?

Numba is fantastic. But after writing enough kernels you end up
rewriting the same things:

- typed collections that work inside `@njit`
- sampling algorithms
- search and selection primitives
- battle-tested parallel patterns
- profiling helpers that measure JIT code *correctly*
- diagnostics for what the compiler actually built

numba-utils packages those building blocks into a single, well-tested
library. It does not compete with Numba: it builds on top of it — no
magic, no hidden internals, everything callable from your own jitted
code.

It was born from years of demanding compute workloads — long-running
Monte Carlo simulations and numerical engines where the same utilities
had to be rewritten, re-optimized and re-benchmarked for every project.
The lessons from that work are part of the library: as code, as
diagnostics, and as documentation.

## Benchmark honesty

This project has an official policy (see
[GUIDELINES.md](GUIDELINES.md)): every algorithm states whether it is
**faster** than the standard alternative, **similar but more ergonomic**,
or **slower but solving a problem unavailable elsewhere** — and
unfavorable results are never hidden. [BENCHMARKS.md](BENCHMARKS.md)
contains losing rows on purpose; they tell you when NOT to use a
function, which is worth more than the wins.

See [VISION.md](VISION.md), [ROADMAP.md](ROADMAP.md) and
[GUIDELINES.md](GUIDELINES.md).

## Status

Phase 1 in development. Available modules:

- `numba_utils.decorators` — `njit_fast`, `parallel`, `cached_njit`, `boundscheck`
- `numba_utils.profiling` — `benchmark`, `compare`, `warmup`, `compile_time`
- `numba_utils.arrays` — `binary_search`, `lower_bound`, `upper_bound`,
  `fast_clip`, `normalize`, `cumulative_sum`, `rolling_sum`, `rolling_mean`,
  `histogram`, `bincount`, `unique_sorted`
- `numba_utils.algorithms` — `nth_element`, `quickselect`, `fast_argpartition`,
  `topk`, `argmax2`, `insertion_sort`, `partial_sort`, `counting_sort`,
  `radix_sort`
- `numba_utils.random` — `seed`, `shuffle`, `permutation`, `choice`,
  `reservoir_sampling`, `weighted_sampling`, `alias_setup`/`alias_draw`/
  `alias_sample` (all over Numba's nopython RNG)
- `numba_utils.collections` — `Stack`, `FixedQueue`, `RingBuffer`,
  `PriorityQueue`, `BitSet`, `SparseSet`, `ObjectPool` (jitclass-based,
  usable inside `@njit`), plus `counter` and `typed_defaultdict`
- `numba_utils.diagnostics` — `show`, `check`, `inspect`: what did Numba
  actually build, and which known issues apply to it
- `numba_utils.configure` / `config` — global decorator policy (cache,
  fastmath, parallel, nogil) from code or environment

Measured results against NumPy live in [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

```python
import numpy as np
from numba_utils import njit_fast, compare

@njit_fast
def total(arr):
    acc = 0.0
    for x in arr:
        acc += x
    return acc

arr = np.random.rand(10_000_000)

result = compare(np.sum, total, args=(arr,), n=50)
print(result.summary())
```

## Development

```
python -m venv .venv
.venv/Scripts/pip install -e .[dev]
.venv/Scripts/python -m pytest
```

`@boundscheck` development mode is enabled with the environment variable
`NUMBA_UTILS_DEV=1` (turns on array bounds checking; it vanishes in
production builds).

## Configuration

Decorator behavior can be overridden globally — from code or from the
environment — without touching call sites:

```python
import numba_utils as nu

nu.configure(cache=False)      # e.g. multi-process farms; see docs/numba-cache.md
```

```
NUMBA_UTILS_CACHE=0            # same, from the environment (CI, worker farms)
```

Global overrides win over per-call arguments by design: they exist for
environment-level policy. Options: `cache`, `fastmath`, `parallel`,
`nogil`.

## Diagnostics

```python
from numba_utils import diagnostics

diagnostics.show(fn)     # signatures, cache state, flags, compile times
diagnostics.check(fn)    # known-issue warnings with concrete recommendations
```

## Performance documentation

Numba behaves differently in production than in tutorials. The knowledge
is first-class documentation here:

- [docs/performance.md](docs/performance.md) — when Numba wins, JIT
  amortization, common mistakes
- [docs/numba-cache.md](docs/numba-cache.md) — the on-disk cache and the
  environments where it crashes
- [docs/parallelism.md](docs/parallelism.md) — prange granularity,
  process-level parallelism, structural rules
- [docs/benchmarking.md](docs/benchmarking.md) — measuring JIT code
  without fooling yourself
