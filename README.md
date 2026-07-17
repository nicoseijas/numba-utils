# numba-utils

High-performance building blocks for [Numba](https://numba.pydata.org/).

Everything that any Numba developer ends up writing over and over again should
live here. It does not compete with Numba: it builds on top of it.

## Why this project exists

This library was born from years of working on demanding compute workloads —
long-running Monte Carlo simulations and numerical engines where the same
utilities had to be rewritten, re-optimized and re-benchmarked for every new
project. numba-utils collects those battle-tested building blocks in one
place, so the next hot loop starts from proven, measured code instead of
boilerplate.

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
