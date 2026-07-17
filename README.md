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
