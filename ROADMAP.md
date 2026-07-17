# Roadmap

## Founding decisions (2026-07-17)

- **Phase 1 order:** start with `decorators/` + `profiling/`, which are the foundation for developing and benchmarking everything else. Then `arrays/` + `algorithms/`.
- **Targets:** latest stable Numba, Python 3.10+.
- **Publishing:** private development for now; GitHub/PyPI once Phase 1 is solid.

## Phase 1 — Foundation

**Goal:** build a solid base.

**Status:** `decorators/`, `profiling/`, `arrays/` and `algorithms/` implemented
and benchmarked (see BENCHMARKS.md). Beyond the original list, shipped:
global configuration (`configure()` / `NUMBA_UTILS_*` env overrides), the
`diagnostics/` module (`show`/`check`/`inspect`), and the permanent
`docs/` knowledge base (performance, numba-cache, parallelism,
benchmarking). Still pending from the lists below: `argpartition_topk`
(covered by `algorithms.topk`/`fast_argpartition`), `stable_argsort` and
`lexsort`. Next up: `random/` and `collections/`.

```
numba_utils/
    decorators/
    arrays/
    algorithms/
    random/
    collections/
    parallel/
    testing/
    profiling/
```

### Decorators

- `@njit_fast`

  ```python
  @njit_fast
  def foo():
      ...
  ```

  Internally:

  ```python
  njit(
      cache=True,
      fastmath=True,
      nogil=True
  )
  ```

- `@parallel`

  ```python
  @parallel
  def foo():
      ...
  ```

  Clean alias for `@njit(parallel=True)`.

- `@cached_njit` — compiles once. Ideal for scripts.
- `@boundscheck` — development version. Adds asserts; they vanish in production.

### Arrays

A lot of value here.

- `binary_search` — `idx = binary_search(arr, value)`
- `lower_bound`
- `upper_bound`
- `argpartition_topk` — more intuitive than NumPy.
- `unique_sorted` — specialized.
- `fast_clip`
- `normalize`
- `cumulative_sum`
- `rolling_sum`
- `rolling_mean`
- `histogram` — optimized for integers.
- `bincount`

### Algorithms

This is where the project can shine.

- `fast_argpartition` — specialized version.
- `nth_element`
- `quickselect`
- `radix_sort`
- `counting_sort`
- `insertion_sort`
- `partial_sort`
- `topk` — `topk(arr, k)`
- `argmax2` — returns index and value.
- `stable_argsort`
- `lexsort` — Numba-compatible.

### Random

- `choice`
- `shuffle`
- `permutation`
- `reservoir_sampling`
- `weighted_sampling`
- `alias_sampler` — very useful.

### Collections

One of the most interesting parts.

- `typed_defaultdict`

  ```python
  d = typed_defaultdict(
      key_type=int64,
      value_type=float64
  )
  ```

- `Counter`
- `Multiset`
- `BitSet`
- `RingBuffer`
- `FixedQueue`
- `PriorityQueue`
- `Stack`
- `SparseSet`
- `ObjectPool`

### Parallel

A huge opportunity here.

Design constraints learned from real Numba workloads (multi-process CFR
farms) that this module must respect and document:

- Fine-grained `prange` over tiny regions loses to serial: every launch
  pays a full thread-team barrier. Parallelism pays on coarse,
  independent, race-free slots — the API should push users there.
- Repeated `prange` launches in one process can crash the threadpool on
  some setups; long-lived processes with process-level parallelism across
  independent work items are often the better architecture.
- Cross-module `njit -> njit` calls into a parallel region have
  segfaulted in the wild; parallel kernels should be co-located with
  their `prange` driver.
- Micro-benchmarks of jitted loops over-report throughput (loop
  hoisting); benchmark helpers must measure realistic call patterns.

- `parallel_range`

  ```python
  for i in parallel_range(n):
      ...
  ```

  Enables: chunks, scheduling, balance.

- `parallel_reduce`
- `parallel_sum`
- `parallel_histogram`
- `parallel_prefix_sum`
- Work stealing helpers

### Atomics

When Numba allows it (CPU atomics are not supported today; CUDA only).

- `atomic_add`
- `atomic_max`
- `atomic_min`
- `atomic_inc`

### Math

- `clamp`
- `lerp`
- `sigmoid`
- `softmax`
- `logsumexp`
- `percentile`
- `quantile`
- `median`

### Geometry

Very useful for simulations.

- `dot2`
- `cross2`
- `norm2`
- `distance2`
- `bounding_box`

### Statistics

- `mean`
- `variance`
- `std`
- `covariance`
- `correlation`
- `weighted_mean`
- `weighted_quantile`

### Graph

- BFS
- DFS
- UnionFind
- Topological Sort
- Dijkstra

### Profiling

Another very valuable part.

- `benchmark()`

  ```python
  with benchmark():
      foo()
  ```

- `compare()` — `compare(foo, bar)` produces: speedup, mean, median, variance.
- `warmup()`
- `compile_time()`

### Testing

- `assert_close`
- `random_arrays`
- `benchmark_assert`
- `deterministic_rng`

## Benchmark Suite

Every algorithm must be measured against the whole chain:

```
Python → NumPy → Numba → numba-utils
```

With reproducible results.
