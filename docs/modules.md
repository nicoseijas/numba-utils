# Module reference

The full API surface, by category. Every function's docstring carries
complexity, memory, and honest positioning; this page is the map.

## Core

### `numba_utils.decorators`

- `njit_fast` — `njit(cache=True, fastmath=True, nogil=True)` for throughput kernels
- `njit_parallel` — `njit(parallel=True, cache=True)`; read [parallelism.md](parallelism.md) first
- `cached_njit` — compile once, reuse across runs; see [numba-cache.md](numba-cache.md)
- `boundscheck` — bounds checking with `NUMBA_UTILS_DEV=1`, plain njit in production

All accept bare and called forms; keyword overrides forward to `njit`
verbatim.

### `numba_utils.arrays`

- Search over sorted arrays: `binary_search`, `lower_bound`, `upper_bound`
- Transforms: `fast_clip`, `normalize`, `cumulative_sum` (all with `out=` buffer reuse)
- Windows: `rolling_sum`, `rolling_mean`
- Counting: `histogram` (single pass, no edge array), `bincount`
- `unique_sorted` — dedup without the sort that dominates `np.unique`

### `numba_utils.algorithms`

- Selection: `nth_element` (in-place, C++ semantics), `quickselect`, `fast_argpartition`
- `topk` — heap path for small k, quickselect for large; `argmax2` (index AND value)
- Sorts: `insertion_sort`, `partial_sort` (in-place); `counting_sort`, `radix_sort` (new array; integer dtypes, honest loss vs NumPy's SIMD sort on full-range keys)

## Performance

### `numba_utils.parallel`

Complete parallel operations, not prange wrappers ([docs](parallelism.md),
[design](design/parallel.md)): `parallel_sum`, `parallel_reduce`
(per-index kernel decorator), `parallel_histogram` (bit-exact),
`parallel_prefix_sum`, `parallel_topk`. All fall back to serial below
`SERIAL_THRESHOLD`.

### `numba_utils.profiling`

- `benchmark` — function mode excludes JIT compilation by default; block mode via `with benchmark():`
- `compare` — two callables, same inputs, warmed up: mean/median/variance + speedup
- `warmup`, `compile_time`, `compile_stats`

### `numba_utils.diagnostics`

- `show(fn)` — signatures, cache state, flags, compile times
- `check(fn)` — known-issue warnings with concrete recommendations
- `inspect(fn)` — the underlying immutable `FunctionReport`

## Data structures

### `numba_utils.collections`

jitclass-based, constructible and usable inside `@njit`
([design](design/collections.md)): `Stack`, `FixedQueue`, `RingBuffer`
(overwrite-oldest), `PriorityQueue` (binary min-heap), `BitSet`,
`SparseSet` (O(1) add/discard/contains/clear), `ObjectPool` (slot
allocator with double-release detection). Plus `counter` and
`typed_defaultdict` over typed dicts. float64 values / int64 indices in
v1; dtype-generic factories are on the roadmap.

### `numba_utils.random`

Over Numba's nopython RNG, which is separate from NumPy's — seed it
with `seed()` ([design](design/rng.md)): `shuffle`, `permutation`,
`choice`, `reservoir_sampling` (Algorithm R), `weighted_sampling`, and
the Walker alias method as `alias_setup` / `alias_draw` /
`alias_sample`.

## Developer tools

### `numba_utils.testing`

- `assert_equivalent(reference, candidate, inputs)` — per-case array copies, failing case named, empty generators fail
- `random_arrays` — generated cases plus the edges that break kernels
- `assert_close`, `deterministic_rng` (pins all three RNG worlds)

Strategy: [testing.md](testing.md).

### Configuration

```python
import numba_utils as nu
nu.configure(cache=False)     # or NUMBA_UTILS_CACHE=0 in the environment
```

Global policy for `cache` / `fastmath` / `parallel` / `nogil`, from code
or environment. Overrides beat per-call arguments by design
([design](design/cache.md)).
