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
- Sorts: `insertion_sort`, `partial_sort` (in-place); `counting_sort`, `radix_sort` (new array; integer dtypes, honest loss vs NumPy's SIMD sort on full-range keys); `stable_argsort` (the stable-argsort spelling that works in nopython), `lexsort` (`np.lexsort` for `@njit` — Numba doesn't implement it; takes a 2-D array, last row is the primary key)
- `combination_table(n, k)` — the C(n, k) index table; loop over `table.shape[0]` instead of hardcoding combo counts (the evaluator bug class)
- `disjoint_rank_aggregate` / `DisjointRankStructure` — reach-weighted all-pairs comparison skipping pairs that share a key, EXACT via inclusion–exclusion over the 2^K−1 key subsets: O((2^K−1)·N log N) vs dense O(N·M). `build` once, `eval` per weight vector (the CFR shape: 133x over dense with the build amortized). Certified against a dense reference with a drop-removal mutation that screams. Driven from Python, not njit-callable.

## Performance

### `numba_utils.parallel`

Complete parallel operations, not prange wrappers ([docs](parallelism.md),
[design](design/parallel.md)): `parallel_sum`, `parallel_reduce`
(per-index kernel decorator), `parallel_histogram` (bit-exact),
`parallel_prefix_sum`, `parallel_topk`. All fall back to serial below
`SERIAL_THRESHOLD`. Plus `chunked_reduce` — one per-chunk kernel,
serial and parallel drivers with **bit-identical** results: chunk
boundaries depend only on `(n_items, n_chunks)`, never on thread
count; pair the chunk index with `philox_uniform` for runs that are
reproducible by construction.

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
`typed_defaultdict` over typed dicts. `Stack`, `FixedQueue`,
`RingBuffer` and `PriorityQueue` are float64 by default; the
`stack_type` / `fixed_queue_type` / `ring_buffer_type` /
`priority_queue_type` factories return the same containers specialized
to any Numba scalar type, cached per type (`stack_type(float64) is
Stack`). Index-domain containers stay int64.

### `numba_utils.graph`

Graph algorithms over CSR adjacency arrays (`indptr`, `indices` — the
`scipy.sparse.csr_matrix` layout; no Graph class, arrays are the
nopython-native currency): `edges_to_csr` (stable, returns an `order`
array to align per-edge payloads like weights), `bfs` (hop distances,
-1 unreachable), `dfs_preorder` (explicit stack, matches the recursive
order), `topological_sort` (Kahn, deterministic lowest-index-first,
raises on cycles), `dijkstra` (lazy-deletion binary heap, rejects
NaN/negative weights, `inf` = unreachable), and `UnionFind` (jitclass;
union by size + path compression, `union` returns whether a merge
happened). The CSR structure is validated up front (`indptr` monotonic
with the right endpoints) and `indices` entries are bounds-checked
during traversal — a malformed CSR raises instead of corrupting
memory.

### `numba_utils.stats`

Numerically hard statistics — only functions whose naive versions are
*wrong* ([design](design/stats.md)): `logsumexp` and `softmax`
(max-shifted; the direct formulas overflow `exp` beyond ~709; `softmax`
takes `out=`), `weighted_quantile` (inverted CDF — exact match
with `np.quantile(..., weights=..., method="inverted_cdf")`; rejects
NaN values and NaN/negative weights up front), and `weighted_mc_mean`
(uniform-subsample-then-weight, Philox-driven — the correct pattern
for the reach² bug that `assert_no_reweight_bias` guards against).

### `numba_utils.random`

Over Numba's nopython RNG, which is separate from NumPy's — seed it
with `seed()` ([design](design/rng.md)): `shuffle`, `permutation`,
`choice`, `reservoir_sampling` (Algorithm R), `sample_without_replacement`
and the in-place `partial_shuffle` (partial Fisher–Yates, the
zero-allocation repeated-draw MC primitive), `weighted_sampling`, and
the Walker alias method as `alias_setup` / `alias_draw` /
`alias_sample`. Plus the stateless counter-based generator
(Philox4x64-10, bit-identical to `np.random.Philox`):
`philox_uniform` / `philox_uniforms` / `philox_randint` /
`philox4x64` — pure functions of `(key, counter)`, reproducible
regardless of threads, processes or call order — and the composed
variants `philox_partial_shuffle` / `philox_sample_without_replacement`
that drive the Fisher–Yates primitives from a Philox stream.

## Developer tools

### `numba_utils.testing`

- `assert_equivalent(reference, candidate, inputs)` — per-case array copies, failing case named, empty generators fail
- `random_arrays` — generated cases plus the edges that break kernels
- `assert_close`, `deterministic_rng` (pins all three RNG worlds)
- Stochastic asserts: `assert_reproducible` (same seed → bit-identical) and `assert_converges` (different seeds → within N standard errors of the truth; the statistic is Student-t, real false-positive rates documented per `n_runs`). Both take `pass_seed=True` for counter-based (Philox) kernels, whose stream comes from an argument rather than global state. `assert_within_se` is the one-sample-set primitive underneath.
- Certification: `mutation_screams` (deliberately break the kernel, assert the check FAILS — a check that cannot fail certifies nothing) and `assert_no_reweight_bias` (screams on the reach² double-weighting bug)

Strategy: [testing.md](testing.md).

### Configuration

```python
import numba_utils as nu
nu.configure(cache=False)     # or NUMBA_UTILS_CACHE=0 in the environment
```

Global policy for `cache` / `fastmath` / `parallel` / `nogil`, from code
or environment. Overrides beat per-call arguments by design
([design](design/cache.md)).
