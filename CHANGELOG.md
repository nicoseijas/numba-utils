# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-21

Phase 4, complete: contributions from a production PLO5 CFR solver.
The flagship `disjoint_rank_aggregate` kernel, certification testing
primitives, the reach² guard, and two documented parallel patterns —
plus the fixes from the 0.3.3 audit verdict. Additive; the one
behavior change (`counting_sort`'s range guard) is a relaxation, not a
restriction.

### Fixed

- **algorithms** — `counting_sort`'s range guard was too aggressive
  (`range > 16n + 4096` rejected canonical inputs: uint16 codes over
  n~1000, a few hundred values over 10k). It now rejects only when the
  range is large both absolutely (> 2**20, 8 MiB counts) AND relative
  to n (`range > 64n`) — the 1 GiB pathological case stays rejected,
  the dense regimes are admitted. (#11)
- **arrays / parallel** — the `bins > 2**30` cap is now library-wide:
  `histogram` enforces it too (it had none), and `parallel_histogram`
  checks it before delegating to the serial path, so the same `bins`
  is accepted or rejected regardless of array length. (#12)
- **arrays** — `histogram` documents the two boundary behaviors of
  binning-by-scaling: edge-adjacent values can land one bin off
  `np.histogram` in narrow ranges (total identical), and the
  degenerate-span rejection (`hi - lo < bins / 1.8e308`) trades a
  silently-false result for a loud error — prefer `np.histogram` for
  extreme-magnitude ranges. (#13)
- **decorators** — the boundscheck→`cache=False` invariant now warns
  when it overrides an EXPLICIT `cache=True` (per-call kwarg), matching
  `configure()`'s fail-fast stance; the override still stands. (#14)

### Docs

- docs/parallelism.md — `python -X faulthandler` for locating an
  intermittent segfault (an OOB read returns garbage as often as it
  crashes, and garbage-agrees-with-garbage passes a comparison test;
  faulthandler prints the exact `file:line`, distinguishing it from a
  teardown crash), and `os._exit(0)` after flush to skip Numba's
  threadpool teardown in a test runner. Completes Phase 4 item 4.
- docs/parallelism.md — two production patterns documented rather than
  packaged (each safe only under a caller-owned precondition): Hogwild
  lock-free racing accumulation (safe for self-correcting iteration,
  wrong for exact counts — the opposite of what `parallel_histogram`
  needs) and factorized independent-opponent aggregation (Θ(items·B·P)
  vs Θ(B^P), valid under factor independence), with the rejected
  drop-removal prefix-sum anti-pattern (44% error) as the
  counterexample. Completes Phase 4 item 5.
- Upgrade note in docs/numba-cache.md: Numba's cache key does not
  include the numba-utils version, so a stale `__pycache__` can make a
  process load an old binary and skip a compile-time validation gate
  added in a release. Clear `__pycache__` (or run once with
  `NUMBA_UTILS_CACHE=0`) after upgrading. (#15)

### Added

- **algorithms** — the Phase 4 flagship: `disjoint_rank_aggregate` and
  `DisjointRankStructure` (build/eval). Reach-weighted all-pairs
  comparison that skips pairs sharing a key — EXACTLY, via
  inclusion–exclusion over the hero's 2^K−1 key subsets (an algebraic
  identity, not tuning): O((2^K−1)·N log N) vs the dense O(N·M).
  Generalized from the contributing solver's showdown kernel: keys are
  ANY int64 values (rank-compressed internally, overflow-guarded),
  K ≤ 12 distinct keys per row (validated — a duplicated key breaks
  the identity), house parallel conventions (serial below the launch
  threshold, cache=False on parfor kernels). The build/eval split is
  the iterative-solver shape: topology built once, weights change —
  3.2x over dense one-shot, 133x with the build amortized. Certified
  in the suite the way the solver certified it: dense reference
  sharing no code path, plus a drop-removal `mutation_screams` that
  proves the removal term is live.
- **testing** — `mutation_screams(fn, threshold=...)`: deliberately
  break the kernel and assert the surrounding check would catch it —
  a check that cannot fail certifies nothing. A non-finite deviation
  counts as a scream.
- **testing** — `assert_within_se(samples, target, k=3)`: the
  one-sample-set primitive now shared by `assert_converges`; the SE is
  measured from the samples, never assumed.
- **testing / stats** — the reach² guard: `weighted_mc_mean` (uniform
  subsample of the support, weighted by the TRUE weights — never both;
  Philox-driven, exact when the support fits, njit-callable) and
  `assert_no_reweight_bias` (feeds values correlated with strongly
  skewed weights — the regime where double-weighting biases hardest —
  and asserts the estimates stay within k measured SE of the exact
  weighted mean). The bug it kills shipped twice in the contributing
  solver: invisible at 0/1 reach, equities above 1 at mixed
  frequencies.
- Declined from the contribution proposal: `assert_vs_reference`
  (already exists as `assert_equivalent`); docs now frame it as the
  independent-reference certification step.

## [0.3.3] - 2026-07-21

Rounds 3 and 4 of the adversarial audit (staleness reviewer, Philox
sampling auditor, fix-attacker, regressions reviewer — the audit's
closing verdict). The recurring lesson this release encodes: the
validation standard must apply per LIBRARY, not per file — the NaN
trap solved carefully in `histogram` was absent from `graph/`'s
validator written the same day.

### Fixed (round 4 — attacks on the 0.3.2 fixes)

- **decorators** — the boundscheck/cache invariant moved to the layer
  where ALL option paths converge: `cached_njit(boundscheck=True)`
  (any decorator, kwarg path) bypassed the dev-mode lock and produced
  a cached boundscheck binary — reintroducing the bidirectional cache
  poisoning 0.3.2 declared fixed. Now any resolved
  `boundscheck=True` forces `cache=False`, after defaults, kwargs and
  global overrides alike.
- **graph** — new SIGSEGV closed: a float64 `indptr` containing NaN
  passed `check_csr` (NaN fails every comparison, so `v < prev` never
  fired) and drove `range(1.0, nan)` — 2^31 iterations of arbitrary
  writes in `topological_sort`. Three lateral fixes: integer-dtype
  gate on `indptr`/`indices` at compile time (`np.zeros` default is
  float64 — arriving there by accident is easy), the monotonicity
  test inverted to `not (v >= prev)` (the histogram NaN lesson,
  applied to the sibling module), and the missing in-loop index guard
  added to `topological_sort` — the only one of the four that lacked
  it.
- **parallel** — new SIGSEGV closed: `parallel_histogram` with
  `bins ~ 2**63` overflowed the cache-line padding arithmetic into a
  negative dimension that parallel lowering turned into out-of-bounds
  writes. `bins > 2**30` (already 8 GiB of counts) now raises before
  the arithmetic.
- **arrays / parallel** — the 0.3.2 histogram clamp was memory-safe
  but silently WRONG in the degenerate regimes: an overflowing
  `hi - lo` (scale 0) or a subnormal one (scale inf) put every value
  in bin 0 with `counts.sum()` still matching n — undetectable
  downstream, and the realistic nearly-constant-data trigger fell in
  the second regime. Both now raise (`hi - lo` overflows or is too
  small for this many bins); NumPy refuses the first regime too. The
  clamp stays as the last line of memory safety.
- **algorithms** — 0.3.2 regression: the integer gate rejected bool
  arrays, which sorted fine in 0.3.0 (`np.int64(bool)` is exact).
  Boolean is accepted again.
- **random** — the `philox_uniforms`/`philox_randint` word overlap
  (found independently by two reviewers, corr = 1.0) is now pinned by
  a test so the shared counter space stays VISIBLE, not just
  documented.
- Honest-costs note the 0.3.2 changelog owed: `check_csr` itself is
  free (0.04–0.8%), but the per-edge index guard inside the hot loops
  costs ~7% in bfs — it is also what saves bfs/dijkstra when the
  structural check is bypassed.

### Fixed (round 3)

- **random** — `shuffle`, `partial_shuffle` and
  `philox_partial_shuffle` reject non-1-D input. On a 2-D array the
  element swap operates on row VIEWS, which alias: rows get silently
  DUPLICATED instead of exchanged — a deck sampling the same card
  twice, plausible to the eye, no exception (reproduced). The
  `*sample_without_replacement` wrappers inherit the guard.
- **profiling** — `compile_time` detects a warm on-disk cache: with
  `cache=True` and a populated `__pycache__` the first call loads the
  binary instead of compiling (measured ~40x smaller than a true cold
  compile) — a `RuntimeWarning` now says the number is cache-load
  time, and the docstring documents the regime.
- **profiling** — `ComparisonResult.speedup` (and therefore
  `summary()`) no longer raises `ZeroDivisionError` when the second
  mean is 0.0 — reachable with trivial kernels under `perf_counter`'s
  finite resolution, especially on Windows. Returns `inf` (or `nan`
  when both means are 0), documented.
- **algorithms** — `counting_sort` guards its O(range) term relative
  to n: `range > 16n + 4096` now raises the "use radix_sort" error
  (**behavior change**: previously 2 elements spanning 2**27 silently
  allocated a 1 GiB counts array and scanned 134M entries).
- **random** — corrected the 0.3.1 `philox_uniforms` docstring
  regression: a float32 `out=` buffer does NOT silently truncate — it
  fails loudly at compile time (type unification), like non-1-D
  buffers. The old text asserted the opposite; the behavior is now
  pinned by a test.
- **random** — Philox module docstring documents that the counter
  space is GLOBAL across all `philox_*` functions (`philox_uniforms`
  consumes all four block words, overlapping `philox_uniform` and
  `philox_randint` at equal counters — measured correlation ~1.0,
  invisible to marginal uniformity tests) and that counters wrap
  modulo 2**64 silently. Partition counters once per key.
- **diagnostics** — `check()` no longer shadows the module-level
  `warnings` import with a local list.

### Changed

- docs/numba-cache.md documents the Windows long-path cache-locator
  failure mode (`cannot cache function ... no locator available`) —
  it can fail an entire suite from a mapped path and mislead audits.
- Stronger uniformity test for `philox_partial_shuffle`: all 20
  ordered pairs of (n=5, k=2) equifrequent, catching bias or
  mis-indexing beyond the first swap.

## [0.3.2] - 2026-07-21

Full-audit follow-up (three adversarial reviewers over the whole
library): two memory-safety fixes reachable with VALID input, a
repaired dev-mode safety tool, a compile-time dtype gate, and one
corrected semantic where "matches NumPy" was asserted but false.

### Fixed

- **arrays / parallel** — `histogram` and `parallel_histogram`: a
  subnormal `hi - lo` (realistic trigger:
  `histogram(a, bins, a.min(), a.max())` on nearly-constant data)
  degenerates `scale` so that `0 * inf = NaN -> int(NaN) = INT64_MIN`
  for IN-range values — an out-of-bounds write with valid input. Both
  kernels now clamp `idx < 0` symmetrically with the existing high
  clamp (which also keeps memory safety if a global fastmath override
  ever weakens the NaN filter), and non-finite `lo`/`hi` are rejected
  up front.
- **graph** — the four algorithms validate CSR structure up front
  (`indptr[0] == 0`, `indptr[-1] == len(indices)`, monotonic): a
  malformed `indptr` previously drove out-of-bounds reads and writes
  (reproduced as SIGSEGV) despite the per-entry `indices` checks. O(n)
  next to O(n + m) algorithms — free.
- **decorators** — `boundscheck` dev mode hard-locks `cache=False`
  against global overrides and per-call arguments. Numba's cache key
  ignores `boundscheck`, so a shared on-disk cache poisoned both
  directions: production loading checked binaries (cost), and dev
  loading unchecked binaries — the safety tool silently checking
  nothing. `locked` options are a new internal tier that beats even
  `configure()`/env overrides, used only for safety invariants.
- **algorithms** — `counting_sort` and `radix_sort` reject non-integer
  dtypes at compile time (via a typing-time gate; dtypes are not
  runtime-comparable in nopython). Float input previously truncated
  through the int64 key conversions and FABRICATED output values.
- **stats** — `weighted_quantile` at `q = 0` with a zero-weight
  minimum now matches NumPy for real: NumPy's weighted `inverted_cdf`
  skips zero-weight values even at `q = 0` (verified empirically —
  the 0.3.0 docstring asserted the opposite behavior as "matching
  NumPy"). Zero-weight values are now never selected, at any q.

Also verified this round: the SparseSet churn benchmark reproduces on
the reference machine (4.5x; a reviewer's 0.82x was environment
noise), and `chunked_reduce`'s bit-exactness held under an adversarial
1e12/1e-12 kernel across chunk counts and 1–8 threads.

## [0.3.1] - 2026-07-21

Contract fixes from the second round of user review: the 0.3.0
primitives now compose with each other, two guarantees are stated no
stronger than the code sustains, and one documented statistic was
plain wrong. Two behavior changes, both declared below.

### Added

- **random** — `philox_partial_shuffle` and
  `philox_sample_without_replacement`: the Fisher–Yates primitives
  driven by the stateless Philox stream (consume counters
  `counter..counter+k-1`). 0.3.0 shipped counter-based RNG and the MC
  sampling primitive without a way to combine them; now the headline
  features compose.
- **testing** — `pass_seed=True` on `assert_reproducible` and
  `assert_converges`: each run calls `fn(run_seed, ...)`, making
  counter-based (Philox) kernels testable — they are pure functions of
  their key, so varying global seeds alone gives zero variance.

### Fixed

- **testing** — `assert_converges` documented normal-distribution
  false-positive rates (~0.27% at 3 sigma), but the SE is estimated
  from the runs, so the statistic is Student-t with `n_runs - 1`
  degrees of freedom: the real rates are ~0.55% at the default 30
  runs, ~4% at 5, ~20% at 2. The docstring now carries the correct
  table, and `n_runs < 5` is rejected (**behavior change**: previously
  2+ was accepted).
- **random** — `philox_randint` now consumes word x1 of the Philox
  block (x0 belongs to `philox_uniform`), so drawing both at the same
  `(key, counter)` yields an independent pair instead of perfectly
  correlated noise (**behavior change**: `philox_randint`'s output
  stream differs from 0.3.0).
- **parallel** — `chunked_reduce`'s bit-exactness guarantee is now
  stated with its real precondition in the headline: it holds for
  kernels that are deterministic functions of `(chunk_id, start,
  end)`; a kernel drawing from Numba's per-thread `np.random` loses it
  silently. The test suite now includes the negative case proving the
  divergence, and the serial driver no longer requests on-disk caching
  for its uncacheable closure (spurious `NumbaWarning`).
- **random** — `alias_draw` validates that `prob` and `alias` lengths
  match (mixed-up tables would index out of range — silent corruption
  in nopython); `weighted_sampling` clamps its search result as
  defense in depth against a rounding-mode/RNG-granularity change ever
  producing `u == total` (analytically unreachable today; verified).
- **algorithms** — `combination_table`'s size cap now bounds total
  ELEMENTS (rows × k, ~1 GiB of int64), not just rows — the old cap
  "protected" against a 10 GiB allocation by allowing it.
- **random** — the `np.uint64` requirement for keys/counters `>= 2**63`
  from Python is documented at module level, not just on `philox4x64`;
  `philox_uniforms` documents that its `out=` must be float64 (dtype
  is uncheckable at runtime in nopython; non-1-D fails loudly at
  compile time, now pinned by a test).

## [0.3.0] - 2026-07-21

Phase 3: Monte Carlo primitives, driven by the first real-user
feedback (a production MC equity engine evaluating the library).
Additive only — no breaking changes; upgrading from 0.2.0 is a
drop-in.

### Added

- **random** — counter-based RNG: `philox4x64` (raw Philox4x64-10
  block), `philox_uniform`, `philox_uniforms` (bulk, block-packed,
  `out=`), `philox_randint` (multiply-shift bound, bias < n/2**64,
  documented). Pure functions of `(key, counter)`: reproducible
  regardless of threads, processes or call order — the primitive that
  prevents "result depends on how many workers ran" artifacts. Tests
  assert BIT EQUALITY against `np.random.Philox` (NumPy increments the
  counter before generating; the tests account for the offset).
- **random** — `partial_shuffle` (in-place partial Fisher–Yates: k
  swaps on a reusable scratch array, the zero-allocation repeated-draw
  MC primitive) and `sample_without_replacement` (copying wrapper).
- **algorithms** — `combination_table(n, k)`: the C(n, k) index table
  in lexicographic order; looping over `table.shape[0]` kills the
  hardcoded-combo-count bug class endemic to hand evaluators. Count
  computed over min(k, n-k) so symmetric cases (C(60, 58)) don't trip
  the row cap at their partial-product peak.
- **parallel** — `chunked_reduce`: one per-chunk kernel, serial and
  parallel drivers with bit-identical results (chunk boundaries depend
  only on `(n_items, n_chunks)`, never on thread count; partials merge
  serially in chunk order). Jitted drivers exposed as
  `.serial`/`.parallel`. Pairs with Philox for runs reproducible by
  construction; 11.7x over its own serial driver at 24 threads.
- **testing** — stochastic asserts: `assert_reproducible` (same seed →
  bit-identical across runs) and `assert_converges` (differently
  seeded runs within N standard errors of the truth; zero-variance
  results detected as "seeds not reaching the sampler"; ~0.27%
  false-positive rate at the default 3 sigma, documented).

### Changed

- `njit_fast` docstring now says explicitly that integer-only kernels
  gain nothing from `fastmath` — `cached_njit` is the right pick
  there.

## [0.2.0] - 2026-07-21

Phase 2 complete: two new modules (`graph/`, `stats/`), dtype-generic
collections, and the sorting gaps (`stable_argsort`, `lexsort`).
Additive only — no breaking changes; upgrading from 0.1.2 is a
drop-in.

### Added

- **stats** — new module for numerics whose naive implementations are
  wrong, not just slow: `logsumexp` and `softmax` (max-shifted — the
  direct formulas overflow `exp` beyond ~709; `softmax` supports
  `out=`), and `weighted_quantile` (inverted CDF convention, exact
  match with `np.quantile(..., weights=..., method="inverted_cdf")`;
  fail-fast validation of NaN values and NaN/negative weights).
  Rationale in docs/design/stats.md.
- **graph** — new module: algorithms over CSR adjacency arrays
  (`scipy.sparse.csr_matrix` layout, no Graph class). `edges_to_csr`
  (stable; returns an `order` array to align per-edge payloads),
  `bfs`, `dfs_preorder` (explicit stack, recursive-identical order),
  `topological_sort` (Kahn, deterministic, raises on cycles),
  `dijkstra` (lazy-deletion binary heap, zero allocation in the loop,
  NaN/negative weights rejected up front), and `UnionFind` (jitclass,
  union by size + path compression, `union` reports whether a merge
  happened). Traversals bounds-check `indices` — malformed CSR raises
  instead of corrupting memory. 10–20x over idiomatic pure Python
  (BENCHMARKS.md, Graph section); rationale in docs/design/graph.md.
- **algorithms** — `stable_argsort` and `lexsort`. `stable_argsort` is
  the honest alias for `np.argsort(kind="mergesort")` — the stable kind
  Numba actually compiles (`kind="stable"` does not). `lexsort` is
  `np.lexsort` for nopython code, which Numba doesn't implement:
  a 2-D array of keys, last row primary (NumPy's convention), results
  identical to `np.lexsort`. Both lose to NumPy from Python
  (BENCHMARKS.md explains why) and exist for inside-`@njit` use.
- **collections** — dtype-generic factories `stack_type`,
  `fixed_queue_type`, `ring_buffer_type` and `priority_queue_type`:
  each returns the container specialized to any Numba scalar type
  (`stack_type(int64)`, `priority_queue_type(float32)`, ...), cached
  per type. The float64 classes are now the factories' own
  specializations (`stack_type(float64) is Stack`) — one
  implementation, no behavior change. `priority_queue_type` rejects
  complex types (no ordering).

### Changed

- ROADMAP.md rewritten: Phase 1 closed as shipped (0.1.0–0.1.2, with
  its scope decisions recorded), Phase 2 defined — dtype-generic
  collections, `stable_argsort`/`lexsort`, a `graph/` module, and the
  numerics that pass the identity filter (`logsumexp`, `softmax`,
  `weighted_quantile`).

## [0.1.2] - 2026-07-21

Second audit round: two out-of-bounds-write fixes, one crash fix in a
parallel edge case, and an important correction to the cache kill-switch
documentation. No API changes.

### Fixed

- **arrays / parallel** — `histogram` and `parallel_histogram` now skip
  NaN values. Previously a NaN passed the range filter (NaN fails every
  comparison) and `int(NaN)` — INT64_MIN in nopython mode — indexed one
  element past the counts buffer: a silent out-of-bounds write next to a
  small heap allocation, plus silently wrong counts.
- **algorithms** — `counting_sort` computes its value range as a uint64
  modular distance, so ranges that overflow int64 now raise the friendly
  "range too large, use radix_sort" error. Previously the wrapped range
  either raised a confusing "negative dimensions" error or — for the
  INT64_MIN/INT64_MAX sentinel pair, whose true range wraps to zero —
  crashed the process with an access violation.
- **parallel** — `parallel_topk` clamps chunk starts to the array
  length. With a thread count high relative to the input (threads ≥
  ~√n, e.g. 300 threads on n just above the serial threshold), trailing
  threads computed negative chunk sizes, undersizing the merge buffer
  and corrupting the heap.
- **diagnostics** — `inspect` no longer assumes Numba's private
  `_cache` attribute exists; it degrades to `cache_enabled=False`,
  matching the module's degrade-don't-fail contract.
- **arrays** — `bincount` rejects a negative `minlength` with a clear
  error instead of a bare "negative dimensions not allowed".

### Changed

- **docs/numba-cache.md** — corrected the kill-switch guidance: the
  library's own kernels are decorated during `import numba_utils`, so
  `configure(cache=False)` cannot reach them. The full kill-switch is
  `NUMBA_UTILS_CACHE=0` set before the first import; `configure()`
  covers functions decorated after the call. `diagnostics.check()` and
  the `cached_njit` docstring now say the same.
- NaN caveats documented on the comparison-based float kernels
  (`nth_element`, `fast_argpartition`, `topk`, `insertion_sort`,
  `partial_sort`), and the mutating-arguments caveat on `benchmark`.

## [0.1.1] - 2026-07-20

Audit follow-up: one silent-wrong-answer fix in weighted sampling, plus
packaging and CI corrections. No API changes.

### Fixed

- **random** — `weighted_sampling` and `alias_setup` now reject non-finite
  weights (NaN, ±inf) and sums that overflow to infinity. Previously a NaN
  weight passed the `w < 0` check and silently produced a degenerate
  distribution (every draw returned index 0).

### Changed

- Ship `numba_utils/py.typed` so type checkers honour the annotations
  (the `Typing :: Typed` classifier was already declared).
- Declare the license as a PEP 639 `license = "MIT"` expression plus
  `license-files`, replacing the deprecated table form and classifier.
- CI now tests every supported Python (3.10–3.13), not just 3.11 and 3.13.
- Workflow actions bumped to current majors (`checkout@v7`,
  `setup-python@v7`, `upload-pages-artifact@v5`, `deploy-pages@v5`) —
  the previous pins ran on the deprecated Node 20 runtime.

## [0.1.0] - 2026-07-17

Foundational release. This first public version focuses on the building
blocks that repeatedly appear in production Numba projects. APIs in this
area are considered stable; future releases expand dtype-generic support
and additional algorithms without breaking them.

### Added

- **decorators** — `njit_fast`, `njit_parallel`, `cached_njit`,
  `boundscheck` (dev-mode bounds checking).
- **arrays** — `binary_search`, `lower_bound`, `upper_bound`,
  `fast_clip`, `normalize`, `cumulative_sum` (with `out=`),
  `rolling_sum`, `rolling_mean`, `histogram`, `bincount`,
  `unique_sorted`.
- **algorithms** — `nth_element`, `quickselect`, `fast_argpartition`,
  `topk`, `argmax2`, `insertion_sort`, `partial_sort`, `counting_sort`,
  `radix_sort`.
- **collections** — jitclass containers callable inside `@njit`:
  `Stack`, `FixedQueue`, `RingBuffer`, `PriorityQueue`, `BitSet`,
  `SparseSet`, `ObjectPool`; plus `counter` and `typed_defaultdict`.
- **random** — over Numba's nopython RNG: `seed`, `shuffle`,
  `permutation`, `choice`, `reservoir_sampling`, `weighted_sampling`,
  and the Walker alias method (`alias_setup` / `alias_draw` /
  `alias_sample`).
- **parallel** — complete parallel operations, not `prange` wrappers:
  `parallel_sum`, `parallel_reduce`, `parallel_histogram` (bit-exact),
  `parallel_prefix_sum`, `parallel_topk`, with serial fallback below
  `SERIAL_THRESHOLD`.
- **profiling** — `benchmark` (JIT compilation excluded by default),
  `compare`, `warmup`, `compile_time`, `compile_stats`.
- **diagnostics** — `show`, `check`, `inspect` for compiled functions.
- **testing** — `assert_equivalent`, `assert_close`, `random_arrays`,
  `deterministic_rng`.
- **config** — `configure()` / `config` with global overrides
  (`cache` / `fastmath` / `parallel` / `nogil`) and `NUMBA_UTILS_*`
  environment variables.

[Unreleased]: https://github.com/nicoseijas/numba-utils/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nicoseijas/numba-utils/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nicoseijas/numba-utils/releases/tag/v0.1.0
