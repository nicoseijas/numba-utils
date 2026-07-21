# Roadmap

## Founding decisions (2026-07-17)

- **Phase 1 order:** start with `decorators/` + `profiling/`, the
  foundation for developing and benchmarking everything else. Then
  `arrays/` + `algorithms/`. *(Done in that order.)*
- **Targets:** latest stable Numba, Python 3.10+.
- **Publishing:** originally "private until Phase 1 is solid" — Phase 1
  got solid: public on GitHub and released on PyPI as 0.1.0
  (2026-07-17), 0.1.1 (2026-07-20) and 0.1.2 (2026-07-21).
- **Direction:** identity over quantity. Ship functions that are
  genuinely hard to implement well with Numba, or where a clear,
  measured improvement exists — not a grab-bag of one-liners.

## Phase 1 — Foundation ✅ (shipped as 0.1.0–0.1.2)

All eight modules shipped, benchmarked (BENCHMARKS.md) and documented
(docs/): `decorators/`, `profiling/`, `arrays/`, `algorithms/`,
`random/`, `collections/`, `parallel/`, `testing/` — plus the unplanned
extras: global configuration (`configure()` / `NUMBA_UTILS_*`), the
`diagnostics/` module (`show`/`check`/`inspect`) and the permanent
docs knowledge base (performance, numba-cache, parallelism,
benchmarking). Two audit rounds (0.1.1, 0.1.2) hardened validation and
fixed out-of-bounds edge cases. The current API is documented in
docs/modules.md; per-item history lives in CHANGELOG.md.

Scope decisions made along the way:

- **Closed as covered:** `argpartition_topk` (→ `topk` /
  `fast_argpartition`), `Multiset` (→ `counter`), `clamp`
  (→ `fast_clip`).
- **Replaced:** `parallel_range` + work-stealing helpers — `parallel/`
  shipped as complete, engineered operations (sum/reduce/histogram/
  prefix_sum/topk with serial fallbacks) instead of prange wrappers;
  the constraint list that drove this lives in docs/parallelism.md.
- **Deferred by design:** `benchmark_assert` (speedup assertions are
  flaky in CI); `alias_sampler` shipped split as
  `alias_setup`/`alias_draw`/`alias_sample`.
- **Blocked upstream:** Atomics (`atomic_add`/`max`/`min`/`inc`) —
  Numba has no CPU atomics today (CUDA only). Revisit if that changes.
- **Deprioritized:** `math/` and `geometry/` wish lists — most entries
  fail the "genuinely hard in Numba" filter. Survivors move to Phase 2.

## Phase 2 — Depth (in progress)

In value order:

1. **dtype-generic collections** *(in progress)* — factory API
   alongside the fixed-dtype classes, per docs/design/collections.md:
   `stack_type(value_type)`, `fixed_queue_type`, `ring_buffer_type`,
   `priority_queue_type`, returning cached jitclass specializations
   (`stack_type(float64) is Stack`). The float64/int64 defaults stay —
   the generic API is additive, nobody pays for it who doesn't use it.
2. **`stable_argsort`, `lexsort`** ✅ — the two algorithms pending from
   Phase 1. Corrected premise: Numba DOES have a stable argsort
   (`kind="mergesort"`; the `kind="stable"` spelling doesn't compile) —
   the real gap was `np.lexsort`, which Numba doesn't implement at all.
   Shipped as an honest alias plus the composed lexsort; both lose to
   NumPy from Python (BENCHMARKS.md explains why) and exist for inside-
   `@njit` use.
3. **`graph/` module** ✅ — structures that are genuinely painful in
   nopython mode: `edges_to_csr` (with payload-aligning `order`),
   `UnionFind` (jitclass), BFS + DFS preorder over CSR, topological
   sort (Kahn), Dijkstra. Dijkstra got its own lazy-deletion pair heap
   rather than the payload-less `PriorityQueue` originally sketched.
   10–20x over idiomatic pure Python (BENCHMARKS.md, Graph section);
   design decisions in docs/design/graph.md.
4. **Numerics that pass the identity filter** ✅ — the `stats/` module:
   `logsumexp`, `softmax` (max-shifted; the naive formulas are wrong,
   not just slow) and `weighted_quantile` (inverted CDF, exact match
   with NumPy's weighted `np.quantile`; honest loss from Python, wins
   by existing inside `@njit`). docs/design/stats.md.

Phase 2 complete — shipped as 0.2.0.

Rules of engagement, unchanged: every addition arrives with tests
(including edge cases), an entry in BENCHMARKS.md measured against the
full chain (Python → NumPy → Numba → numba-utils), honest docs about
when NOT to use it, and NaN/overflow behavior stated or validated.

## Phase 3 — MC primitives (user-driven, in progress)

First real-user feedback: a production Monte Carlo equity engine
evaluated 0.2.0 for adoption and named five gaps, all squarely inside
the project's pitch. In implementation order:

1. **Counter-based RNG (Philox4x64-10)** — stateless
   `philox_uniform(key, counter)` / `philox_uniforms` /
   `philox_randint`: reproducible streams independent of thread count
   and call order, the primitive that prevents "result depends on how
   many chunks/workers ran" artifacts. NumPy ships the same algorithm
   (`np.random.Philox`), so tests assert EXACT equality against an
   independent reference (offset by one block: NumPy increments the
   counter before generating).
2. **`sample_without_replacement` / `partial_shuffle`** — partial
   Fisher–Yates: k of n without shuffling the whole deck; the in-place
   variant is the zero-allocation repeated-draw MC primitive.
3. **`combination_table(n, k)`** — the C(n,k) index table plus its
   true length in one call; kills the hardcoded-combo-count bug class
   endemic to hand evaluators.
4. **`chunked_reduce`** — one kernel body, serial and parallel
   drivers, **bit-exact across both**: chunk boundaries depend only on
   (n_items, n_chunks), never on thread count; partials merge serially
   in chunk order. Pairs with Philox (chunk id -> counter).
5. **`assert_reproducible` / `assert_converges`** — the stochastic
   complement of `assert_equivalent`: same seed -> bit-identical;
   different seeds -> within N sigma of truth, with the false-positive
   rate documented.

## Later / open questions

- Dtype-generic `SparseSet`/`BitSet` universes beyond int64 indices —
  no demand yet; index dtypes are rarely the bottleneck.
- `parallel/` additions (e.g. parallel sort) — only with a benchmark
  that beats NumPy's SIMD sort by enough to matter.
- Radix-based `stable_argsort` fast path for integer keys — would turn
  the current honest loss vs NumPy's stable (radix) sort into a win;
  only worth it if profiling shows argsort inside kernels matters.
- Block-packed `philox_partial_shuffle` — the current version burns a
  full Philox block per swap and discards 3 of its 4 words; consuming
  all four would put the RNG at ~2.7 ns/draw, below `np.random`
  (auditor's measurement). It changes the function's output stream,
  so it is 0.4.0 material (or an opt-in variant), not a patch.
- CUDA variants — out of scope until a real workload pulls them in.
