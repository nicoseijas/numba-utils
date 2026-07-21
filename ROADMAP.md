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
4. **Numerics that pass the identity filter** — `logsumexp`, `softmax`
   (stability-critical), `weighted_quantile` (no NumPy equivalent).

Rules of engagement, unchanged: every addition arrives with tests
(including edge cases), an entry in BENCHMARKS.md measured against the
full chain (Python → NumPy → Numba → numba-utils), honest docs about
when NOT to use it, and NaN/overflow behavior stated or validated.

## Later / open questions

- Dtype-generic `SparseSet`/`BitSet` universes beyond int64 indices —
  no demand yet; index dtypes are rarely the bottleneck.
- `parallel/` additions (e.g. parallel sort) — only with a benchmark
  that beats NumPy's SIMD sort by enough to matter.
- Radix-based `stable_argsort` fast path for integer keys — would turn
  the current honest loss vs NumPy's stable (radix) sort into a win;
  only worth it if profiling shows argsort inside kernels matters.
- CUDA variants — out of scope until a real workload pulls them in.
