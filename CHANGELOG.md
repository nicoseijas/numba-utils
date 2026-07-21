# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
