# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nicoseijas/numba-utils/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nicoseijas/numba-utils/releases/tag/v0.1.0
