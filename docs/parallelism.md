# Parallelism

`parallel=True` + `prange` is the easiest parallelism in Python — and
the easiest to lose money on. These notes come from production Numba
workloads (large multi-process compute farms); they are the design
constraints for the future `numba_utils.parallel` module.

## The launch barrier: why fine-grained prange loses

Every `prange` region launch synchronizes the whole thread team. That
barrier costs roughly the same regardless of how much work the region
does — measured on one workload at ~0.4 ms whether run with 1, 2 or 8
threads over a region of a few thousand rows.

Consequence: a **tiny parallel region called millions of times** is
slower than serial code, while still pinning every core. The symptom is
nasty: CPU monitors show full utilization, but per-item wall time equals
the single-thread time. The cores are burning on barriers.

**Rule: parallelize coarse, independent, race-free units.** A `prange`
over hero buckets where each iteration writes its own output slot scales
near-linearly. A `prange` inside an evaluator called per item does not.

## Process-level parallelism often beats thread-level

For farms of independent work items, the winning layout is usually:

- `NUMBA_NUM_THREADS=1` per worker process,
- K single-threaded workers running in parallel at the **process** level,
- each worker long-lived (JIT is paid once per process — see
  [performance.md](performance.md)).

This also sidesteps a real failure mode: on some setups, **repeatedly
launching parallel regions in one process crashes the threadpool**. If a
pass must re-launch a parallel kernel many times, run each pass in its
own (retryable) subprocess.

## Structural rules

- **Co-locate a parallel kernel with its `prange` driver** in the same
  module, and call it from Python. Cross-module `njit -> njit` calls
  into a parallel region have produced segfaults in the wild.
- **At small per-call N, prefer the serial kernel** and make `prange`
  opt-in for genuinely large N. Thread-spawn overhead dominates tiny
  workloads.
- **Don't mutate a reduction variable to unrank indices inside
  `prange`** — the parfor pass can recurse tracing reductions until the
  compiler itself raises `RecursionError`. Enumerate into an array with
  indexed stores instead. And never raise `sys.setrecursionlimit` to
  silence a Numba `RecursionError`: Numba doesn't honor it, and deep
  native recursion is a crash, not a catchable exception.

## Verify numerics

A parallel reduction reorders floating-point operations. Before trusting
a parallel kernel, diff its output against the serial version on real
data — same-machine serial vs multi-thread should be within (ideally at)
zero difference for order-independent algorithms, and you want to KNOW
the difference for reductions rather than assume it.

`diagnostics.check(fn)` flags `parallel=True` functions with a summary
of these caveats.

## The `numba_utils.parallel` module

These rules are embodied as complete operations, not prange wrappers:
`parallel_sum`, `parallel_reduce` (per-index kernel decorator),
`parallel_histogram` (per-thread private rows, cache-line padded, merged
serially — bit-exact with serial), `parallel_prefix_sum` (two-phase
blocked scan) and `parallel_topk` (per-chunk heaps, merge). Every one
falls back to the serial path below `SERIAL_THRESHOLD`, where the launch
barrier would dominate.
