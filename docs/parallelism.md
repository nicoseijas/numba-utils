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

A related upstream Numba bug: the threadpool teardown at interpreter
shutdown can segfault (`0xC0000005` on Windows) **after all work has
completed correctly**. The crash lives entirely inside Numba's threading
layer while the process exits; user code already finished and its output
is valid. Two consequences:

- **Verify runs by their output artifact, not their exit code.** A
  harness that treats a nonzero return code as failure will flag
  successful runs as broken. Check that the expected file was written or
  the expected output was produced instead.
- Mitigations are the same as above: `NUMBA_NUM_THREADS=1` per worker,
  disposable worker processes (where an exit-time crash is harmless),
  and optionally a different `NUMBA_THREADING_LAYER` — stability varies
  per machine, so treat a layer switch as an experiment, not a fix.

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
- **Exceptions raised inside `prange` don't survive as themselves.**
  The same overflow that raises `ValueError('Stack: full')` in serial
  code surfaces from a parallel region as `SystemError:
  CPUDispatcher(...) returned a result with an exception set` — a
  Numba limitation, and it applies to every raise in this library's
  collections (`Stack`/`FixedQueue` full, `SparseSet` capacity,
  `BitSet` bounds) exactly where you'd use them per-thread. Validate
  capacities BEFORE the parallel region (or size containers so the
  raise is unreachable), and treat any `SystemError` from a parallel
  kernel as a masked exception from inside the loop, not as a Numba
  bug to report.

## Verify numerics

A parallel reduction reorders floating-point operations. Before trusting
a parallel kernel, diff its output against the serial version on real
data — same-machine serial vs multi-thread should be within (ideally at)
zero difference for order-independent algorithms, and you want to KNOW
the difference for reductions rather than assume it.

`diagnostics.check(fn)` flags `parallel=True` functions with a summary
of these caveats.

## Locating an intermittent segfault

Nopython code has no bounds checking, so an out-of-bounds read is
undefined behavior: it *sometimes* crashes and sometimes returns
garbage. The garbage case is the dangerous one — if the same buggy
kernel feeds both sides of a comparison, garbage agrees with garbage
and the test passes. A "flaky segfault" that shows up on some runs and
not others is usually this, not the threadpool.

To locate it, run under **`python -X faulthandler`** (or call
`faulthandler.enable()` at startup). On the crash it prints the exact
`file:line` of the access violation — which distinguishes an OOB read
in your kernel from a threadpool-teardown crash (the one that fires at
interpreter exit, after the work is done) in seconds. Without it, both
look like the same opaque `0xC0000005`.

In a test runner that has finished its real work but risks a teardown
segfault poisoning the exit code, `os._exit(0)` after flushing output
skips Numba's threadpool teardown deterministically. Use it only once
results are safely written — it bypasses `atexit` handlers and buffer
flushing, so flush first.

## Two patterns worth knowing (not shipped as API)

These come from a production CFR solver and are documented rather than
packaged: each is safe only under conditions the caller must own, so a
library function would either hide the precondition or over-constrain
the use. They belong in your kernel, with the caveat attached.

### Hogwild: lock-free racing accumulation

A `prange` over independent tasks, each updating shared accumulator
arrays (regrets, strategies) with **racing, unsynchronized adds** — no
locks, no atomics. Individual `+=` updates can be lost to the race, yet
in the solver this gave ~60x with convergence *unchanged* (the
correlation to the reference fixed point actually strengthened).

The precondition is everything: this is safe only when the algorithm
tolerates lossy updates because it is **iterative and
self-correcting** — a dropped regret increment is re-accumulated on the
next visit, and the fixed point is an average over many iterations.
Apply Hogwild to a computation that needs every write (a histogram, an
exact sum) and it silently produces wrong counts. That is exactly why
`numba_utils.parallel_histogram` uses per-thread private rows merged
serially instead: same shape, opposite requirement. Know which one you
have before racing.

### Factorized independent-opponent aggregation

Replacing a Θ(B^P) joint over P independent factors with a per-item
**product of factor CDFs**, Θ(items·B·P) — linear in P instead of
exponential. In the solver, a P-way all-in pot share that looked like
it needed a joint bucket tensor `share[i,j,k,...]` factorizes, given
independent opponent reaches, into a product of per-opponent
beat/tie CDFs integrated over `[0,1]` (the tie identity
`1/(K+1) = ∫₀¹ xᴷ dx` handles ties exactly). The exponential "wall" was
an artifact of precomputing the joint tensor instead of folding the
factors in at aggregation time.

The precondition is **independence** of the factors. Where it holds,
5/6/7-way is as cheap as 3-way; where the factors are correlated the
product is wrong and there is no cheap fix. A related anti-pattern from
the same solver, kept as a warning: a "fast" prefix-sum variant that
dropped the hero-opponent removal term was rejected at ~44% error —
the shortcut changed the answer, and only a dense cross-check caught
it. Certify a factorized kernel against the dense joint on *random*
(not uniform) inputs before trusting it.

## The `numba_utils.parallel` module

These rules are embodied as complete operations, not prange wrappers:
`parallel_sum`, `parallel_reduce` (per-index kernel decorator),
`parallel_histogram` (per-thread private rows, cache-line padded, merged
serially — bit-exact with serial), `parallel_prefix_sum` (two-phase
blocked scan) and `parallel_topk` (per-chunk heaps, merge). Every one
falls back to the serial path below `SERIAL_THRESHOLD`, where the launch
barrier would dominate.
