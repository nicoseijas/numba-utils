# Design: parallel

## Why complete operations instead of prange wrappers

A `parallel_range` wrapper would add a name over `prange` and nothing
else — worse, it would invite exactly the fine-grained usage that loses
to serial code while pinning every core. The valuable thing to package
is not the loop construct but the **pattern**: chunking, private
per-thread state, merge phase, and the judgment of when not to
parallelize at all. So the module ships finished operations
(`parallel_sum`, `parallel_histogram`, `parallel_prefix_sum`,
`parallel_topk`, `parallel_reduce`) with those patterns built in.

## Why every operation has a serial fallback (`SERIAL_THRESHOLD`)

Each prange launch synchronizes the whole thread team — measured at
~0.4 ms per launch regardless of thread count on one production
workload. Below tens of thousands of elements the barrier dominates any
gain, so every operation checks `n < SERIAL_THRESHOLD` (2^16) and runs
the serial path. The threshold is a heuristic, deliberately a single
shared constant: tunable in one place, not twenty knobs.

## Why per-thread private state, padded, merged serially

`parallel_histogram` gives each thread its own row of counts, padded to
64-byte boundaries. Alternatives: atomics (Numba has no CPU atomics),
or unpadded rows (adjacent threads write the same cache line — false
sharing turns parallel counting into cache-line ping-pong). The serial
merge is O(threads·bins), negligible, and makes the result **bit-exact**
with the serial histogram — which the tests assert.

## Why `parallel_topk` never pads short chunks with sentinels

The tempting implementation pads a short chunk's candidate slots with a
sentinel (or a repeated chunk value) so every chunk contributes exactly
k. Repeating real values corrupts duplicates: a value appearing once in
the input can appear twice among candidates and twice in the answer.
Instead each chunk contributes `min(k, chunk_len)` candidates and a
count array drives the merge. There is a regression test for exactly
this (`test_duplicates_not_inflated`).

## Why the decorator was renamed `njit_parallel`

The original roadmap had both a `@parallel` decorator and a `parallel/`
module. In Python, a package attribute can't reliably be both a
function and a submodule — import order decides which one wins. The
decorator became `njit_parallel` (consistent with the `njit_fast`
family), freeing the `parallel` name for the module that carries the
actual value. Done pre-release, when renames are free.

## Why float results may differ from serial in the last bits

Parallel reduction reorders floating-point addition; that is inherent,
not a bug. The module docstring says so, integer paths are tested for
exactness, and `parallel_histogram` — where exactness is achievable —
promises and tests bit-exactness. Where it isn't achievable, the
contract says so instead of pretending.
