# Five lessons from running Numba in production

Numba's demo is magical: add `@njit`, get C speed. Production is where
the footnotes live. These five lessons come from long-running compute
farms — multi-process Monte Carlo workloads on big machines — and each
one cost real debugging time before it became a rule. All five have
the same shape: a symptom that looks like *your* bug, a cause that
lives in the runtime, and a boring discipline that makes it never
happen again.

## 1. The on-disk cache can segfault another process

`cache=True` is the right default for scripts: compile once, reuse the
binary across runs. But the cache assumes a stored binary is safe for
*any* process to load, and on some setups — multi-process worker
farms, shared filesystems, container layers — a binary compiled by one
process **intermittently crashes another** that loads it. Access
violation, no traceback, no pattern.

The failure signature is distinctive, and recognizing it saves weeks:

- the first run (the one that compiled) works; later runs crash,
- crashes look random — same code, same data, different outcome,
- deleting `__pycache__` "fixes" it… until the cache repopulates,
- it happens even on pure, non-parallel kernels.

If that pattern matches, stop debugging your kernel. It's the cache.
Disable it globally for that environment and pay the recompile.

One subtlety worth internalizing: cache settings are typically read
**at decoration time**. A code-level "disable cache" call cannot reach
functions that were decorated while the library imported — an
environment variable set *before the first import* is the only switch
that covers everything. If you build a kill-switch, build that one.

## 2. The process can crash after your work is done

A rarer cousin: the parallel runtime's threadpool can segfault during
**teardown**, at interpreter shutdown — after every result was
computed correctly. The process does all its work, writes its output,
and dies with an access violation on the way out.

The consequence is operational, and it changes how you build harnesses:
**verify runs by their output artifact, not their exit code.** A
pipeline that treats nonzero exit as failure will re-run (or worse,
discard) perfectly good work. Check that the expected file exists and
parses; treat the exit code of a Numba-heavy process as advisory.

## 3. `prange` charges you per launch, not per element

Every `parallel=True` region launch synchronizes the whole thread team
— a barrier costing roughly the same (~0.4 ms measured, on one
machine) whether the region does a million elements or a hundred. A
tiny parallel region called millions of times is therefore **slower
than serial code while pinning every core**. That's the trap: CPU
monitors show 100% utilization, so it *looks* like parallelism is
working; per-item wall time says otherwise.

Two rules follow. Parallelize coarse, independent, race-free units —
one `prange` over big chunks, never a `prange` inside a per-item
function. And for farms of independent tasks, prefer **process-level
parallelism**: K workers with `NUMBA_NUM_THREADS=1` each routinely
beats one process with K threads, sidesteps the launch barrier
entirely, and (bonus) turns lesson 2's crashes into disposable-worker
noise. Long-lived workers amortize compilation; a subprocess per task
pays JIT every time.

## 4. There is no bounds checking, and NaN knows it

Nopython kernels compile without bounds checks. An out-of-range index
doesn't raise — it reads or writes *somewhere*. The nastiest source of
out-of-range indices is not arithmetic bugs; it's **NaN**, because NaN
fails every comparison and therefore slides through range filters
written the intuitive way:

```python
if x < lo or x > hi:      # NaN: both False -> falls through
    continue
idx = int((x - lo) * scale)   # int(NaN) == INT64_MIN in nopython
counts[idx] += 1              # silent out-of-bounds write
```

In compiled code `int(NaN)` is INT64_MIN (the CPU conversion's
sentinel), and an unchecked `counts[INT64_MIN]` lands one element past
the buffer — a silent heap write. The same blindness applies to
validation: `w < 0` accepts NaN weights and corrupts whatever is built
from them. The disciplines: write range filters inverted
(`if not (lo <= x <= hi)`, which rejects NaN), validate with
`isfinite` rather than sign checks, and run development builds with
`boundscheck=True` so corruption becomes `IndexError` while you can
still afford it.

## 5. Never answer `RecursionError` with `setrecursionlimit`

Two separate traps share this lesson. First, Numba's own compiler can
recurse pathologically — certain reduction patterns in parallel loops
make the parfor analysis trace until *the compiler itself* raises
`RecursionError`. The fix is restructuring the kernel (indexed stores
into an array instead of mutating a reduction variable), not raising
limits. Second, recursion in your own kernels runs on the **native**
stack: Python's recursion limit doesn't govern it, so
`sys.setrecursionlimit(10**6)` trades a clean Python exception for a
native stack overflow — a hard crash. Deep traversals in compiled code
want explicit stacks over preallocated arrays, with a bounded-depth
assert doubling as a cycle detector.

## The meta-lesson

None of these are Numba bugs you can wait out; they are properties of
running a JIT compiler and a threadpool inside CPython processes at
scale. The teams that do this well all converge on the same posture:
recognize failure signatures instead of re-deriving them, encode each
one as a default (a kill-switch, a harness rule, a code-review
pattern), and keep kernels boring.

That posture is why
[numba-utils](https://github.com/nicoseijas/numba-utils) exists: its
[cache page](../numba-cache.md) and
[parallelism page](../parallelism.md) are these lessons in reference
form, its `diagnostics.check()` warns when a function's configuration
matches a known failure mode, and its kernels ship with the NaN and
validation disciplines above already applied. The library is the
residue of the debugging.

---

*Related: [Why most Numba benchmarks are wrong](why-most-numba-benchmarks-are-wrong.md) ·
[Your coverage tool can't see your fastest code](coverage-cant-see-your-kernels.md)*
