# Why `parallel=True` made my code slower

Adding `parallel=True` to an `@njit` function feels like it should be
free speedup: same code, one flag, all your cores. Then you benchmark
it, and the parallel version is slower. Not slightly slower — on the
first kernel below it is **15× slower** than the serial version, while
pinning every core on the machine.

This is the most common disappointment in Numba, and it is not a bug.
`parallel=True` is a tool with a break-even point, a bandwidth ceiling,
and a couple of sharp edges. This article measures each one, and ends
with the checklist I run through before enabling the flag.

!!! note "The numbers below are real"
    Every measurement was taken in one session on one machine: 24
    threads, Numba 0.66, NumPy 2.4. Warmup rounds excluded, medians
    reported, results consumed so the compiler can't delete the work
    (see [Why most Numba benchmarks are
    wrong](why-most-numba-benchmarks-are-wrong.md)). Your absolute
    numbers will differ; the shapes and crossover points are the
    lesson.

## The two-line change that backfires

```python
import numpy as np
from numba import njit, prange

@njit
def serial_sum(x):
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i]
    return total

@njit(parallel=True)
def par_sum(x):
    total = 0.0
    for i in prange(x.shape[0]):   # range -> prange
        total += x[i]
    return total
```

Same algorithm, measured across input sizes:

| n          | serial   | parallel | parallel is |
|-----------:|---------:|---------:|:------------|
| 1,000      | 1.0 µs   | 15.0 µs  | **15× slower** |
| 10,000     | 6.5 µs   | 26.8 µs  | **4× slower** |
| 100,000    | 62 µs    | 22 µs    | 2.8× faster |
| 1,000,000  | 632 µs   | 76 µs    | 8.3× faster |
| 20,000,000 | 12.0 ms  | 2.5 ms   | 4.7× faster |

Three different regimes are visible in that table, and each one is a
separate reason your code got slower. Let's take them in order.

## Reason 1: every prange launch synchronizes the whole team

A `prange` region is not "the loop, but concurrent". Entering it means
distributing chunks to a thread team, and leaving it means a **barrier**:
every thread waits until the last one finishes. That choreography has a
fixed cost that does not care how little work is inside the region — on
one production workload we measured it at roughly **0.4 ms per launch**
whether the region ran with 1, 2 or 8 threads.

At n=1,000 the serial loop finishes in one microsecond. The parallel
version spends its entire 15 µs on coordination — the actual summing is
noise. You paid the barrier and bought nothing.

The failure mode that hurts in production is not one small call, it is a
**small parallel region inside a hot loop**: a `prange` kernel invoked
per item, millions of times. The symptom is nasty and specific — CPU
monitors show every core at 100%, yet per-item wall time equals the
single-thread time. The cores are burning on barriers. If you see full
utilization with no speedup, suspect launch overhead before anything
else.

## Reason 2: below a minimum work size, serial always wins

The table's crossover sits between 10,000 and 100,000 elements — for a
kernel this cheap, the thread team needs tens of thousands of elements
just to amortize its own coordination. The crossover moves down if each
iteration does more work, and up if the machine has more threads to
synchronize, but it never disappears.

Two consequences:

- **Don't parallelize on size you haven't checked.** If the function
  also receives small inputs, the flag makes those calls slower.
- **Branch on n.** The robust pattern is a serial kernel and a parallel
  kernel behind one entry point that picks by input size. Every
  operation in `numba_utils.parallel` does exactly this below a shared
  `SERIAL_THRESHOLD` (2^16), because below it the barrier dominates any
  possible gain.

## Reason 3: memory bandwidth is a shared ceiling

Now the strange row at the bottom of the table: 1M elements ran 8.3×
faster in parallel, but 20M — twenty times more work, the *favorable*
direction for parallelism — only 4.7×. On a 24-thread machine. Where
did the cores go?

A sum does almost no arithmetic per byte: load, add, next. At 1M
elements the 8 MB working set is L3-resident and the cache can feed
many cores at once. At 20M elements (160 MB) the kernel streams from
DRAM, and **DRAM bandwidth is shared by all cores**. Once a few threads
saturate the memory bus, additional threads just queue behind it.
Twenty-four cores wait on the same memory controller that four could
saturate.

Compare a kernel that actually computes something per element:

```python
@njit(parallel=True)
def par_heavy(x):
    total = 0.0
    for i in prange(x.shape[0]):
        total += np.sin(x[i]) * np.exp(-x[i] * x[i])
    return total
```

At 20M elements this gets **8.6×** where the plain sum got 4.7× — same
array, same machine, same flag. More arithmetic per byte means the
cores spend their time computing instead of waiting on memory.

The rule of thumb: parallelism multiplies *compute*, not *bandwidth*.
If your loop body is a load and an add, `parallel=True` competes with a
ceiling no amount of threads can raise — and NumPy's vectorized call
may already be near it.

## Reason 4: false sharing — your threads fight over cache lines

Cores don't share memory byte by byte; they move it in 64-byte cache
lines, and only one core can hold a line for writing at a time. If two
threads write to *different* variables that happen to live in the
*same* line, the line ping-pongs between cores on every write. No race,
no wrong answer — just serialized writes with coherence traffic on top.
That is false sharing, and the classic way to trigger it is per-thread
accumulator slots packed next to each other.

A per-thread histogram, 8 bins per thread — each thread's row of counts
is 64 bytes, directly adjacent to its neighbor's:

```python
counts = np.zeros((n_threads, 8), dtype=np.int64)    # rows touch
counts = np.zeros((n_threads, 64), dtype=np.int64)   # rows padded apart
```

Identical algorithm, identical output, 20M elements:

| layout                    | time     |
|:--------------------------|---------:|
| adjacent rows             | 22.9 ms  |
| padded rows (first 8 used)| 7.6 ms   |

**3× faster from padding alone.** Nothing about the computation
changed; the padded version simply guarantees no two threads ever write
the same cache line. This is why `numba_utils.parallel_histogram` pads
its per-thread rows to cache-line boundaries — the pattern "give every
thread private state, then merge serially" only delivers if the private
state is actually private at the cache-line level.

## Reductions: what prange handles, and what it quietly changes

The `total += x[i]` in `par_sum` works because Numba recognizes the
reduction pattern: each thread accumulates privately and the partials
combine at the end. Two things to know before relying on it:

- **A parallel float reduction reorders additions**, so the result can
  differ from serial in the last bits. That is inherent to the
  reordering, not a bug — but it means "diff parallel vs serial" is a
  check you run *knowingly*, expecting exact zeros only for integer and
  order-independent cases.
- **Only touch a reduction variable with reduction operations.** Using
  one as scratch state — mutating it to unrank indices, for example —
  can send the parfor pass tracing reductions until the *compiler*
  raises `RecursionError`. Enumerate into an array with indexed stores
  instead.

Also be aware that exceptions raised inside a `prange` body do not
survive as themselves: they surface as an opaque `SystemError` from the
dispatcher. Validate inputs and capacities *before* the parallel
region, and treat any `SystemError` from a parallel kernel as a masked
exception from inside the loop.

## When prange is the right call

After all of that, `parallel=True` is still the easiest parallelism in
Python — over the *right* loop. The wins share a shape:

- **Coarse iterations.** Each `prange` step is a meaningful unit of
  work (a row, a simulation, a bucket), not three arithmetic ops. One
  launch, thousands of fat iterations.
- **Independent, race-free writes.** Iteration `i` writes `out[i]` and
  nothing else, or accumulates through a recognized reduction. No
  shared mutable state, no "mostly disjoint" index math.
- **Compute-bound bodies.** Enough arithmetic per byte that the memory
  bus isn't the real limit.
- **Large enough n**, with a serial fallback for everything smaller.

And sometimes the right amount of thread-level parallelism is zero:
for farms of independent work items, K single-threaded worker
*processes* (`NUMBA_NUM_THREADS=1` each) routinely beat one process
with a thread team — no barriers, no sharing, no bandwidth fights
inside the process. See [Parallelism](../parallelism.md) for that
layout and the threadpool failure modes that push you toward it.

## Before enabling `parallel=True`, ask yourself…

1. **Is there enough work per launch?** At least tens of thousands of
   elements, or iterations fat enough to dwarf the launch barrier? A
   parallel region invoked per item in a hot loop is a slowdown
   machine.
2. **Do small inputs also hit this function?** If yes, branch: serial
   kernel below a threshold, parallel above.
3. **Is the loop body compute-bound or a memory stream?** If it's
   load-add-store, measure against serial *and* NumPy first — the
   memory bus may already be the ceiling.
4. **Does any thread write within 64 bytes of another thread's data?**
   Per-thread slots and accumulator rows must be cache-line padded.
5. **Are all cross-iteration interactions recognized reductions?**
   Reduction variables used only as reductions; everything else
   written to disjoint indices.
6. **Can anything inside the loop raise?** Move validation before the
   region; a raise inside `prange` surfaces as a bare `SystemError`.
7. **Did you diff parallel vs serial output on real data?** Bit-exact
   for integers and order-independent algorithms; known, accepted
   last-bit drift for float reductions.
8. **Did you measure with warmup excluded and results consumed?**
   Otherwise you don't know yet whether the flag helped at all.

If a kernel clears the list, `prange` will pay you close to core-count
on compute-bound work. If it doesn't, serial code is not a failure —
it's the faster option, and the honest benchmark will say so.

The operations in
[numba-utils](https://github.com/nicoseijas/numba-utils)'s `parallel`
module — `parallel_sum`, `parallel_histogram`, `parallel_prefix_sum`,
`parallel_topk`, `chunked_reduce` — exist because this checklist kept
producing the same answers: serial fallback under `SERIAL_THRESHOLD`,
cache-line-padded per-thread state merged serially, documented float
semantics. They package the judgment, not just the loop.

---

*The production rules behind this article:
[Parallelism](../parallelism.md). Why the module ships finished
operations instead of prange wrappers:
[Design: parallel](../design/parallel.md). How the numbers were
measured: [Benchmarking JIT-compiled code](../benchmarking.md).*
