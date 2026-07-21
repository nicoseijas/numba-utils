# Your coverage tool can't see your fastest code

Here is an uncomfortable measurement. A library where every compiled
kernel is tested against an independent reference implementation — 275
tests, randomized inputs, edge cases, error paths — reports **~44%
line coverage**. Not because half the code is untested, but because
the coverage tool physically cannot observe the half that matters.

If you ship Numba (or Cython, or anything else that compiles Python
away), your coverage number is systematically lying to you, and the
obvious fixes make it lie in the opposite direction. This article is
about what to measure instead.

## Why the tracer goes blind

`coverage.py` works by hooking CPython's tracing machinery: every
executed line of *bytecode* fires an event. But `@njit` functions
don't run bytecode. Numba lowers them to LLVM IR and executes native
machine code; the Python function object is just a launcher. From the
tracer's point of view, the body of every kernel — the code your users
actually execute, the code with the pointer arithmetic and the
hand-rolled loops — **never runs**.

So a Numba codebase has three different things one could call
"coverage":

1. coverage of the Python wrapper layer (decorators, config, argument
   validation done in Python),
2. coverage of Python-level logic (diagnostics, test generators),
3. coverage of the nopython kernels.

Standard tooling measures 1 and 2 and reports 0% for 3 — and 3 is the
product.

## The tempting fix that measures the wrong program

The well-known workaround is `NUMBA_DISABLE_JIT=1`: run the suite with
compilation off, the kernels execute as plain Python, the tracer sees
every line, the badge turns green.

The number is now real — for a program you don't ship. Interpreted
Python and nopython-compiled code are *not* semantically identical,
and the differences cluster exactly where bugs live. One example from
a real audit: in nopython mode, `int(float("nan"))` is
**-9223372036854775808** (INT64_MIN — that's what the underlying CPU
conversion produces); in interpreted Python the same expression raises
`ValueError`. A histogram kernel that computed `int((x - lo) * scale)`
on data containing NaN passed every test under `NUMBA_DISABLE_JIT=1` —
and in compiled mode silently wrote 8 bytes one element past its
counts buffer. The green badge certified the fallback, not the
artifact.

## Executed is not correct — especially without bounds checking

There is a deeper problem than visibility. Line coverage certifies
that a line *ran*, not that it did the right thing — and nopython code
has **no bounds checking**. An off-by-one that would raise
`IndexError` in Python is a silent out-of-bounds write in a kernel.
The line executes, gets counted as covered, and corrupts memory
without a traceback. Coverage is weakest precisely where compiled code
is most dangerous.

## What to guarantee instead

For numerical code, the strong claim is not "every line ran" — it is
"**the kernel agrees with an independent implementation on inputs
designed to break it**". That is checkable, and it is what a testing
strategy for compiled kernels should be built around:

- **Reference validation.** Every kernel is compared against NumPy, a
  pure-Python implementation, or the standard library (`heapq`, `set`)
  on the same inputs. Not similar-looking — equal, or within a stated
  tolerance with the tolerance justified.
- **Adversarial input generation.** Random cases plus the edges that
  actually break kernels: empty, single element, constant, sorted both
  ways, duplicates, NaN, extreme sentinels (`INT64_MIN` in the same
  array as `INT64_MAX` has crashed real code).
- **Error-path tests.** Every documented `ValueError`/`IndexError` is
  raised on cue. In a bounds-check-free world, validation guards are
  load-bearing; test them like features.
- **Bit-exactness where promised.** If a parallel kernel claims
  bit-identical results to its serial version, assert equality, not
  closeness.
- **Bounds checking in development.** Numba can compile with
  `boundscheck=True` — slow, but it turns silent corruption into an
  `IndexError` during development and CI.

A results table of these guarantees communicates more than any
percentage — and unlike a coverage badge, it cannot be inflated by
testing the interpreter instead of the product.

## The policy

This is ultimately an honesty question, the testing twin of honest
benchmarking: **never publish a metric you know is systematically
misleading, even when it's the metric everyone expects.** A 44% badge
would tell users the kernels are untested (false); a
`NUMBA_DISABLE_JIT` 95% badge would tell them the shipped binaries are
verified (also false). The correct move is to publish neither and
document what is actually guaranteed.

The machinery described here ships in
[numba-utils](https://github.com/nicoseijas/numba-utils):
`assert_equivalent` runs reference-vs-kernel comparisons with per-call
array copies (mutating kernels can't contaminate the comparison) and
fails on an accidentally-empty generator instead of passing vacuously;
`random_arrays` yields the edge cases above; the project's own
[testing page](../testing.md) is the guarantee table this article
argues for.

---

*Related: [Why most Numba benchmarks are wrong](why-most-numba-benchmarks-are-wrong.md) ·
[Testing strategy](../testing.md)*
