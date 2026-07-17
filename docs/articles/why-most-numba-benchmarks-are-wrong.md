# Why most Numba benchmarks are wrong

Numba makes it trivially easy to measure the wrong thing. The same
`@njit` decorator that gives you a 40× speedup can, with a
one-line benchmarking mistake, report a 40× *slowdown* — or a fictional
"200 million evaluations per second" that evaporates the moment the
function runs inside a real application.

This is not a Numba flaw. It is what happens when you point a naive timer
at a compiler. Below are the five mistakes that make almost every casual
Numba benchmark untrustworthy, with real numbers, and the small set of
rules that fix them.

!!! note "The numbers below are real"
    Every measurement here was taken on one laptop with a specific NumPy
    and Numba version. Yours will differ — that is exactly the point of
    the reproducibility section. Treat the *ratios* and *shapes* as the
    lesson, not the absolute milliseconds.

## The kernel we'll measure

A single fused pass: clip each value and accumulate its square. NumPy has
to allocate two temporaries (the clip result, then its square) and sweep
memory three times; the jitted version does it in one pass.

```python
from numba import njit
import numpy as np

@njit(fastmath=True)
def clipped_energy(values, lo, hi):
    total = 0.0
    for i in range(values.shape[0]):
        x = values[i]
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        total += x * x
    return total

def numpy_clipped_energy(values, lo, hi):
    return float(np.sum(np.clip(values, lo, hi) ** 2))
```

On 2,000,000 floats, measured honestly, the kernel runs in **0.137 ms**
versus NumPy's **6.20 ms** — a **45×** win. Now watch how easy it is to
report anything but that number.

## Mistake 1 — Timing the first call

The first call to a jitted function compiles it. On this machine,
compiling `clipped_energy` takes about **237 ms**. The kernel itself runs
in 0.137 ms.

```python
import time

t0 = time.perf_counter()
clipped_energy(values, -1.0, 1.0)   # compiles here!
print(time.perf_counter() - t0)     # ~0.237 s
```

If your benchmark times that first call, you conclude the jitted function
takes 237 ms and is **~40× slower than NumPy**. The exact same code that
is 45× faster now looks like a catastrophe — you measured the compiler,
not the kernel.

**Fix:** always run uncounted warmup rounds before timing. Compile once,
throw the number away, then measure.

```python
clipped_energy(values, -1.0, 1.0)   # warmup: discard
# ...now start the clock
```

## Mistake 2 — Letting the compiler delete your work

This is the subtle one, and it inflates *wins* instead of losses. A tight
loop whose result nobody reads is an invitation for the optimizer to hoist
loop-invariant work, fold constants, or eliminate the computation
entirely:

```python
@njit
def bench(values):
    for _ in range(1000):
        clipped_energy(values, -1.0, 1.0)   # result discarded
```

The compiler can legally notice that the loop body has no observable
effect and run it once — or never. You measure "200 million evals/sec"
and report a number that has nothing to do with your application, where
the result is actually consumed.

Real workloads routinely show an order-of-magnitude gap between a
micro-benchmark ("millions of evals/sec") and in-application throughput
(tens of thousands). If a number looks too good, **assume the compiler
deleted your work until proven otherwise** — accumulate results into a
value you return and print.

## Mistake 3 — An unfair or unrealistic baseline

Two failure modes here:

- **Strawman baselines.** Benchmarking against a pure-Python loop instead
  of the vectorized NumPy call the user would actually write. The baseline
  must be the *best standard tool for the job*.
- **Unrealistic inputs.** Tiny arrays that live in L1 cache, dtypes nobody
  uses, or a call pattern no real program runs. A kernel that wins on
  1,000 elements can lose on 50,000,000 when it becomes bandwidth-bound.

The corollary: sometimes NumPy wins, and an honest suite says so. NumPy's
SIMD sort will beat a hand-written radix sort on full-range keys.
Publishing that loss is what tells a user *when not to reach for your
function* — which saves them more time than any win.

## Mistake 4 — Reporting a mean and nothing else

A single mean hides instability. If variance is large relative to the
mean, your measurement is noise-dominated and the number is an anecdote.
Always report **median and variance** alongside the mean; if they
disagree, increase iterations, pin input sizes up, and isolate the
machine before trusting anything.

## Mistake 5 — Publishing only the runs you won

A benchmark suite that contains only wins is marketing, not engineering.
The losing rows are the most useful rows: they are the ones that change a
reader's decision. Keeping them, next to the wins, in the same table, is a
costly signal that the winning numbers can be trusted too.

## Doing it right

The rules, condensed:

1. **Warm up first.** Never time the call that compiles.
2. **Consume the result.** Return and print it so the optimizer can't
   delete the work.
3. **Realistic baseline and inputs.** Best standard tool, real sizes, real
   dtypes, results used.
4. **Report distribution, not a point.** Mean, median, variance, min/max.
5. **Pin and publish for reproducibility.** Seed, sizes, iteration counts,
   and Python / NumPy / Numba versions.
6. **Keep the losses.**

I got tired of re-implementing this checklist in every project, so it
lives as a function in
[numba-utils](https://github.com/nicoseijas/numba-utils). `compare()`
runs the warmup rounds, times both implementations on identical inputs,
and returns mean / median / variance / speedup — so the JIT never
pollutes the numbers and you see the distribution, not a single figure:

```python
from numba_utils import compare

result = compare(
    numpy_clipped_energy, clipped_energy,
    args=(values, -1.0, 1.0), n=21,
)
print(result.summary())
# clipped_energy vs numpy_clipped_energy: 45.27x
# (warmup rounds excluded; mean / median / variance reported per side)
```

None of this is exotic. It is just the difference between measuring your
kernel and measuring your compiler — and once you have been burned by a
"40× slowdown" that was really 237 ms of compilation, you never point a
naive timer at `@njit` again.

---

*More on the project's benchmarking rules:
[Benchmarking JIT-compiled code](../benchmarking.md). Why there is no
line-coverage badge to match:
[Testing](../testing.md).*
