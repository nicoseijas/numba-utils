# Design decisions

Not manuals — records of *why it was done this way and not another way*.
For users who want to trust the library's judgment, and contributors who
would otherwise re-litigate (or silently break) a settled trade-off.

- [cache.md](cache.md) — why caching defaults on, and why the global
  override beats explicit per-call arguments
- [collections.md](collections.md) — why jitclass, why fixed dtypes,
  why `counter` is a function
- [rng.md](rng.md) — why our own `seed()`, why the alias method is a
  split API
- [parallel.md](parallel.md) — why complete operations instead of prange
  wrappers, and every threshold in them
- [benchmarking.md](benchmarking.md) — why the tooling measures the way
  it does, and what CI deliberately doesn't report

Each document states the decision, the alternatives considered, and the
reason the alternative lost. If a future change invalidates a reason,
the decision is open again — that's what these files are for.
