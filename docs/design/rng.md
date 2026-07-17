# Design: random

## Why the library ships its own `seed()`

Numba's nopython RNG state is per-thread and completely separate from
NumPy's Python-level state: `np.random.seed(...)` called from Python
does not affect jitted code, which is one of the most common
reproducibility surprises in Numba. `numba_utils.random.seed` is a
jitted function that seeds the state jitted code actually uses;
`testing.deterministic_rng` goes further and pins all three worlds
(NumPy legacy, NumPy `Generator`, Numba) in one call.

## Why `alias_setup` and `alias_draw` are separate

The Walker alias method is O(n) preprocessing + O(1) per draw — that
split IS the algorithm's value. A merged `alias_sample(weights, size)`
would silently turn repeated sampling into O(n) per call, destroying
the property users chose the method for. The API mirrors the cost
model: pay setup once, hold the `(prob, alias)` tables, draw for free.
`alias_sample(prob, alias, size)` exists only as a bulk-draw
convenience over already-built tables.

## Why `weighted_sampling` exists alongside the alias method

Different amortization regimes. `weighted_sampling` (cumulative sums +
binary search, O(n + size·log n)) wins for one-shot sampling or weights
that change between calls — no tables to build or hold. The alias
method wins when many draws share fixed weights. Both docstrings point
at each other; BENCHMARKS.md shows the crossover (weighted_sampling at
parity with NumPy, alias 2.7x with setup amortized).

## Why reservoir sampling (Algorithm R)

Without-replacement sampling that needs one pass, O(k) memory, and no
shuffle of the input — the right primitive when n is huge or streaming.
For with-replacement, `choice` is the direct O(size) answer. The two
cover both replacement regimes with the minimal API for each.
