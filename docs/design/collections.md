# Design: collections

## Why jitclass

Three candidate implementations for stateful containers usable in
nopython code:

1. **Arrays + free functions** (`heap_push(data, size, x) -> new_size`):
   C-style, zero overhead, cacheable — but the caller manually threads
   state through every call, and misuse (stale `size`) corrupts silently.
2. **Generic factories** (`make_priority_queue(dtype)`): flexible, but
   every instantiation is a new class compile, and the API surface
   doubles (factory + instance).
3. **jitclass with fixed dtypes**: real objects, constructible and
   usable inside `@njit` with no hacks, methods at native speed.

jitclass won: it is the framework-native expression of a container, and
the ergonomic difference compounds in user code. Cost accepted: jitclass
methods are not cached on disk (first use pays JIT per process) — noted
in the module docstring.

## Why fixed dtypes (float64 values, int64 indices)

Generic jitclass factories increase API complexity and compilation cost.
The fixed variant covers the large majority of numerical workloads
(scores, priorities, entity ids). Generic variants are planned as a
separate, additive API — shipping them first would have made every user
pay the complexity for the minority need.

## Why `counter` is a function, not a Counter class

Counting is one operation over a typed dict — a class would add
construction ceremony for no capability. Inside the kernel, the
membership pattern (`if key in d: d[key] += 1 else: d[key] = 1`) is used
instead of `d.get(key, 0)` because Numba's type inference cannot resolve
`.get` with a default on a freshly-created `dict()` — this is a
compiler constraint, recorded in a comment at the site.

Honest positioning (also in the docstring and BENCHMARKS.md): for
one-shot counting of a materialized array, sort-based `np.unique` is
faster. `counter` exists for incremental counting inside jitted loops.

## Why `ObjectPool` allocates slots, not objects

There are no Python objects in nopython mode. The useful translation of
an object pool is a **slot allocator**: `acquire()` hands out an index
into whatever preallocated arrays the caller owns, `release()` returns
it, double-release raises. Same lifecycle discipline, expressed in the
terms the runtime actually has.

## Why `SparseSet` exists next to `BitSet`

Both store small-int sets, but `SparseSet.clear()` is O(1) (reset a
counter) versus `BitSet.clear()`'s O(capacity/64) — decisive for
simulation loops that reset membership every iteration. `BitSet` wins
on memory (1 bit vs 16 bytes per universe slot). Two structures, two
documented trade-offs, rather than one compromised one.
