# Design: caching

## Why `cache=True` is the default

Numba's own recommendation, and what the majority of users (scripts,
notebooks, single-process services) want: compile once, load from disk
forever after. The environments where the cache is dangerous —
multi-process farms, network filesystems, ephemeral containers (see
[../numba-cache.md](../numba-cache.md)) — are real but specific, and a
library API should not be designed around its worst-case environment.

**Alternative considered:** cache off by default, opt-in. Rejected:
punishes the common case to protect a case that has an explicit,
documented escape hatch.

## Why the global override beats explicit per-call arguments

`configure(cache=False)` / `NUMBA_UTILS_CACHE=0` wins even over
`@cached_njit(cache=True)`. Deliberate, and the opposite of the usual
"most specific wins" rule: the override exists as an **environment
policy tool** — "this machine cannot safely use the cache" — and a
policy that any call site can silently defeat is not a policy. The
operator deploying to the broken environment outranks the author of any
one decorator line.

## Why some parallel functions declare `cache=False` themselves

`parallel_histogram`, `parallel_prefix_sum` and `parallel_topk` trip
Numba's "dynamic globals" cache limitation after the parfor transform
(via `get_num_threads`), and `parallel_reduce` drivers capture a kernel
in a closure, which Numba cannot cache. Left alone, each would emit a
NumbaWarning on every user's first compile. Declaring `cache=False`
with a comment converts an unavoidable limitation into a documented,
silent decision.

## Why `@boundscheck` dev mode disables the cache

Dev builds compile with `boundscheck=True` and are transient by
definition — caching them buys nothing and writes dev artifacts into
the same `__pycache__` namespace production builds read from. Keeping
dev compilation fully in-memory is free and removes a class of "which
binary am I actually running?" questions.
