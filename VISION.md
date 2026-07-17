# Vision

**numba-utils** — High-performance building blocks for Numba.

## Philosophy

Everything that any Numba developer ends up writing over and over again should live here.

It does not compete with Numba: it builds on top of it.

- No magic.
- No hiding of internals.
- Just make writing efficient code much more comfortable.

## Goals

- Reduce boilerplate.
- Provide pre-optimized algorithms.
- Offer reusable data structures.
- Make profiling and benchmarking easy.
- Encapsulate common patterns.
- Stay compatible with `@njit`.

## Priorities

1. Performance first.
2. Small APIs.
3. No unnecessary dependencies.
4. NumPy compatibility.
5. Everything benchmarked.
6. Everything typed.
7. Everything documented with time complexity.

## Key differentiator

There are plenty of libraries with algorithms. Very few teach **how to write efficient code for Numba**. That is the gap to fill: the repository must also be an educational resource.

Every module should include a **"Performance Notes"** section covering things like:

- When Numba actually speeds up a routine.
- Cases where NumPy is still faster.
- JIT compilation cost and strategies to amortize it.
- Common mistakes (Python objects, dynamic lists, boxing, etc.).
- The impact of `parallel=True`, `fastmath=True` and `cache=True`.
- Examples of patterns that scale well and patterns that don't.

Over time, that turns numba-utils not just into a useful library, but into a reference for any developer who wants to write high-performance code with Numba.
