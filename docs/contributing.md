# Contributing

Contributions are welcome. The bar is specific and non-negotiable in two
places: **benchmarks are mandatory, and honesty is policy.** The full
rules live in
[GUIDELINES.md](https://github.com/nicoseijas/numba-utils/blob/main/GUIDELINES.md);
this page is the short version.

## Before you add a function

Every addition must solve a *recurring* problem for Numba users, and must
answer one of the three [benchmark-honesty](benchmark-honesty.md)
questions — faster, similar-but-more-ergonomic, or slower-but-unique.
"It seems faster" is not accepted; a reproducible table is.

## What a PR must include

- A benchmark with a pinned seed, input size, and iteration count, added
  to `benchmarks/` and reflected in `BENCHMARKS.md` — losing rows
  included.
- Tests built on **reference validation**, not just self-consistency:
  compare against an independent NumPy or pure-Python implementation.
  See [Testing](testing.md) for why this is the only trustworthy
  coverage in nopython mode.
- A docstring following the fixed structure: what, why, complexity,
  memory, example, benchmark, limitations, related functions.

## Design decisions

Non-obvious trade-offs are recorded in
[design records](design/README.md) — read the relevant one before
changing behavior around caching, parallelism, collections, or RNG, and
add a new record when you make a decision worth remembering.

## Running the suite

```bash
.venv/Scripts/python -m pytest        # POSIX: .venv/bin/python
```

For the machine-specific cache crash, set `NUMBA_UTILS_CACHE=0` (see
[Installation](getting-started/installation.md)).

## Reporting bugs and asking questions

See
[SUPPORT.md](https://github.com/nicoseijas/numba-utils/blob/main/SUPPORT.md):
questions go to Discussions, bugs to Issues, vulnerabilities to a private
advisory.
