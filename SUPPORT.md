# Getting help

Thanks for using numba-utils. Here is where each kind of message belongs,
so the right thing gets the right attention.

## Questions and ideas → Discussions

For "how do I...", "is this the right tool for...", performance
questions, or feature ideas, use
[GitHub Discussions](https://github.com/nicoseijas/numba-utils/discussions).
Search first — someone may have asked already.

## Bugs → Issues

Open an [issue](https://github.com/nicoseijas/numba-utils/issues) when
something is wrong. A good report includes:

- what you expected versus what happened;
- a minimal `@njit` snippet that reproduces it;
- your Python, NumPy, and Numba versions (`python -c "import numba, numpy, sys; print(sys.version, numba.__version__, numpy.__version__)"`);
- your OS, and whether the cache was involved (see
  [docs/numba-cache.md](https://github.com/nicoseijas/numba-utils/blob/main/docs/numba-cache.md)).

Before filing, run `diagnostics.check(your_fn)` — it flags the common
Numba pitfalls (cache-across-processes, `fastmath` reproducibility) that
account for a large share of surprises.

## Security → private disclosure

Do not open a public issue for vulnerabilities. See
[SECURITY.md](SECURITY.md).

## Documentation

- [Module reference](https://github.com/nicoseijas/numba-utils/blob/main/docs/modules.md)
- [Runnable examples](https://github.com/nicoseijas/numba-utils/tree/main/examples)
- [Performance and cache notes](https://github.com/nicoseijas/numba-utils/tree/main/docs)
