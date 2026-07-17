# Security Policy

## Supported versions

numba-utils is pre-1.0. Security fixes are applied to the latest
released version only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✓         |
| < 0.1   | ✗         |

## Reporting a vulnerability

Please report vulnerabilities **privately** rather than in a public
issue. Open a
[GitHub security advisory](https://github.com/nicoseijas/numba-utils/security/advisories/new),
or email nicoseijas@gmail.com.

You can expect an initial response within a few days. Once a fix is
ready, a patched release is published and the advisory is disclosed.

## Scope

numba-utils is a client-side numerical library with no network,
filesystem, or subprocess surface, and its only runtime dependencies are
NumPy and Numba. The most relevant risks are memory-safety issues in
compiled kernels — Numba's nopython mode has no bounds checking, so an
out-of-bounds access is silent corruption rather than an exception. If
you find an input that drives a kernel out of bounds, that is in scope.
