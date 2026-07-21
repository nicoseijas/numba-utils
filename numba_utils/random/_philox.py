"""Counter-based RNG: Philox4x64-10, stateless.

Why counter-based: a stateful global RNG makes Monte Carlo results
depend on execution order — how many chunks ran, on how many threads,
in what sequence. A counter-based generator is a pure function
``(key, counter) -> randomness``: stream ``key``, index ``counter``,
same numbers every time, no matter who computes which chunk where.
Assign each work unit its own counter range and the run is
reproducible by construction.

This is the exact algorithm behind ``np.random.Philox`` (Philox4x64
with 10 rounds), and the test suite asserts bit-identical output
against it. One layout note: NumPy increments the counter *before*
generating, so ``philox4x64(k0, k1, c + 1, 0, 0, 0)`` equals the first
raw block of ``np.random.Philox(counter=[c, 0, 0, 0], key=[k0, k1])``.
"""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit

_M0 = np.uint64(0xD2E7470EE14C6C93)
_M1 = np.uint64(0xCA5A826395121157)
_W0 = np.uint64(0x9E3779B97F4A7C15)
_W1 = np.uint64(0xBB67AE8584CAA73B)
_MASK32 = np.uint64(0xFFFFFFFF)
_SHIFT32 = np.uint64(32)
_SHIFT11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53


@cached_njit
def _mulhi64(a, b):
    # High 64 bits of the 128-bit product, composed from 32-bit limbs
    # (no native uint128 in nopython).
    a_lo = a & _MASK32
    a_hi = a >> _SHIFT32
    b_lo = b & _MASK32
    b_hi = b >> _SHIFT32
    lo_lo = a_lo * b_lo
    hi_lo = a_hi * b_lo
    lo_hi = a_lo * b_hi
    cross = (lo_lo >> _SHIFT32) + (hi_lo & _MASK32) + lo_hi
    return a_hi * b_hi + (hi_lo >> _SHIFT32) + (cross >> _SHIFT32)


@cached_njit
def philox4x64(key0, key1, c0, c1, c2, c3):
    """One raw Philox4x64-10 block: four uint64 words for counter
    ``(c0, c1, c2, c3)`` under key ``(key0, key1)``.

    The full-width primitive; the ``philox_*`` helpers below cover the
    common single-key single-counter cases. When calling from Python,
    pass values ``>= 2**63`` as ``np.uint64`` — Numba's dispatcher
    types plain Python ints as int64. Complexity: 10 rounds of integer
    math, no memory traffic.
    """
    k0 = np.uint64(key0)
    k1 = np.uint64(key1)
    x0 = np.uint64(c0)
    x1 = np.uint64(c1)
    x2 = np.uint64(c2)
    x3 = np.uint64(c3)
    for _ in range(10):
        hi0 = _mulhi64(_M0, x0)
        lo0 = _M0 * x0
        hi1 = _mulhi64(_M1, x2)
        lo1 = _M1 * x2
        x0 = hi1 ^ x1 ^ k0
        x1 = lo1
        x2 = hi0 ^ x3 ^ k1
        x3 = lo0
        k0 += _W0
        k1 += _W1
    return x0, x1, x2, x3


@cached_njit
def philox_uniform(key, counter):
    """Uniform float64 in ``[0, 1)`` for stream ``key`` at index
    ``counter`` — a pure function, no state anywhere.

    Same ``(key, counter)`` always gives the same number, regardless of
    threads, processes or call order. Give each work unit disjoint
    counters (e.g. ``counter = chunk_id * draws_per_chunk + i``) and
    the whole run is reproducible by construction.

    Cost: one Philox block per call (three of its four words unused —
    the price of the simple contract; for bulk generation use
    :func:`philox_uniforms`).
    """
    x0, _, _, _ = philox4x64(
        np.uint64(key), np.uint64(0), np.uint64(counter),
        np.uint64(0), np.uint64(0), np.uint64(0),
    )
    return np.float64(x0 >> _SHIFT11) * _INV53


@cached_njit
def philox_randint(key, counter, n):
    """Uniform int64 in ``[0, n)`` for stream ``key`` at index ``counter``.

    Multiply-shift bounding: bias is below ``n / 2**64`` — negligible
    for any practical ``n``, and documented rather than hidden. Raises
    ``ValueError`` for ``n < 1``.
    """
    if n < 1:
        raise ValueError("philox_randint: n must be >= 1")
    x0, _, _, _ = philox4x64(
        np.uint64(key), np.uint64(0), np.uint64(counter),
        np.uint64(0), np.uint64(0), np.uint64(0),
    )
    return np.int64(_mulhi64(x0, np.uint64(n)))


@cached_njit
def philox_uniforms(key, counter, size, out=None):
    """``size`` uniform float64 in ``[0, 1)`` for stream ``key``
    starting at block index ``counter`` — the bulk version, using all
    four words of each block.

    Consumes ``ceil(size / 4)`` counter values: the next independent
    call should start at ``counter + ceil(size / 4)``. Layout is fixed
    (word ``i % 4`` of block ``counter + i // 4``), so results are
    reproducible and splittable across workers.
    """
    if size < 0:
        raise ValueError("philox_uniforms: size must be >= 0")
    if out is None:
        out = np.empty(size, np.float64)
    if out.shape[0] != size:
        raise ValueError("philox_uniforms: out has wrong length")
    k = np.uint64(key)
    c = np.uint64(counter)
    zero = np.uint64(0)
    i = 0
    while i < size:
        x0, x1, x2, x3 = philox4x64(k, zero, c, zero, zero, zero)
        out[i] = np.float64(x0 >> _SHIFT11) * _INV53
        if i + 1 < size:
            out[i + 1] = np.float64(x1 >> _SHIFT11) * _INV53
        if i + 2 < size:
            out[i + 2] = np.float64(x2 >> _SHIFT11) * _INV53
        if i + 3 < size:
            out[i + 3] = np.float64(x3 >> _SHIFT11) * _INV53
        i += 4
        c += np.uint64(1)
    return out
